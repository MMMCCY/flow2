"""External offline evaluation drives the preregistered finite development sequence."""
import argparse
import os
import statistics
import subprocess
import sys
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.stage21v2.common import *
from scripts.stage21.common import validate_registry, load_conditions
from scripts.stage21.run import initial_noise
from scripts.stage21.evaluate import voxel_metrics, connected, match_objects


def verified_receipt(folder, registry):
    receipt = read_json(folder / "receipt.json")
    sealed = read_json(folder / "completion_receipt.json")
    if (sealed["receipt_sha256"] != file_hash(folder / "receipt.json") or receipt["status"] != "completed"
        or receipt["source_sha256"] != registry["source_sha256"]
        or receipt["protocol_sha256"] != registry["protocol_sha256"]):
        raise ValueError("sealed prediction receipt mismatch")
    verify_tensor(receipt["prediction"])
    return receipt


def reuse_baseline(case, seed, registry):
    old = read_json(OLD / "inference_registry.json")
    validate_registry(old)
    if old["source_sha256"] != source_hashes():
        raise ValueError("old sampling source differs")
    state = read_json(OLD / "formal/sampling_state.json")
    receipt = read_json(OLD / "formal/completion_receipt.json")
    if (state["status"] != "completed" or not state["base_model_unchanged"]
        or receipt["state_sha256"] != file_hash(OLD / "formal/sampling_state.json")
        or receipt["records_sha256"] != json_hash(state["records"])
        or state["binding"]["registry_file_sha256"] != file_hash(OLD / "inference_registry.json")):
        raise ValueError("old completion not sealed")
    if (state["environment"]["torch"] != torch.__version__
        or state["environment"]["cuda"] != torch.version.cuda
        or state["environment"]["cudnn"] != torch.backends.cudnn.version()
        or state["environment"]["gpu"] != torch.cuda.get_device_name(0)):
        raise ValueError("old inference environment differs")
    orig = load_conditions(old["conditions"][case]["9"])
    new = check_condition(registry["conditions"][case]["A_9"])
    if (not torch.equal(orig["condition_values"],new["values"])
        or not torch.equal(orig["condition_mask"],new["mask"])):
        raise ValueError("baseline conditions differ")
    r = next(r for r in state["records"] if (r["case_id"],r["source_seed"],r["well_count"]) == (case,seed,9))
    if r["condition_record_sha256"] != json_hash(old["conditions"][case]["9"]):
        raise ValueError("old condition binding")
    path = OLD / "formal" / r["prediction_path"]
    asset = dict(path=str(path.relative_to(REPO)),file_sha256=r["prediction_file_sha256"],tensor_sha256=r["prediction_tensor_sha256"])
    pred = verify_tensor(asset)
    if ((pred != new["values"]) & new["mask"]).any():
        raise ValueError("baseline violations")
    noise_path = OLD / "formal/noise" / f"seed{seed}.pt"
    noise = load_tensor(noise_path)
    if (file_hash(noise_path) != r["initial_noise_file_sha256"]
        or tensor_hash(noise) != r["initial_noise_tensor_sha256"]
        or not torch.equal(noise, initial_noise(seed,noise.shape[1],(64,64,64)))):
        raise ValueError("old source noise mismatch")
    trace_path = OLD / "formal" / r["trace_path"]
    trace = read_json(trace_path)
    if (file_hash(trace_path) != r["trace_file_sha256"] or len(trace) != 32
        or any(abs(t["time"]-(i+.5)/32) > 1e-8 for i,t in enumerate(trace))):
        raise ValueError("old Euler time grid")
    return dict(status="completed", case_id=case,source_seed=seed,well_count=9,arm="A",
                prediction=asset, reused=True, velocity_evaluations=0, historical_velocity_evaluations=32,
                initial_noise_tensor_sha256=r["initial_noise_tensor_sha256"],
                base_hash_before=state["base_hash_before"],base_hash_after=r["base_hash_after"],
                original_condition_violations=0, added_condition_violations=0,
                elapsed_seconds=0, source_sha256=registry["source_sha256"],protocol_sha256=registry["protocol_sha256"],
                reuse_audit=dict(old_registry_sha256=file_hash(OLD / "inference_registry.json"),
                                 old_state_sha256=file_hash(OLD / "formal/sampling_state.json"),
                                 old_record=r, checkpoint_sha256=CHECKPOINT_SHA,
                                 source_exact=True, conditions_exact=True, noise_exact=True,
                                 environment_exact=True, fp32_deterministic_32_midpoint_euler=True,
                                 original_prediction_pointer_only=True))


def metrics(pred, truth, domain, classes):
    r = voxel_metrics(pred,truth,domain,classes)
    r["target_absolute_volume_error"] = abs(r["target_pred_volume"]-r["target_truth_volume"])
    for label in range(14):
        a,b = (pred == label)&domain,(truth == label)&domain
        inter,union,tn,pn = int((a&b).sum()),int((a|b).sum()),int(b.sum()),int(a.sum())
        r.update({f"class_{label}_intersection":inter,f"class_{label}_union":union,
                  f"class_{label}_truth_voxels":tn,f"class_{label}_pred_voxels":pn,
                  f"class_{label}_iou":inter/union if tn else None,
                  f"class_{label}_precision":inter/pn if tn and pn else None,
                  f"class_{label}_recall":inter/tn if tn else None})
    return r


class Experiment:
    def __init__(self,root):
        self.root = root
        self.registry = validate_inference(read_json(root / "inference_registry.json"))
        self.ev = read_json(root / "evaluation_registry.json")
        self.plan = read_json(root / "layout.json")
        if sources() != self.registry["source_sha256"]:
            raise ValueError("frozen source changed")
        prep = read_json(root / "preparation_receipt.json")
        if (file_hash(root / "evaluation_registry.json") != prep["evaluation_sha256"]
            or file_hash(root / "inference_registry.json") != prep["registry_sha256"]
            or json_hash(read_json(root / "PROTOCOL.json")) != prep["protocol_sha256"]):
            raise ValueError("preparation seal mismatch")
        self.samples=[]
        self.decisions=[]
        self.skips=[]
        self.case_cache={}

    def case(self,case):
        if case not in self.case_cache:
            ev = self.ev["cases"][case]
            truth = verify_tensor(ev["truth"])
            orig = check_condition(self.registry["conditions"][case]["A_9"])
            domains = {name:(verify_tensor(r["tensor"]),r["classes"]) for name,r in ev["domains"].items()}
            self.case_cache[case]=(truth,orig,domains)
        return self.case_cache[case]

    def flow(self,case,seed,arm,n):
        folder=self.root / "predictions" / case / f"seed{seed}" / f"{arm}_{n}"
        if folder.exists():
            r=verified_receipt(folder,self.registry)
        else:
            r=None
            if arm == "A":
                try:
                    r=reuse_baseline(case,seed,self.registry)
                except ValueError as exc:
                    self.decisions.append(dict(action="baseline_recompute",case=case,seed=seed,reason=str(exc)))
                if r:
                    folder.mkdir(parents=True)
                    write_json(folder / "receipt.json",r)
                    write_json(folder / "completion_receipt.json",dict(receipt_sha256=file_hash(folder / "receipt.json")))
            if r is None:
                cmd=[sys.executable,str(PROJECT / "scripts/stage21v2/run.py"),"--registry",str(self.root / "inference_registry.json"),
                     "--case",case,"--seed",str(seed),"--arm",arm,"--n",str(n),"--out",str(folder)]
                log=self.root / "logs" / f"{case}_seed{seed}_{arm}_{n}.log"
                log.parent.mkdir(exist_ok=True)
                with log.open("x") as stream:
                    subprocess.run(cmd,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,check=True)
                r=verified_receipt(folder,self.registry)
        if (r["case_id"],r["source_seed"],r["arm"],r["well_count"]) != (case,seed,arm,n):
            raise ValueError("receipt identity mismatch")
        pred=verify_tensor(r["prediction"])
        c=check_condition(self.registry["conditions"][case][f"{arm}_{n}"])
        if ((pred != c["values"]) & c["mask"]).any():
            raise ValueError("condition violations")
        if arm != "A":
            a=next(s for s in self.samples if (s["case_id"],s["source_seed"],s["arm"]) == (case,seed,"A"))
            if (r["initial_noise_tensor_sha256"] != a["initial_noise_tensor_sha256"]
                or r["base_hash_after"] != a["base_hash_after"]):
                raise ValueError("paired noise/model mismatch")
        row=dict(case_id=case,source_seed=seed,arm=arm,well_count=n,prediction=r["prediction"],
                 receipt_path=str((folder / "receipt.json").relative_to(REPO)),
                 receipt_sha256=file_hash(folder / "receipt.json"),
                 velocity_evaluations=r["velocity_evaluations"],elapsed_seconds=r["elapsed_seconds"],
                 reused=r.get("reused",False),initial_noise_tensor_sha256=r["initial_noise_tensor_sha256"],
                 base_hash_after=r["base_hash_after"])
        self.evaluate(row,pred)
        print(f"Evaluated {case} seed{seed} {arm}_{n}: holdout mIoU={row['metrics']['E_holdout']['fixed_class_miou']:.6f}",flush=True)
        return row

    def evaluate(self,row,pred):
        case=row["case_id"]
        truth,orig,domains=self.case(case)
        row["metrics"]={}
        for name,(domain,classes) in domains.items():
            row["metrics"][name]=metrics(pred.numpy()[0,0],truth.numpy()[0,0],domain.numpy()[0,0],classes)
        self.samples.append(row)
        write_json(self.root / "execution_state.json",dict(status="running",samples=self.samples,decisions=self.decisions,skips=self.skips))

    def offline(self,case,seed,n,a):
        orig=self.case(case)[1]
        conditions=check_condition(self.registry["conditions"][case][f"C_{n}"])
        baseline=verify_tensor(a["prediction"])
        copied=copy_only(baseline,conditions["values"],conditions["mask"])
        held=self.case(case)[2]["E_holdout"][0]
        if not torch.equal(copied[held],baseline[held]):
            raise ValueError("COPY_ONLY changed holdout")
        folder=self.root / "offline" / case / f"seed{seed}" / f"D_{n}.pt"
        asset=self.offline_asset(folder,copied)
        self.evaluate(dict(case_id=case,source_seed=seed,well_count=n,arm="D",prediction=asset,
                           velocity_evaluations=0,elapsed_seconds=0,copy_holdout_equals_A=True),copied)

    @staticmethod
    def offline_asset(path,value):
        if path.exists():
            old=load_tensor(path)
            if not torch.equal(old,value):
                raise ValueError("offline resume mismatch")
            return dict(path=str(path.relative_to(REPO)),file_sha256=file_hash(path),tensor_sha256=tensor_hash(value))
        return save_tensor(path,value)

    def property(self,case):
        if any(r["case_id"] == case and r["arm"] == "P" for r in self.samples):
            return
        asset=self.ev["cases"][case]["pseudo"]
        self.evaluate(dict(case_id=case,source_seed=None,well_count=None,arm="P",prediction=asset,
                           velocity_evaluations=0,elapsed_seconds=0),verify_tensor(asset))

    def improves(self,b,a,domain="E_holdout",metric="fixed_class_miou",equal=False):
        bv,av=b["metrics"][domain][metric],a["metrics"][domain][metric]
        return bv is not None and av is not None and (bv >= av if equal else bv > av)

    def candidate(self,c,a):
        return (self.improves(c,a) and self.improves(c,a,"E_original_free",equal=True)
                and self.improves(c,a,"E_original_free","target_iou"))

    def execute(self):
        torch.set_num_threads(4)
        case=CASES[0]
        self.property(case)
        a=self.flow(case,42,"A",9)
        candidate=None
        for n in LEVELS:
            b=self.flow(case,42,"B",n)
            self.offline(case,42,n,a)
            if not self.improves(b,a):
                self.skips.append(dict(case=case,seed=42,arm="C",n=n,reason="B holdout mIoU not strictly above A"))
                continue
            c=self.flow(case,42,"C",n)
            passed=self.candidate(c,a)
            self.decisions.append(dict(case=case,seed=42,n=n,stage="screen",candidate=passed,
                                       A_receipt_sha256=a["receipt_sha256"],B_receipt_sha256=b["receipt_sha256"],C_receipt_sha256=c["receipt_sha256"]))
            if passed:
                candidate=n
                for higher in LEVELS:
                    if higher > n:
                        self.skips.append(dict(case=case,seed=42,n=higher,arm="B/C",reason="first candidate; higher density untested"))
                break
        confirmed=False
        paired=[]
        if candidate:
            for seed in (142,242):
                aa=self.flow(case,seed,"A",9)
                self.flow(case,seed,"B",candidate)
                self.flow(case,seed,"C",candidate)
                self.offline(case,seed,candidate,aa)
            for seed in (42,142,242):
                aa=next(r for r in self.samples if (r["case_id"],r["source_seed"],r["arm"]) == (case,seed,"A"))
                cc=next(r for r in self.samples if (r["case_id"],r["source_seed"],r["arm"],r["well_count"]) == (case,seed,"C",candidate))
                paired.append(dict(source_seed=seed,
                    holdout_miou=cc["metrics"]["E_holdout"]["fixed_class_miou"]-aa["metrics"]["E_holdout"]["fixed_class_miou"],
                    original_free_miou=cc["metrics"]["E_original_free"]["fixed_class_miou"]-aa["metrics"]["E_original_free"]["fixed_class_miou"],
                    original_free_target_iou=cc["metrics"]["E_original_free"]["target_iou"]-aa["metrics"]["E_original_free"]["target_iou"]))
            confirmed=(all(r["holdout_miou"]>0 for r in paired)
                       and statistics.median(r["original_free_miou"] for r in paired)>0
                       and statistics.median(r["original_free_target_iou"] for r in paired)>0)
            if confirmed:
                for case in CASES[1:]:
                    self.property(case)
                    aa=self.flow(case,42,"A",9)
                    self.flow(case,42,"B",candidate)
                    self.flow(case,42,"C",candidate)
                    self.offline(case,42,candidate,aa)
            else:
                self.skips.append(dict(arm="cross_case_A/B/C",reason="development confirmation inconsistent; stop with mixed evidence"))
        else:
            self.skips.append(dict(arm="confirmation_and_cross_case",reason="no candidate in three fixed screening levels"))
        # Enumerate every unexecuted Flow configuration in the finite protocol.
        done={(s["case_id"],s["source_seed"],s["arm"],s["well_count"]) for s in self.samples}
        for case in CASES:
            for seed in ((42,142,242) if case == CASES[0] else (42,)):
                for arm,n in [("A",9), *[(arm,n) for n in LEVELS for arm in "BC"]]:
                    if (case,seed,arm,n) not in done:
                        self.skips.append(dict(case=case,seed=seed,arm=arm,n=n,reason="not reached under fixed progression; no result inferred"))
        result=dict(status="completed",samples=self.samples,decisions=self.decisions,skips=self.skips,
                    candidate_density=candidate,confirmation_passed=confirmed,paired_differences=paired,
                    flow_outputs=sum(s["arm"] in "ABC" and not s.get("reused",False) for s in self.samples),
                    reused_baselines=sum(s.get("reused",False) for s in self.samples),
                    velocity_evaluations=sum(s["velocity_evaluations"] for s in self.samples),
                    elapsed_seconds=sum(s["elapsed_seconds"] for s in self.samples))
        if result["flow_outputs"] > 19 or result["velocity_evaluations"] > 608:
            raise ValueError("network budget exceeded")
        write_json(self.root / "execution_state.json",result)
        self.tables(result)

    def tables(self,result):
        rows,objects,pred_objects,exposures=[],[],[],[]
        for sample in self.samples:
            case=sample["case_id"]
            truth,orig,domains=self.case(case)
            pred=verify_tensor(sample["prediction"])
            ti,_=connected(truth.numpy()[0,0] == 9)
            pi,_=connected(pred.numpy()[0,0] == 9)
            identity={k:sample[k] for k in ("case_id","source_seed","arm","well_count")}
            for name,(domain,classes) in domains.items():
                oo,pp,ss,mat=match_objects(ti,pi,domain.numpy()[0,0],OBJECT_RULES)
                row={**identity,"domain":name,"domain_sha256":tensor_hash(domain),
                     "fixed_classes":str(classes),**sample["metrics"][name],**ss}
                rows.append(row)
                for obj in oo:
                    mask=torch.from_numpy(ti == obj["object_id"]).reshape_as(truth)
                    full=int(mask.sum())
                    hit=int((mask & orig["original_mask"]).sum())
                    extra=0 if sample["well_count"] in (None,9) else int((mask & well_mask(self.plan["levels"][str(sample["well_count"])]) & ~orig["original_mask"]).sum())
                    objects.append({**identity,"domain":name,**obj,"full_voxels":full,
                                    "major":full>=20,"original_M9_hits":hit,
                                    "original_zero_M9_major":hit==0 and full>=20,"new_well_hit_voxels":extra})
                pred_objects.extend({**identity,"domain":name,**r} for r in pp)
                name_seed="none" if sample["source_seed"] is None else str(sample["source_seed"])
                path=self.root / "overlap_matrices" / f"{case}_{name_seed}_{sample['arm']}_{sample['well_count']}_{name}.npz"
                path.parent.mkdir(exist_ok=True)
                if not path.exists():
                    np.savez_compressed(path,**mat)
            # Confusion of actual added pseudo conditions, independently of whether C ran.
        for case in {s["case_id"] for s in self.samples}:
            truth,orig,domains=self.case(case)
            raw=verify_tensor(self.ev["cases"][case]["pseudo_raw"])
            for n in LEVELS:
                added=well_mask(self.plan["levels"][str(n)]) & ~orig["original_mask"]
                for true_label in range(-1,14):
                    for pseudo_label in range(14):
                        count=int(((truth==true_label)&(raw==pseudo_label)&added).sum())
                        exposures.append(dict(case_id=case,well_count=n,true_label=true_label,pseudo_label=pseudo_label,count=count))
            confs={"full_nonair":truth!=-1,**{k:v[0] for k,v in domains.items()}}
            for name,domain in confs.items():
                conf=np.zeros((15,15),dtype=np.int64)
                t=truth[domain].long().numpy()+1
                p=raw[domain].long().numpy()+1
                np.add.at(conf,(t,p),1)
                write_json(self.root / "confusion" / f"{case}_{name}.json",dict(labels=list(range(-1,14)),matrix=conf.tolist(),
                           domain_voxels=int(domain.sum()),pseudo_raw_sha256=tensor_hash(raw)))
        write_csv(self.root / "sample_metrics.csv",rows)
        write_csv(self.root / "object_metrics.csv",objects)
        write_csv(self.root / "predicted_object_metrics.csv",pred_objects)
        write_csv(self.root / "pseudo_well_confusion.csv",exposures)
        write_csv(self.root / "paired_differences.csv",result["paired_differences"] or [dict(status="not reached")])
        write_csv(self.root / "execution_checklist.csv",[
            {**{k:s[k] for k in ("case_id","source_seed","arm","well_count","velocity_evaluations","elapsed_seconds")},
             "status":"reused" if s.get("reused") else "executed","prediction_file_sha256":s["prediction"]["file_sha256"]}
            for s in self.samples]+[{**r,"status":"skipped"} for r in self.skips])


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True,type=Path)
    args=parser.parse_args()
    Experiment(args.root.resolve()).execute()
