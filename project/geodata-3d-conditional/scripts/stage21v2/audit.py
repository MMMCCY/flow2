"""Independent delivery checks on conditions, pairing, sealed assets and fixed domains."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.stage21v2.common import *


def audit(root):
    reg=validate_inference(read_json(root / "inference_registry.json"))
    state=read_json(root / "execution_state.json")
    ev=read_json(root / "evaluation_registry.json")
    plan=read_json(root / "layout.json")
    if json_hash(plan) != json_hash(layout()) or state["status"] != "completed":
        raise ValueError("layout/completion")
    if sources() != reg["source_sha256"] or file_hash(REPO / CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError("source/checkpoint changed")
    if json_hash(read_json(root / "PROTOCOL.json")) != reg["protocol_sha256"]:
        raise ValueError("protocol seal")
    originals={}
    for case in CASES:
        original=check_condition(reg["conditions"][case]["A_9"])
        originals[case]=original
        truth=verify_tensor(ev["cases"][case]["truth"])
        expected=evaluation_domains(truth,original["mask"],plan["holdout"])
        for name,domain in expected.items():
            registered=ev["cases"][case]["domains"][name]
            if not torch.equal(domain,verify_tensor(registered["tensor"])):
                raise ValueError("fixed domain")
            classes=sorted(int(x) for x in truth[domain].unique())
            if classes != registered["classes"] or int(domain.sum()) != registered["voxels"]:
                raise ValueError("domain classes/count")
        pseudo=verify_tensor(ev["cases"][case]["pseudo"])
        for n in LEVELS:
            b,c=(check_condition(reg["conditions"][case][f"{a}_{n}"]) for a in "BC")
            if not torch.equal(b["mask"],c["mask"]):
                raise ValueError("B/C masks")
            if (b["mask"] & expected["E_holdout"]).any():
                raise ValueError("holdout conditioned")
            for ts,labels in ((b,truth),(c,pseudo)):
                added=ts["mask"] & ~original["mask"]
                if not torch.equal(ts["values"][added],labels[added].to(ts["values"].dtype)):
                    raise ValueError("added condition accuracy/declared pseudo")
    baselines={}
    for s in state["samples"]:
        if s["arm"]=="A":
            baselines[(s["case_id"],s["source_seed"])]=s
    count=0
    for s in state["samples"]:
        prediction=verify_tensor(s["prediction"])
        case=s["case_id"]
        if s["arm"] in "ABC":
            path=REPO / s["receipt_path"]
            receipt=read_json(path)
            if file_hash(path) != s["receipt_sha256"] or receipt["status"] != "completed":
                raise ValueError("sample receipt changed")
            if read_json(path.parent / "completion_receipt.json")["receipt_sha256"] != file_hash(path):
                raise ValueError("sample seal")
            ts=check_condition(reg["conditions"][case][f"{s['arm']}_{s['well_count']}"])
            if ((prediction != ts["values"]) & ts["mask"]).any():
                raise ValueError("hard-condition violation")
            if not receipt.get("reused"):
                if receipt["base_hash_before"] != receipt["base_hash_after"] or receipt["velocity_evaluations"] != 32:
                    raise ValueError("model/budget")
                if file_hash(path.parent / "trace.json") != receipt["trace_sha256"]:
                    raise ValueError("trace hash")
                trace=read_json(path.parent / "trace.json")
                if len(trace)!=32 or any(t["time"] != (i+.5)/32 for i,t in enumerate(trace)):
                    raise ValueError("Euler times")
                noise=verify_tensor(receipt["noise"])
                if tensor_hash(noise) != receipt["initial_noise_tensor_sha256"]:
                    raise ValueError("noise asset")
                count+=32
            a=baselines[(case,s["source_seed"])]
            if (s["initial_noise_tensor_sha256"] != a["initial_noise_tensor_sha256"]
                or s["base_hash_after"] != a["base_hash_after"]):
                raise ValueError("paired model/noise")
        elif s["arm"]=="D":
            a=verify_tensor(baselines[(case,s["source_seed"])]["prediction"])
            domain=verify_tensor(ev["cases"][case]["domains"]["E_holdout"]["tensor"])
            if not torch.equal(prediction[domain],a[domain]):
                raise ValueError("COPY_ONLY holdout")
        # Independent fixed-domain per-class arithmetic, without calling the evaluator.
        truth=verify_tensor(ev["cases"][case]["truth"])
        for name,d in ev["cases"][case]["domains"].items():
            domain=verify_tensor(d["tensor"])
            scores=[]
            for label in d["classes"]:
                p=(prediction==label)&domain
                t=(truth==label)&domain
                inter,union=int((p&t).sum()),int((p|t).sum())
                score=inter/union
                scores.append(score)
                if s["metrics"][name][f"class_{label}_intersection"] != inter or s["metrics"][name][f"class_{label}_union"] != union:
                    raise ValueError("class counts")
            if abs(float(np.mean(scores))-s["metrics"][name]["fixed_class_miou"])>1e-12:
                raise ValueError("fixed mIoU")
    if count != state["velocity_evaluations"] or count > 608:
        raise ValueError("total budget")
    for rel,h in read_json(root / "report/artifact_hashes.json").items():
        if file_hash(root / rel) != h:
            raise ValueError("report artifact hash")
    audit=dict(status="passed",new_velocity_evaluations=count,new_flow_outputs=count//32,
               reused_baselines=state["reused_baselines"],fixed_domains_independently_verified=True,
               class_intersections_unions_independently_recomputed=True,
               COPY_ONLY_holdout_exact=True,original_and_added_conditions_exact=True,
               paired_noise_and_model_exact=True,checkpoint_and_sampling_sources_unchanged=True,
               source_sha256=sources())
    write_json(root / "DELIVERY_AUDIT.json",audit)
    print(audit["status"],flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True,type=Path)
    audit(parser.parse_args().root.resolve())
