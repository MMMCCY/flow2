"""Offline evaluation of sealed preprocessing; external gates allow at most 1+2 Flow outputs."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.stage22.common import (
    PROJECT,REPO,OLD,CASE,RULES,guard,utc,source_hashes,read_json,write_json,write_csv,
    file_hash,json_hash,tensor_hash,verify_tensor,save_tensor,check_condition,
)
from scripts.stage21.evaluate import voxel_metrics,connected,match_objects
import torch


def evaluate(root):
    guard("evaluate")
    seal=read_json(root/"PREPROCESSING_RECEIPT.json")
    freeze=read_json(root/"METHOD_FREEZE.json")
    registry=read_json(root/"inference_registry.json")
    if (seal["status"]!="completed" or seal["freeze_file_sha256"]!=file_hash(root/"METHOD_FREEZE.json")
        or seal["inference_registry_sha256"]!=file_hash(root/"inference_registry.json")
        or freeze["source_sha256"]!=source_hashes()
        or json_hash(read_json(root/"PROTOCOL.json"))!=freeze["protocol_sha256"]):
        raise ValueError("Method or preprocessing seal mismatch")
    if (root/"EXECUTION.json").exists() and read_json(root/"EXECUTION.json")["status"]=="completed":
        raise FileExistsError("Retain completed Stage22 experiment")
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    evaluation_started=utc()
    # First full-truth access is AFTER method/config/conditions/final iterate seal.
    e=read_json(OLD/"evaluation_registry.json")["cases"][CASE]
    truth=verify_tensor(e["truth"])
    conditions=check_condition(registry["conditions"])
    added=conditions["mask"]&~conditions["original_mask"]
    if json_hash(registry["conditions"])!=seal["condition_record_sha256"]:
        raise ValueError("Conditions changed after sealing")
    domains={name:(verify_tensor(record["tensor"]),record["classes"]) for name,record in e["domains"].items()}
    if int(((truth==9)&domains["E_original_free"][0]).sum())!=1733 or int(((truth==9)&domains["E_holdout"][0]).sum())!=103:
        raise ValueError("Fixed domain identity/counts")
    ti,_=connected(truth.numpy()[0,0]==9)
    object1=torch.from_numpy(ti==1).reshape_as(truth)
    if int(object1.sum())!=572:
        raise ValueError("Original object ID mismatch")
    tp=int((added&(truth==9)).sum())
    fp=int((added&(truth!=9)).sum())
    count=int(added.sum())
    precision=tp/count if count else None
    hit=int((added&object1).sum())
    offline_pass=bool(seal["converged"] and precision is not None and precision>=.95 and hit>=5)
    gate=dict(added_voxels=count,columns=int(added.any(-1).sum()),TP=tp,FP=fp,precision=precision,
              object1_correct_added_voxels=hit,solver_converged=seal["converged"],passed=offline_pass,
              evaluation_started_utc=evaluation_started,preprocessing_sealed_at_utc=seal["sealed_at_utc"],
              no_conditions_modified_after_truth=True,gate_is_development_truth_check_not_deployment=True)
    write_json(root/"OFFLINE_GATE.json",gate)
    samples=[]
    object_rows=[]
    predicted_objects=[]
    allmetrics=[]
    def measure(arm,seed,pred,asset,receipt=None):
        pi,_=connected(pred.numpy()[0,0]==9)
        identity=dict(arm=arm,seed=seed)
        result=dict(**identity,prediction=asset,metrics={},objects={},receipt=receipt)
        for name,(domain,classes) in domains.items():
            dm=domain.numpy()[0,0]
            row=voxel_metrics(pred.numpy()[0,0],truth.numpy()[0,0],dm,classes)
            oo,pp,stats,_=match_objects(ti,pi,dm,RULES)
            subset=object1&domain&~added
            outside_count=int(subset.sum())
            outside_tp=int(((pred==9)&subset).sum())
            row.update(outside_added_object1_voxels=outside_count,outside_added_object1_TP=outside_tp,
                outside_added_object1_coverage=outside_tp/outside_count if outside_count else None,**stats)
            result["metrics"][name]=row
            result["objects"][name]=oo
            allmetrics.append({**identity,"domain":name,**row})
            object_rows.extend({**identity,"domain":name,**o,"full_voxels":int((ti==o["object_id"]).sum()),
                "new_correct_conditions_inside_object":int((added&torch.from_numpy(ti==o["object_id"]).reshape_as(truth)).sum())} for o in oo)
            predicted_objects.extend({**identity,"domain":name,**p} for p in pp)
        samples.append(result)
        return result
    def reference(seed,arm):
        folder=OLD/"predictions"/CASE/f"seed{seed}"/arm
        r=read_json(folder/"receipt.json")
        if read_json(folder/"completion_receipt.json")["receipt_sha256"]!=file_hash(folder/"receipt.json") or r["status"]!="completed":
            raise ValueError("Old reference not sealed")
        pred=verify_tensor(r["prediction"])
        expected=registry["noise"][str(seed)]
        if r["initial_noise_tensor_sha256"]!=expected["asset"]["tensor_sha256"] or r["base_hash_after"]!=expected["expected_model_hash"]:
            raise ValueError("Reference noise/model mismatch")
        return measure(arm,seed,pred,r["prediction"],dict(path=str((folder/"receipt.json").relative_to(REPO)),file_sha256=file_hash(folder/"receipt.json")))
    def copy_only(seed,a):
        base=verify_tensor(a["prediction"])
        pred=torch.where(conditions["mask"],conditions["values"].to(base.dtype),base)
        held=domains["E_holdout"][0]
        if not torch.equal(pred[held],base[held]):
            raise ValueError("Direct-copy control changed fixed holdout")
        asset=save_tensor(root/"offline"/f"seed{seed}_copy_only.pt",pred)
        return measure("COPY_ONLY_STAGE22",seed,pred,asset)
    def flow(seed):
        out=root/"predictions"/f"seed{seed}"
        if not out.exists():
            log=root/f"flow_seed{seed}.log"
            with log.open("x") as stream:
                subprocess.run([sys.executable,str(PROJECT/"scripts/stage22/run.py"),"--registry",str(root/"inference_registry.json"),
                    "--seed",str(seed),"--out",str(out)],cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,check=True)
        r=read_json(out/"receipt.json")
        if r["status"]!="completed" or read_json(out/"completion_receipt.json")["receipt_sha256"]!=file_hash(out/"receipt.json"):
            raise ValueError("New prediction seal")
        if r["condition_record_sha256"]!=json_hash(registry["conditions"]) or r["velocity_evaluations"]!=32:
            raise ValueError("Shared conditions/budget mismatch")
        pred=verify_tensor(r["prediction"])
        if ((pred!=conditions["values"])&conditions["mask"]).any():
            raise ValueError("Condition violation")
        return measure("FLOW_STAGE22",seed,pred,r["prediction"],dict(path=str((out/"receipt.json").relative_to(REPO)),file_sha256=file_hash(out/"receipt.json")))
    def criteria(new,a):
        n,b=new["metrics"]["E_original_free"],a["metrics"]["E_original_free"]
        obj=next(o for o in new["objects"]["E_original_free"] if o["object_id"]==1)
        checks=dict(recall_improves=n["target_recall"]>b["target_recall"],
            precision_not_worse=n["target_precision"] is not None and n["target_precision"]>=b["target_precision"],
            object1_matched_iou=obj["matched_iou"]>=.10,object1_matched_coverage=obj["matched_coverage"]>=.10,
            holdout_miou_guard=new["metrics"]["E_holdout"]["fixed_class_miou"]>=a["metrics"]["E_holdout"]["fixed_class_miou"]-.01)
        return dict(seed=new["seed"],**checks,passed=all(checks.values()),
            outside_added_object1_TP_delta=n["outside_added_object1_TP"]-b["outside_added_object1_TP"])
    assets=read_json(root/"preprocessing_assets.json")
    raw=verify_tensor(assets["classification"])
    direct=torch.where(conditions["original_mask"],conditions["original_values"].to(raw.dtype),raw)
    measure("RESTORED_CLASSIFICATION",None,direct,save_tensor(root/"offline/restored_direct.pt",direct))
    smooth=verify_tensor(e["pseudo"])
    measure("SMOOTH_CLASSIFICATION_P",None,smooth,e["pseudo"])
    gates=[]
    a=reference(42,"A_9")
    reference(42,"B_512")
    reference(42,"C_512")
    copy_only(42,a)
    if offline_pass:
        print(f"Offline gate passed: TP={tp}, FP={fp}, precision={precision:.6f}, object1 hits={hit}",flush=True)
        new=flow(42)
        first=criteria(new,a)
        gates.append(first)
        if first["passed"]:
            for seed in (142,242):
                aa=reference(seed,"A_9")
                reference(seed,"B_512")
                reference(seed,"C_512")
                copy_only(seed,aa)
                nn=flow(seed)
                gates.append(criteria(nn,aa))
            reason="three_seed_confirmation_complete"
        else:
            reason="seed42_research_gate_failed_no_confirmation"
    else:
        reason="offline_condition_gate_failed" if seal["converged"] else "numerical_solver_unconverged_not_scientific_failure"
        print(f"Stop before Flow: {reason}; TP={tp}, FP={fp}, precision={precision}, object1 hits={hit}",flush=True)
    outputs=sum(s["arm"]=="FLOW_STAGE22" for s in samples)
    if outputs>3:
        raise ValueError("Flow budget exceeded")
    result=dict(status="completed",stop_reason=reason,offline_gate=gate,flow_gates=gates,
        new_Flow_outputs=outputs,velocity_evaluations=32*outputs,samples=samples,
        paired_repeat_supported=len(gates)==3 and all(g["passed"] for g in gates),
        outside_added_TP_increases_for_all_new_seeds=bool(gates) and all(g["outside_added_object1_TP_delta"]>0 for g in gates),
        source_sha256=source_hashes(),completed_at_utc=utc())
    write_json(root/"EXECUTION.json",result)
    write_csv(root/"sample_metrics.csv",allmetrics)
    write_csv(root/"object_metrics.csv",object_rows)
    write_csv(root/"predicted_object_metrics.csv",predicted_objects)
    write_json(root/"EVALUATION_INPUTS.json",dict(truth=e["truth"],domains=e["domains"],
        preprocessing_receipt_sha256=file_hash(root/"PREPROCESSING_RECEIPT.json"),started_at_utc=evaluation_started))
    from scripts.stage22.report import report
    report(root,result)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    evaluate(parser.parse_args().root.resolve())
