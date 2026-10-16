"""Freeze method, solve without full truth, and seal sparse target-only conditions."""
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"]=""
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.stage22.common import (
    PROJECT,REPO,OLD,ROOT,CASE,CHECKPOINT,CHECKPOINT_SHA,utc,guard,protocol,source_hashes,
    read_json,write_json,write_csv,file_hash,json_hash,tensor_hash,verify_tensor,save_tensor,
    well_mask,git_snapshot,check_condition,classify_properties,sparse_conditions,
)
from scripts.stage22.solver import solve,blur
from guidance.property_volume import property_table_from_config,hard_labels_to_properties,_gaussian_kernel_1d
import torch


def prepare(out):
    guard("prepare")
    if out.exists():
        raise FileExistsError(f"Immutable run exists: {out}")
    torch.set_num_threads(4)
    p=protocol()
    old_inference=read_json(OLD/"inference_registry.json")
    # Registry metadata is needed only to resolve the registered PROPERTY input.
    # The guarded process cannot open any truth tensor or object/diagnostic asset.
    property_record=read_json(OLD/"evaluation_registry.json")["cases"][CASE]["properties"]
    classifier=read_json(OLD/"classifier.json")
    config=read_json(OLD/"codebook.json")
    layout=read_json(OLD/"layout.json")
    original=check_condition(old_inference["conditions"][CASE]["A_9"])
    properties=verify_tensor(property_record)
    table,weights,_=property_table_from_config(config,15)
    scales=table[:,1:].max(1).values-table[:,1:].min(1).values
    if scales.tolist()!=classifier["scales"] or weights.tolist()!=classifier["weights"]:
        raise ValueError("Frozen classifier mismatch")
    if properties.dtype!=torch.float32 or properties.shape!=(1,2,64,64,64):
        raise ValueError("Observation shape/dtype mismatch")
    out.mkdir(parents=True)
    write_json(out/"PROTOCOL.json",p)
    sources=source_hashes()
    for relative in sources:
        destination=out/"source_snapshot"/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(REPO/relative,destination)
    shutil.copyfile(REPO/"STAGE22_BLUR_AWARE_BRIDGE_PROMPT_2026-10-16.md",out/"source_snapshot/TASK_SPEC.md")
    shutil.copyfile(OLD/"codebook.json",out/"codebook.json")
    binding=dict(protocol_sha256=json_hash(p),source_sha256=sources,
        property_observation=property_record,original_conditions=old_inference["conditions"][CASE]["A_9"],
        codebook_sha256=file_hash(OLD/"codebook.json"),classifier_sha256=file_hash(OLD/"classifier.json"),
        layout_sha256=file_hash(OLD/"layout.json"),kernel_f32_sha256=tensor_hash(_gaussian_kernel_1d(1.,torch.device("cpu"),torch.float32)),
        frozen_at_utc=utc(),git=git_snapshot())
    write_json(out/"METHOD_FREEZE.json",binding)
    print("Method frozen before solving and before any full-truth access",flush=True)
    normalized=properties.double()/scales.double()[None,:,None,None,None]
    known=hard_labels_to_properties(original["values"],table).double()/scales.double()[None,:,None,None,None]
    lower=(table.double().min(1).values/scales.double())[None,:,None,None,None]
    upper=(table.double().max(1).values/scales.double())[None,:,None,None,None]
    mask=original["mask"]
    start=time.monotonic()
    def progress(row):
        if row["iteration"]%100==0:
            write_json(out/"SOLVER_PROGRESS.json",{**row,"elapsed_seconds":time.monotonic()-start})
            print(f"iteration{row['iteration']} objective={row['objective']:.9g} rel={row['relative_change']:.3g} stable={row['stable_count']}",flush=True)
    recovered,history,converged=solve(normalized,lower,upper,known,mask,callback=progress)
    projection_error=float((recovered-known).abs().masked_select(mask.expand_as(recovered)).max())
    bound_error=max(float((lower-recovered).clamp_min(0).max()),float((recovered-upper).clamp_min(0).max()))
    if projection_error!=0 or bound_error!=0 or not torch.isfinite(recovered).all():
        raise ValueError("Recovered-field constraints violated")
    restored=(recovered*scales.double()[None,:,None,None,None]).float()
    pred,d9,margin=classify_properties(restored,table,weights)
    new_values,new_mask,added=sparse_conditions(original["values"],mask,pred,d9,margin,
                                               well_mask(layout["levels"]["512"]))
    if (added&well_mask(layout["holdout"])).any():
        raise ValueError("Added condition enters fixed holdout")
    records={"original_mask":save_tensor(out/"conditions/original_mask.pt",mask),
             "original_values":save_tensor(out/"conditions/original_values.pt",original["values"]),
             "values":save_tensor(out/"conditions/values.pt",new_values),
             "mask":save_tensor(out/"conditions/mask.pt",new_mask)}
    check_condition(records)
    preprocess=dict(z=save_tensor(out/"preprocessing/recovered_z.pt",recovered),
        properties=save_tensor(out/"preprocessing/restored_properties.pt",restored),
        classification=save_tensor(out/"preprocessing/restored_classification.pt",pred),
        d9=save_tensor(out/"preprocessing/d9.pt",d9),margin=save_tensor(out/"preprocessing/margin9.pt",margin),
        added=save_tensor(out/"conditions/added_mask.pt",added))
    noise_records={}
    for seed in (42,142,242):
        a_path=OLD/"predictions"/CASE/f"seed{seed}/A_9/receipt.json"
        a=read_json(a_path)
        if a["status"]!="completed":
            raise ValueError("Paired baseline incomplete")
        if "noise" in a:
            noise_asset=a["noise"]
        else:
            # Verified original source location used by the audited reused A.
            noise_path=PROJECT/"experiments/stage21_virtual_borehole_conditioning/run_20261014_v2/formal/noise"/f"seed{seed}.pt"
            old_record=a["reuse_audit"]["old_record"]
            noise_asset=dict(path=str(noise_path.relative_to(REPO)),
                file_sha256=old_record["initial_noise_file_sha256"],tensor_sha256=old_record["initial_noise_tensor_sha256"])
        noise=verify_tensor(noise_asset)
        if tensor_hash(noise)!=a["initial_noise_tensor_sha256"]:
            raise ValueError("Actual A noise hash mismatch")
        noise_records[str(seed)]=dict(asset=noise_asset,expected_model_hash=a["base_hash_after"],
            A_receipt_path=str(a_path.relative_to(REPO)),A_receipt_sha256=file_hash(a_path))
    inference=dict(schema="stage22_conditions_only_v1",protocol_sha256=json_hash(p),source_sha256=sources,
        checkpoint=CHECKPOINT,checkpoint_sha256=CHECKPOINT_SHA,conditions=records,noise=noise_records)
    write_json(out/"inference_registry.json",inference)
    write_json(out/"preprocessing_assets.json",preprocess)
    write_csv(out/"solver_history.csv",history)
    if source_hashes()!=sources:
        raise ValueError("Source changed during frozen solve")
    write_json(out/"PREPROCESSING_RECEIPT.json",dict(status="completed",converged=converged,
        freeze_file_sha256=file_hash(out/"METHOD_FREEZE.json"),protocol_sha256=json_hash(p),
        sealed_at_utc=utc(),assets=preprocess,condition_record_sha256=json_hash(records),
        inference_registry_sha256=file_hash(out/"inference_registry.json"),
        solver_history_sha256=file_hash(out/"solver_history.csv"),iterations=len(history)-1,
        initial_objective=history[0]["objective"],final=history[-1],
        hard_constraint_error=projection_error,bounds_error=bound_error,
        added_condition_voxels=int(added.sum()),columns_with_additions=int(added.any(-1).sum()),
        elapsed_seconds=time.monotonic()-start,truth_access=False,checkpoint_loaded=False))
    print(f"Sealed final iterate: converged={converged}, added={int(added.sum())}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True,type=Path)
    prepare(parser.parse_args().out.resolve())
