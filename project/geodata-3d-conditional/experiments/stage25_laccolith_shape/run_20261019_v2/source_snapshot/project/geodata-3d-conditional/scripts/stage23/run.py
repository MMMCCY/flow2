"""Single frozen native Flow from sealed Stage23 conditions and actual paired A noise."""
import argparse
import os
import platform
import sys
import time
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.stage23.common import (
    REPO,CHECKPOINT,CHECKPOINT_SHA,guard,source_hashes,read_json,write_json,
    file_hash,json_hash,tensor_hash,verify_tensor,save_tensor,check_condition,utc,
)
from scripts.stage21.run import initial_noise,sample_native
from guidance.dflow import model_state_dict_hashes
import inference_runtime as runtime
import torch


def run(registry_path,seed,out):
    registry_path=registry_path.resolve()
    out=out.resolve()
    guard("flow")
    if out.resolve()!=(registry_path.parent/"predictions/seed42").resolve():
        raise ValueError("One canonical output only per sealed experiment")
    r=read_json(registry_path)
    if set(r)!={"schema","protocol_sha256","source_sha256","checkpoint","checkpoint_sha256","conditions","noise"}:
        raise ValueError("Conditions-only registry fields")
    if r["schema"]!="stage23_conditions_only_v1" or r["checkpoint"]!=CHECKPOINT or r["checkpoint_sha256"]!=CHECKPOINT_SHA:
        raise ValueError("Model contract")
    if source_hashes()!=r["source_sha256"] or file_hash(REPO/CHECKPOINT)!=CHECKPOINT_SHA:
        raise ValueError("Frozen source/checkpoint changed")
    ts=check_condition(r["conditions"])
    extra=ts["mask"]&~ts["original_mask"]
    if not ((ts["values"][extra]>=0)&(ts["values"][extra]<=13)).all():
        raise ValueError("Added labels must use raw0..13")
    if seed!=42 or set(r["noise"])!={"42"}:
        raise ValueError("Only source seed42 authorized")
    seal=read_json(registry_path.parent/"CONDITION_RECEIPT.json")
    if seal["inference_registry_sha256"]!=file_hash(registry_path) or seal["condition_record_sha256"]!=json_hash(r["conditions"]):
        raise ValueError("Conditions are not sealed")
    paired=r["noise"][str(seed)]
    if file_hash(REPO/paired["A_receipt_path"])!=paired["A_receipt_sha256"]:
        raise ValueError("Original A binding changed")
    noise=verify_tensor(paired["asset"])
    if not torch.equal(noise,initial_noise(seed,noise.shape[1],(64,64,64))):
        raise ValueError("Actual paired CPU source differs from registered Gaussian generator")
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    state=dict(status="initializing",seed=seed,arm="FLOW_STAGE23",registry_sha256=file_hash(registry_path),
        source_sha256=r["source_sha256"],protocol_sha256=r["protocol_sha256"],
        condition_record_sha256=json_hash(r["conditions"]),source_noise_asset=paired["asset"],
        initial_noise_tensor_sha256=tensor_hash(noise),velocity_evaluations=0,started_at_utc=utc())
    write_json(out/"receipt.json",state)
    start=time.monotonic()
    try:
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        if not torch.cuda.is_available():
            raise ValueError("Real CUDA required")
        from model_train_sh_inference_cond import Geo3DStochInterp
        model,report=runtime.load_model_with_weight_policy(Geo3DStochInterp,REPO/CHECKPOINT,map_location="cuda",weight_source="ema")
        if report["ema_entry_count"]!=411 or report["ema_excluded_frozen_parameters"]!=["embedding.weight"]:
            raise ValueError("EMA/embedding policy")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            parameter.grad=None
        model.eval().float().to("cuda")
        before,entries=model_state_dict_hashes(model)
        if before!=paired["expected_model_hash"]:
            raise ValueError("Model hash differs from paired A")
        state.update(base_hash_before=before,model_load_report=report,
            environment=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),precision="FP32_no_TF32_no_autocast",
                deterministic=True,CUBLAS_WORKSPACE_CONFIG=os.environ["CUBLAS_WORKSPACE_CONFIG"]))
        write_json(out/"receipt.json",state)
        def count_call(net,inputs):
            state["velocity_evaluations"]+=1
            if state["velocity_evaluations"]>32:
                raise ValueError("32-call limit exceeded")
        counter=model.net.register_forward_pre_hook(count_call)
        try:
            with torch.no_grad():
                pred,continuous,trace=sample_native(model,noise.to("cuda"),ts["values"].to("cuda"),ts["mask"].to("cuda"),32)
        finally:
            counter.remove()
        del continuous
        after,after_entries=model_state_dict_hashes(model)
        if before!=after or after_entries!=entries or any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise ValueError("Frozen model changed")
        ov=int(((pred!=ts["values"])&ts["original_mask"]).sum())
        av=int(((pred!=ts["values"])&extra).sum())
        if ov or av or len(trace)!=32 or state["velocity_evaluations"]!=32:
            raise ValueError("Hard condition/budget violation")
        write_json(out/"trace.json",trace)
        state.update(status="completed",prediction=save_tensor(out/"decoded_geology.pt",pred),
            trace_sha256=file_hash(out/"trace.json"),velocity_evaluations=len(trace),
            original_condition_violations=ov,added_condition_violations=av,base_hash_after=after,
            elapsed_seconds=time.monotonic()-start,completed_at_utc=utc())
        if source_hashes()!=r["source_sha256"]:
            raise ValueError("Source changed during inference")
        write_json(out/"receipt.json",state)
        write_json(out/"completion_receipt.json",dict(receipt_sha256=file_hash(out/"receipt.json")))
        print(f"seed{seed}: 32 velocity calls, {state['elapsed_seconds']:.2f}s",flush=True)
    except Exception as exc:
        state.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        write_json(out/"receipt.json",state)
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry",type=Path,required=True)
    parser.add_argument("--seed",type=int,choices=[42],required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    run(args.registry,args.seed,args.out)
