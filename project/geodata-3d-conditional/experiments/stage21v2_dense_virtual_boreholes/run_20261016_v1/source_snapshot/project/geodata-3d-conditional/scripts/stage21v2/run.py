"""One sealed CUDA prediction from prepared categorical conditions only."""
import argparse
import os
import platform
import sys
import time
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.stage21v2.common import *
from scripts.stage21.run import initial_noise, sample_native, install_firewall
from guidance.dflow import model_state_dict_hashes
import inference_runtime as runtime


def firewall():
    install_firewall()
    def guard(event, args):
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            p = Path(os.fsdecode(args[0])).resolve()
            if "evaluation" in p.parts or p.name.startswith("pseudo_"):
                raise PermissionError(f"stage21v2 evaluation asset firewall: {p}")
    sys.addaudithook(guard)


def run(args):
    firewall()
    registry = validate_inference(read_json(args.registry))
    if sources() != registry["source_sha256"]:
        raise ValueError("frozen source mismatch")
    if file_hash(REPO / CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError("checkpoint hash")
    key = f"{args.arm}_{args.n}"
    record = registry["conditions"][args.case][key]
    conditions = check_condition(record)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if args.out.exists():
        raise FileExistsError(f"retain immutable output {args.out}")
    args.out.mkdir(parents=True)
    start = time.monotonic()
    state = dict(status="initializing", case_id=args.case, source_seed=args.seed, well_count=args.n,
                 arm=args.arm, registry_sha256=file_hash(args.registry),
                 protocol_sha256=registry["protocol_sha256"], source_sha256=sources(),
                 condition_record_sha256=json_hash(record), checkpoint_sha256=CHECKPOINT_SHA)
    write_json(args.out / "receipt.json", state)
    try:
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        from model_train_sh_inference_cond import Geo3DStochInterp
        model, report = runtime.load_model_with_weight_policy(
            Geo3DStochInterp, REPO / CHECKPOINT, map_location="cuda", weight_source="ema")
        if report["ema_entry_count"] != 411 or report["ema_excluded_frozen_parameters"] != ["embedding.weight"]:
            raise ValueError("EMA contract")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        model.eval().float().to("cuda")
        before, entries = model_state_dict_hashes(model)
        noise = initial_noise(args.seed, model.embedding_dim, (64, 64, 64))
        state.update(model_load_report=report, base_hash_before=before,
                     model_tensor_hashes=entries, initial_noise_tensor_sha256=tensor_hash(noise),
                     noise=save_tensor(args.out / "initial_noise.pt", noise),
                     environment=dict(python=platform.python_version(), torch=torch.__version__,
                                      cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                                      gpu=torch.cuda.get_device_name(0), precision="float32_no_autocast_no_tf32",
                                      determinism="torch_deterministic_algorithms_cublas4096_8_cudnn_benchmark_false"))
        state["status"] = "sampling"
        write_json(args.out / "receipt.json", state)
        torch.cuda.synchronize()
        t0 = time.monotonic()
        with torch.no_grad():
            pred, final, trace = sample_native(model, noise.to("cuda"), conditions["values"].to("cuda"),
                                               conditions["mask"].to("cuda"), 32)
        torch.cuda.synchronize()
        state["sampling_seconds"] = time.monotonic()-t0
        del final
        after, after_entries = model_state_dict_hashes(model)
        if before != after or entries != after_entries or any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise ValueError("frozen model changed")
        om = conditions["original_mask"]
        violations = int(((pred != conditions["values"]) & om).sum())
        added_violations = int(((pred != conditions["values"]) & conditions["mask"] & ~om).sum())
        if violations or added_violations or len(trace) != 32:
            raise ValueError("prediction contract")
        prediction = save_tensor(args.out / "decoded_geology.pt", pred)
        write_json(args.out / "trace.json", trace)
        if sources() != registry["source_sha256"]:
            raise ValueError("source changed during inference")
        state.update(status="completed", prediction=prediction, trace_sha256=file_hash(args.out / "trace.json"),
                     velocity_evaluations=len(trace), original_condition_violations=violations,
                     added_condition_violations=added_violations, base_hash_after=after,
                     model_unchanged=True, truth_firewall_enabled=True,
                     elapsed_seconds=time.monotonic()-start)
        write_json(args.out / "receipt.json", state)
        write_json(args.out / "completion_receipt.json", dict(receipt_sha256=file_hash(args.out / "receipt.json")))
        print(f"{args.case} seed{args.seed} {key}: {state['sampling_seconds']:.2f}s, 32 forwards", flush=True)
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(args.out / "receipt.json", state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--seed", type=int, choices=[42,142,242], required=True)
    parser.add_argument("--arm", choices=list("ABC"), required=True)
    parser.add_argument("--n", type=int, choices=[9,*LEVELS], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.arm == "A") != (args.n == 9):
        parser.error("A uses N9; B/C use dense levels")
    run(args)
