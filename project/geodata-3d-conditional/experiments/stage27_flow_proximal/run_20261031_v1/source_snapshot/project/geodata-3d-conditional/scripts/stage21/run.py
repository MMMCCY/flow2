"""Truth-blind frozen native Flow runner. Only accepts the stripped condition registry."""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

# Set before importing torch / initializing CUDA.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch

PROJECT = Path(__file__).resolve().parents[2]
REPO = PROJECT.parents[1]
for path in (PROJECT, REPO, REPO / "src", REPO / "StructuralGeo-main/src"):
    sys.path.insert(0, str(path))
import inference_runtime as runtime
from guidance.dflow import model_state_dict_hashes
from guidance.generator_posterior import projected_fixed_euler_prior_sample
from scripts.stage21.common import (
    file_hash, git_snapshot, json_hash, load_conditions, read_json, source_hashes,
    tensor_hash, validate_registry, verify_tensor, write_json,
)


def install_firewall():
    """Fail closed on opening evaluation assets, including through torch.load."""
    def guard(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        parts = set(path.parts)
        if ("truth" in parts or ("geometry" in parts and "experiments" in parts) or "evaluation_registry.json" in parts
            or "full_structuralgeo_benchmark" in parts or "object_exposure.csv" in parts):
            raise PermissionError(f"Stage21 inference truth firewall: {path}")
    sys.addaudithook(guard)


def initial_noise(seed, embedding_dim, shape):
    return torch.randn((1, embedding_dim, *shape),
                       generator=torch.Generator(device="cpu").manual_seed(seed), dtype=torch.float32)


def sample_native(model, initial, values, mask, n_steps):
    """Only categorical conditions are passed to the existing pure sampler."""
    embedded = model.embed(values)
    conditioning = embedded * mask
    roundtrip = (model.decode(embedded) - 1).unsqueeze(1)
    if not torch.equal(roundtrip[mask], values[mask].long()):
        raise ValueError("raw-label / embedding-index roundtrip mismatch")
    trace = []
    def prehook(net, inputs):
        state, atb, time_value = inputs[:3]
        error = float((state - embedded).abs().masked_select(mask.expand_as(state)).max())
        if error != 0 or not torch.isfinite(state).all() or not torch.equal(atb, conditioning):
            raise ValueError("per-step condition/finite/input audit failed")
        trace.append({"step": len(trace), "time": float(time_value[0]),
                      "condition_embedding_max_error": error, "state_finite": True})
    def hook(net, inputs, velocity):
        if not torch.isfinite(velocity).all():
            raise FloatingPointError("nonfinite velocity")
        trace[-1]["velocity_finite"] = True
    h1 = model.net.register_forward_pre_hook(prehook)
    h2 = model.net.register_forward_hook(hook)
    try:
        state = projected_fixed_euler_prior_sample(model, initial, conditioning, embedded, mask, n_steps=n_steps)
    finally:
        h1.remove()
        h2.remove()
    if not torch.isfinite(state).all() or not torch.equal(state.masked_select(mask.expand_as(state)), embedded.masked_select(mask.expand_as(state))):
        raise ValueError("final state audit failed")
    if len(trace) != n_steps:
        raise ValueError("velocity evaluation budget mismatch")
    decoded = (model.decode(state) - 1).unsqueeze(1).long()
    if int(((decoded != values) & mask).sum()):
        raise ValueError("decoded hard-condition violation")
    return decoded.detach().cpu(), state, trace


def independent_n9_reference(model, initial, values, mask, steps):
    """Independent explicit baseline loop for equivalence, no posterior/guidance API."""
    embedded = model.embed(values)
    expanded = mask.expand_as(initial)
    state = torch.where(expanded, embedded, initial)
    condition = embedded * mask
    for i in range(steps):
        t = torch.full((1,), (i + .5) / steps, device=initial.device, dtype=initial.dtype)
        state = torch.where(expanded, embedded, state + (1.0 / steps) * model.net(state, condition, t))
    return state


def verify_completed(record, folder):
    path = folder / record["prediction_path"]
    if file_hash(path) != record["prediction_file_sha256"]:
        raise ValueError("resume prediction file hash mismatch")
    pred = runtime.load_tensor(path)
    if tensor_hash(pred) != record["prediction_tensor_sha256"]:
        raise ValueError("resume prediction tensor hash mismatch")
    if file_hash(folder / record["trace_path"]) != record["trace_file_sha256"]:
        raise ValueError("resume trace hash mismatch")
    return pred


def run(args):
    install_firewall()
    registry = read_json(args.registry)
    p = validate_registry(registry)
    if source_hashes() != registry["source_sha256"]:
        raise ValueError("frozen inference source hash mismatch")
    if file_hash(REPO / p["checkpoint"]) != p["checkpoint_sha256"]:
        raise ValueError("checkpoint hash mismatch")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise ValueError("real-checkpoint run requires CUDA; CPU fixtures are tested separately")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cases = p["cases"] if args.mode == "formal" else [p["smoke_case"]]
    seeds = p["source_seeds"] if args.mode == "formal" else [p["smoke_seed"]]
    binding = {"registry_file_sha256": file_hash(args.registry), "mode": args.mode,
               "source_sha256": registry["source_sha256"], "protocol_sha256": registry["protocol_sha256"]}
    out = args.out.resolve()
    ledger_path = out / "sampling_state.json"
    if out.exists():
        if not args.resume:
            raise FileExistsError(f"immutable run exists: {out}; use verified --resume")
        state = read_json(ledger_path)
        if state["binding"] != binding:
            raise ValueError("resume binding mismatch")
        if state["status"] == "failed":
            raise ValueError("failed engineering run retained; diagnose and use new run_id")
    else:
        out.mkdir(parents=True)
        state = {"binding": binding, "status": "running", "records": [], "failures": [],
                 "expected_outputs": len(cases)*len(seeds)*3, "git": git_snapshot()}
    # Formal execution requires a completed smoke under the identical frozen registry/source.
    smoke_receipt = None
    if args.mode == "formal":
        if args.smoke is None:
            raise ValueError("formal inference needs --smoke receipt")
        smoke = read_json(args.smoke / "sampling_state.json")
        if (smoke["status"] != "completed" or smoke["binding"] != {**binding, "mode": "smoke"}
            or len(smoke["records"]) != 3 or not smoke["independent_n9_equivalence"]
            or not smoke["base_model_unchanged"]):
            raise ValueError("smoke gate mismatch")
        for r in smoke["records"]:
            verify_completed(r, args.smoke)
        smoke_receipt = {"path": str(args.smoke), "state_sha256": file_hash(args.smoke / "sampling_state.json")}
    if state.get("status") == "completed":
        receipt = read_json(out / "completion_receipt.json")
        if receipt["state_sha256"] != file_hash(ledger_path) or receipt["records_sha256"] != json_hash(state["records"]):
            raise ValueError("completed resume receipt mismatch")
        expected = {(c, s, n) for c in cases for s in seeds for n in p["well_counts"]}
        keys = {(r["case_id"], r["source_seed"], r["well_count"]) for r in state["records"]}
        if keys != expected or len(state["records"]) != len(expected):
            raise ValueError("completed resume set mismatch")
        for r in state["records"]:
            pred = verify_completed(r, out)
            ts = load_conditions(registry["conditions"][r["case_id"]][str(r["well_count"])])
            if ((pred != ts["condition_values"]) & ts["condition_mask"]).any():
                raise ValueError("completed resume condition violation")
            noise_path = out / "noise" / f"seed{r['source_seed']}.pt"
            saved_noise = runtime.load_tensor(noise_path)
            noise = initial_noise(r["source_seed"], saved_noise.shape[1], p["shape"])
            if (tensor_hash(noise) != r["initial_noise_tensor_sha256"]
                or not torch.equal(noise, saved_noise) or file_hash(noise_path) != r["initial_noise_file_sha256"]):
                raise ValueError("completed resume noise mismatch")
        print(f"Verified complete immutable {args.mode} run; no files changed: {out}", flush=True)
        return
    from model_train_sh_inference_cond import Geo3DStochInterp
    model, report = runtime.load_model_with_weight_policy(Geo3DStochInterp, REPO / p["checkpoint"],
                                                        map_location="cuda", weight_source="ema")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    model.eval().float().to("cuda")
    before, entries = model_state_dict_hashes(model)
    if "base_hash_before" in state and state["base_hash_before"] != before:
        raise ValueError("resume model hash mismatch")
    state.update(base_hash_before=before, model_tensor_hashes=entries, model_load_report=report,
                 smoke_receipt=smoke_receipt,
                 environment={"python": platform.python_version(), "torch": torch.__version__,
                              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                              "cudnn": torch.backends.cudnn.version(), "precision": p["precision"],
                              "determinism": p["determinism"]})
    write_json(ledger_path, state)
    completed = {(r["case_id"], r["source_seed"], r["well_count"]): r for r in state["records"]}
    if len(completed) != len(state["records"]):
        raise ValueError("duplicate records in resume")
    start = time.monotonic()
    try:
        with torch.no_grad():
            for case in cases:
                conditions = {n: load_conditions(registry["conditions"][case][str(n)]) for n in p["well_counts"]}
                base_mask = conditions[9]["condition_mask"]
                for n in (16, 25):
                    if (not torch.equal(conditions[n]["surface_mask"], conditions[9]["surface_mask"])
                        or not torch.equal(conditions[n]["condition_values"][base_mask], conditions[9]["condition_values"][base_mask])
                        or torch.any(base_mask & ~conditions[n]["condition_mask"])):
                        raise ValueError("original surface/well condition changed")
                for seed in seeds:
                    noise = initial_noise(seed, model.embedding_dim, p["shape"])
                    noise_hash = tensor_hash(noise)  # before any condition projection
                    noise_path = out / "noise" / f"seed{seed}.pt"
                    noise_path.parent.mkdir(exist_ok=True)
                    if noise_path.exists():
                        if tensor_hash(runtime.load_tensor(noise_path)) != noise_hash:
                            raise ValueError("saved source noise mismatch")
                    else:
                        torch.save(noise, noise_path)
                    for n in p["well_counts"]:
                        key = (case, seed, n)
                        if key in completed:
                            r = completed[key]
                            pred = verify_completed(r, out)
                            if r["initial_noise_tensor_sha256"] != noise_hash or r["base_hash_after"] != before:
                                raise ValueError("resume pairing/model mismatch")
                            cm = conditions[n]["condition_mask"]
                            if ((pred != conditions[n]["condition_values"]) & cm).any():
                                raise ValueError("resume condition violation")
                            continue
                        folder = out / case / f"seed{seed}" / f"n{n}"
                        if folder.exists():
                            raise FileExistsError(f"uncommitted sample retained: {folder}")
                        values = conditions[n]["condition_values"].to("cuda")
                        mask = conditions[n]["condition_mask"].to("cuda")
                        t0 = time.monotonic()
                        pred, final, trace = sample_native(model, noise.to("cuda"), values, mask, p["n_steps"])
                        if args.mode == "smoke" and n == 9:
                            reference = independent_n9_reference(model, noise.to("cuda"), values, mask, p["n_steps"])
                            if not torch.equal(reference, final):
                                raise ValueError("independent N9 baseline continuous-state mismatch")
                            state["independent_n9_equivalence"] = True
                            state["independent_n9_state_sha256"] = tensor_hash(final)
                            del reference
                        del final
                        after, after_entries = model_state_dict_hashes(model)
                        if after != before or after_entries != entries or any(x.requires_grad or x.grad is not None for x in model.parameters()):
                            raise ValueError("base model/EMA/embedding changed")
                        folder.mkdir(parents=True)
                        prediction_path = folder / "decoded_geology.pt"
                        torch.save(pred, prediction_path)
                        trace_path = folder / "trace.json"
                        write_json(trace_path, trace)
                        record = {"case_id": case, "source_seed": seed, "well_count": n,
                                  "initial_noise_tensor_sha256": noise_hash,
                                  "initial_noise_file_sha256": file_hash(noise_path),
                                  "condition_record_sha256": json_hash(registry["conditions"][case][str(n)]),
                                  "prediction_path": str(prediction_path.relative_to(out)),
                                  "prediction_file_sha256": file_hash(prediction_path),
                                  "prediction_tensor_sha256": tensor_hash(pred),
                                  "trace_path": str(trace_path.relative_to(out)), "trace_file_sha256": file_hash(trace_path),
                                  "velocity_evaluations": len(trace), "condition_violations": 0,
                                  "base_hash_after": after, "elapsed_seconds": time.monotonic()-t0}
                        state["records"].append(record)
                        write_json(ledger_path, state)
                        print(f"{args.mode} {len(state['records'])}/{state['expected_outputs']} {case} seed={seed} n={n} {record['elapsed_seconds']:.1f}s", flush=True)
        if len(state["records"]) != state["expected_outputs"]:
            raise ValueError("incomplete output set")
        if source_hashes() != registry["source_sha256"]:
            raise ValueError("source changed during inference")
        state.update(status="completed", base_model_unchanged=True, base_hash_after=before,
                     invocation_elapsed_seconds=time.monotonic()-start, truth_firewall_enabled=True)
        write_json(ledger_path, state)
        # This small inference-only receipt is sealed before any retrospective evaluator is invoked.
        write_json(out / "completion_receipt.json", {"state_sha256": file_hash(ledger_path),
                   "records_sha256": json_hash(state["records"]), "binding": binding})
    except Exception as exc:
        state["status"] = "failed"
        state["failures"].append({"error": f"{type(exc).__name__}: {exc}"})
        write_json(ledger_path, state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "formal"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", type=Path)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        # Initialization/import/hash failures must be retained as well as sampling failures.
        if args.out.exists():
            write_json(args.out / "invocation_failure.json", {"mode": args.mode,
                       "error": f"{type(exc).__name__}: {exc}", "sampling_state_present": (args.out / "sampling_state.json").exists()})
        raise
