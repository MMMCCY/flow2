#!/usr/bin/env python3
"""Truth-blind five-arm Stage19 test inference runner."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime
from guidance.property_sampling import fixed_euler_property_sample
from guidance.property_volume import property_table_from_config
from guidance.residual_velocity_adapter import ResidualVelocityAdapter, fixed_euler_adapter_sample
from guidance.seismic import tensor_sha256
from scripts.stage15.common import base_manifest, normalize_volume, read_json, refuse_nonempty, write_csv, write_json
from scripts.stage19.common import CONFIG_DIR, ROOT, asset, freeze_base_model, require_config, resolve_project_path, validate_asset

ARMS = ("FLOW_ONLY", "STAGE18_CONTINUOUS_TARGET", "ADAPTER_CORRECT", "ADAPTER_ZERO", "ADAPTER_WRONG_CASE")
DEFAULT_CONFIG = CONFIG_DIR / "inference_v1.json"
DEFAULT_TRAINING_CONFIG = CONFIG_DIR / "training_v1.json"
DEFAULT_EVIDENCE = ROOT / "evidence"
DEFAULT_CHECKPOINT = ROOT / "checkpoints/formal_v1/adapter_checkpoint.pt"
DEFAULT_OUTPUT = ROOT / "formal/inference_v1"
PROPERTY_CONFIG = PROJECT_DIR / "experiments/stage15_binary_seismic_consensus/configs/binary_trace_property_indicator_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--adapter-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _parameter_hash(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode()); digest.update(tensor_sha256(parameter.detach().cpu()).encode())
    return digest.hexdigest()


def _case_inputs(case: dict[str, object]) -> dict[str, torch.Tensor]:
    case_id = str(case["case_id"])
    obs = case["observation_assets"]
    return {
        "condition_values": normalize_volume(runtime.load_tensor(validate_asset(obs["condition_values"], f"{case_id}/condition values")), "condition values").long(),
        "condition_mask": normalize_volume(runtime.load_tensor(validate_asset(obs["condition_mask"], f"{case_id}/condition mask")), "condition mask").bool(),
        "subsurface": normalize_volume(runtime.load_tensor(validate_asset(obs["subsurface_mask"], f"{case_id}/subsurface")), "subsurface").bool(),
        "score": normalize_volume(runtime.load_tensor(validate_asset(case["evidence"], f"{case_id}/evidence")), "evidence", torch.float32),
    }


def main() -> None:
    args = parse_args()
    refuse_nonempty(args.output_dir)
    cfg = require_config(args.config, "stage19_inference_v1")
    training_cfg = require_config(args.training_config, "stage19_learned_evidence_adapter_training_v1")
    if cfg.get("truth_loaded_by_runner") is not False or tuple(cfg["arms"]) != ARMS:
        raise ValueError("Stage19 inference truth firewall or arm order changed")
    registry_path = args.evidence_dir / "evidence_registry.json"
    registry = read_json(registry_path)
    cases = [case for case in registry["cases"] if case["split"] == "test"]
    if len(cases) != 12:
        raise ValueError("Stage19 inference requires exactly 12 TEST cases")
    wrong = {str(case["case_id"]): str(cases[(index + 1) % len(cases)]["case_id"]) for index, case in enumerate(cases)}
    if any(key == value for key, value in wrong.items()):
        raise ValueError("wrong-case evidence mapping self-maps")
    executed_cases = cases[:1] if args.smoke else cases
    seeds = cfg["source_seeds"][:1] if args.smoke else cfg["source_seeds"]
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Stage19 inference requires CUDA")
    checkpoint = resolve_project_path(training_cfg["base_model"]["checkpoint"])
    from model_train_sh_inference_cond import Geo3DStochInterp
    model, model_report = runtime.load_model_with_weight_policy(model_class=Geo3DStochInterp, checkpoint_path=checkpoint, map_location=device, weight_source="ema")
    model = model.to(device).eval(); freeze_base_model(model)
    base_hash_before = _parameter_hash(model)
    payload = torch.load(args.adapter_checkpoint, map_location=device, weights_only=False)
    if payload.get("schema") != "stage19_adapter_checkpoint_v1" or payload.get("base_checkpoint_sha256") != training_cfg["base_model"]["checkpoint_sha256"]:
        raise ValueError("invalid Stage19 adapter checkpoint")
    adapter = ResidualVelocityAdapter(model.embedding_dim, geophysics_channels=1, base_width=int(training_cfg["adapter"]["base_width"]), dilations=training_cfg["adapter"]["dilations"]).to(device)
    adapter.load_state_dict(payload["adapter_state_dict"]); adapter.eval()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    table, channel_weights, _ = property_table_from_config(read_json(PROPERTY_CONFIG), model.num_categories)
    table, channel_weights = table.to(device), channel_weights.to(device)
    inputs = {str(case["case_id"]): _case_inputs(case) for case in cases}

    args.output_dir.mkdir(parents=True)
    manifest = base_manifest("stage19_inference_run_v1", Path(__file__), args.config)
    manifest.update({"run_status": "running", "run_class": "engineering_smoke" if args.smoke else "formal_test_inference", "smoke_subset": args.smoke, "truth_loaded_by_runner": False, "wrong_case_mapping": wrong})
    write_json(args.output_dir / "run_manifest.json", manifest)
    rows, traces = [], []
    try:
        for case in executed_cases:
            case_id = str(case["case_id"]); tensors = inputs[case_id]
            values, mask, support, score = tensors["condition_values"], tensors["condition_mask"], tensors["subsurface"], tensors["score"]
            with torch.no_grad():
                embedded = model.embed(values.to(device))
                conditioning = embedded * mask.to(device).expand_as(embedded)
            free = support & ~mask
            q_correct = (score * support.float()).to(device)
            q_zero = torch.zeros_like(q_correct)
            wrong_id = wrong[case_id]
            q_wrong = (inputs[wrong_id]["score"] * support.float()).to(device)
            for seed in seeds:
                initial_cpu = torch.randn((1, model.embedding_dim, *model.data_shape), generator=torch.Generator(device="cpu").manual_seed(int(seed)), dtype=embedded.dtype).contiguous()
                initial_sha = tensor_sha256(initial_cpu)
                for arm in ARMS:
                    if arm == "FLOW_ONLY":
                        final, trace = fixed_euler_adapter_sample(model=model, adapter=adapter, initial_state=initial_cpu.to(device), conditioning=conditioning, embedded_conditions=embedded, condition_mask=mask.to(device), geophysics=q_correct, n_steps=int(cfg["n_steps"]), adapter_scale=0.0, max_residual_ratio=float(cfg["max_residual_ratio"]))
                    elif arm == "STAGE18_CONTINUOUS_TARGET":
                        guide = cfg["stage18"]
                        final, trace = fixed_euler_property_sample(model=model, initial_state=initial_cpu.to(device), conditioning=conditioning, embedded_truth=embedded, truth_model=values.to(device), condition_mask=mask.to(device), target_properties=score.to(device), property_table=table, confidence=(score * free.float()).to(device), property_sigmas=guide["property_sigmas"], property_scale_weights=guide["property_scale_weights"], property_channel_weights=channel_weights, n_steps=int(cfg["n_steps"]), alpha=float(guide["alpha"]), max_guidance_ratio=float(guide["max_guidance_ratio"]), tau_start=float(guide["tau_start"]), tau_end=float(guide["tau_end"]), tau_schedule=str(guide["tau_schedule"]), guidance_start=float(guide["guidance_start"]), guidance_schedule=str(guide["guidance_schedule"]), grad_clip_norm=float(guide["grad_clip_norm"]), guidance_scaling_mode=str(guide["guidance_scaling_mode"]), sample_id=0, loss_mode=str(guide["property_loss_mode"]))
                    else:
                        q = q_correct if arm == "ADAPTER_CORRECT" else q_zero if arm == "ADAPTER_ZERO" else q_wrong
                        final, trace = fixed_euler_adapter_sample(model=model, adapter=adapter, initial_state=initial_cpu.to(device), conditioning=conditioning, embedded_conditions=embedded, condition_mask=mask.to(device), geophysics=q, n_steps=int(cfg["n_steps"]), adapter_scale=float(cfg["adapter_scale"]), max_residual_ratio=float(cfg["max_residual_ratio"]))
                    with torch.no_grad():
                        decoded = (model.decode(final).detach().cpu() - 1).unsqueeze(1).long()
                    violations = int(((decoded != values) & mask).sum())
                    if violations:
                        raise RuntimeError(f"hard condition violation: {case_id}/{seed}/{arm}")
                    out = args.output_dir / case_id / f"seed_{seed}" / arm
                    out.mkdir(parents=True)
                    torch.save(decoded, out / "decoded_geology.pt")
                    for item in trace:
                        traces.append({"case_id": case_id, "source_seed": int(seed), "arm": arm, **item})
                    rows.append({"case_id": case_id, "source_seed": int(seed), "arm": arm, "initial_noise_sha256": initial_sha, "input_evidence_case_id": wrong_id if arm == "ADAPTER_WRONG_CASE" else case_id, "input_evidence_sha256": tensor_sha256(inputs[wrong_id]["score"] if arm == "ADAPTER_WRONG_CASE" else score if arm != "ADAPTER_ZERO" else torch.zeros_like(score)), "decoded_geology_sha256": tensor_sha256(decoded), "condition_violation_count": violations, "decoded_path": str((out / "decoded_geology.pt").relative_to(args.output_dir))})
                    print(f"Stage19 inference {case_id} seed={seed} arm={arm}", flush=True)
                if args.smoke:
                    # Independent alpha-zero property path must match FLOW_ONLY exactly.
                    guide = cfg["stage18"]
                    reference, _ = fixed_euler_property_sample(model=model, initial_state=initial_cpu.to(device), conditioning=conditioning, embedded_truth=embedded, truth_model=values.to(device), condition_mask=mask.to(device), target_properties=score.to(device), property_table=table, confidence=(score * free.float()).to(device), property_sigmas=guide["property_sigmas"], property_scale_weights=guide["property_scale_weights"], property_channel_weights=channel_weights, n_steps=int(cfg["n_steps"]), alpha=0.0, max_guidance_ratio=float(guide["max_guidance_ratio"]), tau_start=float(guide["tau_start"]), tau_end=float(guide["tau_end"]), tau_schedule=str(guide["tau_schedule"]), guidance_start=float(guide["guidance_start"]), guidance_schedule=str(guide["guidance_schedule"]), grad_clip_norm=float(guide["grad_clip_norm"]), guidance_scaling_mode=str(guide["guidance_scaling_mode"]), sample_id=0, loss_mode=str(guide["property_loss_mode"]))
                    flow_row = next(row for row in rows if row["case_id"] == case_id and row["source_seed"] == int(seed) and row["arm"] == "FLOW_ONLY")
                    ref_decoded = (model.decode(reference).detach().cpu() - 1).unsqueeze(1).long()
                    if tensor_sha256(ref_decoded) != flow_row["decoded_geology_sha256"]:
                        raise RuntimeError("adapter scale=0 does not reproduce FLOW_ONLY")
        expected = 5 if args.smoke else 180
        if len(rows) != expected:
            raise RuntimeError(f"unexpected formal output count: {len(rows)}")
        base_hash_after = _parameter_hash(model)
        if base_hash_before != base_hash_after:
            raise RuntimeError("base model changed during inference")
        write_csv(args.output_dir / "sample_manifest.csv", rows)
        write_csv(args.output_dir / "sampling_trace.csv", traces)
        manifest.update({"run_status": "completed", "output_count": len(rows), "executed_case_count": len(executed_cases), "source_seed_count": len(seeds), "all_condition_violations_zero": True, "base_model_unchanged": True, "base_model_hash_before": base_hash_before, "base_model_hash_after": base_hash_after, "model_load_report": model_report, "evidence_registry": asset(registry_path), "adapter_checkpoint": asset(args.adapter_checkpoint), "truth_loaded_by_runner": False, "scale_zero_flow_equivalence_checked": bool(args.smoke)})
        write_json(args.output_dir / "run_manifest.json", manifest)
    except Exception as exc:
        manifest.update({"run_status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        write_json(args.output_dir / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
