#!/usr/bin/env python3
"""Truth-blind Stage17C FLOW_ONLY/TRAJECTORY_EVIDENCE paired runner."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime
from guidance.dflow import model_state_dict_hashes
from guidance.property_sampling import fixed_euler_property_sample
from guidance.property_volume import property_table_from_config
from guidance.probability_volume import tensor_sha256
from scripts.stage15.common import base_manifest, normalize_volume, read_json, refuse_nonempty, write_csv, write_json
from scripts.stage17.common import require_frozen_config, validate_asset_record


ROOT = PROJECT_DIR / "experiments/stage17_evidence_coupling_attribution"
DEFAULT_CONFIG = ROOT / "configs/coupling_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _validate(config: dict[str, object]) -> None:
    require_frozen_config(config, "stage17_same_evidence_coupling_v1")
    if config["authorized_arms"] != ["FLOW_ONLY", "TRAJECTORY_EVIDENCE"]:
        raise ValueError("Stage17C trajectory-only arm set changed")
    if config.get("dflow_arm_authorized") is not False:
        raise ValueError("weak Stage17B gate forbids the D-Flow geophysical arm")
    if config.get("truth_loaded_by_runner") is not False:
        raise ValueError("Stage17C runner must remain truth-blind")
    if config.get("primary_statistical_unit") != "independent_geology_case":
        raise ValueError("Stage17C statistical unit changed")


def main() -> None:
    args = parse_args()
    refuse_nonempty(args.output_dir)
    config = read_json(args.config)
    _validate(config)
    validate_asset_record(config["stage17a_gate"], "Stage17A gate")
    validate_asset_record(config["stage17b_gate"], "Stage17B gate")
    checkpoint = validate_asset_record(config["checkpoint"], "checkpoint")
    property_path = validate_asset_record(config["property_config"], "property_config")
    validate_asset_record(config["authoritative_stage15h_reference_config"], "Stage15-H reference")
    device = torch.device(args.device or str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    cases = config["cases"][:1] if args.smoke else config["cases"]
    seeds = config["source_seeds"][:1] if args.smoke else config["source_seeds"]

    from model_train_sh_inference_cond import Geo3DStochInterp

    model, model_report = runtime.load_model_with_weight_policy(
        model_class=Geo3DStochInterp,
        checkpoint_path=checkpoint,
        map_location=device,
        weight_source="ema",
    )
    model = model.to(device).eval()
    model_hash_before, tensor_hash_before = model_state_dict_hashes(model)
    table, channel_weights, property_metadata = property_table_from_config(
        read_json(property_path), model.num_categories
    )
    table = table.to(device)
    channel_weights = channel_weights.to(device)
    guidance = config["guidance"]

    args.output_dir.mkdir(parents=True)
    shutil.copyfile(args.config, args.output_dir / "config.json")
    manifest = base_manifest("stage17_same_evidence_coupling_run_v1", Path(__file__), args.config)
    manifest.update(
        {
            "run_status": "running",
            "run_class": "engineering_smoke" if args.smoke else "formal_coupling_confirmation",
            "smoke_subset": bool(args.smoke),
            "truth_loaded_by_runner": False,
            "dflow_arm_executed": False,
            "dflow_stop_reason": config["dflow_stop_reason"],
        }
    )
    write_json(args.output_dir / "run_manifest.json", manifest)
    records: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []
    try:
        for case in cases:
            case_id = str(case["case_id"])
            evidence_path = validate_asset_record(case["evidence"], f"{case_id}/evidence")
            validate_asset_record(case["evidence_case_manifest"], f"{case_id}/evidence_manifest")
            assets = case["observation_assets"]
            condition_values_path = validate_asset_record(assets["condition_values"], f"{case_id}/condition_values")
            condition_mask_path = validate_asset_record(assets["condition_mask"], f"{case_id}/condition_mask")
            support_path = validate_asset_record(assets["subsurface_mask"], f"{case_id}/subsurface")
            condition_cpu = normalize_volume(runtime.load_tensor(condition_values_path), "condition_values").long()
            condition_mask_cpu = normalize_volume(runtime.load_tensor(condition_mask_path), "condition_mask").bool()
            support_cpu = normalize_volume(runtime.load_tensor(support_path), "subsurface").bool()
            score = normalize_volume(runtime.load_tensor(evidence_path), "evidence", torch.float32)
            condition_values = condition_cpu.to(device)
            condition_mask = condition_mask_cpu.to(device)
            embedded = model.embed(condition_values)
            conditioning = embedded * condition_mask.expand(-1, embedded.shape[1], -1, -1, -1)
            confidence = (score * (support_cpu & ~condition_mask_cpu).float()).to(device)
            channels = table.shape[0]
            target_properties = torch.ones(
                (1, channels, *score.shape[2:]), device=device, dtype=table.dtype
            )
            for sample_id, seed in enumerate(seeds):
                generator = torch.Generator(device="cpu").manual_seed(int(seed))
                initial_cpu = torch.randn(
                    1, model.embedding_dim, *model.data_shape, generator=generator, dtype=embedded.dtype
                )
                for arm, alpha in (
                    ("FLOW_ONLY", 0.0),
                    ("TRAJECTORY_EVIDENCE", float(guidance["alpha"])),
                ):
                    final, trace = fixed_euler_property_sample(
                        model=model,
                        initial_state=initial_cpu.to(device),
                        conditioning=conditioning,
                        embedded_truth=embedded,
                        truth_model=condition_values,
                        condition_mask=condition_mask,
                        target_properties=target_properties,
                        property_table=table,
                        confidence=confidence,
                        property_sigmas=guidance["property_sigmas"],
                        property_scale_weights=guidance["property_scale_weights"],
                        property_channel_weights=channel_weights,
                        n_steps=int(config["n_steps"]),
                        alpha=alpha,
                        max_guidance_ratio=float(guidance["max_guidance_ratio"]),
                        tau_start=float(guidance["tau_start"]),
                        tau_end=float(guidance["tau_end"]),
                        tau_schedule=str(guidance["tau_schedule"]),
                        guidance_start=float(guidance["guidance_start"]),
                        guidance_schedule=str(guidance["guidance_schedule"]),
                        grad_clip_norm=float(guidance["grad_clip_norm"]),
                        guidance_scaling_mode=str(guidance["guidance_scaling_mode"]),
                        sample_id=sample_id,
                        loss_mode=str(guidance["property_loss_mode"]),
                    )
                    decoded = (model.decode(final).detach().cpu() - 1).unsqueeze(1).long()
                    violations = int(((decoded != condition_cpu) & condition_mask_cpu).sum())
                    if violations:
                        raise RuntimeError(f"condition violation: {case_id}/{seed}/{arm}")
                    output = args.output_dir / case_id / f"seed_{seed}" / arm / "decoded_geology.pt"
                    output.parent.mkdir(parents=True)
                    torch.save(decoded, output)
                    for row in trace:
                        traces.append({"case_id": case_id, "source_seed": int(seed), "arm": arm, **row})
                    records.append(
                        {
                            "case_id": case_id,
                            "source_seed": int(seed),
                            "sample_id": sample_id,
                            "arm": arm,
                            "initial_noise_sha256": tensor_sha256(initial_cpu),
                            "evidence_tensor_sha256": tensor_sha256(score),
                            "decoded_geology_sha256": tensor_sha256(decoded),
                            "condition_violations": violations,
                            "decoded_path": str(output.relative_to(args.output_dir)),
                        }
                    )
                print(f"Stage17C {case_id} seed={seed} completed", flush=True)
        model_hash_after, tensor_hash_after = model_state_dict_hashes(model)
        if model_hash_before != model_hash_after or tensor_hash_before != tensor_hash_after:
            raise RuntimeError("frozen model changed during Stage17C")
        write_csv(args.output_dir / "sample_manifest.csv", records)
        write_csv(args.output_dir / "guidance_traces.csv", traces)
        manifest.update(
            {
                "run_status": "completed",
                "sample_arm_count": len(records),
                "executed_case_count": len(cases),
                "executed_source_seed_count_per_case": len(seeds),
                "model_load_report": model_report,
                "property_metadata": property_metadata,
                "model_state_dict_sha256_before": model_hash_before,
                "model_state_dict_sha256_after": model_hash_after,
                "model_state_dict_unchanged": True,
                "all_condition_violations_zero": True,
                "same_evidence_hash_across_arms": True,
                "same_initial_noise_hash_across_arms": True,
                "scientific_conclusion_allowed": not args.smoke,
                "training_performed": False,
                "parameter_sweep_performed": False,
            }
        )
        write_json(args.output_dir / "run_manifest.json", manifest)
    except Exception as exc:
        manifest.update({"run_status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        write_json(args.output_dir / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
