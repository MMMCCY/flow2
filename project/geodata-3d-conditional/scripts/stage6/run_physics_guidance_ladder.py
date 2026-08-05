#!/usr/bin/env python3
"""Run the frozen Phase-6P extreme trajectory-guidance ladder."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time
from typing import Dict, Mapping

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import inference_runtime as runtime
from guidance import probability_sampling as controller_module
from guidance import seismic as seismic_module
from guidance import seismic_sampling as sampling_module
from guidance.generator_posterior import projected_fixed_euler_prior_sample
from guidance.physics_attainment import field_attainment_diagnostics
from guidance.seismic import tensor_sha256
from guidance.seismic_sampling import fixed_euler_seismic_sample
from scripts.stage4 import run_seismic_guidance as phase4_runner
from scripts.stage4.run_seismic_guidance import (
    add_hard_seismic_metrics,
    load_observation_assets,
    read_json,
    write_json,
    write_rows,
)
from scripts.stage5.run_generator_posterior import _validate_historical_baseline
from scripts.stage6.run_physics_attainment_limit import (
    _condition_violations,
    _model_tensor_hash,
    _save_tensor_record,
)


PHASE6P_LADDER_CONFIG_SCHEMA = "phase6p_trajectory_ladder_config_v1"
PHASE6P_LADDER_RUN_SCHEMA = "phase6p_trajectory_ladder_run_v1"
PHASE6P_LADDER_STAGE = "phase6p_seismic_extreme_trajectory_ladder_v1"


def parse_args() -> argparse.Namespace:
    experiment = PROJECT_DIR / "experiments/stage6_geo_adapter"
    case_dir = PROJECT_DIR / "samples/jupyter-demo/cond_generation_0"
    parser = argparse.ArgumentParser(
        description="Run paired 0.25x through 4x seismic trajectory guidance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=experiment
        / "configs/physics_attainment_seismic_trajectory_ladder_v1.json",
    )
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        default=PROJECT_DIR / "demo_model/conditional-weights.ckpt",
    )
    parser.add_argument("--samples-dir", type=Path, default=case_dir)
    parser.add_argument("--truth-model", type=Path, default=None)
    parser.add_argument("--boreholes", type=Path, default=None)
    parser.add_argument(
        "--observation-dir",
        type=Path,
        default=PROJECT_DIR
        / "experiments/stage4_seismic/observations/cond_generation_0"
        / "distinct_upper_bound_v1_fix2",
    )
    parser.add_argument(
        "--historical-baseline-dir",
        type=Path,
        default=PROJECT_DIR
        / "experiments/stage4_seismic/runs/cond_generation_0/phase4c_seismic_v1"
        / "seed42_n1_s32_a025_c025/baseline",
    )
    parser.add_argument(
        "--historical-alpha025-dir",
        type=Path,
        default=PROJECT_DIR
        / "experiments/stage4_seismic/runs/cond_generation_0/phase4c_seismic_v1"
        / "seed42_n1_s32_a025_c025/alpha025",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def validate_ladder_config(config: Mapping[str, object]) -> Dict[str, object]:
    expected = {
        "schema": PHASE6P_LADDER_CONFIG_SCHEMA,
        "status": "frozen_before_cuda_output",
        "truth_metrics_computed_by_runner": False,
        "base_model_trainable": False,
        "publication_evidence": False,
        "allow_hyperparameter_sweep_on_case": False,
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(f"Phase-6P ladder config {field} must be {value!r}")
    levels = config.get("levels")
    sampling = config.get("sampling")
    thresholds = config.get("attainment_thresholds")
    geology = config.get("geology_audit_thresholds")
    if not isinstance(levels, list) or not levels or not all(
        isinstance(value, Mapping) for value in (sampling, thresholds, geology)
    ):
        raise ValueError("Phase-6P ladder config sections are incomplete")
    resolved_levels: list[Dict[str, object]] = []
    seen: set[str] = set()
    for level in levels:
        if not isinstance(level, Mapping):
            raise ValueError("Phase-6P ladder levels must be mappings")
        level_id = str(level.get("id", ""))
        alpha = float(level.get("alpha", float("nan")))
        cap = float(level.get("max_guidance_ratio", float("nan")))
        if not level_id or level_id in seen:
            raise ValueError("Phase-6P ladder level ids must be nonempty and unique")
        if not math.isfinite(alpha) or alpha <= 0 or not math.isclose(
            alpha, cap, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("each ladder level requires positive alpha=cap")
        seen.add(level_id)
        resolved_levels.append(
            {"id": level_id, "alpha": alpha, "max_guidance_ratio": cap}
        )
    expected_ladder = [0.25, 0.5, 1.0, 2.0, 4.0]
    if [value["alpha"] for value in resolved_levels] != expected_ladder:
        raise ValueError("frozen Phase-6P ladder must be [0.25,0.5,1,2,4]")
    if sampling.get("tau_schedule") not in controller_module.TEMPERATURE_SCHEDULES:
        raise ValueError("invalid ladder temperature schedule")
    if sampling.get("guidance_schedule") not in controller_module.GUIDANCE_SCHEDULES:
        raise ValueError("invalid ladder guidance schedule")
    if sampling.get("guidance_scaling_mode") != (
        controller_module.REFERENCE_GUIDANCE_SCALING_MODE
    ):
        raise ValueError("ladder must retain Phase-4C reference scaling")
    resolved: Dict[str, object] = {
        "levels": resolved_levels,
        "seed": int(sampling["seed"]),
        "n_steps": int(sampling["n_steps"]),
        "tau_start": float(sampling["tau_start"]),
        "tau_end": float(sampling["tau_end"]),
        "tau_schedule": str(sampling["tau_schedule"]),
        "guidance_start": float(sampling["guidance_start"]),
        "guidance_schedule": str(sampling["guidance_schedule"]),
        "guidance_scaling_mode": str(sampling["guidance_scaling_mode"]),
        "gradient_clip_norm": float(sampling["gradient_clip_norm"]),
        "low_attainment_upper": float(thresholds["low_upper"]),
        "high_attainment_lower": float(thresholds["high_lower"]),
        "target_iou_material_delta": float(geology["target_iou_delta"]),
        "truth_present_mean_iou_material_delta": float(
            geology["truth_present_mean_iou_delta"]
        ),
    }
    if int(resolved["n_steps"]) <= 0 or float(resolved["gradient_clip_norm"]) <= 0:
        raise ValueError("ladder sampling steps and gradient clip must be positive")
    if not 0 <= float(resolved["guidance_start"]) < 1:
        raise ValueError("ladder guidance start must lie in [0,1)")
    if float(resolved["tau_start"]) <= 0 or float(resolved["tau_end"]) <= 0:
        raise ValueError("ladder temperatures must be positive")
    low = float(resolved["low_attainment_upper"])
    high = float(resolved["high_attainment_lower"])
    if not 0 < low < high < 1:
        raise ValueError("ladder attainment thresholds are invalid")
    return resolved


def _attainment_band(value: float, resolved: Mapping[str, object]) -> str:
    if value < float(resolved["low_attainment_upper"]):
        return "low"
    if value < float(resolved["high_attainment_lower"]):
        return "partial"
    return "high"


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty; refusing to overwrite: {args.output_dir}"
        )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the frozen Phase-6P ladder requires CUDA")
    protocol_config = read_json(args.config)
    resolved = validate_ladder_config(protocol_config)
    truth_path = args.truth_model or args.samples_dir / "true_model.pt"
    boreholes_path = args.boreholes or args.samples_dir / "boreholes.pt"
    truth_cpu = runtime.normalize_single_geology(
        runtime.load_tensor(truth_path, map_location="cpu"), str(truth_path)
    ).long()
    boreholes_cpu = runtime.normalize_single_geology(
        runtime.load_tensor(boreholes_path, map_location="cpu"), str(boreholes_path)
    ).long()

    from model_train_sh_inference_cond import Geo3DStochInterp

    model, model_load_report = runtime.load_model_with_weight_policy(
        model_class=Geo3DStochInterp,
        checkpoint_path=args.ckpt_path,
        map_location=device,
        weight_source="ema",
    )
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    base_hash_before = _model_tensor_hash(model)
    conditioning_report = runtime.validate_conditioning_pair(
        truth_cpu, boreholes_cpu, model.num_categories, target_label=9
    )
    tensors, observation_manifest, forward_operator, resolved_observation = (
        load_observation_assets(
            args.observation_dir,
            truth_cpu,
            truth_path=truth_path,
            num_categories=model.num_categories,
        )
    )
    condition_mask_cpu = (boreholes_cpu != -1) | (truth_cpu == -1)
    guidance_confidence_cpu = ((truth_cpu != -1) & ~condition_mask_cpu).float()
    truth_device = truth_cpu.to(device)
    condition_mask_device = condition_mask_cpu.to(device)
    embedded_truth = model.embed(truth_device)
    conditioning = embedded_truth * condition_mask_device.expand_as(embedded_truth)

    source_assets = runtime.experiment_asset_records(
        protocol_config=args.config,
        protocol_spec=PROJECT_DIR / "docs/PHASE6P_INFERENCE_LIMIT_SPEC.md",
        truth_model=truth_path,
        boreholes=boreholes_path,
        observation_manifest=args.observation_dir / "manifest.json",
        historical_baseline_config=args.historical_baseline_dir / "config.json",
        historical_alpha025_config=args.historical_alpha025_dir / "config.json",
        runner_source=Path(__file__),
        sampling_source=Path(sampling_module.__file__),
        controller_source=Path(controller_module.__file__),
        runtime_source=Path(runtime.__file__),
        seismic_source=Path(seismic_module.__file__),
        phase4_loader_source=Path(phase4_runner.__file__),
    )
    source_assets["checkpoint"] = model_load_report["checkpoint"]
    run_config: Dict[str, object] = {
        "schema": PHASE6P_LADDER_RUN_SCHEMA,
        "stage": PHASE6P_LADDER_STAGE,
        "run_status": "running",
        "protocol_config": protocol_config,
        "resolved_protocol": resolved,
        "truth_metrics_computed_by_runner": False,
        "truth_used_for": (
            "condition_construction_immutable_observation_validation_and_"
            "condition_violation_checks_only"
        ),
        "selection_policy": "all_predeclared_levels_reported_no_truth_selection_v1",
        "model_weight_source": "ema",
        "ema_applied": bool(model_load_report["ema_applied"]),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_device_name": torch.cuda.get_device_name(device),
        "output_dir": str(args.output_dir),
        "asset_records": source_assets,
        "model_load_report": model_load_report,
        "conditioning_report": conditioning_report,
        "observation_config_resolved": resolved_observation,
        "base_model_tensor_sha256_before": base_hash_before,
    }
    run_config.update(runtime.flatten_asset_hashes(source_assets))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "config.json", run_config)
    write_json(args.output_dir / "model_load_report.json", model_load_report)
    write_json(args.output_dir / "input_validation.json", conditioning_report)
    write_json(args.output_dir / "observation_manifest.json", observation_manifest)

    generator = torch.Generator(device="cpu").manual_seed(int(resolved["seed"]))
    initial_noise_cpu = torch.randn(
        1,
        model.embedding_dim,
        *model.data_shape,
        generator=generator,
        dtype=embedded_truth.dtype,
    ).contiguous()
    initial_noise_hash = tensor_sha256(initial_noise_cpu)
    baseline_state = projected_fixed_euler_prior_sample(
        model,
        initial_noise_cpu.to(device),
        conditioning,
        embedded_truth,
        condition_mask_cpu,
        n_steps=int(resolved["n_steps"]),
    )
    with torch.no_grad():
        baseline_decoded = (model.decode(baseline_state).detach().cpu() - 1).long()
    historical_config = read_json(args.historical_baseline_dir / "config.json")
    historical_validation = _validate_historical_baseline(
        baseline_dir=args.historical_baseline_dir,
        baseline_config=historical_config,
        initial_decoded=baseline_decoded,
        initial_noise_sha256=initial_noise_hash,
        ckpt_path=args.ckpt_path,
        truth_path=truth_path,
        boreholes_path=boreholes_path,
        observation_dir=args.observation_dir,
        initial_seed=int(resolved["seed"]),
        n_steps=int(resolved["n_steps"]),
    )

    def evaluate_hard(prediction: torch.Tensor) -> tuple[Dict[str, object], torch.Tensor]:
        row: Dict[str, object] = {}
        field = add_hard_seismic_metrics(
            row,
            prediction=prediction,
            target_acoustic=tensors["truth_acoustic.pt"],
            condition_mask=condition_mask_cpu,
            property_table=tensors["acoustic_property_table.pt"],
            subsurface_mask=tensors["subsurface_mask.pt"],
            forward_operator=forward_operator,
            observed=tensors["observed_seismic.pt"],
            sample_mask=tensors["sample_mask.pt"],
            uncertainty=tensors["uncertainty_amplitude.pt"],
            device=device,
        )
        row["condition_violations"] = _condition_violations(
            prediction, truth_cpu, condition_mask_cpu
        )
        return row, field

    baseline_metrics, baseline_field = evaluate_hard(baseline_decoded)
    output_records: Dict[str, object] = {
        "baseline_sample.pt": _save_tensor_record(
            args.output_dir / "baseline_sample.pt",
            baseline_decoded.to(torch.int8),
            categorical=True,
        ),
        "baseline_hard_seismic.pt": _save_tensor_record(
            args.output_dir / "baseline_hard_seismic.pt", baseline_field
        ),
    }
    historical_alpha025 = runtime.normalize_single_geology(
        runtime.load_tensor(
            args.historical_alpha025_dir / "sample_0.pt", map_location="cpu"
        ),
        "historical_alpha025",
    ).long()

    level_rows: list[Dict[str, object]] = []
    trace_rows: list[Dict[str, object]] = []
    alpha025_exact = False
    torch.cuda.reset_peak_memory_stats(device)
    total_started = time.perf_counter()
    for level in resolved["levels"]:
        level_started = time.perf_counter()
        final_state, level_trace = fixed_euler_seismic_sample(
            model=model,
            initial_state=initial_noise_cpu.to(device),
            conditioning=conditioning,
            embedded_truth=embedded_truth,
            truth_model=truth_device,
            condition_mask=condition_mask_device,
            target_acoustic=tensors["truth_acoustic.pt"],
            property_table=tensors["acoustic_property_table.pt"],
            guidance_confidence=guidance_confidence_cpu,
            subsurface_mask=tensors["subsurface_mask.pt"],
            forward_operator=forward_operator,
            observed=tensors["observed_seismic.pt"],
            sample_mask=tensors["sample_mask.pt"],
            uncertainty=tensors["uncertainty_amplitude.pt"],
            n_steps=int(resolved["n_steps"]),
            alpha=float(level["alpha"]),
            max_guidance_ratio=float(level["max_guidance_ratio"]),
            tau_start=float(resolved["tau_start"]),
            tau_end=float(resolved["tau_end"]),
            tau_schedule=str(resolved["tau_schedule"]),
            guidance_start=float(resolved["guidance_start"]),
            guidance_schedule=str(resolved["guidance_schedule"]),
            grad_clip_norm=float(resolved["gradient_clip_norm"]),
            guidance_scaling_mode=str(resolved["guidance_scaling_mode"]),
            sample_id=0,
        )
        if not torch.isfinite(final_state).all():
            raise FloatingPointError(f"ladder level {level['id']} is non-finite")
        with torch.no_grad():
            decoded = (model.decode(final_state).detach().cpu() - 1).long()
        hard_metrics, hard_field = evaluate_hard(decoded)
        diagnostics = field_attainment_diagnostics(
            tensors["observed_seismic.pt"],
            baseline_field,
            hard_field,
            tensors["sample_mask.pt"],
        )
        diagnostics["attainment_band"] = _attainment_band(
            float(diagnostics["attainment"]), resolved
        )
        changed = decoded != baseline_decoded
        level_id = str(level["id"])
        level_rows.append(
            {
                **dict(level),
                **hard_metrics,
                **diagnostics,
                "changed_from_baseline_voxels": int(changed.sum().item()),
                "changed_from_baseline_fraction": float(changed.float().mean()),
                "active_guidance_steps": sum(
                    float(row["used_guidance_ratio"]) > 0 for row in level_trace
                ),
                "cap_hit_steps": sum(
                    float(row["guidance_cap_fraction"]) > 0 for row in level_trace
                ),
                "mean_used_guidance_ratio": sum(
                    float(row["used_guidance_ratio"]) for row in level_trace
                )
                / len(level_trace),
                "max_used_guidance_ratio": max(
                    float(row["used_guidance_ratio"]) for row in level_trace
                ),
                "level_seconds": time.perf_counter() - level_started,
            }
        )
        for row in level_trace:
            trace_rows.append({"level_id": level_id, **row})
        output_records[f"{level_id}_sample.pt"] = _save_tensor_record(
            args.output_dir / f"{level_id}_sample.pt",
            decoded.to(torch.int8),
            categorical=True,
        )
        output_records[f"{level_id}_hard_seismic.pt"] = _save_tensor_record(
            args.output_dir / f"{level_id}_hard_seismic.pt", hard_field
        )
        if level_id == "ratio025":
            alpha025_exact = torch.equal(
                runtime.normalize_single_geology(decoded, "new_alpha025").long(),
                historical_alpha025,
            )

    total_seconds = time.perf_counter() - total_started
    write_rows(args.output_dir / "ladder_metrics.csv", level_rows)
    write_rows(args.output_dir / "guidance_trace.csv", trace_rows)
    write_json(
        args.output_dir / "historical_baseline_validation.json",
        historical_validation,
    )
    base_hash_after = _model_tensor_hash(model)
    base_gradients_absent = all(
        parameter.grad is None for parameter in model.parameters()
    )
    engineering_gates = {
        "historical_baseline_exact": bool(
            historical_validation["exact_initial_hard_regression"]
        ),
        "historical_alpha025_exact": alpha025_exact,
        "base_model_hash_unchanged": base_hash_after == base_hash_before,
        "base_model_gradients_absent": base_gradients_absent,
        "all_levels_completed": len(level_rows) == len(resolved["levels"]),
        "all_conditions_exact": all(
            int(row["condition_violations"]) == 0 for row in level_rows
        ),
        "all_physics_metrics_finite": all(
            math.isfinite(float(row["hard_seismic_loss"]))
            and math.isfinite(float(row["attainment"]))
            for row in level_rows
        ),
    }
    physically_best = min(level_rows, key=lambda row: float(row["hard_seismic_loss"]))
    run_config.update(
        {
            "run_status": "completed",
            "initial_noise_sha256": initial_noise_hash,
            "historical_baseline_validation": historical_validation,
            "historical_alpha025_exact": alpha025_exact,
            "baseline_physics_metrics": baseline_metrics,
            "level_physics_metrics": level_rows,
            "physically_best_level_id": physically_best["id"],
            "maximum_attainment": max(
                float(row["attainment"]) for row in level_rows
            ),
            "total_seconds": total_seconds,
            "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "base_model_tensor_sha256_after": base_hash_after,
            "base_model_tensor_hash_unchanged": base_hash_after
            == base_hash_before,
            "base_model_gradients_absent": base_gradients_absent,
            "engineering_gates": engineering_gates,
            "engineering_pass": all(engineering_gates.values()),
            "output_tensor_records": output_records,
            "geology_audit_status": "not_run_by_truth_blind_runner",
        }
    )
    write_json(args.output_dir / "config.json", run_config)
    print(
        "Phase-6P trajectory ladder complete: "
        f"best={physically_best['id']}, max_attainment={run_config['maximum_attainment']:.6f}, "
        f"engineering={run_config['engineering_pass']}, output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
