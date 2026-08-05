#!/usr/bin/env python3
"""Run the frozen Phase-6P seismic endpoint-attainment audit."""

from __future__ import annotations

import argparse
import hashlib
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
from guidance import physics_attainment as attainment
from guidance import seismic as seismic_module
from guidance.generator_posterior import projected_fixed_euler_prior_sample
from guidance.physics_attainment import (
    ENDPOINT_CONDITION_POLICY,
    ENDPOINT_SELECTION_POLICY,
    PHYSICS_ATTAINMENT_OPTIMIZER_VERSION,
    expand_temperature_schedule,
    field_attainment_diagnostics,
    optimize_endpoint_state,
)
from guidance.seismic import seismic_volume_loss, tensor_sha256
from scripts.stage4 import run_seismic_guidance as phase4_runner
from scripts.stage4.run_seismic_guidance import (
    add_hard_seismic_metrics,
    load_observation_assets,
    read_json,
    write_json,
    write_rows,
)
from scripts.stage5.run_generator_posterior import _validate_historical_baseline


PHASE6P_CONFIG_SCHEMA = "phase6p_physics_attainment_config_v1"
PHASE6P_RUN_SCHEMA = "phase6p_physics_attainment_run_v1"
PHASE6P_STAGE = "phase6p_seismic_endpoint_attainment_v1"


def parse_args() -> argparse.Namespace:
    experiment = PROJECT_DIR / "experiments/stage6_geo_adapter"
    case_dir = PROJECT_DIR / "samples/jupyter-demo/cond_generation_0"
    parser = argparse.ArgumentParser(
        description="Run frozen-network seismic endpoint physics fitting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=experiment
        / "configs/physics_attainment_seismic_endpoint_v1.json",
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def validate_config(config: Mapping[str, object]) -> Dict[str, object]:
    """Validate and resolve the frozen endpoint-attainment configuration."""
    expected = {
        "schema": PHASE6P_CONFIG_SCHEMA,
        "status": "frozen_before_cuda_output",
        "method": PHYSICS_ATTAINMENT_OPTIMIZER_VERSION,
        "truth_metrics_computed_by_optimizer": False,
        "base_model_trainable": False,
        "publication_evidence": False,
        "allow_hyperparameter_sweep_on_case": False,
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(f"Phase-6P config {field} must be {value!r}")
    optimizer = config.get("optimizer")
    sampling = config.get("sampling")
    thresholds = config.get("attainment_thresholds")
    geology_thresholds = config.get("geology_audit_thresholds")
    schedule = config.get("temperature_schedule")
    if not all(
        isinstance(value, Mapping)
        for value in (optimizer, sampling, thresholds, geology_thresholds)
    ) or not isinstance(schedule, list):
        raise ValueError("Phase-6P config sections are incomplete")
    if optimizer.get("type") != "adam":
        raise ValueError("Phase-6P endpoint optimizer must be Adam")
    temperatures = expand_temperature_schedule(schedule)
    resolved: Dict[str, object] = {
        "learning_rate": float(optimizer["learning_rate"]),
        "weight_decay": float(optimizer["weight_decay"]),
        "gradient_clip_norm": float(optimizer["gradient_clip_norm"]),
        "hard_check_interval": int(optimizer["hard_check_interval"]),
        "max_state_norm_to_embedding_norm": float(
            optimizer["max_state_norm_to_embedding_norm"]
        ),
        "hard_improvement_tolerance": float(
            optimizer["hard_improvement_tolerance"]
        ),
        "temperature_schedule": [dict(value) for value in schedule],
        "optimization_steps": len(temperatures),
        "sampling_seed": int(sampling["seed"]),
        "sampling_steps": int(sampling["n_steps"]),
        "low_attainment_upper": float(thresholds["low_upper"]),
        "high_attainment_lower": float(thresholds["high_lower"]),
        "target_iou_material_delta": float(
            geology_thresholds["target_iou_delta"]
        ),
        "truth_present_mean_iou_material_delta": float(
            geology_thresholds["truth_present_mean_iou_delta"]
        ),
    }
    positive = (
        "learning_rate",
        "gradient_clip_norm",
        "hard_check_interval",
        "max_state_norm_to_embedding_norm",
        "optimization_steps",
        "sampling_steps",
    )
    for field in positive:
        value = float(resolved[field])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"resolved Phase-6P field must be positive: {field}")
    if float(resolved["weight_decay"]) < 0 or float(
        resolved["hard_improvement_tolerance"]
    ) < 0:
        raise ValueError("weight decay and improvement tolerance must be non-negative")
    low = float(resolved["low_attainment_upper"])
    high = float(resolved["high_attainment_lower"])
    if not 0 < low < high < 1:
        raise ValueError("attainment thresholds must satisfy 0 < low < high < 1")
    return resolved


def _model_tensor_hash(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(tensor_sha256(parameter.detach().cpu()).encode("ascii"))
    return digest.hexdigest()


def _condition_violations(
    decoded: torch.Tensor,
    condition_values: torch.Tensor,
    condition_mask: torch.Tensor,
) -> int:
    prediction = runtime.normalize_single_geology(decoded, "decoded").long()
    target = runtime.normalize_single_geology(
        condition_values, "condition_values"
    ).long()
    return int(((prediction != target) & condition_mask.bool()).sum().item())


def _save_tensor_record(
    path: Path,
    value: torch.Tensor,
    *,
    categorical: bool = False,
) -> Dict[str, object]:
    saved = value.detach().cpu().contiguous()
    torch.save(saved, path)
    record: Dict[str, object] = {
        "path": str(path),
        "shape": list(saved.shape),
        "dtype": str(saved.dtype),
        "file_sha256": runtime.file_sha256(path),
        "raw_tensor_sha256": tensor_sha256(saved),
    }
    if categorical:
        record["canonical_int64_content_sha256"] = tensor_sha256(saved.long())
    return record


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
        raise RuntimeError("the frozen Phase-6P production run requires CUDA")

    protocol_config = read_json(args.config)
    resolved = validate_config(protocol_config)
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
    embedded_truth = model.embed(truth_cpu.to(device))
    condition_mask_device = condition_mask_cpu.to(device)
    conditioning = embedded_truth * condition_mask_device.expand_as(embedded_truth)

    source_assets = runtime.experiment_asset_records(
        protocol_config=args.config,
        protocol_spec=PROJECT_DIR / "docs/PHASE6P_INFERENCE_LIMIT_SPEC.md",
        truth_model=truth_path,
        boreholes=boreholes_path,
        observation_manifest=args.observation_dir / "manifest.json",
        historical_baseline_config=args.historical_baseline_dir / "config.json",
        runner_source=Path(__file__),
        optimizer_source=Path(attainment.__file__),
        runtime_source=Path(runtime.__file__),
        seismic_source=Path(seismic_module.__file__),
        phase4_loader_source=Path(phase4_runner.__file__),
    )
    source_assets["checkpoint"] = model_load_report["checkpoint"]
    run_config: Dict[str, object] = {
        "schema": PHASE6P_RUN_SCHEMA,
        "stage": PHASE6P_STAGE,
        "run_status": "running",
        "protocol_config": protocol_config,
        "resolved_protocol": resolved,
        "optimizer_version": PHYSICS_ATTAINMENT_OPTIMIZER_VERSION,
        "condition_policy": ENDPOINT_CONDITION_POLICY,
        "selection_policy": ENDPOINT_SELECTION_POLICY,
        "truth_metrics_computed_by_optimizer": False,
        "truth_used_for": (
            "condition_construction_immutable_observation_validation_and_"
            "condition_violation_checks_only"
        ),
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

    generator = torch.Generator(device="cpu").manual_seed(
        int(resolved["sampling_seed"])
    )
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
        n_steps=int(resolved["sampling_steps"]),
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
        initial_seed=int(resolved["sampling_seed"]),
        n_steps=int(resolved["sampling_steps"]),
    )

    property_table = tensors["acoustic_property_table.pt"].to(device)
    target_acoustic = tensors["truth_acoustic.pt"].to(device)
    subsurface_mask = tensors["subsurface_mask.pt"].to(device)
    observed = tensors["observed_seismic.pt"].to(device)
    sample_mask = tensors["sample_mask.pt"].to(device)
    uncertainty = tensors["uncertainty_amplitude.pt"].to(device)
    embedding_weight = model.embedding.weight.detach()

    def hard_evaluate(
        state: torch.Tensor,
    ) -> tuple[Mapping[str, object], Mapping[str, torch.Tensor]]:
        with torch.no_grad():
            decoded = (model.decode(state).detach().cpu() - 1).long()
        row: Dict[str, object] = {}
        field = add_hard_seismic_metrics(
            row,
            prediction=decoded,
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
        metrics: Dict[str, object] = {
            "hard_loss": row["hard_seismic_loss"],
            "hard_rmse_amplitude": row["hard_seismic_rmse_amplitude"],
            "hard_mae_amplitude": row["hard_seismic_mae_amplitude"],
            "condition_violations": _condition_violations(
                decoded, truth_cpu, condition_mask_cpu
            ),
            "canonical_sample_sha256": tensor_sha256(decoded),
        }
        return metrics, {
            "decoded": decoded.to(torch.int8),
            "hard_seismic": field,
        }

    def soft_loss(
        state: torch.Tensor, temperature: float
    ) -> tuple[torch.Tensor, Mapping[str, object]]:
        loss, diagnostics = seismic_volume_loss(
            state,
            embedding_weight,
            property_table,
            target_acoustic,
            condition_mask_device,
            subsurface_mask,
            forward_operator,
            observed,
            sample_mask,
            uncertainty,
            tau=temperature,
        )
        return loss, {
            "seismic_rmse_amplitude": diagnostics["seismic_rmse_amplitude"],
            "seismic_mae_amplitude": diagnostics["seismic_mae_amplitude"],
        }

    max_embedding_norm = float(
        torch.linalg.vector_norm(embedding_weight, dim=1).max().detach().cpu()
    )
    configured_max_voxel_norm = max_embedding_norm * float(
        resolved["max_state_norm_to_embedding_norm"]
    )
    baseline_max_voxel_norm = float(
        torch.linalg.vector_norm(baseline_state, dim=1).max().detach().cpu()
    )
    # The guard may stop endpoint-state expansion but must never perturb the
    # exact historical starting point before the first hard evaluation.
    max_voxel_norm = max(configured_max_voxel_norm, baseline_max_voxel_norm)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = optimize_endpoint_state(
        initial_state=baseline_state,
        embedded_conditions=embedded_truth,
        condition_mask=condition_mask_device,
        soft_loss=soft_loss,
        hard_evaluate=hard_evaluate,
        temperature_schedule=resolved["temperature_schedule"],
        learning_rate=float(resolved["learning_rate"]),
        weight_decay=float(resolved["weight_decay"]),
        gradient_clip_norm=float(resolved["gradient_clip_norm"]),
        hard_check_interval=int(resolved["hard_check_interval"]),
        max_voxel_norm=max_voxel_norm,
        hard_loss_key="hard_loss",
        improvement_tolerance=float(resolved["hard_improvement_tolerance"]),
    )
    torch.cuda.synchronize(device)
    optimization_seconds = time.perf_counter() - started

    baseline_payload = result["initial_payload"]
    best_payload = result["best_payload"]
    optimizer_initial_hard_exact = torch.equal(
        runtime.normalize_single_geology(
            baseline_payload["decoded"], "optimizer_initial_decoded"
        ).long(),
        runtime.normalize_single_geology(
            baseline_decoded, "reconstructed_baseline_decoded"
        ).long(),
    )
    baseline_field = baseline_payload["hard_seismic"]
    best_field = best_payload["hard_seismic"]
    physical_diagnostics = field_attainment_diagnostics(
        tensors["observed_seismic.pt"],
        baseline_field,
        best_field,
        tensors["sample_mask.pt"],
    )
    physical_diagnostics["attainment_band"] = _attainment_band(
        float(physical_diagnostics["attainment"]), resolved
    )

    trace = result["trace"]
    write_rows(args.output_dir / "optimization_trace.csv", trace)
    write_json(
        args.output_dir / "historical_baseline_validation.json",
        historical_validation,
    )
    output_records = {
        "baseline_state.pt": _save_tensor_record(
            args.output_dir / "baseline_state.pt", result["initial_state"]
        ),
        "best_state.pt": _save_tensor_record(
            args.output_dir / "best_state.pt", result["best_state"]
        ),
        "final_state.pt": _save_tensor_record(
            args.output_dir / "final_state.pt", result["final_state"]
        ),
        "baseline_sample.pt": _save_tensor_record(
            args.output_dir / "baseline_sample.pt",
            baseline_payload["decoded"],
            categorical=True,
        ),
        "best_sample.pt": _save_tensor_record(
            args.output_dir / "best_sample.pt",
            best_payload["decoded"],
            categorical=True,
        ),
        "baseline_hard_seismic.pt": _save_tensor_record(
            args.output_dir / "baseline_hard_seismic.pt", baseline_field
        ),
        "best_hard_seismic.pt": _save_tensor_record(
            args.output_dir / "best_hard_seismic.pt", best_field
        ),
    }
    base_hash_after = _model_tensor_hash(model)
    base_gradients_absent = all(
        parameter.grad is None for parameter in model.parameters()
    )
    checked_losses = [
        float(row["hard_loss"])
        for row in trace
        if bool(row["hard_checked"])
    ]
    best_loss = float(result["best_metrics"]["hard_loss"])
    exact_best_selection = math.isclose(
        best_loss, min(checked_losses), rel_tol=0.0, abs_tol=1e-12
    )
    engineering_gates = {
        "historical_baseline_exact": bool(
            historical_validation["exact_initial_hard_regression"]
        ),
        "optimizer_initial_hard_exact": optimizer_initial_hard_exact,
        "base_model_hash_unchanged": base_hash_after == base_hash_before,
        "base_model_gradients_absent": base_gradients_absent,
        "initial_conditions_exact": int(
            result["initial_metrics"]["condition_violations"]
        )
        == 0,
        "best_conditions_exact": int(
            result["best_metrics"]["condition_violations"]
        )
        == 0,
        "minimum_hard_loss_selection_exact": exact_best_selection,
        "hard_loss_nonregression": best_loss
        <= float(result["initial_metrics"]["hard_loss"]),
        "all_updates_completed": int(result["updates_completed"])
        == int(resolved["optimization_steps"]),
    }
    run_config.update(
        {
            "run_status": "completed",
            "initial_noise_sha256": initial_noise_hash,
            "historical_baseline_validation": historical_validation,
            "max_embedding_norm": max_embedding_norm,
            "configured_max_voxel_norm": configured_max_voxel_norm,
            "baseline_max_voxel_norm": baseline_max_voxel_norm,
            "max_voxel_norm": max_voxel_norm,
            "updates_completed": result["updates_completed"],
            "hard_evaluations": result["hard_evaluations"],
            "best_step": result["best_step"],
            "initial_physics_metrics": result["initial_metrics"],
            "best_physics_metrics": result["best_metrics"],
            "physical_attainment_diagnostics": physical_diagnostics,
            "optimization_seconds": optimization_seconds,
            "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "base_model_tensor_sha256_after": base_hash_after,
            "base_model_tensor_hash_unchanged": base_hash_after
            == base_hash_before,
            "base_model_gradients_absent": base_gradients_absent,
            "engineering_gates": engineering_gates,
            "engineering_pass": all(engineering_gates.values()),
            "output_tensor_records": output_records,
            "geology_audit_status": "not_run_by_truth_blind_optimizer",
        }
    )
    write_json(args.output_dir / "config.json", run_config)
    print(
        "Phase-6P endpoint audit complete: "
        f"attainment={physical_diagnostics['attainment']:.6f}, "
        f"band={physical_diagnostics['attainment_band']}, "
        f"best_step={result['best_step']}, engineering={all(engineering_gates.values())}, "
        f"output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
