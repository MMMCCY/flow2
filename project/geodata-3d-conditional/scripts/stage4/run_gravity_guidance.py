#!/usr/bin/env python3
"""Run strict paired Phase-4a full-support gravity guidance."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Dict, Mapping, Sequence

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import inference_runtime as runtime
from guidance import gravity as gravity_module
from guidance import gravity_sampling as gravity_sampling_module
from guidance import probability_sampling as controller_module
from guidance import property_evaluation as evaluation_module
from guidance.gravity import (
    GRAVITY_FORWARD_MODE,
    GRAVITY_LOSS_MODE,
    gravity_field_loss,
    gravity_operator_from_config,
    hard_labels_to_density,
    overwrite_exact_condition_density,
    probabilities_to_density,
    tensor_sha256,
)
from guidance.gravity_sampling import fixed_euler_gravity_sample
from guidance.probability_evaluation import (
    class_transition_records,
    ensemble_diversity_summary,
    summarize_rows,
)
from guidance.probability_volume import build_target_mask, dilate_mask
from guidance.property_evaluation import (
    paired_per_class_deltas,
    paired_property_metric_deltas,
    paired_truth_component_recovery_deltas,
    per_class_hard_metrics,
    sample_property_hard_metrics,
    summarize_per_class_rows,
    truth_component_recovery_rows,
)
from guided_geophysical_sampling import soft_decode_to_probs


PHASE4_PROTOCOL_VERSION = 1
PHASE4_STAGE = "phase4a_gravity_v1"
PHASE4_DESCRIPTION = (
    "Truth-derived full-support rectangular-prism surface gravity guidance; "
    "synthetic inverse crime, not measured geophysics."
)
OBSERVATION_TENSOR_FILES = (
    "density_table_kg_m3.pt",
    "truth_density_kg_m3.pt",
    "observed_gravity_mgal.pt",
    "noiseless_gravity_mgal.pt",
    "gravity_noise_mgal.pt",
    "survey_mask.pt",
    "uncertainty_mgal.pt",
)
SAVED_PAIR_TENSORS = {
    "density_table.pt": "density_table_sha256",
    "target_density.pt": "target_density_sha256",
    "observed_gravity_mgal.pt": "observed_gravity_sha256",
    "noiseless_gravity_mgal.pt": "noiseless_gravity_sha256",
    "gravity_noise_mgal.pt": "gravity_noise_sha256",
    "survey_mask.pt": "survey_mask_sha256",
    "uncertainty_mgal.pt": "uncertainty_sha256",
    "guidance_confidence.pt": "guidance_confidence_sha256",
    "target_mask.pt": "target_mask_sha256",
    "target_roi_mask.pt": "target_roi_mask_sha256",
}
PHASE4_PAIR_FIELDS = (
    "protocol_version",
    "phase4_protocol_version",
    "stage",
    "checkpoint_sha256",
    "model_weight_source",
    "ema_applied",
    "truth_model_sha256",
    "boreholes_sha256",
    "observation_manifest_sha256",
    "controller_manifest_sha256",
    "runner_source_sha256",
    "runtime_source_sha256",
    "gravity_source_sha256",
    "gravity_sampling_source_sha256",
    "property_evaluation_source_sha256",
    "controller_source_sha256",
    "density_table_sha256",
    "target_density_sha256",
    "observed_gravity_sha256",
    "noiseless_gravity_sha256",
    "gravity_noise_sha256",
    "survey_mask_sha256",
    "uncertainty_sha256",
    "guidance_confidence_sha256",
    "target_mask_sha256",
    "target_roi_mask_sha256",
    "density_config_id",
    "controller_level_id",
    "controller_intended_alpha",
    "density_unit",
    "gravity_forward_mode",
    "gravity_loss_mode",
    "grid_shape",
    "cell_size_m",
    "origin_m",
    "station_height_m",
    "observation_height_above_top_m",
    "gravity_output_unit",
    "survey_station_count",
    "truth_derived",
    "measured_geophysics",
    "inverse_crime",
    "target_label",
    "target_roi_radius",
    "n_samples",
    "n_steps",
    "integrator",
    "initial_noise_policy",
    "seed",
    "tau_start",
    "tau_end",
    "tau_schedule",
    "guidance_start",
    "guidance_schedule",
    "guidance_scaling_mode",
    "guidance_gradient_reference_policy",
    "max_guidance_ratio",
    "grad_clip_norm",
    "condition_projection",
    "known_density_policy",
    "device",
)


def read_json(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=PHASE4_DESCRIPTION,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--model-weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--truth-model", type=Path, default=None)
    parser.add_argument("--boreholes", type=Path, default=None)
    parser.add_argument("--observation-dir", type=Path, required=True)
    parser.add_argument("--controller-manifest", type=Path, required=True)
    parser.add_argument("--controller-level", required=True)
    parser.add_argument("--target-label", type=int, default=9)
    parser.add_argument("--target-roi-radius", type=int, default=6)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--max-guidance-ratio", type=float, default=0.25)
    parser.add_argument("--tau-start", type=float, default=0.50)
    parser.add_argument("--tau-end", type=float, default=0.10)
    parser.add_argument(
        "--tau-schedule",
        choices=controller_module.TEMPERATURE_SCHEDULES,
        default="cosine",
    )
    parser.add_argument("--guidance-start", type=float, default=0.25)
    parser.add_argument(
        "--guidance-schedule",
        choices=controller_module.GUIDANCE_SCHEDULES,
        default="windowed_sine",
    )
    parser.add_argument(
        "--guidance-scaling-mode",
        choices=controller_module.GUIDANCE_SCALING_MODES,
        default=controller_module.REFERENCE_GUIDANCE_SCALING_MODE,
    )
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--baseline-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.model_weights != "ema":
        raise ValueError("Phase 4 requires the canonical EMA inference policy")
    if args.n_samples <= 0 or args.n_steps <= 0:
        raise ValueError("n-samples and n-steps must be positive")
    if args.alpha < 0:
        raise ValueError("alpha must be non-negative")
    if args.alpha == 0 and args.baseline_dir is not None:
        raise ValueError("alpha=0 defines the baseline and takes no baseline-dir")
    if args.alpha > 0 and args.baseline_dir is None:
        raise ValueError("positive alpha requires --baseline-dir")
    if args.max_guidance_ratio < 0 or args.grad_clip_norm < 0:
        raise ValueError("guidance cap and gradient clip must be non-negative")
    if not 0 <= args.guidance_start < 1:
        raise ValueError("guidance-start must satisfy 0 <= start < 1")
    if args.target_roi_radius < 0:
        raise ValueError("target ROI radius must be non-negative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty; refusing to overwrite: {args.output_dir}"
        )


def paired_gravity_config_verdict(
    baseline: Mapping[str, object],
    guided: Mapping[str, object],
) -> tuple[bool, str]:
    equal, reason = runtime.require_equal_fields(baseline, guided, PHASE4_PAIR_FIELDS)
    if not equal:
        return False, reason
    if float(baseline.get("alpha", float("nan"))) != 0:
        return False, "paired baseline alpha is not zero"
    if float(guided.get("alpha", 0)) <= 0:
        return False, "guided alpha must be positive"
    if baseline.get("run_status") != "completed":
        return False, "paired baseline did not complete"
    if int(baseline.get("samples_written", -1)) != int(guided["n_samples"]):
        return False, "paired baseline sample count is incomplete"
    return True, "strict Phase-4 gravity assets, noise, and sampler settings match"


def load_controller_level(
    path: Path,
    level_id: str,
    *,
    run_alpha: float,
    max_guidance_ratio: float,
) -> dict[str, object]:
    manifest = read_json(path)
    if manifest.get("schema") != "phase4a_gravity_controller_screen_v1":
        raise ValueError("invalid Phase-4a controller manifest schema")
    levels = manifest.get("levels")
    if not isinstance(levels, list) or not levels:
        raise ValueError("controller manifest must contain levels")
    indexed = {str(level.get("id")): level for level in levels}
    if len(indexed) != len(levels) or level_id not in indexed:
        raise ValueError(f"unknown controller level: {level_id}")
    level = indexed[level_id]
    intended_alpha = float(level["alpha"])
    intended_cap = float(level["max_guidance_ratio"])
    if run_alpha > 0 and not math.isclose(run_alpha, intended_alpha, abs_tol=1e-12):
        raise ValueError(
            f"guided alpha {run_alpha} does not match controller level {intended_alpha}"
        )
    if not math.isclose(max_guidance_ratio, intended_cap, abs_tol=1e-12):
        raise ValueError(
            f"guidance cap {max_guidance_ratio} does not match controller level {intended_cap}"
        )
    return dict(level)


def _recorded_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _validate_source_record(
    record: Mapping[str, object],
    *,
    expected_path: Path | None = None,
) -> Path:
    path = expected_path or _recorded_path(record.get("path"))
    digest = runtime.file_sha256(path)
    if digest != record.get("sha256"):
        raise ValueError(f"source asset hash mismatch: {path}")
    return path


def load_observation_assets(
    observation_dir: Path,
    truth: torch.Tensor,
    *,
    truth_path: Path,
    num_categories: int,
) -> tuple[dict[str, torch.Tensor], dict[str, object], object, dict[str, object]]:
    """Load and validate one immutable observation directory read-only."""
    manifest_path = observation_dir / "manifest.json"
    manifest = read_json(manifest_path)
    required = {
        "status": "complete",
        "truth_derived": True,
        "measured_geophysics": False,
        "inverse_crime": True,
    }
    for field, expected in required.items():
        if manifest.get(field) != expected:
            raise ValueError(f"observation manifest {field} must be {expected!r}")
    source_assets = manifest.get("source_assets")
    if not isinstance(source_assets, Mapping):
        raise ValueError("observation manifest lacks source_assets")
    truth_record = source_assets.get("truth_model")
    gravity_record = source_assets.get("gravity_source")
    density_config_record = source_assets.get("density_config")
    observation_config_record = source_assets.get("observation_config")
    if not all(
        isinstance(record, Mapping)
        for record in (
            truth_record,
            gravity_record,
            density_config_record,
            observation_config_record,
        )
    ):
        raise ValueError("observation manifest source records are incomplete")
    if runtime.file_sha256(truth_path) != truth_record.get("sha256"):
        raise ValueError("current truth file does not match observation source")
    _validate_source_record(truth_record)
    _validate_source_record(gravity_record, expected_path=Path(gravity_module.__file__))
    _validate_source_record(density_config_record)
    _validate_source_record(observation_config_record)

    tensor_records = manifest.get("generated_tensors")
    if not isinstance(tensor_records, Mapping):
        raise ValueError("observation manifest lacks generated_tensors")
    tensors: dict[str, torch.Tensor] = {}
    for filename in OBSERVATION_TENSOR_FILES:
        record = tensor_records.get(filename)
        if not isinstance(record, Mapping):
            raise ValueError(f"observation manifest lacks tensor record: {filename}")
        value = runtime.load_tensor(observation_dir / filename, map_location="cpu")
        if list(value.shape) != record.get("shape") or str(value.dtype) != record.get("dtype"):
            raise ValueError(f"observation tensor shape/dtype mismatch: {filename}")
        if tensor_sha256(value) != record.get("sha256"):
            raise ValueError(f"observation tensor hash mismatch: {filename}")
        tensors[filename] = value

    table = tensors["density_table_kg_m3.pt"]
    target_density = tensors["truth_density_kg_m3.pt"]
    if table.ndim != 1 or table.numel() != num_categories:
        raise ValueError("observation density table does not match model categories")
    expected_density = hard_labels_to_density(truth, table).to(target_density.dtype)
    if not torch.equal(expected_density, target_density):
        raise ValueError("observation truth density does not match current truth/codebook")
    resolved = manifest.get("observation_config_resolved")
    if not isinstance(resolved, Mapping):
        raise ValueError("observation manifest lacks resolved observation config")
    operator, resolved_validated = gravity_operator_from_config(
        resolved, grid_shape=truth.shape[2:]
    )
    field_shape = (1, 1, truth.shape[2], truth.shape[3])
    for filename in (
        "observed_gravity_mgal.pt",
        "noiseless_gravity_mgal.pt",
        "gravity_noise_mgal.pt",
        "survey_mask.pt",
        "uncertainty_mgal.pt",
    ):
        if tuple(tensors[filename].shape) != field_shape:
            raise ValueError(f"invalid field shape in {filename}")
    if not torch.equal(
        tensors["observed_gravity_mgal.pt"],
        tensors["noiseless_gravity_mgal.pt"] + tensors["gravity_noise_mgal.pt"],
    ):
        raise ValueError("observed gravity is inconsistent with noiseless field plus noise")
    if bool((tensors["survey_mask.pt"] < 0).any()) or float(
        tensors["survey_mask.pt"].sum()
    ) <= 0:
        raise ValueError("survey mask is invalid")
    if bool((tensors["uncertainty_mgal.pt"] <= 0).any()):
        raise ValueError("uncertainty field must be positive")
    return tensors, manifest, operator, resolved_validated


def _build_config(
    *,
    args: argparse.Namespace,
    truth_path: Path,
    boreholes_path: Path,
    model_load_report: Mapping[str, object],
    conditioning_report: Mapping[str, object],
    manifest: Mapping[str, object],
    resolved_observation: Mapping[str, object],
    tensor_assets: Mapping[str, torch.Tensor],
    target_mask: torch.Tensor,
    target_roi: torch.Tensor,
    guidance_confidence: torch.Tensor,
    controller_level: Mapping[str, object],
) -> Dict[str, object]:
    assets = runtime.experiment_asset_records(
        truth_model=truth_path,
        boreholes=boreholes_path,
        observation_manifest=args.observation_dir / "manifest.json",
        controller_manifest=args.controller_manifest,
        runner_source=Path(__file__),
        runtime_source=Path(runtime.__file__),
        gravity_source=Path(gravity_module.__file__),
        gravity_sampling_source=Path(gravity_sampling_module.__file__),
        property_evaluation_source=Path(evaluation_module.__file__),
        controller_source=Path(controller_module.__file__),
    )
    assets["checkpoint"] = model_load_report["checkpoint"]
    density_metadata = manifest["density"]
    observation_metadata = manifest["observation"]
    config: Dict[str, object] = {
        "protocol_version": runtime.PROTOCOL_VERSION,
        "phase4_protocol_version": PHASE4_PROTOCOL_VERSION,
        "stage": PHASE4_STAGE,
        "description": PHASE4_DESCRIPTION,
        "integrator": runtime.PAIRED_INTEGRATOR,
        "initial_noise_policy": runtime.INITIAL_NOISE_POLICY,
        "condition_projection": "clean_embedding_before_first_step_and_after_every_step_v1",
        "known_density_policy": "exact_truth_density_before_gravity_zero_condition_gradient_v1",
        "ckpt_path": str(args.ckpt_path),
        "model_weight_source": args.model_weights,
        "ema_applied": bool(model_load_report["ema_applied"]),
        "samples_dir": str(args.samples_dir),
        "truth_model": str(truth_path),
        "boreholes": str(boreholes_path),
        "observation_dir": str(args.observation_dir),
        "output_dir": str(args.output_dir),
        "baseline_dir": str(args.baseline_dir) if args.baseline_dir else None,
        "density_config_id": density_metadata["id"],
        "controller_level_id": str(controller_level["id"]),
        "controller_intended_alpha": float(controller_level["alpha"]),
        "controller_level": dict(controller_level),
        "density_unit": density_metadata["unit"],
        "density_table_sha256": tensor_sha256(tensor_assets["density_table_kg_m3.pt"]),
        "target_density_sha256": tensor_sha256(tensor_assets["truth_density_kg_m3.pt"]),
        "observed_gravity_sha256": tensor_sha256(tensor_assets["observed_gravity_mgal.pt"]),
        "noiseless_gravity_sha256": tensor_sha256(tensor_assets["noiseless_gravity_mgal.pt"]),
        "gravity_noise_sha256": tensor_sha256(tensor_assets["gravity_noise_mgal.pt"]),
        "survey_mask_sha256": tensor_sha256(tensor_assets["survey_mask.pt"]),
        "uncertainty_sha256": tensor_sha256(tensor_assets["uncertainty_mgal.pt"]),
        "guidance_confidence_sha256": tensor_sha256(guidance_confidence),
        "target_mask_sha256": tensor_sha256(target_mask),
        "target_roi_mask_sha256": tensor_sha256(target_roi),
        "gravity_forward_mode": observation_metadata["forward_mode"],
        "gravity_loss_mode": observation_metadata["loss_mode"],
        "grid_shape": resolved_observation["grid_shape"],
        "cell_size_m": resolved_observation["cell_size_m"],
        "origin_m": resolved_observation["origin_m"],
        "station_height_m": observation_metadata["station_height_m"],
        "observation_height_above_top_m": resolved_observation[
            "observation_height_above_top_m"
        ],
        "gravity_output_unit": observation_metadata["output_unit"],
        "survey_station_count": int(tensor_assets["survey_mask.pt"].sum().item()),
        "truth_derived": True,
        "measured_geophysics": False,
        "inverse_crime": True,
        "target_label": int(args.target_label),
        "target_roi_radius": int(args.target_roi_radius),
        "n_samples": int(args.n_samples),
        "n_steps": int(args.n_steps),
        "alpha": float(args.alpha),
        "max_guidance_ratio": float(args.max_guidance_ratio),
        "tau_start": float(args.tau_start),
        "tau_end": float(args.tau_end),
        "tau_schedule": args.tau_schedule,
        "guidance_start": float(args.guidance_start),
        "guidance_schedule": args.guidance_schedule,
        "guidance_scaling_mode": args.guidance_scaling_mode,
        "guidance_gradient_reference_policy": (
            controller_module.GUIDANCE_GRADIENT_REFERENCE_POLICY
            if args.guidance_scaling_mode == controller_module.REFERENCE_GUIDANCE_SCALING_MODE
            else None
        ),
        "grad_clip_norm": float(args.grad_clip_norm),
        "seed": int(args.seed),
        "device": str(torch.device(args.device)),
        "gravity_sampler_version": gravity_sampling_module.GRAVITY_SAMPLER_VERSION,
        "asset_records": assets,
        "model_load_report": dict(model_load_report),
        "conditioning_report": dict(conditioning_report),
        "observation_manifest": dict(manifest),
        "run_status": "running",
    }
    config.update(runtime.flatten_asset_hashes(assets))
    return config


def add_hard_gravity_metrics(
    row: Dict[str, object],
    *,
    prediction: torch.Tensor,
    truth_density: torch.Tensor,
    condition_mask: torch.Tensor,
    density_table: torch.Tensor,
    forward_operator,
    observed_mgal: torch.Tensor,
    survey_mask: torch.Tensor,
    uncertainty_mgal: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    labels = runtime.normalize_single_geology(prediction, "prediction").long().to(device)
    predicted_density = hard_labels_to_density(labels, density_table.to(device))
    known_density = overwrite_exact_condition_density(
        predicted_density,
        truth_density.to(device=device, dtype=predicted_density.dtype),
        condition_mask.to(device),
    )
    with torch.no_grad():
        field = forward_operator(known_density)
        loss, diagnostics = gravity_field_loss(
            field,
            observed_mgal.to(device=device, dtype=field.dtype),
            survey_mask.to(device=device, dtype=field.dtype),
            uncertainty_mgal.to(device=device, dtype=field.dtype),
        )
    row["hard_gravity_loss"] = float(loss.cpu())
    row["hard_gravity_rmse_mgal"] = float(diagnostics["gravity_rmse_mgal"].cpu())
    row["hard_gravity_mae_mgal"] = float(diagnostics["gravity_mae_mgal"].cpu())
    row["hard_observation_loss"] = row["hard_gravity_loss"]
    row["hard_observation_mae"] = row["hard_gravity_mae_mgal"]
    return field.cpu()


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable; use the GPU terminal or --device cpu for smoke tests"
        )
    truth_path = args.truth_model or args.samples_dir / "true_model.pt"
    boreholes_path = args.boreholes or args.samples_dir / "boreholes.pt"
    truth_cpu = runtime.normalize_single_geology(
        runtime.load_tensor(truth_path, map_location="cpu"), str(truth_path)
    ).long()
    boreholes_cpu = runtime.normalize_single_geology(
        runtime.load_tensor(boreholes_path, map_location="cpu"), str(boreholes_path)
    ).long()
    controller_level = load_controller_level(
        args.controller_manifest,
        args.controller_level,
        run_alpha=args.alpha,
        max_guidance_ratio=args.max_guidance_ratio,
    )

    from model_train_sh_inference_cond import Geo3DStochInterp

    model, model_load_report = runtime.load_model_with_weight_policy(
        model_class=Geo3DStochInterp,
        checkpoint_path=args.ckpt_path,
        map_location=device,
        weight_source=args.model_weights,
    )
    model = model.to(device)
    conditioning_report = runtime.validate_conditioning_pair(
        truth_cpu, boreholes_cpu, model.num_categories, target_label=args.target_label
    )
    tensor_assets, observation_manifest, forward_operator, resolved_observation = (
        load_observation_assets(
            args.observation_dir,
            truth_cpu,
            truth_path=truth_path,
            num_categories=model.num_categories,
        )
    )
    density_table = tensor_assets["density_table_kg_m3.pt"]
    target_density = tensor_assets["truth_density_kg_m3.pt"]
    observed_mgal = tensor_assets["observed_gravity_mgal.pt"]
    survey_mask = tensor_assets["survey_mask.pt"]
    uncertainty_mgal = tensor_assets["uncertainty_mgal.pt"]
    condition_mask_cpu = (boreholes_cpu != -1) | (truth_cpu == -1)
    guidance_confidence_cpu = ((truth_cpu != -1) & ~condition_mask_cpu).float()
    if int(guidance_confidence_cpu.sum()) == 0:
        raise ValueError("gravity guidance region has no unconstrained non-air voxels")
    target_mask, target_metadata = build_target_mask(
        truth_cpu, target_label=args.target_label, component_mode="all"
    )
    target_roi = dilate_mask(target_mask, args.target_roi_radius)

    config = _build_config(
        args=args,
        truth_path=truth_path,
        boreholes_path=boreholes_path,
        model_load_report=model_load_report,
        conditioning_report=conditioning_report,
        manifest=observation_manifest,
        resolved_observation=resolved_observation,
        tensor_assets=tensor_assets,
        target_mask=target_mask,
        target_roi=target_roi,
        guidance_confidence=guidance_confidence_cpu,
        controller_level=controller_level,
    )
    baseline_config: Dict[str, object] | None = None
    if args.baseline_dir is not None:
        baseline_config = read_json(args.baseline_dir / "config.json")
        paired, reason = paired_gravity_config_verdict(baseline_config, config)
        if not paired:
            raise ValueError(f"strict baseline pairing failed: {reason}")
        config["pairing_validation"] = {
            "paired": True,
            "reason": reason,
            "baseline_config": str(args.baseline_dir / "config.json"),
        }
    else:
        config["pairing_validation"] = None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved = {
        "density_table.pt": density_table,
        "target_density.pt": target_density,
        "observed_gravity_mgal.pt": observed_mgal,
        "noiseless_gravity_mgal.pt": tensor_assets["noiseless_gravity_mgal.pt"],
        "gravity_noise_mgal.pt": tensor_assets["gravity_noise_mgal.pt"],
        "survey_mask.pt": survey_mask,
        "uncertainty_mgal.pt": uncertainty_mgal,
        "guidance_confidence.pt": guidance_confidence_cpu,
        "target_mask.pt": target_mask,
        "target_roi_mask.pt": target_roi,
    }
    for filename, tensor in saved.items():
        torch.save(tensor, args.output_dir / filename)
    write_json(args.output_dir / "observation_manifest.json", observation_manifest)
    write_json(args.output_dir / "model_load_report.json", model_load_report)
    write_json(args.output_dir / "input_validation.json", conditioning_report)
    write_json(
        args.output_dir / "evaluation_manifest.json",
        {
            "target_metadata": target_metadata,
            "hard_density_loss": "single_scale_full_3d_density_context_only",
            "hard_gravity_loss": GRAVITY_LOSS_MODE,
            "continuous_loss_alone_is_not_success": True,
        },
    )
    write_json(args.output_dir / "config.json", config)

    truth = truth_cpu.to(device)
    condition_mask = condition_mask_cpu.to(device)
    embedded_truth = model.embed(truth)
    embedded_mask = condition_mask.expand(-1, embedded_truth.shape[1], -1, -1, -1)
    conditioning = embedded_truth * embedded_mask
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    expected_initial_hashes = (
        list(baseline_config.get("initial_noise_sha256", []))
        if baseline_config is not None
        else []
    )
    if baseline_config is not None and len(expected_initial_hashes) != args.n_samples:
        raise ValueError("paired baseline lacks complete initial-noise hashes")

    traces: list[Dict[str, object]] = []
    metrics_rows: list[Dict[str, object]] = []
    class_rows: list[Dict[str, object]] = []
    component_rows: list[Dict[str, object]] = []
    baseline_metrics_rows: list[Dict[str, object]] = []
    baseline_class_rows: list[Dict[str, object]] = []
    baseline_component_rows: list[Dict[str, object]] = []
    delta_rows: list[Dict[str, object]] = []
    transition_rows: list[Dict[str, object]] = []
    decoded_samples: list[torch.Tensor] = []
    baseline_samples: list[torch.Tensor] = []
    soft_gravity_fields: list[torch.Tensor] = []
    hard_gravity_fields: list[torch.Tensor] = []
    initial_hashes: list[str] = []
    sample_hashes: list[str] = []
    all_class_ids = list(range(0, model.num_categories - 1))
    density_property_table = density_table.unsqueeze(0)
    density_channel_weight = torch.ones(1)

    for sample_id in range(args.n_samples):
        initial_cpu = torch.randn(
            1,
            model.embedding_dim,
            *model.data_shape,
            generator=generator,
            dtype=embedded_truth.dtype,
        )
        initial_hash = tensor_sha256(initial_cpu)
        if baseline_config is not None and initial_hash != expected_initial_hashes[sample_id]:
            raise ValueError(f"sample {sample_id} initial noise differs from paired baseline")
        initial_hashes.append(initial_hash)
        final_state, sample_trace = fixed_euler_gravity_sample(
            model=model,
            initial_state=initial_cpu.to(device),
            conditioning=conditioning,
            embedded_truth=embedded_truth,
            truth_model=truth,
            condition_mask=condition_mask,
            target_density=target_density,
            density_table=density_table,
            guidance_confidence=guidance_confidence_cpu,
            forward_operator=forward_operator,
            observed_mgal=observed_mgal,
            survey_mask=survey_mask,
            uncertainty_mgal=uncertainty_mgal,
            n_steps=args.n_steps,
            alpha=args.alpha,
            max_guidance_ratio=args.max_guidance_ratio,
            tau_start=args.tau_start,
            tau_end=args.tau_end,
            tau_schedule=args.tau_schedule,
            guidance_start=args.guidance_start,
            guidance_schedule=args.guidance_schedule,
            grad_clip_norm=args.grad_clip_norm,
            guidance_scaling_mode=args.guidance_scaling_mode,
            sample_id=sample_id,
        )
        if not torch.isfinite(final_state).all():
            raise FloatingPointError(f"sample {sample_id} contains NaN or Inf")
        with torch.no_grad():
            probabilities = soft_decode_to_probs(
                final_state, model.embedding.weight, tau=args.tau_end
            )
            soft_density = probabilities_to_density(probabilities, density_table.to(device))
            known_density = overwrite_exact_condition_density(
                soft_density,
                target_density.to(device=device, dtype=soft_density.dtype),
                condition_mask,
            )
            soft_gravity_fields.append(forward_operator(known_density).cpu())
        decoded = (model.decode(final_state).detach().cpu() - 1)[0]
        sample_path = args.output_dir / f"sample_{sample_id}.pt"
        torch.save(decoded, sample_path)
        sample_hashes.append(tensor_sha256(decoded))
        decoded_samples.append(decoded)

        baseline_decoded = None
        if args.baseline_dir is not None:
            baseline_decoded = runtime.load_tensor(
                args.baseline_dir / f"sample_{sample_id}.pt", map_location="cpu"
            )
        metrics = sample_property_hard_metrics(
            prediction=decoded,
            truth_model=truth_cpu,
            condition_mask=condition_mask_cpu,
            target_mask=target_mask,
            target_roi_mask=target_roi,
            target_label=args.target_label,
            property_table=density_property_table,
            property_confidence=guidance_confidence_cpu,
            property_sigmas=(0.0,),
            property_scale_weights=(1.0,),
            property_channel_weights=density_channel_weight,
            sample_id=sample_id,
            baseline_prediction=baseline_decoded,
        )
        hard_gravity_fields.append(
            add_hard_gravity_metrics(
                metrics,
                prediction=decoded,
                truth_density=target_density,
                condition_mask=condition_mask_cpu,
                density_table=density_table,
                forward_operator=forward_operator,
                observed_mgal=observed_mgal,
                survey_mask=survey_mask,
                uncertainty_mgal=uncertainty_mgal,
                device=device,
            )
        )
        metrics["path"] = str(sample_path)
        metrics_rows.append(metrics)
        current_class_rows = per_class_hard_metrics(
            decoded, truth_cpu, sample_id, class_ids=all_class_ids
        )
        class_rows.extend(current_class_rows)
        component_rows.extend(
            truth_component_recovery_rows(decoded, truth_cpu, args.target_label, sample_id)
        )

        if baseline_decoded is not None:
            baseline_samples.append(baseline_decoded.squeeze())
            baseline_metrics = sample_property_hard_metrics(
                prediction=baseline_decoded,
                truth_model=truth_cpu,
                condition_mask=condition_mask_cpu,
                target_mask=target_mask,
                target_roi_mask=target_roi,
                target_label=args.target_label,
                property_table=density_property_table,
                property_confidence=guidance_confidence_cpu,
                property_sigmas=(0.0,),
                property_scale_weights=(1.0,),
                property_channel_weights=density_channel_weight,
                sample_id=sample_id,
            )
            add_hard_gravity_metrics(
                baseline_metrics,
                prediction=baseline_decoded,
                truth_density=target_density,
                condition_mask=condition_mask_cpu,
                density_table=density_table,
                forward_operator=forward_operator,
                observed_mgal=observed_mgal,
                survey_mask=survey_mask,
                uncertainty_mgal=uncertainty_mgal,
                device=device,
            )
            baseline_metrics_rows.append(baseline_metrics)
            deltas = paired_property_metric_deltas(baseline_metrics, metrics)
            for field in (
                "hard_gravity_loss",
                "hard_gravity_rmse_mgal",
                "hard_gravity_mae_mgal",
                "hard_observation_loss",
                "hard_observation_mae",
            ):
                deltas[f"delta_{field}"] = float(metrics[field]) - float(
                    baseline_metrics[field]
                )
            delta_rows.append(deltas)
            current_baseline_classes = per_class_hard_metrics(
                baseline_decoded, truth_cpu, sample_id, class_ids=all_class_ids
            )
            baseline_class_rows.extend(current_baseline_classes)
            baseline_component_rows.extend(
                truth_component_recovery_rows(
                    baseline_decoded, truth_cpu, args.target_label, sample_id
                )
            )
            transition_rows.extend(
                class_transition_records(baseline_decoded, decoded, sample_id)
            )
        traces.extend(sample_trace)

    soft_fields = torch.cat(soft_gravity_fields, dim=0)
    hard_fields = torch.cat(hard_gravity_fields, dim=0)
    torch.save(soft_fields, args.output_dir / "final_soft_gravity_mgal.pt")
    torch.save(hard_fields, args.output_dir / "hard_gravity_fields_mgal.pt")
    write_rows(args.output_dir / "guidance_trace.csv", traces)
    write_rows(args.output_dir / "sample_metrics.csv", metrics_rows)
    write_rows(args.output_dir / "per_class_metrics.csv", class_rows)
    write_rows(args.output_dir / "truth_component_recovery.csv", component_rows)
    write_json(args.output_dir / "metrics_summary.json", summarize_rows(metrics_rows))
    write_json(args.output_dir / "per_class_summary.json", summarize_per_class_rows(class_rows))

    ensemble_summary: Dict[str, object] = {
        "current": ensemble_diversity_summary(
            torch.stack(decoded_samples),
            target_mask,
            target_roi,
            args.target_label,
            sample_hashes=sample_hashes,
        )
    }
    if delta_rows:
        class_delta_rows = paired_per_class_deltas(baseline_class_rows, class_rows)
        write_rows(args.output_dir / "paired_baseline_metrics.csv", baseline_metrics_rows)
        write_rows(args.output_dir / "paired_deltas.csv", delta_rows)
        write_rows(
            args.output_dir / "paired_baseline_per_class_metrics.csv", baseline_class_rows
        )
        write_rows(args.output_dir / "paired_per_class_deltas.csv", class_delta_rows)
        write_rows(
            args.output_dir / "paired_baseline_truth_component_recovery.csv",
            baseline_component_rows,
        )
        write_rows(
            args.output_dir / "paired_truth_component_recovery_deltas.csv",
            paired_truth_component_recovery_deltas(
                baseline_component_rows, component_rows
            ),
        )
        write_rows(args.output_dir / "paired_class_transitions.csv", transition_rows)
        write_json(args.output_dir / "paired_delta_summary.json", summarize_rows(delta_rows))
        ensemble_summary["baseline"] = ensemble_diversity_summary(
            torch.stack(baseline_samples),
            target_mask,
            target_roi,
            args.target_label,
            sample_hashes=list(baseline_config.get("sample_sha256", [])),
        )
    write_json(args.output_dir / "ensemble_summary.json", ensemble_summary)

    config.update(
        {
            "run_status": "completed",
            "samples_written": len(metrics_rows),
            "initial_noise_sha256": initial_hashes,
            "sample_sha256": sample_hashes,
            "final_soft_gravity_sha256": tensor_sha256(soft_fields),
            "hard_gravity_fields_sha256": tensor_sha256(hard_fields),
            "max_post_projection_condition_violations": max(
                int(row["post_projection_condition_violations"]) for row in traces
            ),
        }
    )
    write_json(args.output_dir / "config.json", config)


if __name__ == "__main__":
    main()
