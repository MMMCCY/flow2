#!/usr/bin/env python3
"""Run strict paired Phase-4c convolutional seismic guidance."""

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
from guidance import probability_sampling as controller_module
from guidance import property_evaluation as evaluation_module
from guidance import seismic as seismic_module
from guidance import seismic_sampling as seismic_sampling_module
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
from guidance.seismic import (
    SEISMIC_FORWARD_MODE,
    SEISMIC_LOSS_MODE,
    SUBSURFACE_SOFT_ACOUSTIC_POLICY,
    hard_labels_to_acoustic,
    overwrite_exact_condition_acoustic,
    probabilities_to_subsurface_acoustic,
    seismic_field_loss,
    seismic_operator_from_config,
    tensor_sha256,
    validate_contiguous_subsurface_mask,
)
from guidance.seismic_sampling import fixed_euler_seismic_sample
from guided_geophysical_sampling import soft_decode_to_probs


PHASE4C_PROTOCOL_VERSION = 1
PHASE4C_STAGE = "phase4c_seismic_v1"
PHASE4C_DESCRIPTION = (
    "Truth-derived normal-incidence convolutional seismic guidance; "
    "synthetic inverse crime, not measured geophysics."
)
OBSERVATION_TENSOR_FILES = (
    "density_table_kg_m3.pt",
    "velocity_table_m_s.pt",
    "impedance_table_kg_m2_s.pt",
    "slowness_table_s_m.pt",
    "acoustic_property_table.pt",
    "truth_acoustic.pt",
    "subsurface_mask.pt",
    "observed_seismic.pt",
    "noiseless_seismic.pt",
    "seismic_noise.pt",
    "sample_mask.pt",
    "uncertainty_amplitude.pt",
    "wavelet.pt",
)
SAVED_PAIR_TENSORS = {
    "acoustic_property_table.pt": "acoustic_property_table_sha256",
    "target_acoustic.pt": "target_acoustic_sha256",
    "subsurface_mask.pt": "subsurface_mask_sha256",
    "observed_seismic.pt": "observed_seismic_sha256",
    "noiseless_seismic.pt": "noiseless_seismic_sha256",
    "seismic_noise.pt": "seismic_noise_sha256",
    "sample_mask.pt": "sample_mask_sha256",
    "uncertainty_amplitude.pt": "uncertainty_sha256",
    "guidance_confidence.pt": "guidance_confidence_sha256",
    "target_mask.pt": "target_mask_sha256",
    "target_roi_mask.pt": "target_roi_mask_sha256",
}
PHASE4C_PAIR_FIELDS = (
    "protocol_version",
    "phase4c_protocol_version",
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
    "seismic_source_sha256",
    "seismic_sampling_source_sha256",
    "property_evaluation_source_sha256",
    "controller_source_sha256",
    "acoustic_property_table_sha256",
    "target_acoustic_sha256",
    "subsurface_mask_sha256",
    "observed_seismic_sha256",
    "noiseless_seismic_sha256",
    "seismic_noise_sha256",
    "sample_mask_sha256",
    "uncertainty_sha256",
    "guidance_confidence_sha256",
    "target_mask_sha256",
    "target_roi_mask_sha256",
    "acoustic_config_id",
    "controller_level_id",
    "controller_intended_alpha",
    "seismic_forward_mode",
    "seismic_loss_mode",
    "grid_shape",
    "cell_size_m",
    "num_time_samples",
    "sample_interval_ms",
    "wavelet_sha256",
    "trace_sample_count",
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
    "known_acoustic_policy",
    "subsurface_soft_acoustic_policy",
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
        description=PHASE4C_DESCRIPTION,
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
        "--tau-schedule", choices=controller_module.TEMPERATURE_SCHEDULES, default="cosine"
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


def paired_seismic_config_verdict(
    baseline: Mapping[str, object],
    guided: Mapping[str, object],
) -> tuple[bool, str]:
    equal, reason = runtime.require_equal_fields(baseline, guided, PHASE4C_PAIR_FIELDS)
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
    return True, "strict Phase-4c seismic assets, noise, and sampler settings match"


def load_controller_level(
    path: Path,
    level_id: str,
    *,
    run_alpha: float,
    max_guidance_ratio: float,
) -> dict[str, object]:
    manifest = read_json(path)
    if manifest.get("schema") != "phase4c_seismic_controller_screen_v1":
        raise ValueError("invalid Phase-4c controller manifest schema")
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
    record: Mapping[str, object], *, expected_path: Path | None = None
) -> Path:
    path = expected_path or _recorded_path(record.get("path"))
    if runtime.file_sha256(path) != record.get("sha256"):
        raise ValueError(f"source asset hash mismatch: {path}")
    return path


def load_observation_assets(
    observation_dir: Path,
    truth: torch.Tensor,
    *,
    truth_path: Path,
    num_categories: int,
) -> tuple[dict[str, torch.Tensor], dict[str, object], object, dict[str, object]]:
    """Load and validate one immutable seismic observation directory read-only."""
    manifest = read_json(observation_dir / "manifest.json")
    for field, expected in {
        "status": "complete",
        "truth_derived": True,
        "measured_geophysics": False,
        "inverse_crime": True,
    }.items():
        if manifest.get(field) != expected:
            raise ValueError(f"observation manifest {field} must be {expected!r}")
    source_assets = manifest.get("source_assets")
    if not isinstance(source_assets, Mapping):
        raise ValueError("observation manifest lacks source_assets")
    records = {
        name: source_assets.get(name)
        for name in ("truth_model", "seismic_source", "acoustic_config", "observation_config")
    }
    if not all(isinstance(record, Mapping) for record in records.values()):
        raise ValueError("observation manifest source records are incomplete")
    if runtime.file_sha256(truth_path) != records["truth_model"].get("sha256"):
        raise ValueError("current truth file does not match observation source")
    _validate_source_record(records["truth_model"])
    _validate_source_record(records["seismic_source"], expected_path=Path(seismic_module.__file__))
    _validate_source_record(records["acoustic_config"])
    _validate_source_record(records["observation_config"])

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

    table = tensors["acoustic_property_table.pt"]
    target = tensors["truth_acoustic.pt"]
    if table.shape != (2, num_categories):
        raise ValueError("observation acoustic table does not match model categories")
    expected_target = hard_labels_to_acoustic(truth, table).to(target.dtype)
    if not torch.equal(expected_target, target):
        raise ValueError("observation truth acoustic volume does not match truth/codebook")
    expected_subsurface = truth != -1
    if not torch.equal(tensors["subsurface_mask.pt"].bool(), expected_subsurface):
        raise ValueError("observation subsurface mask does not match known surface")
    validate_contiguous_subsurface_mask(tensors["subsurface_mask.pt"])
    resolved = manifest.get("observation_config_resolved")
    if not isinstance(resolved, Mapping):
        raise ValueError("observation manifest lacks resolved observation config")
    operator, resolved_validated = seismic_operator_from_config(
        resolved, grid_shape=truth.shape[2:]
    )
    field_shape = (
        1,
        1,
        truth.shape[2],
        truth.shape[3],
        operator.num_time_samples,
    )
    for filename in (
        "observed_seismic.pt",
        "noiseless_seismic.pt",
        "seismic_noise.pt",
        "sample_mask.pt",
        "uncertainty_amplitude.pt",
    ):
        if tuple(tensors[filename].shape) != field_shape:
            raise ValueError(f"invalid seismic field shape in {filename}")
    if not torch.equal(
        tensors["observed_seismic.pt"],
        tensors["noiseless_seismic.pt"] + tensors["seismic_noise.pt"],
    ):
        raise ValueError("observed seismic is inconsistent with noiseless data plus noise")
    if bool((tensors["sample_mask.pt"] < 0).any()) or float(
        tensors["sample_mask.pt"].sum()
    ) <= 0:
        raise ValueError("seismic sample mask is invalid")
    if bool((tensors["uncertainty_amplitude.pt"] <= 0).any()):
        raise ValueError("seismic uncertainty must be positive")
    if not torch.equal(
        tensors["wavelet.pt"], operator.wavelet(torch.device("cpu"), torch.float64)
    ):
        raise ValueError("saved wavelet does not match resolved operator")
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
        seismic_source=Path(seismic_module.__file__),
        seismic_sampling_source=Path(seismic_sampling_module.__file__),
        property_evaluation_source=Path(evaluation_module.__file__),
        controller_source=Path(controller_module.__file__),
    )
    assets["checkpoint"] = model_load_report["checkpoint"]
    acoustic_metadata = manifest["acoustic"]
    observation_metadata = manifest["observation"]
    time_sampling = resolved_observation["time_sampling"]
    config: Dict[str, object] = {
        "protocol_version": runtime.PROTOCOL_VERSION,
        "phase4c_protocol_version": PHASE4C_PROTOCOL_VERSION,
        "stage": PHASE4C_STAGE,
        "description": PHASE4C_DESCRIPTION,
        "integrator": runtime.PAIRED_INTEGRATOR,
        "initial_noise_policy": runtime.INITIAL_NOISE_POLICY,
        "condition_projection": "clean_embedding_before_first_step_and_after_every_step_v1",
        "known_acoustic_policy": "exact_truth_acoustic_before_seismic_zero_condition_gradient_v1",
        "subsurface_soft_acoustic_policy": SUBSURFACE_SOFT_ACOUSTIC_POLICY,
        "ckpt_path": str(args.ckpt_path),
        "model_weight_source": args.model_weights,
        "ema_applied": bool(model_load_report["ema_applied"]),
        "samples_dir": str(args.samples_dir),
        "truth_model": str(truth_path),
        "boreholes": str(boreholes_path),
        "observation_dir": str(args.observation_dir),
        "output_dir": str(args.output_dir),
        "baseline_dir": str(args.baseline_dir) if args.baseline_dir else None,
        "acoustic_config_id": acoustic_metadata["id"],
        "controller_level_id": str(controller_level["id"]),
        "controller_intended_alpha": float(controller_level["alpha"]),
        "controller_level": dict(controller_level),
        "acoustic_property_table_sha256": tensor_sha256(
            tensor_assets["acoustic_property_table.pt"]
        ),
        "target_acoustic_sha256": tensor_sha256(tensor_assets["truth_acoustic.pt"]),
        "subsurface_mask_sha256": tensor_sha256(tensor_assets["subsurface_mask.pt"]),
        "observed_seismic_sha256": tensor_sha256(tensor_assets["observed_seismic.pt"]),
        "noiseless_seismic_sha256": tensor_sha256(tensor_assets["noiseless_seismic.pt"]),
        "seismic_noise_sha256": tensor_sha256(tensor_assets["seismic_noise.pt"]),
        "sample_mask_sha256": tensor_sha256(tensor_assets["sample_mask.pt"]),
        "uncertainty_sha256": tensor_sha256(tensor_assets["uncertainty_amplitude.pt"]),
        "guidance_confidence_sha256": tensor_sha256(guidance_confidence),
        "target_mask_sha256": tensor_sha256(target_mask),
        "target_roi_mask_sha256": tensor_sha256(target_roi),
        "seismic_forward_mode": observation_metadata["forward_mode"],
        "seismic_loss_mode": observation_metadata["loss_mode"],
        "grid_shape": resolved_observation["grid_shape"],
        "cell_size_m": resolved_observation["cell_size_m"],
        "num_time_samples": int(time_sampling["num_samples"]),
        "sample_interval_ms": float(time_sampling["sample_interval_ms"]),
        "wavelet_sha256": tensor_sha256(tensor_assets["wavelet.pt"]),
        "trace_sample_count": int(tensor_assets["sample_mask.pt"].sum().item()),
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
        "seismic_sampler_version": seismic_sampling_module.SEISMIC_SAMPLER_VERSION,
        "asset_records": assets,
        "model_load_report": dict(model_load_report),
        "conditioning_report": dict(conditioning_report),
        "observation_manifest": dict(manifest),
        "run_status": "running",
    }
    config.update(runtime.flatten_asset_hashes(assets))
    return config


def add_hard_seismic_metrics(
    row: Dict[str, object],
    *,
    prediction: torch.Tensor,
    target_acoustic: torch.Tensor,
    condition_mask: torch.Tensor,
    property_table: torch.Tensor,
    subsurface_mask: torch.Tensor,
    forward_operator,
    observed: torch.Tensor,
    sample_mask: torch.Tensor,
    uncertainty: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    labels = runtime.normalize_single_geology(prediction, "prediction").long().to(device)
    predicted = hard_labels_to_acoustic(labels, property_table.to(device))
    known = overwrite_exact_condition_acoustic(
        predicted,
        target_acoustic.to(device=device, dtype=predicted.dtype),
        condition_mask.to(device),
    )
    with torch.no_grad():
        field = forward_operator(
            known[:, 0:1], known[:, 1:2], subsurface_mask.to(device)
        )
        loss, diagnostics = seismic_field_loss(
            field,
            observed.to(device=device, dtype=field.dtype),
            sample_mask.to(device=device, dtype=field.dtype),
            uncertainty.to(device=device, dtype=field.dtype),
        )
    row["hard_seismic_loss"] = float(loss.cpu())
    row["hard_seismic_rmse_amplitude"] = float(
        diagnostics["seismic_rmse_amplitude"].cpu()
    )
    row["hard_seismic_mae_amplitude"] = float(
        diagnostics["seismic_mae_amplitude"].cpu()
    )
    row["hard_observation_loss"] = row["hard_seismic_loss"]
    row["hard_observation_mae"] = row["hard_seismic_mae_amplitude"]
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
    property_table = tensor_assets["acoustic_property_table.pt"]
    target_acoustic = tensor_assets["truth_acoustic.pt"]
    subsurface_mask = tensor_assets["subsurface_mask.pt"]
    observed = tensor_assets["observed_seismic.pt"]
    sample_mask = tensor_assets["sample_mask.pt"]
    uncertainty = tensor_assets["uncertainty_amplitude.pt"]
    condition_mask_cpu = (boreholes_cpu != -1) | (truth_cpu == -1)
    guidance_confidence_cpu = ((truth_cpu != -1) & ~condition_mask_cpu).float()
    if int(guidance_confidence_cpu.sum()) == 0:
        raise ValueError("seismic guidance region has no unconstrained non-air voxels")
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
        paired, reason = paired_seismic_config_verdict(baseline_config, config)
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
        "acoustic_property_table.pt": property_table,
        "target_acoustic.pt": target_acoustic,
        "subsurface_mask.pt": subsurface_mask,
        "observed_seismic.pt": observed,
        "noiseless_seismic.pt": tensor_assets["noiseless_seismic.pt"],
        "seismic_noise.pt": tensor_assets["seismic_noise.pt"],
        "sample_mask.pt": sample_mask,
        "uncertainty_amplitude.pt": uncertainty,
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
            "hard_acoustic_loss": "single_scale_full_3d_impedance_slowness_context_only",
            "hard_seismic_loss": SEISMIC_LOSS_MODE,
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
    soft_fields: list[torch.Tensor] = []
    hard_fields: list[torch.Tensor] = []
    initial_hashes: list[str] = []
    sample_hashes: list[str] = []
    all_class_ids = list(range(0, model.num_categories - 1))
    property_channel_weight = torch.ones(2)

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
        final_state, sample_trace = fixed_euler_seismic_sample(
            model=model,
            initial_state=initial_cpu.to(device),
            conditioning=conditioning,
            embedded_truth=embedded_truth,
            truth_model=truth,
            condition_mask=condition_mask,
            target_acoustic=target_acoustic,
            property_table=property_table,
            guidance_confidence=guidance_confidence_cpu,
            subsurface_mask=subsurface_mask,
            forward_operator=forward_operator,
            observed=observed,
            sample_mask=sample_mask,
            uncertainty=uncertainty,
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
            soft_acoustic = probabilities_to_subsurface_acoustic(
                probabilities,
                property_table.to(device),
                subsurface_mask.to(device),
            )
            known_acoustic = overwrite_exact_condition_acoustic(
                soft_acoustic,
                target_acoustic.to(device=device, dtype=soft_acoustic.dtype),
                condition_mask,
            )
            soft_fields.append(
                forward_operator(
                    known_acoustic[:, 0:1],
                    known_acoustic[:, 1:2],
                    subsurface_mask.to(device),
                ).cpu()
            )
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
            property_table=property_table,
            property_confidence=guidance_confidence_cpu,
            property_sigmas=(0.0,),
            property_scale_weights=(1.0,),
            property_channel_weights=property_channel_weight,
            sample_id=sample_id,
            baseline_prediction=baseline_decoded,
        )
        hard_fields.append(
            add_hard_seismic_metrics(
                metrics,
                prediction=decoded,
                target_acoustic=target_acoustic,
                condition_mask=condition_mask_cpu,
                property_table=property_table,
                subsurface_mask=subsurface_mask,
                forward_operator=forward_operator,
                observed=observed,
                sample_mask=sample_mask,
                uncertainty=uncertainty,
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
                property_table=property_table,
                property_confidence=guidance_confidence_cpu,
                property_sigmas=(0.0,),
                property_scale_weights=(1.0,),
                property_channel_weights=property_channel_weight,
                sample_id=sample_id,
            )
            add_hard_seismic_metrics(
                baseline_metrics,
                prediction=baseline_decoded,
                target_acoustic=target_acoustic,
                condition_mask=condition_mask_cpu,
                property_table=property_table,
                subsurface_mask=subsurface_mask,
                forward_operator=forward_operator,
                observed=observed,
                sample_mask=sample_mask,
                uncertainty=uncertainty,
                device=device,
            )
            baseline_metrics_rows.append(baseline_metrics)
            deltas = paired_property_metric_deltas(baseline_metrics, metrics)
            for field in (
                "hard_seismic_loss",
                "hard_seismic_rmse_amplitude",
                "hard_seismic_mae_amplitude",
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

    soft_field = torch.cat(soft_fields, dim=0)
    hard_field = torch.cat(hard_fields, dim=0)
    torch.save(soft_field, args.output_dir / "final_soft_seismic.pt")
    torch.save(hard_field, args.output_dir / "hard_seismic_fields.pt")
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
            "final_soft_seismic_sha256": tensor_sha256(soft_field),
            "hard_seismic_fields_sha256": tensor_sha256(hard_field),
            "max_post_projection_condition_violations": max(
                int(row["post_projection_condition_violations"]) for row in traces
            ),
        }
    )
    write_json(args.output_dir / "config.json", config)


if __name__ == "__main__":
    main()
