#!/usr/bin/env python3
"""Run strictly paired truth-derived 3-D full-lithology property guidance."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import inference_runtime as runtime
from guidance import probability_sampling as controller_module
from guidance import property_evaluation as evaluation_module
from guidance import property_sampling as sampling_module
from guidance import property_volume as property_module
from guidance.probability_evaluation import (
    class_transition_records,
    ensemble_diversity_summary,
    summarize_rows,
)
from guidance.probability_volume import build_target_mask, dilate_mask, tensor_sha256
from guidance.property_evaluation import (
    paired_per_class_deltas,
    paired_property_metric_deltas,
    paired_truth_component_recovery_deltas,
    per_class_hard_metrics,
    sample_property_hard_metrics,
    summarize_per_class_rows,
    truth_component_recovery_rows,
)
from guidance.property_sampling import fixed_euler_property_sample
from guidance.property_volume import (
    hard_labels_to_properties,
    probabilities_to_expected_properties,
    property_table_from_config,
)
from guided_geophysical_sampling import soft_decode_to_probs


PHASE2_PROTOCOL_VERSION = 1
PHASE2_EXPERIMENT_STAGES = {
    "phase2a_ideal_3d_property": (
        "Truth-derived full-resolution full-lithology 3-D property upper bound; "
        "not measured or inverted geophysics."
    ),
    "phase2b_codebook_ambiguity_v1": (
        "Truth-derived full-resolution 3-D property codebook ambiguity/contrast "
        "ablation; not measured or inverted geophysics."
    ),
    "phase5b_inversion_property_bridge_v1": (
        "No-training flow bridge using the truth-blind Phase-5a posterior mean "
        "log-impedance and spread-derived confidence; synthetic inverse-crime, "
        "not measured geophysics."
    ),
}
PHASE2_PAIR_FIELDS = (
    "protocol_version",
    "phase2_protocol_version",
    "stage",
    "checkpoint_sha256",
    "model_weight_source",
    "ema_applied",
    "truth_model_sha256",
    "boreholes_sha256",
    "property_config_sha256",
    "runner_source_sha256",
    "runtime_source_sha256",
    "property_volume_source_sha256",
    "property_sampling_source_sha256",
    "property_evaluation_source_sha256",
    "controller_source_sha256",
    "property_table_sha256",
    "target_properties_sha256",
    "property_confidence_sha256",
    "target_mask_sha256",
    "target_roi_mask_sha256",
    "property_config_schema",
    "property_channel_names",
    "property_channel_units",
    "property_channel_weights",
    "property_sigmas",
    "property_scale_weights",
    "property_loss_mode",
    "confidence_mode",
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
    "device",
)


def _parse_float_list(
    values: Sequence[float] | None,
    defaults: Sequence[float],
) -> list[float]:
    return [float(value) for value in (values if values is not None else defaults)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strict fixed-Euler guidance with a truth-derived complete 3-D "
            "property volume. This is an ideal property oracle, not measured geophysics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--model-weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--truth-model", type=Path, default=None)
    parser.add_argument("--boreholes", type=Path, default=None)
    parser.add_argument("--property-config", type=Path, required=True)
    parser.add_argument(
        "--experiment-stage",
        choices=tuple(PHASE2_EXPERIMENT_STAGES),
        default="phase2a_ideal_3d_property",
    )
    parser.add_argument(
        "--confidence-mode",
        choices=("unconditioned_nonair_v1", "external_posterior_spread_v1"),
        default="unconditioned_nonair_v1",
    )
    parser.add_argument(
        "--external-property-dir",
        type=Path,
        default=None,
        help="Completed Phase-5b target/confidence directory; allowed only in the Phase-5b stage.",
    )
    parser.add_argument("--property-sigma", action="append", type=float, default=None)
    parser.add_argument(
        "--property-scale-weight",
        action="append",
        type=float,
        default=None,
    )
    parser.add_argument("--target-label", type=int, default=9)
    parser.add_argument("--target-roi-radius", type=int, default=6)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--max-guidance-ratio", type=float, default=0.10)
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
        raise ValueError("Phase 2 requires the canonical EMA inference policy")
    if args.n_samples <= 0 or args.n_steps <= 0:
        raise ValueError("n_samples and n_steps must be positive")
    if args.alpha < 0:
        raise ValueError("alpha must be non-negative")
    if args.alpha == 0 and args.baseline_dir is not None:
        raise ValueError("alpha=0 defines the paired baseline and takes no baseline-dir")
    if args.alpha > 0 and args.baseline_dir is None:
        raise ValueError("positive alpha requires --baseline-dir for strict pairing")
    if args.max_guidance_ratio < 0 or args.grad_clip_norm < 0:
        raise ValueError("guidance cap and gradient clip must be non-negative")
    if not 0 <= args.guidance_start < 1:
        raise ValueError("guidance_start must satisfy 0 <= start < 1")
    if args.target_roi_radius < 0:
        raise ValueError("target ROI radius must be non-negative")
    is_bridge = getattr(args, "experiment_stage", "phase2a_ideal_3d_property") == (
        "phase5b_inversion_property_bridge_v1"
    )
    external_property_dir = getattr(args, "external_property_dir", None)
    confidence_mode = getattr(args, "confidence_mode", "unconditioned_nonair_v1")
    if is_bridge and external_property_dir is None:
        raise ValueError("Phase-5b requires --external-property-dir")
    if is_bridge and confidence_mode != "external_posterior_spread_v1":
        raise ValueError("Phase-5b requires external_posterior_spread_v1 confidence")
    if not is_bridge and external_property_dir is not None:
        raise ValueError("external property assets are allowed only in Phase-5b")
    if not is_bridge and confidence_mode != "unconditioned_nonair_v1":
        raise ValueError("Phase-2 stages require unconditioned_nonair_v1 confidence")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty; refusing to overwrite: {args.output_dir}"
        )


def paired_property_config_verdict(
    baseline: Mapping[str, object],
    guided: Mapping[str, object],
) -> tuple[bool, str]:
    equal, reason = runtime.require_equal_fields(
        baseline,
        guided,
        PHASE2_PAIR_FIELDS,
    )
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
    return True, "strict Phase-2 property assets, noise, and sampler settings match"


def _read_json(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
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


def _normalized_scale_weights(values: Sequence[float]) -> list[float]:
    parsed = [float(value) for value in values]
    if any(value < 0 for value in parsed) or sum(parsed) <= 0:
        raise ValueError("property scale weights must be non-negative and sum positive")
    total = sum(parsed)
    return [value / total for value in parsed]


def _load_external_property_assets(
    directory: Path,
    *,
    property_table: torch.Tensor,
    condition_mask: torch.Tensor,
    expected_shape: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Load and validate the completed truth-blind Phase-5b bridge assets."""
    manifest_path = directory / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != "phase5b_inversion_property_assets_v1":
        raise ValueError("external property manifest has the wrong schema")
    if manifest.get("status") != "complete":
        raise ValueError("external property manifest is not complete")
    for field, expected in {
        "truth_geology_loaded": False,
        "truth_acoustic_loaded": False,
        "truth_metrics_used_for_construction": False,
        "phase5a_pass_bit_used_as_stop_gate": True,
    }.items():
        if manifest.get(field) is not expected:
            raise ValueError(f"external property manifest {field} must be {expected}")
    records = manifest.get("generated_tensors")
    if not isinstance(records, Mapping):
        raise ValueError("external property manifest lacks generated tensors")

    def load(filename: str) -> torch.Tensor:
        record = records.get(filename)
        if not isinstance(record, Mapping):
            raise ValueError(f"external property tensor is missing: {filename}")
        path = directory / filename
        if runtime.file_sha256(path) != record.get("sha256"):
            raise ValueError(f"external property file hash mismatch: {filename}")
        value = runtime.load_tensor(path)
        if tensor_sha256(value) != record.get("tensor_sha256"):
            raise ValueError(f"external property tensor hash mismatch: {filename}")
        return value

    saved_table = load("property_table.pt")
    target = load("target_properties.pt")
    confidence = load("property_confidence.pt")
    saved_condition = load("condition_mask.pt").bool()
    if not torch.equal(saved_table, property_table):
        raise ValueError("external property table differs from parsed config")
    if tuple(target.shape) != tuple(expected_shape):
        raise ValueError("external target shape does not match the model volume")
    if confidence.shape != (expected_shape[0], 1, *expected_shape[2:]):
        raise ValueError("external confidence shape does not match the model volume")
    if not torch.equal(saved_condition, condition_mask.bool()):
        raise ValueError("external condition mask differs from current conditions")
    if not torch.isfinite(target).all() or not torch.isfinite(confidence).all():
        raise ValueError("external property assets must be finite")
    if bool((confidence < 0).any()) or bool((confidence > 1).any()):
        raise ValueError("external confidence must remain in [0,1]")
    if bool(confidence[condition_mask.bool()].any()):
        raise ValueError("external confidence must be zero at hard conditions")
    if float(confidence.sum()) <= 0:
        raise ValueError("external confidence contains no active voxels")
    return target, confidence, manifest


def _build_config(
    args: argparse.Namespace,
    truth_path: Path,
    boreholes_path: Path,
    property_sigmas: Sequence[float],
    property_scale_weights: Sequence[float],
    property_metadata: Mapping[str, object],
    model_load_report: Mapping[str, object],
    conditioning_report: Mapping[str, object],
    property_table: torch.Tensor,
    target_properties: torch.Tensor,
    confidence: torch.Tensor,
    target_mask: torch.Tensor,
    target_roi: torch.Tensor,
    external_property_manifest: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    assets = runtime.experiment_asset_records(
        truth_model=truth_path,
        boreholes=boreholes_path,
        property_config=args.property_config,
        runner_source=Path(__file__),
        runtime_source=Path(runtime.__file__),
        property_volume_source=Path(property_module.__file__),
        property_sampling_source=Path(sampling_module.__file__),
        property_evaluation_source=Path(evaluation_module.__file__),
        controller_source=Path(controller_module.__file__),
        external_property_manifest=(
            args.external_property_dir / "manifest.json"
            if args.external_property_dir is not None
            else None
        ),
    )
    assets["checkpoint"] = model_load_report["checkpoint"]
    config: Dict[str, object] = {
        "protocol_version": runtime.PROTOCOL_VERSION,
        "phase2_protocol_version": PHASE2_PROTOCOL_VERSION,
        "stage": args.experiment_stage,
        "description": PHASE2_EXPERIMENT_STAGES[args.experiment_stage],
        "integrator": runtime.PAIRED_INTEGRATOR,
        "initial_noise_policy": runtime.INITIAL_NOISE_POLICY,
        "condition_projection": "clean_embedding_before_first_step_and_after_every_step_v1",
        "ckpt_path": str(args.ckpt_path),
        "model_weight_source": args.model_weights,
        "ema_applied": bool(model_load_report["ema_applied"]),
        "samples_dir": str(args.samples_dir),
        "truth_model": str(truth_path),
        "boreholes": str(boreholes_path),
        "property_config": str(args.property_config),
        "output_dir": str(args.output_dir),
        "baseline_dir": str(args.baseline_dir) if args.baseline_dir else None,
        "property_config_schema": property_metadata["schema"],
        "property_channel_names": property_metadata["channel_names"],
        "property_channel_units": property_metadata["channel_units"],
        "property_channel_weights": property_metadata["channel_weights"],
        "property_table_sha256": tensor_sha256(property_table),
        "property_codebook_diagnostics": property_module.property_codebook_diagnostics(
            property_table,
            torch.as_tensor(property_metadata["channel_weights"]),
            target_raw_label=args.target_label,
        ),
        "target_properties_sha256": tensor_sha256(target_properties),
        "property_confidence_sha256": tensor_sha256(confidence),
        "target_mask_sha256": tensor_sha256(target_mask),
        "target_roi_mask_sha256": tensor_sha256(target_roi),
        "property_sigmas": list(property_sigmas),
        "property_scale_weights": list(property_scale_weights),
        "property_loss_mode": property_module.PROPERTY_LOSS_MODE,
        "property_sampler_version": sampling_module.PROPERTY_SAMPLER_VERSION,
        "confidence_mode": args.confidence_mode,
        "external_property_dir": (
            str(args.external_property_dir)
            if args.external_property_dir is not None
            else None
        ),
        "external_property_manifest_schema": (
            external_property_manifest.get("schema")
            if external_property_manifest is not None
            else None
        ),
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
            if args.guidance_scaling_mode
            == controller_module.REFERENCE_GUIDANCE_SCALING_MODE
            else None
        ),
        "grad_clip_norm": float(args.grad_clip_norm),
        "seed": int(args.seed),
        "device": str(torch.device(args.device)),
        "asset_records": assets,
        "model_load_report": dict(model_load_report),
        "conditioning_report": dict(conditioning_report),
        "property_metadata": dict(property_metadata),
        "run_status": "running",
    }
    config.update(runtime.flatten_asset_hashes(assets))
    return config


def _add_external_observation_metrics(
    row: Dict[str, object],
    *,
    prediction: torch.Tensor,
    target_properties: torch.Tensor,
    property_table: torch.Tensor,
    confidence: torch.Tensor,
    property_sigmas: Sequence[float],
    property_scale_weights: Sequence[float],
    property_channel_weights: torch.Tensor,
) -> None:
    predicted = runtime.normalize_single_geology(prediction, "prediction").long()
    predicted_properties = hard_labels_to_properties(predicted, property_table)
    loss, diagnostics = property_module.matched_multiscale_property_loss(
        predicted_properties,
        target_properties,
        confidence,
        sigmas=property_sigmas,
        scale_weights=property_scale_weights,
        channel_weights=property_channel_weights,
    )
    row["hard_observation_loss"] = float(loss.detach().cpu())
    row["hard_observation_mae"] = float(
        diagnostics["property_mae_mean"].detach().cpu()
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable in this process; run in the "
            "GPU-enabled terminal or use --device cpu only for utility smoke tests"
        )

    truth_path = args.truth_model or args.samples_dir / "true_model.pt"
    boreholes_path = args.boreholes or args.samples_dir / "boreholes.pt"
    truth_cpu = runtime.normalize_single_geology(
        runtime.load_tensor(truth_path, map_location="cpu"),
        str(truth_path),
    ).long()
    boreholes_cpu = runtime.normalize_single_geology(
        runtime.load_tensor(boreholes_path, map_location="cpu"),
        str(boreholes_path),
    ).long()
    property_config = _read_json(args.property_config)
    property_sigmas = _parse_float_list(args.property_sigma, (0.0, 1.5, 3.0))
    property_scale_weights = _normalized_scale_weights(
        _parse_float_list(args.property_scale_weight, (0.50, 0.30, 0.20))
    )
    if len(property_sigmas) != len(property_scale_weights):
        raise ValueError("property sigmas and weights must have the same length")

    from model_train_sh_inference_cond import Geo3DStochInterp

    model, model_load_report = runtime.load_model_with_weight_policy(
        model_class=Geo3DStochInterp,
        checkpoint_path=args.ckpt_path,
        map_location=device,
        weight_source=args.model_weights,
    )
    model = model.to(device)
    conditioning_report = runtime.validate_conditioning_pair(
        truth=truth_cpu,
        boreholes=boreholes_cpu,
        num_categories=model.num_categories,
        target_label=args.target_label,
    )
    property_table, property_channel_weights, property_metadata = (
        property_table_from_config(property_config, model.num_categories)
    )
    condition_mask_cpu = (boreholes_cpu != -1) | (truth_cpu == -1)
    external_property_manifest: dict[str, object] | None = None
    if args.experiment_stage == "phase5b_inversion_property_bridge_v1":
        target_properties, confidence_cpu, external_property_manifest = (
            _load_external_property_assets(
                args.external_property_dir,
                property_table=property_table,
                condition_mask=condition_mask_cpu,
                expected_shape=(
                    1,
                    property_table.shape[0],
                    *truth_cpu.shape[2:],
                ),
            )
        )
    else:
        target_properties = hard_labels_to_properties(truth_cpu, property_table)
        confidence_cpu = ((truth_cpu != -1) & ~condition_mask_cpu).float()
    if int(confidence_cpu.sum().item()) == 0:
        raise ValueError("property confidence contains no active voxels")
    target_mask, target_metadata = build_target_mask(
        truth_cpu,
        target_label=args.target_label,
        component_mode="all",
    )
    target_roi = dilate_mask(target_mask, args.target_roi_radius)

    config = _build_config(
        args,
        truth_path,
        boreholes_path,
        property_sigmas,
        property_scale_weights,
        property_metadata,
        model_load_report,
        conditioning_report,
        property_table,
        target_properties,
        confidence_cpu,
        target_mask,
        target_roi,
        external_property_manifest,
    )
    baseline_config: Dict[str, object] | None = None
    if args.baseline_dir is not None:
        baseline_config = _read_json(args.baseline_dir / "config.json")
        paired, reason = paired_property_config_verdict(baseline_config, config)
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
    torch.save(property_table, args.output_dir / "property_table.pt")
    torch.save(target_properties, args.output_dir / "target_properties.pt")
    torch.save(confidence_cpu, args.output_dir / "property_confidence.pt")
    torch.save(target_mask, args.output_dir / "target_mask.pt")
    torch.save(target_roi, args.output_dir / "target_roi_mask.pt")
    _write_json(
        args.output_dir / "property_config_resolved.json",
        property_config,
    )
    _write_json(
        args.output_dir / "property_manifest.json",
        {
            **property_metadata,
            "property_config": str(args.property_config),
            "property_config_sha256": config["property_config_sha256"],
            "property_table_sha256": config["property_table_sha256"],
            "target_properties_sha256": config["target_properties_sha256"],
            "property_confidence_sha256": config["property_confidence_sha256"],
            "confidence_mode": args.confidence_mode,
            "active_confidence_voxels": int(confidence_cpu.sum().item()),
            "active_confidence_fraction": float(confidence_cpu.mean().item()),
            "property_sigmas": property_sigmas,
            "property_scale_weights": property_scale_weights,
            "target_label": args.target_label,
            "target_metadata": target_metadata,
            "target_roi_radius": args.target_roi_radius,
            "target_roi_voxels": int(target_roi.sum().item()),
            "truth_derived": args.experiment_stage != "phase5b_inversion_property_bridge_v1",
            "inversion_posterior_derived": args.experiment_stage
            == "phase5b_inversion_property_bridge_v1",
            "external_property_manifest": (
                str(args.external_property_dir / "manifest.json")
                if args.external_property_dir is not None
                else None
            ),
            "is_measured_geophysics": False,
        },
    )
    _write_json(args.output_dir / "model_load_report.json", model_load_report)
    _write_json(args.output_dir / "input_validation.json", conditioning_report)
    _write_json(args.output_dir / "config.json", config)

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
    baseline_metrics_rows: list[Dict[str, object]] = []
    baseline_class_rows: list[Dict[str, object]] = []
    truth_component_rows: list[Dict[str, object]] = []
    baseline_truth_component_rows: list[Dict[str, object]] = []
    delta_rows: list[Dict[str, object]] = []
    transition_rows: list[Dict[str, object]] = []
    decoded_samples: list[torch.Tensor] = []
    baseline_samples: list[torch.Tensor] = []
    final_expected_properties: list[torch.Tensor] = []
    initial_hashes: list[str] = []
    sample_hashes: list[str] = []
    all_class_ids = list(range(0, model.num_categories - 1))

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
            raise ValueError(
                f"sample {sample_id} initial-noise hash differs from paired baseline"
            )
        initial_hashes.append(initial_hash)
        final_state, sample_trace = fixed_euler_property_sample(
            model=model,
            initial_state=initial_cpu.to(device),
            conditioning=conditioning,
            embedded_truth=embedded_truth,
            truth_model=truth,
            condition_mask=condition_mask,
            target_properties=target_properties,
            property_table=property_table,
            confidence=confidence_cpu,
            property_sigmas=property_sigmas,
            property_scale_weights=property_scale_weights,
            property_channel_weights=property_channel_weights,
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
            final_probabilities = soft_decode_to_probs(
                final_state,
                model.embedding.weight,
                tau=args.tau_end,
            )
            final_properties = probabilities_to_expected_properties(
                final_probabilities,
                property_table,
            ).cpu()
        final_expected_properties.append(final_properties)
        decoded = (model.decode(final_state).detach().cpu() - 1)[0]
        sample_path = args.output_dir / f"sample_{sample_id}.pt"
        torch.save(decoded, sample_path)
        sample_hashes.append(tensor_sha256(decoded))
        decoded_samples.append(decoded)

        baseline_decoded = None
        if args.baseline_dir is not None:
            baseline_decoded = runtime.load_tensor(
                args.baseline_dir / f"sample_{sample_id}.pt",
                map_location="cpu",
            )
        metrics = sample_property_hard_metrics(
            prediction=decoded,
            truth_model=truth_cpu,
            condition_mask=condition_mask_cpu,
            target_mask=target_mask,
            target_roi_mask=target_roi,
            target_label=args.target_label,
            property_table=property_table,
            property_confidence=confidence_cpu,
            property_sigmas=property_sigmas,
            property_scale_weights=property_scale_weights,
            property_channel_weights=property_channel_weights,
            sample_id=sample_id,
            baseline_prediction=baseline_decoded,
        )
        if external_property_manifest is not None:
            _add_external_observation_metrics(
                metrics,
                prediction=decoded,
                target_properties=target_properties,
                property_table=property_table,
                confidence=confidence_cpu,
                property_sigmas=property_sigmas,
                property_scale_weights=property_scale_weights,
                property_channel_weights=property_channel_weights,
            )
        metrics["path"] = str(sample_path)
        metrics_rows.append(metrics)
        current_class_rows = per_class_hard_metrics(
            decoded,
            truth_cpu,
            sample_id,
            class_ids=all_class_ids,
        )
        class_rows.extend(current_class_rows)
        truth_component_rows.extend(
            truth_component_recovery_rows(
                decoded,
                truth_cpu,
                args.target_label,
                sample_id,
            )
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
                property_confidence=confidence_cpu,
                property_sigmas=property_sigmas,
                property_scale_weights=property_scale_weights,
                property_channel_weights=property_channel_weights,
                sample_id=sample_id,
            )
            if external_property_manifest is not None:
                _add_external_observation_metrics(
                    baseline_metrics,
                    prediction=baseline_decoded,
                    target_properties=target_properties,
                    property_table=property_table,
                    confidence=confidence_cpu,
                    property_sigmas=property_sigmas,
                    property_scale_weights=property_scale_weights,
                    property_channel_weights=property_channel_weights,
                )
            baseline_metrics_rows.append(baseline_metrics)
            paired_deltas = paired_property_metric_deltas(baseline_metrics, metrics)
            if external_property_manifest is not None:
                paired_deltas["delta_hard_observation_loss"] = float(
                    metrics["hard_observation_loss"]
                ) - float(baseline_metrics["hard_observation_loss"])
                paired_deltas["delta_hard_observation_mae"] = float(
                    metrics["hard_observation_mae"]
                ) - float(baseline_metrics["hard_observation_mae"])
            delta_rows.append(paired_deltas)
            current_baseline_class_rows = per_class_hard_metrics(
                baseline_decoded,
                truth_cpu,
                sample_id,
                class_ids=all_class_ids,
            )
            baseline_class_rows.extend(current_baseline_class_rows)
            baseline_truth_component_rows.extend(
                truth_component_recovery_rows(
                    baseline_decoded,
                    truth_cpu,
                    args.target_label,
                    sample_id,
                )
            )
            transition_rows.extend(
                class_transition_records(baseline_decoded, decoded, sample_id)
            )
        traces.extend(sample_trace)

    final_expected = torch.cat(final_expected_properties, dim=0)
    torch.save(final_expected, args.output_dir / "final_expected_properties.pt")
    _write_rows(args.output_dir / "guidance_trace.csv", traces)
    _write_rows(args.output_dir / "sample_metrics.csv", metrics_rows)
    _write_rows(args.output_dir / "per_class_metrics.csv", class_rows)
    _write_rows(
        args.output_dir / "truth_component_recovery.csv",
        truth_component_rows,
    )
    _write_json(args.output_dir / "metrics_summary.json", summarize_rows(metrics_rows))
    _write_json(
        args.output_dir / "per_class_summary.json",
        summarize_per_class_rows(class_rows),
    )

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
        class_delta_rows = paired_per_class_deltas(
            baseline_class_rows,
            class_rows,
        )
        _write_rows(
            args.output_dir / "paired_baseline_metrics.csv",
            baseline_metrics_rows,
        )
        _write_rows(args.output_dir / "paired_deltas.csv", delta_rows)
        _write_rows(
            args.output_dir / "paired_baseline_per_class_metrics.csv",
            baseline_class_rows,
        )
        _write_rows(args.output_dir / "paired_per_class_deltas.csv", class_delta_rows)
        _write_rows(
            args.output_dir / "paired_baseline_truth_component_recovery.csv",
            baseline_truth_component_rows,
        )
        _write_rows(
            args.output_dir / "paired_truth_component_recovery_deltas.csv",
            paired_truth_component_recovery_deltas(
                baseline_truth_component_rows,
                truth_component_rows,
            ),
        )
        _write_rows(
            args.output_dir / "paired_class_transitions.csv",
            transition_rows,
        )
        _write_json(
            args.output_dir / "paired_delta_summary.json",
            summarize_rows(delta_rows),
        )
        ensemble_summary["baseline"] = ensemble_diversity_summary(
            torch.stack(baseline_samples),
            target_mask,
            target_roi,
            args.target_label,
            sample_hashes=list(baseline_config.get("sample_sha256", [])),
        )
    _write_json(args.output_dir / "ensemble_summary.json", ensemble_summary)

    config.update(
        {
            "run_status": "completed",
            "samples_written": len(metrics_rows),
            "initial_noise_sha256": initial_hashes,
            "sample_sha256": sample_hashes,
            "final_expected_properties_sha256": tensor_sha256(final_expected),
            "max_post_projection_condition_violations": max(
                int(row["post_projection_condition_violations"]) for row in traces
            ),
        }
    )
    _write_json(args.output_dir / "config.json", config)


if __name__ == "__main__":
    main()
