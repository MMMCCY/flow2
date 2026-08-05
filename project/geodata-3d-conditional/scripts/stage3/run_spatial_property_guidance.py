#!/usr/bin/env python3
"""Run strict Phase-3 spatially degraded 3-D property guidance."""

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
from guidance import spatial_property as spatial_module
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
from guidance.spatial_property import (
    SPATIAL_PROPERTY_LOSS_MODE,
    apply_spatial_property_operator,
    build_spatial_property_observation,
    hard_spatial_observation_loss,
    overwrite_known_properties,
    spatial_property_volume_loss,
)
from guided_geophysical_sampling import soft_decode_to_probs


PHASE3_PROTOCOL_VERSION = 1
PHASE3_STAGE = "phase3_spatial_property_v1"
PHASE3_DESCRIPTION = (
    "Truth-derived spatially degraded 3-D full-lithology property observation; "
    "not measured or acquisition-domain geophysics."
)
PHASE3_PAIR_FIELDS = (
    "protocol_version",
    "phase3_protocol_version",
    "stage",
    "checkpoint_sha256",
    "model_weight_source",
    "ema_applied",
    "truth_model_sha256",
    "boreholes_sha256",
    "property_config_sha256",
    "observation_config_sha256",
    "runner_source_sha256",
    "runtime_source_sha256",
    "property_volume_source_sha256",
    "property_sampling_source_sha256",
    "property_evaluation_source_sha256",
    "spatial_property_source_sha256",
    "controller_source_sha256",
    "property_table_sha256",
    "target_properties_sha256",
    "base_property_confidence_sha256",
    "observation_values_sha256",
    "noiseless_observation_sha256",
    "observation_confidence_sha256",
    "observation_noise_sha256",
    "target_mask_sha256",
    "target_roi_mask_sha256",
    "property_config_schema",
    "spatial_property_config_schema",
    "spatial_property_config_id",
    "spatial_operator",
    "spatial_confidence",
    "spatial_noise",
    "known_property_policy",
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


def _float_list(values: Sequence[float] | None, default: Sequence[float]) -> list[float]:
    return [float(value) for value in (values if values is not None else default)]


def _normalized_weights(values: Sequence[float]) -> list[float]:
    parsed = [float(value) for value in values]
    if any(value < 0 for value in parsed) or sum(parsed) <= 0:
        raise ValueError("property scale weights must be non-negative and sum positive")
    total = sum(parsed)
    return [value / total for value in parsed]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=PHASE3_DESCRIPTION,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--model-weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--truth-model", type=Path, default=None)
    parser.add_argument("--boreholes", type=Path, default=None)
    parser.add_argument("--property-config", type=Path, required=True)
    parser.add_argument("--observation-config", type=Path, required=True)
    parser.add_argument(
        "--confidence-mode",
        choices=("unconditioned_nonair_v1",),
        default="unconditioned_nonair_v1",
    )
    parser.add_argument("--property-sigma", action="append", type=float, default=None)
    parser.add_argument("--property-scale-weight", action="append", type=float, default=None)
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
        raise ValueError("Phase 3 requires the canonical EMA inference policy")
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
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty; refusing to overwrite: {args.output_dir}"
        )


def paired_spatial_property_config_verdict(
    baseline: Mapping[str, object],
    guided: Mapping[str, object],
) -> tuple[bool, str]:
    equal, reason = runtime.require_equal_fields(baseline, guided, PHASE3_PAIR_FIELDS)
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
    return True, "strict Phase-3 observation assets, noise, and sampler settings match"


def _build_config(
    *,
    args: argparse.Namespace,
    truth_path: Path,
    boreholes_path: Path,
    property_metadata: Mapping[str, object],
    observation_metadata: Mapping[str, object],
    model_load_report: Mapping[str, object],
    conditioning_report: Mapping[str, object],
    property_table: torch.Tensor,
    target_properties: torch.Tensor,
    base_confidence: torch.Tensor,
    target_mask: torch.Tensor,
    target_roi: torch.Tensor,
    property_sigmas: Sequence[float],
    property_scale_weights: Sequence[float],
) -> Dict[str, object]:
    assets = runtime.experiment_asset_records(
        truth_model=truth_path,
        boreholes=boreholes_path,
        property_config=args.property_config,
        observation_config=args.observation_config,
        runner_source=Path(__file__),
        runtime_source=Path(runtime.__file__),
        property_volume_source=Path(property_module.__file__),
        property_sampling_source=Path(sampling_module.__file__),
        property_evaluation_source=Path(evaluation_module.__file__),
        spatial_property_source=Path(spatial_module.__file__),
        controller_source=Path(controller_module.__file__),
    )
    assets["checkpoint"] = model_load_report["checkpoint"]
    config: Dict[str, object] = {
        "protocol_version": runtime.PROTOCOL_VERSION,
        "phase3_protocol_version": PHASE3_PROTOCOL_VERSION,
        "stage": PHASE3_STAGE,
        "description": PHASE3_DESCRIPTION,
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
        "observation_config": str(args.observation_config),
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
        "base_property_confidence_sha256": tensor_sha256(base_confidence),
        "observation_values_sha256": observation_metadata["observation_values_sha256"],
        "noiseless_observation_sha256": observation_metadata[
            "noiseless_observation_sha256"
        ],
        "observation_confidence_sha256": observation_metadata[
            "observation_confidence_sha256"
        ],
        "observation_noise_sha256": observation_metadata["observation_noise_sha256"],
        "target_mask_sha256": tensor_sha256(target_mask),
        "target_roi_mask_sha256": tensor_sha256(target_roi),
        "spatial_property_config_schema": observation_metadata["schema"],
        "spatial_property_config_id": observation_metadata["id"],
        "spatial_operator": observation_metadata["operator"],
        "spatial_confidence": observation_metadata["confidence"],
        "spatial_noise": observation_metadata["noise"],
        "known_property_policy": observation_metadata["known_property_policy"],
        "property_sigmas": list(property_sigmas),
        "property_scale_weights": list(property_scale_weights),
        "property_loss_mode": SPATIAL_PROPERTY_LOSS_MODE,
        "property_sampler_version": sampling_module.PROPERTY_SAMPLER_VERSION,
        "confidence_mode": args.confidence_mode,
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
        "observation_metadata": dict(observation_metadata),
        "run_status": "running",
    }
    config.update(runtime.flatten_asset_hashes(assets))
    return config


def _add_hard_observation_metrics(
    row: Dict[str, object],
    *,
    prediction: torch.Tensor,
    truth: torch.Tensor,
    condition_mask: torch.Tensor,
    property_table: torch.Tensor,
    observation,
    observation_config: Mapping[str, object],
    property_sigmas: Sequence[float],
    property_scale_weights: Sequence[float],
    property_channel_weights: torch.Tensor,
) -> None:
    predicted_labels = runtime.normalize_single_geology(
        prediction,
        "prediction",
    ).long()
    truth_labels = runtime.normalize_single_geology(
        truth,
        "truth",
    ).long()
    predicted_properties = hard_labels_to_properties(
        predicted_labels,
        property_table,
    )
    target_properties = hard_labels_to_properties(truth_labels, property_table)
    loss, diagnostics = hard_spatial_observation_loss(
        predicted_properties,
        target_properties,
        condition_mask,
        observation,
        observation_config,
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
        runtime.load_tensor(truth_path, map_location="cpu"), str(truth_path)
    ).long()
    boreholes_cpu = runtime.normalize_single_geology(
        runtime.load_tensor(boreholes_path, map_location="cpu"), str(boreholes_path)
    ).long()
    property_config = _read_json(args.property_config)
    observation_config = _read_json(args.observation_config)
    property_sigmas = _float_list(args.property_sigma, (0.0, 1.5, 3.0))
    property_scale_weights = _normalized_weights(
        _float_list(args.property_scale_weight, (0.50, 0.30, 0.20))
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
    property_table, property_channel_weights, property_metadata = property_table_from_config(
        property_config,
        model.num_categories,
    )
    target_properties = hard_labels_to_properties(truth_cpu, property_table)
    condition_mask_cpu = (boreholes_cpu != -1) | (truth_cpu == -1)
    base_confidence_cpu = ((truth_cpu != -1) & ~condition_mask_cpu).float()
    if int(base_confidence_cpu.sum().item()) == 0:
        raise ValueError("base property confidence contains no active voxels")
    observation = build_spatial_property_observation(
        target_properties,
        base_confidence_cpu,
        truth_cpu != -1,
        observation_config,
    )
    target_mask, target_metadata = build_target_mask(
        truth_cpu,
        target_label=args.target_label,
        component_mode="all",
    )
    target_roi = dilate_mask(target_mask, args.target_roi_radius)

    config = _build_config(
        args=args,
        truth_path=truth_path,
        boreholes_path=boreholes_path,
        property_metadata=property_metadata,
        observation_metadata=observation.metadata,
        model_load_report=model_load_report,
        conditioning_report=conditioning_report,
        property_table=property_table,
        target_properties=target_properties,
        base_confidence=base_confidence_cpu,
        target_mask=target_mask,
        target_roi=target_roi,
        property_sigmas=property_sigmas,
        property_scale_weights=property_scale_weights,
    )
    baseline_config: Dict[str, object] | None = None
    if args.baseline_dir is not None:
        baseline_config = _read_json(args.baseline_dir / "config.json")
        paired, reason = paired_spatial_property_config_verdict(baseline_config, config)
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
    torch.save(base_confidence_cpu, args.output_dir / "base_property_confidence.pt")
    torch.save(observation.values, args.output_dir / "observation_values.pt")
    torch.save(observation.noiseless_values, args.output_dir / "observation_noiseless.pt")
    torch.save(observation.confidence, args.output_dir / "observation_confidence.pt")
    torch.save(observation.noise, args.output_dir / "observation_noise.pt")
    torch.save(target_mask, args.output_dir / "target_mask.pt")
    torch.save(target_roi, args.output_dir / "target_roi_mask.pt")
    _write_json(args.output_dir / "property_config_resolved.json", property_config)
    _write_json(args.output_dir / "observation_config_resolved.json", observation_config)
    _write_json(
        args.output_dir / "observation_manifest.json",
        {
            **observation.metadata,
            "property_config": str(args.property_config),
            "observation_config": str(args.observation_config),
            "property_config_sha256": config["property_config_sha256"],
            "observation_config_sha256": config["observation_config_sha256"],
            "property_table_sha256": config["property_table_sha256"],
            "target_label": int(args.target_label),
            "target_metadata": target_metadata,
            "truth_derived": True,
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
    observation_values_device = observation.values.to(device)
    observation_confidence_device = observation.confidence.to(device)
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
    truth_component_rows: list[Dict[str, object]] = []
    baseline_metrics_rows: list[Dict[str, object]] = []
    baseline_class_rows: list[Dict[str, object]] = []
    baseline_truth_component_rows: list[Dict[str, object]] = []
    delta_rows: list[Dict[str, object]] = []
    transition_rows: list[Dict[str, object]] = []
    decoded_samples: list[torch.Tensor] = []
    baseline_samples: list[torch.Tensor] = []
    final_expected_observations: list[torch.Tensor] = []
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
            confidence=base_confidence_cpu,
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
            loss_function=spatial_property_volume_loss,
            loss_extra_kwargs={
                "observed_properties": observation_values_device,
                "observation_confidence": observation_confidence_device,
                "observation_config": observation_config,
                "condition_mask": condition_mask,
            },
            loss_mode=SPATIAL_PROPERTY_LOSS_MODE,
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
                property_table.to(device),
            )
            final_known = overwrite_known_properties(
                final_properties,
                target_properties,
                condition_mask,
            )
            final_observation = apply_spatial_property_operator(
                final_known,
                observation.metadata["operator"],
            ).cpu()
        final_expected_observations.append(final_observation)
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
            property_confidence=base_confidence_cpu,
            property_sigmas=property_sigmas,
            property_scale_weights=property_scale_weights,
            property_channel_weights=property_channel_weights,
            sample_id=sample_id,
            baseline_prediction=baseline_decoded,
        )
        _add_hard_observation_metrics(
            metrics,
            prediction=decoded,
            truth=truth_cpu,
            condition_mask=condition_mask_cpu,
            property_table=property_table,
            observation=observation,
            observation_config=observation_config,
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
                property_confidence=base_confidence_cpu,
                property_sigmas=property_sigmas,
                property_scale_weights=property_scale_weights,
                property_channel_weights=property_channel_weights,
                sample_id=sample_id,
            )
            _add_hard_observation_metrics(
                baseline_metrics,
                prediction=baseline_decoded,
                truth=truth_cpu,
                condition_mask=condition_mask_cpu,
                property_table=property_table,
                observation=observation,
                observation_config=observation_config,
                property_sigmas=property_sigmas,
                property_scale_weights=property_scale_weights,
                property_channel_weights=property_channel_weights,
            )
            baseline_metrics_rows.append(baseline_metrics)
            deltas = paired_property_metric_deltas(baseline_metrics, metrics)
            deltas["delta_hard_observation_loss"] = float(
                metrics["hard_observation_loss"]
            ) - float(baseline_metrics["hard_observation_loss"])
            deltas["delta_hard_observation_mae"] = float(
                metrics["hard_observation_mae"]
            ) - float(baseline_metrics["hard_observation_mae"])
            delta_rows.append(deltas)
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

    final_expected = torch.cat(final_expected_observations, dim=0)
    torch.save(final_expected, args.output_dir / "final_expected_observations.pt")
    _write_rows(args.output_dir / "guidance_trace.csv", traces)
    _write_rows(args.output_dir / "sample_metrics.csv", metrics_rows)
    _write_rows(args.output_dir / "per_class_metrics.csv", class_rows)
    _write_rows(args.output_dir / "truth_component_recovery.csv", truth_component_rows)
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
        class_delta_rows = paired_per_class_deltas(baseline_class_rows, class_rows)
        _write_rows(args.output_dir / "paired_baseline_metrics.csv", baseline_metrics_rows)
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
        _write_rows(args.output_dir / "paired_class_transitions.csv", transition_rows)
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
            "final_expected_observations_sha256": tensor_sha256(final_expected),
            "max_post_projection_condition_violations": max(
                int(row["post_projection_condition_violations"]) for row in traces
            ),
        }
    )
    _write_json(args.output_dir / "config.json", config)


if __name__ == "__main__":
    main()
