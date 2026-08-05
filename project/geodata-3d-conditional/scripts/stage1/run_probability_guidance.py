#!/usr/bin/env python3
"""Run the Phase-1 oracle 3-D probability-volume guidance experiment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for import_root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import inference_runtime as runtime
from guidance import probability_evaluation as evaluation_module
from guidance import probability_sampling as sampling_module
from guidance import probability_volume as probability_module
from guidance.probability_evaluation import (
    class_transition_records,
    ensemble_diversity_summary,
    paired_metric_deltas,
    sample_hard_metrics,
    summarize_rows,
)
from guidance.probability_sampling import fixed_euler_probability_sample
from guidance.probability_volume import (
    build_probability_volume,
    build_target_mask,
    compute_target_soft_fields,
    dilate_mask,
    paired_target_soft_deltas,
    target_soft_region_stats,
    tensor_sha256,
)


PHASE1_PROTOCOL_VERSION = 4
PHASE1_PAIR_FIELDS = (
    "protocol_version",
    "phase1_protocol_version",
    "stage",
    "checkpoint_sha256",
    "model_weight_source",
    "ema_applied",
    "truth_model_sha256",
    "boreholes_sha256",
    "runner_source_sha256",
    "runtime_source_sha256",
    "probability_volume_source_sha256",
    "probability_sampling_source_sha256",
    "probability_evaluation_source_sha256",
    "target_mask_sha256",
    "target_probability_sha256",
    "roi_mask_sha256",
    "target_label",
    "component_mode",
    "component_rank",
    "connectivity",
    "target_sigmas",
    "target_scale_weights",
    "roi_radius",
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
    "probability_loss_mode",
    "bce_weight",
    "dice_weight",
    "spatial_gradient_loss",
    "spatial_gradient_weight",
    "soft_diagnostic_version",
    "soft_diagnostic_tau",
    "soft_boundary_similarity_thresholds",
    "condition_projection",
    "device",
)


def _parse_float_list(values: Sequence[float] | None, defaults: Sequence[float]) -> list[float]:
    return [float(value) for value in (values if values is not None else defaults)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly paired fixed-Euler sampling with an oracle three-dimensional "
            "target-label probability volume. This is a mechanism upper bound, "
            "not a real geophysical observation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--model-weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--truth-model", type=Path, default=None)
    parser.add_argument("--boreholes", type=Path, default=None)
    parser.add_argument("--target-label", type=int, default=9)
    parser.add_argument(
        "--component-mode",
        choices=probability_module.COMPONENT_MODES,
        default="all",
    )
    parser.add_argument("--component-rank", type=int, default=None)
    parser.add_argument(
        "--target-sigma",
        action="append",
        type=float,
        default=None,
        help="Repeat for each Gaussian target scale; defaults to 0 and 1.5.",
    )
    parser.add_argument(
        "--target-scale-weight",
        action="append",
        type=float,
        default=None,
        help="Optional repeated weights matching --target-sigma.",
    )
    parser.add_argument("--roi-radius", type=int, default=6)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--max-guidance-ratio", type=float, default=0.10)
    parser.add_argument("--tau-start", type=float, default=0.50)
    parser.add_argument("--tau-end", type=float, default=0.10)
    parser.add_argument(
        "--tau-schedule",
        choices=sampling_module.TEMPERATURE_SCHEDULES,
        default="cosine",
    )
    parser.add_argument("--guidance-start", type=float, default=0.25)
    parser.add_argument(
        "--guidance-schedule",
        choices=sampling_module.GUIDANCE_SCHEDULES,
        default="late_quadratic",
    )
    parser.add_argument(
        "--guidance-scaling-mode",
        choices=sampling_module.GUIDANCE_SCALING_MODES,
        default=sampling_module.LEGACY_GUIDANCE_SCALING_MODE,
        help=(
            "Legacy unit-norm relative scaling or reference-norm scaling that "
            "retains gradient decay while remaining capped by prior velocity."
        ),
    )
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--probability-loss-mode",
        choices=probability_module.PROBABILITY_LOSS_MODES,
        default=probability_module.LEGACY_PROBABILITY_LOSS_MODE,
        help=(
            "Legacy class-balanced continuous-target BCE+soft Dice, or the "
            "calibrated unweighted soft BCE+hard-core Dice formulation."
        ),
    )
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument(
        "--spatial-gradient-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for normalized 3-D target-probability gradient matching; "
            "zero preserves the BCE+Dice-only path."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--baseline-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.model_weights != "ema":
        raise ValueError("Phase 1 requires the canonical EMA inference policy")
    if args.n_samples <= 0 or args.n_steps <= 0:
        raise ValueError("n_samples and n_steps must be positive")
    if args.alpha < 0:
        raise ValueError("alpha must be non-negative")
    if args.alpha == 0 and args.baseline_dir is not None:
        raise ValueError("alpha=0 defines the paired baseline and takes no baseline-dir")
    if args.alpha > 0 and args.baseline_dir is None:
        raise ValueError("positive alpha requires --baseline-dir for strict pairing")
    if args.component_mode == "selected" and args.component_rank is None:
        raise ValueError("selected component mode requires --component-rank")
    if args.component_mode != "selected" and args.component_rank is not None:
        raise ValueError("--component-rank is only valid for selected mode")
    if not 0.0 <= args.guidance_start < 1.0:
        raise ValueError("guidance_start must satisfy 0 <= start < 1")
    if args.max_guidance_ratio < 0 or args.grad_clip_norm < 0:
        raise ValueError("guidance ratio and gradient clipping must be non-negative")
    if args.bce_weight < 0 or args.dice_weight < 0:
        raise ValueError("BCE and Dice weights must be non-negative")
    if args.bce_weight + args.dice_weight <= 0:
        raise ValueError("at least one of BCE and Dice weights must be positive")
    if args.spatial_gradient_weight < 0:
        raise ValueError("spatial gradient weight must be non-negative")
    if args.roi_radius < 0:
        raise ValueError("roi_radius must be non-negative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty; refusing to overwrite: {args.output_dir}"
        )


def paired_probability_config_verdict(
    baseline: Mapping[str, object],
    guided: Mapping[str, object],
) -> tuple[bool, str]:
    """Verify strict Phase-1 equality except for positive guided alpha."""
    equal, reason = runtime.require_equal_fields(
        baseline,
        guided,
        PHASE1_PAIR_FIELDS,
    )
    if not equal:
        return False, reason
    if float(baseline.get("alpha", float("nan"))) != 0.0:
        return False, "paired baseline alpha is not zero"
    if float(guided.get("alpha", 0.0)) <= 0.0:
        return False, "guided alpha must be positive"
    if baseline.get("run_status") != "completed":
        return False, "paired baseline did not complete"
    if int(baseline.get("samples_written", -1)) != int(guided["n_samples"]):
        return False, "paired baseline sample count is incomplete"
    return True, "strict Phase-1 assets, target, noise policy, and sampler settings match"


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_config(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"baseline config not found: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"baseline config must contain a JSON object: {path}")
    return payload


def _load_soft_fields(path: Path) -> Dict[str, torch.Tensor]:
    """Load the tensor-only final soft-field mapping written by this runner."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    required = {
        "target_probability",
        "target_probability_margin",
        "target_similarity_margin",
        "soft_hard_target",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise TypeError(f"invalid final soft-field mapping: {path}")
    if any(not isinstance(value, torch.Tensor) for value in payload.values()):
        raise TypeError(f"final soft fields must all be tensors: {path}")
    return {str(key): value.detach().cpu() for key, value in payload.items()}


def _summarize_region_rows(
    rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    regions = sorted({str(row["region"]) for row in rows})
    return {
        "regions": {
            region: summarize_rows(
                [row for row in rows if str(row["region"]) == region]
            )
            for region in regions
        }
    }


def _build_config(
    args: argparse.Namespace,
    truth_path: Path,
    boreholes_path: Path,
    model_load_report: Mapping[str, object],
    conditioning_report: Mapping[str, object],
    target_metadata: Mapping[str, object],
    probability_metadata: Mapping[str, object],
    roi_mask: torch.Tensor,
) -> Dict[str, object]:
    asset_records = runtime.experiment_asset_records(
        truth_model=truth_path,
        boreholes=boreholes_path,
        runner_source=Path(__file__),
        runtime_source=Path(runtime.__file__),
        probability_volume_source=Path(probability_module.__file__),
        probability_sampling_source=Path(sampling_module.__file__),
        probability_evaluation_source=Path(evaluation_module.__file__),
    )
    asset_records["checkpoint"] = model_load_report["checkpoint"]
    config: Dict[str, object] = {
        "protocol_version": runtime.PROTOCOL_VERSION,
        "phase1_protocol_version": PHASE1_PROTOCOL_VERSION,
        "stage": "phase1_oracle_3d_probability",
        "description": (
            "Oracle truth-derived 3-D probability-volume mechanism experiment; "
            "not a measured geophysical observation."
        ),
        "integrator": runtime.PAIRED_INTEGRATOR,
        "initial_noise_policy": runtime.INITIAL_NOISE_POLICY,
        "condition_projection": "clean_embedding_before_first_step_and_after_every_step_v1",
        "ckpt_path": str(args.ckpt_path),
        "model_weight_source": args.model_weights,
        "ema_applied": bool(model_load_report["ema_applied"]),
        "samples_dir": str(args.samples_dir),
        "truth_model": str(truth_path),
        "boreholes": str(boreholes_path),
        "output_dir": str(args.output_dir),
        "baseline_dir": str(args.baseline_dir) if args.baseline_dir else None,
        "target_label": int(args.target_label),
        "component_mode": args.component_mode,
        "component_rank": args.component_rank,
        "connectivity": 6,
        "target_sigmas": probability_metadata["target_sigmas"],
        "target_scale_weights": probability_metadata["target_scale_weights"],
        "roi_radius": int(args.roi_radius),
        "target_mask_sha256": target_metadata["target_mask_sha256"],
        "target_probability_sha256": probability_metadata[
            "target_probability_sha256"
        ],
        "roi_mask_sha256": tensor_sha256(roi_mask),
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
            sampling_module.GUIDANCE_GRADIENT_REFERENCE_POLICY
            if args.guidance_scaling_mode
            == sampling_module.REFERENCE_GUIDANCE_SCALING_MODE
            else None
        ),
        "grad_clip_norm": float(args.grad_clip_norm),
        "probability_loss_mode": args.probability_loss_mode,
        "bce_weight": float(args.bce_weight),
        "dice_weight": float(args.dice_weight),
        "spatial_gradient_loss": probability_module.SPATIAL_GRADIENT_LOSS,
        "spatial_gradient_weight": float(args.spatial_gradient_weight),
        "soft_diagnostic_version": probability_module.SOFT_DIAGNOSTIC_VERSION,
        "soft_diagnostic_tau": float(args.tau_end),
        "soft_boundary_similarity_thresholds": list(
            probability_module.SOFT_BOUNDARY_SIMILARITY_THRESHOLDS
        ),
        "seed": int(args.seed),
        "device": str(torch.device(args.device)),
        "asset_records": asset_records,
        "model_load_report": dict(model_load_report),
        "conditioning_report": dict(conditioning_report),
        "target_metadata": dict(target_metadata),
        "probability_metadata": dict(probability_metadata),
        "run_status": "running",
    }
    config.update(runtime.flatten_asset_hashes(asset_records))
    return config


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable in this process; run the printed "
            "command in a GPU-enabled terminal or pass --device cpu for utility tests"
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

    sigmas = _parse_float_list(args.target_sigma, (0.0, 1.5))
    scale_weights = (
        None
        if args.target_scale_weight is None
        else _parse_float_list(args.target_scale_weight, ())
    )
    target_mask, target_metadata = build_target_mask(
        truth_cpu,
        target_label=args.target_label,
        component_mode=args.component_mode,
        component_rank=args.component_rank,
    )
    target_probability, probability_metadata = build_probability_volume(
        target_mask,
        sigmas=sigmas,
        scale_weights=scale_weights,
    )
    roi_mask = dilate_mask(target_mask, args.roi_radius)

    # Keep heavyweight training imports outside utility/test imports.
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
    condition_mask_cpu = (boreholes_cpu != -1) | (truth_cpu == -1)
    target_metadata["selected_conditioned_voxels"] = int(
        (target_mask & condition_mask_cpu).sum().item()
    )
    target_metadata["selected_conditioned_fraction"] = (
        target_metadata["selected_conditioned_voxels"]
        / target_metadata["selected_target_voxels"]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(target_mask.cpu(), args.output_dir / "target_mask.pt")
    torch.save(target_probability.cpu(), args.output_dir / "target_probability.pt")
    torch.save(roi_mask.cpu(), args.output_dir / "target_roi_mask.pt")
    _write_json(
        args.output_dir / "target_manifest.json",
        {
            **target_metadata,
            **probability_metadata,
            "roi_radius": args.roi_radius,
            "roi_voxels": int(roi_mask.sum().item()),
            "roi_mask_sha256": tensor_sha256(roi_mask),
            "target_mask_path": str(args.output_dir / "target_mask.pt"),
            "target_probability_path": str(args.output_dir / "target_probability.pt"),
            "target_roi_mask_path": str(args.output_dir / "target_roi_mask.pt"),
        },
    )

    config = _build_config(
        args,
        truth_path,
        boreholes_path,
        model_load_report,
        conditioning_report,
        target_metadata,
        probability_metadata,
        roi_mask,
    )
    baseline_config: Dict[str, object] | None = None
    if args.baseline_dir is not None:
        baseline_config = _load_config(args.baseline_dir / "config.json")
        paired, reason = paired_probability_config_verdict(baseline_config, config)
        if not paired:
            raise ValueError(f"strict baseline pairing failed: {reason}")
        config["pairing_validation"] = {
            "paired": True,
            "reason": reason,
            "baseline_config": str(args.baseline_dir / "config.json"),
        }
    else:
        config["pairing_validation"] = None

    _write_json(args.output_dir / "model_load_report.json", model_load_report)
    _write_json(args.output_dir / "input_validation.json", conditioning_report)
    _write_json(args.output_dir / "config.json", config)

    truth = truth_cpu.to(device)
    condition_mask = condition_mask_cpu.to(device)
    embedded_truth = model.embed(truth)
    embedded_mask = condition_mask.expand(-1, embedded_truth.shape[1], -1, -1, -1)
    conditioning = embedded_truth * embedded_mask
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    traces: list[Dict[str, object]] = []
    metric_rows: list[Dict[str, object]] = []
    baseline_metric_rows: list[Dict[str, object]] = []
    delta_rows: list[Dict[str, object]] = []
    transition_rows: list[Dict[str, object]] = []
    decoded_samples: list[torch.Tensor] = []
    baseline_samples: list[torch.Tensor] = []
    initial_hashes: list[str] = []
    sample_hashes: list[str] = []
    soft_region_rows: list[Dict[str, object]] = []
    paired_soft_rows: list[Dict[str, object]] = []
    soft_decoder_mismatch_counts: list[int] = []
    final_soft_field_samples: Dict[str, list[torch.Tensor]] = {
        "target_probability": [],
        "target_probability_margin": [],
        "target_similarity_margin": [],
        "soft_hard_target": [],
    }
    expected_initial_hashes = (
        list(baseline_config.get("initial_noise_sha256", []))
        if baseline_config is not None
        else []
    )
    if baseline_config is not None and len(expected_initial_hashes) != args.n_samples:
        raise ValueError("paired baseline lacks complete initial-noise hashes")
    baseline_soft_fields: Dict[str, torch.Tensor] | None = None
    if baseline_config is not None:
        baseline_soft_fields = _load_soft_fields(
            args.baseline_dir / "final_soft_fields.pt"
        )
        if any(value.shape[0] != args.n_samples for value in baseline_soft_fields.values()):
            raise ValueError("paired baseline soft fields have incomplete sample count")
        expected_soft_hashes = baseline_config.get("final_soft_field_sha256")
        if not isinstance(expected_soft_hashes, Mapping):
            raise ValueError("paired baseline lacks final soft-field hashes")
        for name, value in baseline_soft_fields.items():
            if expected_soft_hashes.get(name) != tensor_sha256(value):
                raise ValueError(
                    f"paired baseline final soft-field hash differs for {name}"
                )

    for sample_index in range(args.n_samples):
        initial_cpu = torch.randn(
            1,
            model.embedding_dim,
            *model.data_shape,
            generator=generator,
            dtype=embedded_truth.dtype,
        )
        initial_hash = tensor_sha256(initial_cpu)
        if baseline_config is not None and initial_hash != expected_initial_hashes[sample_index]:
            raise ValueError(
                f"sample {sample_index} initial-noise hash differs from paired baseline"
            )
        initial_hashes.append(initial_hash)

        final_state, sample_trace = fixed_euler_probability_sample(
            model=model,
            initial_state=initial_cpu.to(device),
            conditioning=conditioning,
            embedded_truth=embedded_truth,
            truth_model=truth,
            condition_mask=condition_mask,
            target_probability=target_probability,
            target_mask=target_mask,
            roi_mask=roi_mask,
            target_label=args.target_label,
            n_steps=args.n_steps,
            alpha=args.alpha,
            max_guidance_ratio=args.max_guidance_ratio,
            tau_start=args.tau_start,
            tau_end=args.tau_end,
            tau_schedule=args.tau_schedule,
            guidance_start=args.guidance_start,
            guidance_schedule=args.guidance_schedule,
            grad_clip_norm=args.grad_clip_norm,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
            spatial_gradient_weight=args.spatial_gradient_weight,
            probability_loss_mode=args.probability_loss_mode,
            guidance_scaling_mode=args.guidance_scaling_mode,
            sample_id=sample_index,
        )
        if not torch.isfinite(final_state).all():
            raise FloatingPointError(f"sample {sample_index} contains NaN or Inf")
        final_soft_fields = compute_target_soft_fields(
            final_state,
            model.embedding.weight,
            target_label=args.target_label,
            tau=args.tau_end,
        )
        decoded = (model.decode(final_state).detach().cpu() - 1)[0]
        final_soft_fields_cpu = {
            name: value.detach().cpu() for name, value in final_soft_fields.items()
        }
        mismatch_count = int(
            (
                final_soft_fields_cpu["soft_hard_target"][0, 0]
                != (decoded == int(args.target_label))
            )
            .sum()
            .item()
        )
        if mismatch_count != 0:
            raise RuntimeError(
                "soft cosine argmax and model hard decoder disagree for target label: "
                f"sample={sample_index}, mismatches={mismatch_count}"
            )
        soft_decoder_mismatch_counts.append(mismatch_count)
        for name, value in final_soft_fields_cpu.items():
            final_soft_field_samples[name].append(value)
        sample_soft_rows = target_soft_region_stats(
            final_soft_fields_cpu,
            truth_cpu,
            target_mask,
            roi_mask,
            condition_mask_cpu,
            target_label=args.target_label,
            sample_id=sample_index,
        )
        for row in sample_soft_rows:
            row["decoder_target_mismatch_count"] = mismatch_count
        soft_region_rows.extend(sample_soft_rows)
        sample_path = args.output_dir / f"sample_{sample_index}.pt"
        torch.save(decoded, sample_path)
        sample_hashes.append(tensor_sha256(decoded))
        decoded_samples.append(decoded)

        baseline_decoded = None
        if args.baseline_dir is not None:
            baseline_path = args.baseline_dir / f"sample_{sample_index}.pt"
            baseline_decoded = runtime.load_tensor(baseline_path, map_location="cpu")
        metrics = sample_hard_metrics(
            prediction=decoded,
            truth_model=truth_cpu,
            target_mask=target_mask,
            roi_mask=roi_mask,
            condition_mask=condition_mask_cpu,
            target_label=args.target_label,
            sample_id=sample_index,
            baseline_prediction=baseline_decoded,
        )
        metrics["path"] = str(sample_path)
        metric_rows.append(metrics)

        if baseline_decoded is not None:
            baseline_samples.append(baseline_decoded.squeeze())
            baseline_metrics = sample_hard_metrics(
                prediction=baseline_decoded,
                truth_model=truth_cpu,
                target_mask=target_mask,
                roi_mask=roi_mask,
                condition_mask=condition_mask_cpu,
                target_label=args.target_label,
                sample_id=sample_index,
            )
            baseline_metric_rows.append(baseline_metrics)
            delta_rows.append(paired_metric_deltas(baseline_metrics, metrics))
            transition_rows.extend(
                class_transition_records(
                    baseline_decoded,
                    decoded,
                    sample_id=sample_index,
                )
            )
            if baseline_soft_fields is None:
                raise RuntimeError("paired baseline soft fields were not loaded")
            paired_soft_rows.extend(
                paired_target_soft_deltas(
                    baseline_fields={
                        name: value[sample_index : sample_index + 1]
                        for name, value in baseline_soft_fields.items()
                    },
                    guided_fields=final_soft_fields_cpu,
                    truth_model=truth_cpu,
                    selected_target_mask=target_mask,
                    roi_mask=roi_mask,
                    condition_mask=condition_mask_cpu,
                    baseline_decoded=baseline_decoded,
                    guided_decoded=decoded,
                    target_label=args.target_label,
                    sample_id=sample_index,
                )
            )
        traces.extend(sample_trace)

    final_soft_fields_stacked = {
        name: torch.cat(values, dim=0)
        for name, values in final_soft_field_samples.items()
    }
    torch.save(final_soft_fields_stacked, args.output_dir / "final_soft_fields.pt")
    _write_rows(args.output_dir / "guidance_trace.csv", traces)
    _write_rows(args.output_dir / "sample_metrics.csv", metric_rows)
    _write_rows(args.output_dir / "final_soft_region_stats.csv", soft_region_rows)
    _write_json(args.output_dir / "metrics_summary.json", summarize_rows(metric_rows))
    _write_json(
        args.output_dir / "final_soft_summary.json",
        _summarize_region_rows(soft_region_rows),
    )
    ensemble_summary: Dict[str, object] = {
        "current": ensemble_diversity_summary(
            torch.stack(decoded_samples, dim=0),
            target_mask,
            roi_mask,
            args.target_label,
            sample_hashes=sample_hashes,
        )
    }
    if delta_rows:
        _write_rows(
            args.output_dir / "paired_baseline_metrics.csv",
            baseline_metric_rows,
        )
        _write_rows(args.output_dir / "paired_deltas.csv", delta_rows)
        _write_rows(
            args.output_dir / "paired_class_transitions.csv",
            transition_rows,
        )
        _write_json(
            args.output_dir / "paired_delta_summary.json",
            summarize_rows(delta_rows),
        )
        _write_rows(
            args.output_dir / "paired_soft_deltas.csv",
            paired_soft_rows,
        )
        _write_json(
            args.output_dir / "paired_soft_delta_summary.json",
            _summarize_region_rows(paired_soft_rows),
        )
        ensemble_summary["baseline"] = ensemble_diversity_summary(
            torch.stack(baseline_samples, dim=0),
            target_mask,
            roi_mask,
            args.target_label,
            sample_hashes=list(baseline_config.get("sample_sha256", [])),
        )
    _write_json(args.output_dir / "ensemble_summary.json", ensemble_summary)

    config.update(
        {
            "run_status": "completed",
            "samples_written": len(metric_rows),
            "initial_noise_sha256": initial_hashes,
            "sample_sha256": sample_hashes,
            "final_soft_fields_path": str(args.output_dir / "final_soft_fields.pt"),
            "final_soft_field_sha256": {
                name: tensor_sha256(value)
                for name, value in final_soft_fields_stacked.items()
            },
            "soft_decoder_mismatch_counts": soft_decoder_mismatch_counts,
            "max_post_projection_condition_violations": max(
                int(row["post_projection_condition_violations"]) for row in traces
            ),
        }
    )
    _write_json(args.output_dir / "config.json", config)


if __name__ == "__main__":
    main()
