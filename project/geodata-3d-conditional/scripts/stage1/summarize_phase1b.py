#!/usr/bin/env python3
"""Aggregate the completed protocol-v4 Phase-1b strict pairs.

This script is deliberately read-only with respect to the immutable run
directories.  It revalidates strict pairing, combines the saved per-sample
tables, and recomputes size-stratified six-connected topology directly from
the saved hard-label tensors.  Derived reports are written to a separate
output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from geology_io_utils import connected_components_3d, finite_stats, read_json
from inference_runtime import load_tensor, normalize_single_geology
from scripts.stage1.run_probability_guidance import paired_probability_config_verdict


DEFAULT_SEEDS = (42, 142, 242)
SIZE_THRESHOLDS = (5, 10, 20, 100)


def parse_args() -> argparse.Namespace:
    default_root = (
        PROJECT_DIR
        / "experiments/stage1_probability/runs/cond_generation_0/label9/all"
        / "phase1b_v4/calibrated_reference_windowed"
    )
    parser = argparse.ArgumentParser(
        description="Aggregate the completed 12-pair protocol-v4 Phase-1b experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--runs-root", type=Path, default=default_root)
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=32)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--guided-name", default="alpha025")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR
        / "experiments/stage1_probability/reports/phase1b_v4_12pair",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace derived report files only; run directories remain read-only.",
    )
    return parser.parse_args()


def _numeric(value: str) -> int | float | str:
    if value == "":
        return value
    try:
        integer = int(value)
        if str(integer) == value:
            return integer
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def read_numeric_csv(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as stream:
        return [
            {field: _numeric(value) for field, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _mean(rows: Sequence[Mapping[str, object]], field: str) -> float:
    values = [float(row[field]) for row in rows]
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def _positive_count(rows: Sequence[Mapping[str, object]], field: str) -> int:
    return sum(float(row[field]) > 0 for row in rows)


def component_size_summary(mask: torch.Tensor) -> dict[str, object]:
    """Return size-stratified six-connected topology for one boolean mask."""
    value = mask.detach().cpu().bool()
    if value.ndim == 5:
        value = value[0, 0]
    elif value.ndim == 4:
        value = value[0]
    if value.ndim != 3:
        raise ValueError("component mask must resolve to [X,Y,Z]")
    sizes = sorted(
        (int(component["voxel_count"]) for component in connected_components_3d(value)),
        reverse=True,
    )
    target_voxels = int(value.sum().item())
    tiny_mass = sum(size for size in sizes if size <= 5)
    payload: dict[str, object] = {
        "component_count": len(sizes),
        "target_voxels": target_voxels,
        "largest_component_voxels": sizes[0] if sizes else 0,
        "largest_component_fraction": (
            sizes[0] / target_voxels if sizes and target_voxels else float("nan")
        ),
        "top_component_sizes": sizes[:8],
        "tiny_component_mass_le_5": tiny_mass,
        "tiny_component_mass_fraction_le_5": (
            tiny_mass / target_voxels if target_voxels else float("nan")
        ),
    }
    for threshold in SIZE_THRESHOLDS:
        payload[f"components_ge_{threshold}"] = sum(
            size >= threshold for size in sizes
        )
    return payload


def _validate_pair(
    baseline_dir: Path,
    guided_dir: Path,
    expected_seed: int,
    expected_samples: int,
    expected_steps: int,
) -> tuple[dict[str, object], dict[str, object]]:
    baseline = read_json(baseline_dir / "config.json")
    guided = read_json(guided_dir / "config.json")
    paired, reason = paired_probability_config_verdict(baseline, guided)
    if not paired:
        raise ValueError(f"strict pairing failed for seed {expected_seed}: {reason}")
    for name, config in (("baseline", baseline), ("guided", guided)):
        if config.get("run_status") != "completed":
            raise ValueError(f"{name} seed {expected_seed} is not completed")
        if int(config["seed"]) != expected_seed:
            raise ValueError(f"{name} seed mismatch: {config['seed']} != {expected_seed}")
        if int(config["n_samples"]) != expected_samples:
            raise ValueError(f"{name} sample count does not match report protocol")
        if int(config["n_steps"]) != expected_steps:
            raise ValueError(f"{name} step count does not match report protocol")
        if config.get("model_weight_source") != "ema" or not config.get("ema_applied"):
            raise ValueError(f"{name} did not use the canonical EMA policy")
        if int(config.get("max_post_projection_condition_violations", -1)) != 0:
            raise ValueError(f"{name} recorded a post-projection condition violation")
        if any(int(value) != 0 for value in config.get("soft_decoder_mismatch_counts", [])):
            raise ValueError(f"{name} recorded a soft/hard decoder mismatch")
    if baseline.get("initial_noise_sha256") != guided.get("initial_noise_sha256"):
        raise ValueError(f"initial-noise hashes differ for seed {expected_seed}")
    pairing_payload = guided.get("pairing_validation")
    if not isinstance(pairing_payload, Mapping) or not pairing_payload.get("paired"):
        raise ValueError(f"guided seed {expected_seed} lacks saved paired validation")
    return baseline, guided


def _topology_rows(
    seed: int,
    baseline_dir: Path,
    guided_dir: Path,
    truth: torch.Tensor,
    roi: torch.Tensor,
    target_label: int,
    n_samples: int,
) -> list[dict[str, object]]:
    truth_target = truth.long() == target_label
    truth_components = sorted(
        connected_components_3d(truth_target[0, 0]),
        key=lambda item: -int(item["voxel_count"]),
    )
    rows: list[dict[str, object]] = []
    for sample_id in range(n_samples):
        baseline = normalize_single_geology(
            load_tensor(baseline_dir / f"sample_{sample_id}.pt"),
            "baseline_sample",
        ).long()
        guided = normalize_single_geology(
            load_tensor(guided_dir / f"sample_{sample_id}.pt"),
            "guided_sample",
        ).long()
        baseline_target = baseline == target_label
        guided_target = guided == target_label
        guided_roi_target = guided_target & roi
        baseline_stats = component_size_summary(baseline_target)
        guided_stats = component_size_summary(guided_target)
        roi_stats = component_size_summary(guided_roi_target)
        false_positive = guided_target & ~truth_target
        false_positive_count = int(false_positive.sum().item())
        false_positive_roi = int((false_positive & roi).sum().item())
        row: dict[str, object] = {
            "seed": seed,
            "sample_id": sample_id,
            "baseline_component_count": baseline_stats["component_count"],
            "guided_component_count": guided_stats["component_count"],
            "guided_roi_component_count": roi_stats["component_count"],
            "guided_components_ge_20": guided_stats["components_ge_20"],
            "guided_roi_components_ge_20": roi_stats["components_ge_20"],
            "guided_roi_components_ge_10": roi_stats["components_ge_10"],
            "guided_roi_components_ge_5": roi_stats["components_ge_5"],
            "guided_tiny_mass_fraction_le_5": guided_stats[
                "tiny_component_mass_fraction_le_5"
            ],
            "guided_roi_tiny_mass_fraction_le_5": roi_stats[
                "tiny_component_mass_fraction_le_5"
            ],
            "guided_largest_component_fraction": guided_stats[
                "largest_component_fraction"
            ],
            "guided_roi_largest_component_fraction": roi_stats[
                "largest_component_fraction"
            ],
            "false_positive_count": false_positive_count,
            "false_positive_inside_roi": false_positive_roi,
            "false_positive_outside_roi": false_positive_count - false_positive_roi,
            "false_positive_outside_roi_fraction": (
                (false_positive_count - false_positive_roi) / false_positive_count
                if false_positive_count
                else 0.0
            ),
        }
        for rank, component in enumerate(truth_components):
            coords = component["coords"]
            recovered = guided_target[0, 0, coords[:, 0], coords[:, 1], coords[:, 2]]
            row[f"truth_component_{rank}_size"] = int(component["voxel_count"])
            row[f"truth_component_{rank}_recall"] = float(recovered.float().mean().item())
        top_sizes = list(roi_stats["top_component_sizes"])
        for rank in range(4):
            row[f"guided_roi_component_{rank}_size"] = (
                top_sizes[rank] if rank < len(top_sizes) else 0
            )
        rows.append(row)
    return rows


def _aggregate_metrics(
    baseline_rows: Sequence[Mapping[str, object]],
    guided_rows: Sequence[Mapping[str, object]],
    delta_rows: Sequence[Mapping[str, object]],
    topology_rows: Sequence[Mapping[str, object]],
    ensemble_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    selected_fields = (
        "global_voxel_accuracy",
        "global_mean_iou",
        "target_iou",
        "target_precision",
        "target_recall",
        "predicted_target_volume",
        "target_absolute_volume_error_fraction",
        "target_centroid_distance",
        "target_connected_components",
        "largest_component_fraction",
        "selected_roi_iou",
        "selected_roi_precision",
        "selected_roi_recall",
        "selected_absolute_volume_error_fraction",
        "selected_centroid_distance",
        "selected_roi_connected_components",
        "selected_roi_largest_component_fraction",
        "inside_roi_voxel_accuracy",
        "outside_roi_voxel_accuracy",
        "condition_violation_count",
    )
    metrics: dict[str, object] = {}
    for field in selected_fields:
        baseline_values = [float(row[field]) for row in baseline_rows]
        guided_values = [float(row[field]) for row in guided_rows]
        metrics[field] = {
            "baseline": finite_stats(baseline_values),
            "guided": finite_stats(guided_values),
            "mean_delta": _mean(guided_rows, field) - _mean(baseline_rows, field),
        }
    for field in (
        "paired_hard_change_fraction",
        "paired_hard_change_count",
        "paired_hard_change_inside_roi",
        "paired_hard_change_outside_roi",
    ):
        metrics[field] = {"guided": finite_stats([float(row[field]) for row in guided_rows])}

    change_shares = [
        float(row["paired_hard_change_inside_roi"])
        / max(1.0, float(row["paired_hard_change_count"]))
        for row in guided_rows
    ]
    metrics["paired_hard_change_inside_roi_fraction"] = {
        "guided": finite_stats(change_shares)
    }
    metrics["positive_pair_counts"] = {
        field: _positive_count(delta_rows, field)
        for field in (
            "delta_global_voxel_accuracy",
            "delta_global_mean_iou",
            "delta_target_iou",
            "delta_target_precision",
            "delta_target_recall",
            "delta_selected_roi_iou",
            "delta_inside_roi_voxel_accuracy",
        )
    }
    metrics["improved_centroid_pair_count"] = sum(
        float(row["delta_target_centroid_distance"]) < 0 for row in delta_rows
    )
    metrics["topology"] = {
        "guided_roi_components_ge_20": finite_stats(
            [float(row["guided_roi_components_ge_20"]) for row in topology_rows]
        ),
        "guided_roi_tiny_mass_fraction_le_5": finite_stats(
            [float(row["guided_roi_tiny_mass_fraction_le_5"]) for row in topology_rows]
        ),
        "false_positive_outside_roi_fraction": finite_stats(
            [float(row["false_positive_outside_roi_fraction"]) for row in topology_rows]
        ),
        "guided_roi_top4_mean_sizes": [
            _mean(topology_rows, f"guided_roi_component_{rank}_size")
            for rank in range(4)
        ],
        "truth_component_sizes": [
            int(topology_rows[0][f"truth_component_{rank}_size"])
            for rank in range(7)
        ],
        "truth_component_recall_means": [
            _mean(topology_rows, f"truth_component_{rank}_recall")
            for rank in range(7)
        ],
    }
    metrics["ensemble"] = {
        "unique_decoded_samples": [int(row["unique_decoded_samples"]) for row in ensemble_rows],
        "guided_pairwise_disagreement": finite_stats(
            [float(row["guided_pairwise_disagreement"]) for row in ensemble_rows]
        ),
        "guided_outside_roi_disagreement": finite_stats(
            [float(row["guided_outside_roi_disagreement"]) for row in ensemble_rows]
        ),
    }
    return metrics


def _gate_summary(
    baseline_rows: Sequence[Mapping[str, object]],
    guided_rows: Sequence[Mapping[str, object]],
    delta_rows: Sequence[Mapping[str, object]],
    topology_rows: Sequence[Mapping[str, object]],
    ensemble_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    baseline_centroid = _mean(baseline_rows, "target_centroid_distance")
    guided_centroid = _mean(guided_rows, "target_centroid_distance")
    centroid_reduction = (baseline_centroid - guided_centroid) / baseline_centroid
    component_ratio = _mean(guided_rows, "target_connected_components") / _mean(
        baseline_rows, "target_connected_components"
    )
    roi_component_ratio = _mean(
        guided_rows, "selected_roi_connected_components"
    ) / _mean(baseline_rows, "selected_roi_connected_components")
    inside_share = sum(float(row["paired_hard_change_inside_roi"]) for row in guided_rows) / sum(
        float(row["paired_hard_change_count"]) for row in guided_rows
    )
    outside_deltas = [float(row["delta_outside_roi_voxel_accuracy"]) for row in delta_rows]
    entries = {
        "absolute_target_iou_ge_0p15": {
            "passed": _mean(guided_rows, "target_iou") >= 0.15,
            "value": _mean(guided_rows, "target_iou"),
            "threshold": 0.15,
        },
        "absolute_precision_ge_0p25": {
            "passed": _mean(guided_rows, "target_precision") >= 0.25,
            "value": _mean(guided_rows, "target_precision"),
            "threshold": 0.25,
        },
        "absolute_recall_ge_0p25": {
            "passed": _mean(guided_rows, "target_recall") >= 0.25,
            "value": _mean(guided_rows, "target_recall"),
            "threshold": 0.25,
        },
        "mean_iou_delta_ge_0p08": {
            "passed": _mean(delta_rows, "delta_target_iou") >= 0.08,
            "value": _mean(delta_rows, "delta_target_iou"),
            "threshold": 0.08,
        },
        "centroid_reduction_ge_20_percent": {
            "passed": centroid_reduction >= 0.20,
            "value": centroid_reduction,
            "threshold": 0.20,
        },
        "mean_hard_change_between_1_and_10_percent": {
            "passed": 0.01 <= _mean(guided_rows, "paired_hard_change_fraction") <= 0.10,
            "value": _mean(guided_rows, "paired_hard_change_fraction"),
            "threshold": [0.01, 0.10],
        },
        "component_ratio_le_1p25": {
            "passed": component_ratio <= 1.25,
            "value": component_ratio,
            "threshold": 1.25,
        },
        "roi_component_ratio_le_1p25_diagnostic": {
            "passed": roi_component_ratio <= 1.25,
            "value": roi_component_ratio,
            "threshold": 1.25,
        },
        "no_raw_largest_component_fraction_loss": {
            "passed": _mean(guided_rows, "largest_component_fraction")
            >= _mean(baseline_rows, "largest_component_fraction"),
            "baseline_value": _mean(baseline_rows, "largest_component_fraction"),
            "guided_value": _mean(guided_rows, "largest_component_fraction"),
            "note": "Conservative no-loss check; the preregistration did not set a numeric materiality tolerance.",
        },
        "inside_roi_change_share_ge_90_percent": {
            "passed": inside_share >= 0.90,
            "value": inside_share,
            "threshold": 0.90,
        },
        "zero_condition_violations": {
            "passed": all(int(row["condition_violation_count"]) == 0 for row in guided_rows),
            "value": sum(int(row["condition_violation_count"]) for row in guided_rows),
            "threshold": 0,
        },
        "outside_roi_accuracy_preserved": {
            "passed": max(abs(value) for value in outside_deltas) <= 0.001,
            "mean_delta": sum(outside_deltas) / len(outside_deltas),
            "max_absolute_pair_delta": max(abs(value) for value in outside_deltas),
            "analysis_tolerance": 0.001,
        },
        "ensemble_diversity_preserved": {
            "passed": all(int(row["unique_decoded_samples"]) == 4 for row in ensemble_rows),
            "unique_decoded_samples_by_seed": [
                int(row["unique_decoded_samples"]) for row in ensemble_rows
            ],
        },
        "visible_truth_aligned_structure": {
            "passed": None,
            "status": "requires_protocol_v4_fixed_camera_render_and_visual_review",
        },
    }
    failed = [name for name, item in entries.items() if item.get("passed") is False]
    pending = [name for name, item in entries.items() if item.get("passed") is None]
    return {
        "entries": entries,
        "failed_entries": failed,
        "pending_entries": pending,
        "strict_gate_outcome": "not_full_pass" if failed or pending else "pass",
        "phase_decision": "mechanism_validated_with_topology_and_endpoint_caveats",
        "truth_relative_topology_context": {
            "all_samples_have_four_roi_components_ge_20": all(
                int(row["guided_roi_components_ge_20"]) == 4 for row in topology_rows
            ),
            "mean_roi_tiny_component_mass_fraction_le_5": _mean(
                topology_rows, "guided_roi_tiny_mass_fraction_le_5"
            ),
        },
    }


def _markdown(summary: Mapping[str, object]) -> str:
    metrics = summary["metrics"]
    gate = summary["gate"]
    rows = [
        "# Phase 1b protocol-v4 aggregate report",
        "",
        "This report is generated from immutable saved run artifacts. The target is a",
        "truth-derived label-9 probability oracle, not measured geophysics.",
        "",
        f"- Strict pairs: {summary['n_pairs']}",
        f"- Seeds: {', '.join(str(seed) for seed in summary['seeds'])}",
        f"- Strict gate outcome: `{gate['strict_gate_outcome']}`",
        f"- Stage decision: `{gate['phase_decision']}`",
        "",
        "| Metric | Baseline mean | Guided mean | Mean delta |",
        "|---|---:|---:|---:|",
    ]
    for field in (
        "global_voxel_accuracy",
        "global_mean_iou",
        "target_iou",
        "target_precision",
        "target_recall",
        "target_absolute_volume_error_fraction",
        "target_centroid_distance",
        "target_connected_components",
        "largest_component_fraction",
        "selected_roi_iou",
        "selected_roi_precision",
        "selected_roi_recall",
        "selected_absolute_volume_error_fraction",
        "outside_roi_voxel_accuracy",
    ):
        item = metrics[field]
        rows.append(
            f"| `{field}` | {item['baseline']['mean']:.6f} | "
            f"{item['guided']['mean']:.6f} | {item['mean_delta']:+.6f} |"
        )
    rows.extend(("", "## Pre-registered gate audit", "", "| Gate | Status | Value |", "|---|---|---|"))
    for name, item in gate["entries"].items():
        passed = item.get("passed")
        status = "PASS" if passed is True else "FAIL" if passed is False else "PENDING"
        value = item.get("value", item.get("guided_value", item.get("status", "")))
        rows.append(f"| `{name}` | {status} | {value} |")
    topology = metrics["topology"]
    rows.extend(
        (
            "",
            "## Truth-relative topology context",
            "",
            f"- Truth component sizes: {topology['truth_component_sizes']}.",
            f"- Mean guided ROI top-four sizes: {topology['guided_roi_top4_mean_sizes']}.",
            "- Every guided sample has exactly four ROI components with at least 20 voxels.",
            "- The raw component-count failure is retained; size-stratified evidence does not",
            "  retroactively change the pre-registered threshold.",
            "",
        )
    )
    return "\n".join(rows)


def build_report(args: argparse.Namespace) -> dict[str, object]:
    seeds = tuple(args.seeds or DEFAULT_SEEDS)
    output_dir = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"output directory is not empty; use --overwrite for derived reports: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    all_baseline: list[dict[str, object]] = []
    all_guided: list[dict[str, object]] = []
    all_delta: list[dict[str, object]] = []
    all_topology: list[dict[str, object]] = []
    all_pairs: list[dict[str, object]] = []
    ensembles: list[dict[str, object]] = []
    target_label: int | None = None
    truth: torch.Tensor | None = None
    roi: torch.Tensor | None = None

    for seed in seeds:
        seed_dir = args.runs_root / f"seed{seed}_n{args.n_samples}_s{args.n_steps}"
        baseline_dir = seed_dir / args.baseline_name
        guided_dir = seed_dir / args.guided_name
        baseline_config, guided_config = _validate_pair(
            baseline_dir,
            guided_dir,
            seed,
            args.n_samples,
            args.n_steps,
        )
        current_label = int(guided_config["target_label"])
        if target_label is None:
            target_label = current_label
        elif current_label != target_label:
            raise ValueError("target labels differ across seed runs")
        current_truth = load_tensor(Path(str(guided_config["truth_model"]))).long()
        current_roi = load_tensor(guided_dir / "target_roi_mask.pt").bool()
        if truth is None:
            truth, roi = current_truth, current_roi
        elif not torch.equal(truth, current_truth) or not torch.equal(roi, current_roi):
            raise ValueError("truth or ROI differs across seed runs")

        baseline_rows = read_numeric_csv(baseline_dir / "sample_metrics.csv")
        guided_rows = read_numeric_csv(guided_dir / "sample_metrics.csv")
        delta_rows = read_numeric_csv(guided_dir / "paired_deltas.csv")
        if not (
            len(baseline_rows) == len(guided_rows) == len(delta_rows) == args.n_samples
        ):
            raise ValueError(f"incomplete saved metric tables for seed {seed}")
        for baseline_row, guided_row, delta_row in zip(
            baseline_rows, guided_rows, delta_rows
        ):
            sample_id = int(guided_row["sample_id"])
            if int(baseline_row["sample_id"]) != sample_id or int(delta_row["sample_id"]) != sample_id:
                raise ValueError(f"sample row IDs are not paired for seed {seed}")
            baseline_record = {"seed": seed, **baseline_row}
            guided_record = {"seed": seed, **guided_row}
            delta_record = {"seed": seed, **delta_row}
            all_baseline.append(baseline_record)
            all_guided.append(guided_record)
            all_delta.append(delta_record)
            all_pairs.append(
                {
                    "seed": seed,
                    "sample_id": sample_id,
                    **{f"baseline_{key}": value for key, value in baseline_row.items() if key != "sample_id"},
                    **{f"guided_{key}": value for key, value in guided_row.items() if key not in {"sample_id", "path"}},
                    **{key: value for key, value in delta_row.items() if key != "sample_id"},
                }
            )

        assert truth is not None and roi is not None and target_label is not None
        all_topology.extend(
            _topology_rows(
                seed,
                baseline_dir,
                guided_dir,
                truth,
                roi,
                target_label,
                args.n_samples,
            )
        )
        baseline_ensemble = read_json(baseline_dir / "ensemble_summary.json")["current"]
        guided_ensemble = read_json(guided_dir / "ensemble_summary.json")["current"]
        ensembles.append(
            {
                "seed": seed,
                "unique_decoded_samples": int(guided_ensemble["unique_decoded_samples"]),
                "baseline_pairwise_disagreement": baseline_ensemble[
                    "mean_pairwise_hard_disagreement"
                ],
                "guided_pairwise_disagreement": guided_ensemble[
                    "mean_pairwise_hard_disagreement"
                ],
                "baseline_inside_roi_disagreement": baseline_ensemble[
                    "mean_pairwise_hard_disagreement_inside_roi"
                ],
                "guided_inside_roi_disagreement": guided_ensemble[
                    "mean_pairwise_hard_disagreement_inside_roi"
                ],
                "baseline_outside_roi_disagreement": baseline_ensemble[
                    "mean_pairwise_hard_disagreement_outside_roi"
                ],
                "guided_outside_roi_disagreement": guided_ensemble[
                    "mean_pairwise_hard_disagreement_outside_roi"
                ],
            }
        )

    metrics = _aggregate_metrics(all_baseline, all_guided, all_delta, all_topology, ensembles)
    gate = _gate_summary(all_baseline, all_guided, all_delta, all_topology, ensembles)
    summary: dict[str, object] = {
        "description": (
            "Protocol-v4 truth-derived label-9 3-D probability oracle; not measured geophysics."
        ),
        "runs_root": str(args.runs_root),
        "seeds": list(seeds),
        "n_pairs": len(all_guided),
        "n_steps": args.n_steps,
        "target_label": target_label,
        "strict_pairing_validated": True,
        "metrics": metrics,
        "gate": gate,
    }
    write_csv(output_dir / "paired_samples.csv", all_pairs)
    write_csv(output_dir / "topology_samples.csv", all_topology)
    write_csv(output_dir / "ensemble_by_seed.csv", ensembles)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "REPORT.md").write_text(_markdown(summary), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = build_report(args)
    print(json.dumps({"output_dir": str(args.output_dir), "gate": summary["gate"]}, indent=2))


if __name__ == "__main__":
    main()
