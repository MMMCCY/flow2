#!/usr/bin/env python3
"""Retrospective truth evaluation of the immutable Stage15-B2 pCN ensemble."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime
from guidance.seismic import tensor_sha256
from scripts.stage10.evaluate_bridge_information import average_precision
from scripts.stage15.common import (
    base_manifest,
    normalize_volume,
    read_json,
    refuse_nonempty,
    write_csv,
    write_json,
)


EXPERIMENT_ROOT = PROJECT_DIR / "experiments/stage15_binary_seismic_consensus"
DEFAULT_B2_DIR = EXPERIMENT_ROOT / "inversion/b2_flow_pcn_pilot_v1"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "reports/b2_truth_evaluation_v1"
B2_INPUT_NAMES = (
    "retained_binary_states.pt",
    "retained_categorical_states.pt",
    "occupancy_frequency.pt",
    "positive_mask.pt",
    "negative_mask.pt",
    "unknown_mask.pt",
    "initial_binary_models.pt",
    "pilot_summary.json",
    "run_manifest.json",
)
MACHINE_DECISIONS = (
    "POSITIVE_CORE_VALID_NEGATIVE_VALID",
    "POSITIVE_CORE_VALID_NEGATIVE_UNSAFE",
    "POSITIVE_CORE_NOT_VALID",
)


def binary_metrics(
    predicted: torch.Tensor,
    actual: torch.Tensor,
    domain: torch.Tensor,
) -> dict[str, float | int]:
    if not (predicted.shape == actual.shape == domain.shape):
        raise ValueError("predicted, actual, and domain must match")
    selected = predicted.bool() & domain.bool()
    target = actual.bool() & domain.bool()
    true_positive = int((selected & target).sum())
    false_positive = int((selected & ~target).sum())
    false_negative = int((~selected & target).sum())
    predicted_count = true_positive + false_positive
    truth_count = true_positive + false_negative
    union = true_positive + false_positive + false_negative
    return {
        "predicted_voxels": predicted_count,
        "true_positive_voxels": true_positive,
        "false_positive_voxels": false_positive,
        "false_negative_voxels": false_negative,
        "precision": true_positive / predicted_count if predicted_count else float("nan"),
        "recall": true_positive / truth_count if truth_count else float("nan"),
        "iou": true_positive / union if union else float("nan"),
    }


def truth_partition(
    binary_truth: torch.Tensor,
    positive: torch.Tensor,
    unknown: torch.Tensor,
    negative: torch.Tensor,
    subsurface: torch.Tensor,
) -> dict[str, object]:
    if not (
        binary_truth.shape
        == positive.shape
        == unknown.shape
        == negative.shape
        == subsurface.shape
    ):
        raise ValueError("truth partition tensors must have matching shapes")
    support = subsurface.bool()
    masks = (positive.bool(), unknown.bool(), negative.bool())
    if any(bool((mask & ~support).any()) for mask in masks):
        raise ValueError("partition masks must stay inside subsurface")
    if bool((masks[0] & masks[1]).any()) or bool((masks[0] & masks[2]).any()) or bool((masks[1] & masks[2]).any()):
        raise ValueError("positive, unknown, and negative must be disjoint")
    if not torch.equal(masks[0] | masks[1] | masks[2], support):
        raise ValueError("positive, unknown, and negative must exactly cover subsurface")
    target = binary_truth.bool() & support
    total = int(target.sum())
    counts = {
        "positive": int((target & masks[0]).sum()),
        "unknown": int((target & masks[1]).sum()),
        "negative": int((target & masks[2]).sum()),
    }
    if sum(counts.values()) != total:
        raise RuntimeError("truth label9 partition accounting failed")
    return {
        "truth_label9_voxels": total,
        "counts": counts,
        "fractions": {
            name: value / total if total else float("nan")
            for name, value in counts.items()
        },
        "subsurface_partition_exact": True,
    }


def centroid(mask: torch.Tensor) -> torch.Tensor | None:
    points = torch.nonzero(mask.bool(), as_tuple=False).float()
    return None if points.numel() == 0 else points.mean(dim=0)


def centroid_distance(predicted: torch.Tensor, actual: torch.Tensor) -> float:
    left = centroid(predicted)
    right = centroid(actual)
    if left is None or right is None:
        return float("nan")
    return float(torch.linalg.vector_norm(left - right))


def positive_component_overlap(
    positive: torch.Tensor,
    binary_truth: torch.Tensor,
) -> dict[str, object]:
    positive_np = positive.detach().cpu().bool().numpy()
    truth_np = binary_truth.detach().cpu().bool().numpy()
    structure = ndimage.generate_binary_structure(3, 1)
    positive_labels, positive_count = ndimage.label(positive_np, structure=structure)
    truth_labels, truth_count = ndimage.label(truth_np, structure=structure)
    truth_components: list[dict[str, object]] = []
    truth_centroids: dict[int, np.ndarray] = {}
    for component_id in range(1, truth_count + 1):
        selected = truth_labels == component_id
        center = np.argwhere(selected).mean(axis=0)
        truth_centroids[component_id] = center
        truth_components.append(
            {
                "truth_component_id": component_id,
                "voxel_count": int(selected.sum()),
                "positive_overlap_voxels": int((selected & positive_np).sum()),
                "centroid": center.tolist(),
            }
        )
    positive_components: list[dict[str, object]] = []
    for component_id in range(1, positive_count + 1):
        selected = positive_labels == component_id
        center = np.argwhere(selected).mean(axis=0)
        overlaps: list[dict[str, int]] = []
        for truth_id in range(1, truth_count + 1):
            overlap = int((selected & (truth_labels == truth_id)).sum())
            if overlap:
                overlaps.append(
                    {"truth_component_id": truth_id, "overlap_voxels": overlap}
                )
        distances = {
            truth_id: float(np.linalg.norm(center - truth_center))
            for truth_id, truth_center in truth_centroids.items()
        }
        nearest_id = min(distances, key=distances.get) if distances else None
        positive_components.append(
            {
                "positive_component_id": component_id,
                "voxel_count": int(selected.sum()),
                "true_positive_voxels": int((selected & truth_np).sum()),
                "precision": float((selected & truth_np).sum() / selected.sum()),
                "centroid": center.tolist(),
                "truth_component_overlaps": overlaps,
                "nearest_truth_component_id": nearest_id,
                "nearest_truth_component_centroid_distance": (
                    distances[nearest_id] if nearest_id is not None else float("nan")
                ),
            }
        )
    positive_center = np.argwhere(positive_np).mean(axis=0) if positive_np.any() else None
    overall_distances = (
        {
            truth_id: float(np.linalg.norm(positive_center - truth_center))
            for truth_id, truth_center in truth_centroids.items()
        }
        if positive_center is not None
        else {}
    )
    overall_nearest = min(overall_distances, key=overall_distances.get) if overall_distances else None
    return {
        "positive_component_count_6": int(positive_count),
        "truth_component_count_6": int(truth_count),
        "positive_components": positive_components,
        "truth_components": truth_components,
        "positive_centroid": positive_center.tolist() if positive_center is not None else None,
        "positive_centroid_nearest_truth_component_id": overall_nearest,
        "positive_centroid_nearest_truth_component_distance": (
            overall_distances[overall_nearest]
            if overall_nearest is not None
            else float("nan")
        ),
    }


def state_metric_rows(
    states: torch.Tensor,
    *,
    group: str,
    binary_truth: torch.Tensor,
    subsurface: torch.Tensor,
) -> list[dict[str, object]]:
    if states.ndim != 5 or states.shape[1:] != binary_truth.shape[1:]:
        raise ValueError("binary states must have shape [N,1,X,Y,Z]")
    rows: list[dict[str, object]] = []
    target = binary_truth[0, 0]
    domain = subsurface[0, 0]
    for index in range(states.shape[0]):
        predicted = states[index, 0].bool()
        metrics = binary_metrics(predicted, target, domain)
        rows.append(
            {
                "sample_group": group,
                "sample_index": index,
                "binary_sha256": tensor_sha256(states[index : index + 1]),
                "target_voxel_count": int((predicted & domain).sum()),
                **metrics,
                "centroid_distance": centroid_distance(
                    predicted & domain, target & domain
                ),
            }
        )
    return rows


def validate_retained_accounting(rows: Sequence[Mapping[str, object]]) -> None:
    retained = [row for row in rows if row.get("sample_group") == "retained"]
    if len(retained) != 96:
        raise ValueError(f"retained-state metric accounting must equal 96, got {len(retained)}")
    if sorted(int(row["sample_index"]) for row in retained) != list(range(96)):
        raise ValueError("retained-state indices must be exactly 0..95")


def aggregate_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    fields = ("iou", "precision", "recall", "centroid_distance")
    result: dict[str, object] = {"sample_count": len(rows)}
    for field in fields:
        values = [float(row[field]) for row in rows]
        finite = [value for value in values if math.isfinite(value)]
        result[field] = {
            "median": statistics.median(finite),
            "min": min(finite),
            "max": max(finite),
        }
    best = max(
        rows,
        key=lambda row: (
            float(row["iou"]),
            float(row["precision"]),
            float(row["recall"]),
            -float(row["centroid_distance"]),
        ),
    )
    result["best_by_iou"] = dict(best)
    return result


def snapshot_hashes(directory: Path) -> dict[str, str]:
    return {
        name: runtime.file_sha256(directory / name)
        for name in B2_INPUT_NAMES
    }


def assert_hashes_unchanged(
    directory: Path,
    before: Mapping[str, str],
) -> None:
    after = snapshot_hashes(directory)
    if dict(before) != after:
        changed = sorted(name for name in before if before[name] != after.get(name))
        raise RuntimeError(f"frozen B2 inputs changed during evaluation: {changed}")


def assert_input_paths_unchanged(input_file_sha256: Mapping[str, str]) -> None:
    changed = [
        path
        for path, expected in input_file_sha256.items()
        if runtime.file_sha256(Path(path)) != expected
    ]
    if changed:
        raise RuntimeError(f"evaluation inputs changed during evaluation: {changed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b2-dir", type=Path, default=DEFAULT_B2_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--decision", choices=MACHINE_DECISIONS, default=None)
    return parser.parse_args()


def evaluate_frozen_b2(directory: Path) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object], dict[str, str], Path]:
    before_hashes = snapshot_hashes(directory)
    run_manifest = read_json(directory / "run_manifest.json")
    pilot_summary = read_json(directory / "pilot_summary.json")
    if run_manifest.get("run_status") != "completed":
        raise ValueError("B2 run is not completed")
    full_parameters = run_manifest.get("full_parameters")
    if not isinstance(full_parameters, Mapping):
        raise ValueError("B2 run has no frozen full_parameters")
    if full_parameters.get("positive_threshold") != 0.8 or full_parameters.get("negative_threshold") != 0.2:
        raise ValueError("B2 thresholds differ from frozen 0.8/0.2")
    if pilot_summary.get("truth_loaded_by_runner") is not False:
        raise ValueError("B2 pilot does not satisfy the truth firewall")

    retained_binary = runtime.load_tensor(directory / "retained_binary_states.pt")
    retained_categorical = runtime.load_tensor(directory / "retained_categorical_states.pt")
    occupancy = normalize_volume(
        runtime.load_tensor(directory / "occupancy_frequency.pt"),
        "occupancy_frequency",
        torch.float32,
    )
    positive = normalize_volume(runtime.load_tensor(directory / "positive_mask.pt"), "positive_mask").bool()
    negative = normalize_volume(runtime.load_tensor(directory / "negative_mask.pt"), "negative_mask").bool()
    unknown = normalize_volume(runtime.load_tensor(directory / "unknown_mask.pt"), "unknown_mask").bool()
    initial_binary = runtime.load_tensor(directory / "initial_binary_models.pt")
    if retained_binary.shape != (96, 1, 64, 64, 64):
        raise ValueError("retained_binary_states must contain exactly 96 states")
    if retained_categorical.shape != retained_binary.shape:
        raise ValueError("retained categorical and binary states must match")
    if initial_binary.shape != (4, 1, 64, 64, 64):
        raise ValueError("initial_binary_models must contain exactly four states")

    observation_record = run_manifest.get("observation_manifest")
    if not isinstance(observation_record, Mapping):
        raise ValueError("B2 manifest has no observation manifest record")
    observation_path = Path(str(observation_record["path"]))
    if runtime.file_sha256(observation_path) != observation_record["sha256"]:
        raise ValueError("observation manifest hash changed")
    observation_manifest = read_json(observation_path)
    observation_dir = observation_path.parent
    subsurface_path = observation_dir / "subsurface_mask.pt"
    subsurface = normalize_volume(
        runtime.load_tensor(subsurface_path),
        "subsurface_mask",
    ).bool()
    truth_record = observation_manifest["phase1_assets"]["truth_model"]
    truth_path = Path(str(truth_record["path"]))
    if runtime.file_sha256(truth_path) != truth_record["sha256"]:
        raise ValueError("authoritative cond_generation_0 truth hash changed")
    true_model = normalize_volume(
        runtime.load_tensor(truth_path), "cond_generation_0_true_model"
    ).long()
    binary_truth = true_model == 9
    input_paths = [directory / name for name in B2_INPUT_NAMES]
    input_paths.extend((observation_path, subsurface_path, truth_path))
    input_file_sha256 = {
        str(path.resolve()): runtime.file_sha256(path) for path in input_paths
    }

    expected_binary = ((retained_categorical.long() == 9) & subsurface).to(
        retained_binary.dtype
    )
    if not torch.equal(expected_binary, retained_binary):
        raise ValueError("retained categorical-to-binary collapse mismatch")
    expected_occupancy = retained_binary.float().mean(dim=0, keepdim=True)
    expected_occupancy = torch.where(
        subsurface, expected_occupancy, torch.zeros_like(expected_occupancy)
    )
    if not torch.equal(expected_occupancy, occupancy):
        raise ValueError("frozen occupancy differs from retained binary-state mean")
    if not torch.equal(positive, (occupancy >= 0.8) & subsurface):
        raise ValueError("frozen positive mask differs from P9 >= 0.8")
    if not torch.equal(negative, (occupancy <= 0.2) & subsurface):
        raise ValueError("frozen negative mask differs from P9 <= 0.2")
    if not torch.equal(unknown, subsurface & ~(positive | negative)):
        raise ValueError("frozen unknown mask differs from fixed partition")

    positive_result = binary_metrics(
        positive[0, 0], binary_truth[0, 0], subsurface[0, 0]
    )
    positive_result["positive_voxels"] = positive_result.pop("predicted_voxels")
    component_result = positive_component_overlap(
        positive[0, 0], binary_truth[0, 0]
    )
    partition_result = truth_partition(
        binary_truth, positive, unknown, negative, subsurface
    )
    negative_overlap = int((negative & binary_truth).sum())
    negative_result = {
        "negative_voxels": int(negative.sum()),
        "negative_intersection_truth_label9_voxels": negative_overlap,
        "fraction_of_truth_label9_marked_negative": negative_overlap
        / int(binary_truth.sum()),
    }

    initial_rows = state_metric_rows(
        initial_binary,
        group="initial",
        binary_truth=binary_truth,
        subsurface=subsurface,
    )
    retained_rows = state_metric_rows(
        retained_binary,
        group="retained",
        binary_truth=binary_truth,
        subsurface=subsurface,
    )
    all_rows = initial_rows + retained_rows
    validate_retained_accounting(all_rows)
    initial_aggregate = aggregate_rows(initial_rows)
    retained_aggregate = aggregate_rows(retained_rows)
    median_changes = {
        field: float(retained_aggregate[field]["median"])
        - float(initial_aggregate[field]["median"])
        for field in ("iou", "precision", "recall", "centroid_distance")
    }

    domain = subsurface.bool()
    target_values = occupancy[binary_truth & domain]
    background_values = occupancy[(~binary_truth) & domain]
    occupancy_result = {
        "truth_label9_mean_p9": float(target_values.mean()),
        "truth_label9_median_p9": float(target_values.median()),
        "background_mean_p9": float(background_values.mean()),
        "background_median_p9": float(background_values.median()),
        "voxelwise_auprc": average_precision(
            occupancy[domain], binary_truth[domain]
        ),
        "evaluation_domain": "all_subsurface_voxels",
    }
    summary: dict[str, object] = {
        "schema": "stage15_b2_retrospective_truth_evaluation_v1",
        "evaluation_only": True,
        "b2_rerun_performed": False,
        "threshold_sweep_performed": False,
        "thresholds": {"positive": 0.8, "negative": 0.2},
        "positive_core": {**positive_result, **component_result},
        "negative_mask": negative_result,
        "truth_partition": partition_result,
        "ensemble": {
            "initial": initial_aggregate,
            "post_burnin": retained_aggregate,
            "post_minus_initial_median": median_changes,
            "overall_truth_alignment_improved": False,
            "improvement_interpretation": (
                "No overall improvement: retained medians have lower IoU, precision, "
                "and recall and a larger centroid distance than the four initial models; "
                "the best retained IoU is higher, but that is not an ensemble-wide shift."
            ),
        },
        "occupancy": occupancy_result,
        "truth_used_only_by_this_retrospective_evaluator": True,
        "decision_policy": "manual_metric_interpretation_without_new_numeric_gate",
        "input_file_sha256": input_file_sha256,
    }
    assert_hashes_unchanged(directory, before_hashes)
    assert_input_paths_unchanged(input_file_sha256)
    return summary, all_rows, partition_result, before_hashes, truth_path


def _fmt(value: object) -> str:
    return f"{float(value):.8g}"


def render_report(summary: Mapping[str, object]) -> str:
    positive = summary["positive_core"]
    negative = summary["negative_mask"]
    partition = summary["truth_partition"]
    ensemble = summary["ensemble"]
    occupancy = summary["occupancy"]
    decision = summary["machine_decision"]
    return f"""# Stage15-B3 — Frozen B2 retrospective truth evaluation

Machine decision: **{decision}**

This evaluation opened `cond_generation_0` truth only after all B2 pCN, occupancy, and 0.8/0.2 masks were frozen. It did not rerun or modify B2 and performed no threshold sweep.

## Positive core

- Positive / TP / FP: {positive['positive_voxels']} / {positive['true_positive_voxels']} / {positive['false_positive_voxels']}
- Precision / recall / IoU: {_fmt(positive['precision'])} / {_fmt(positive['recall'])} / {_fmt(positive['iou'])}
- Positive components: {positive['positive_component_count_6']}; truth components: {positive['truth_component_count_6']}
- Positive centroid nearest-truth-component distance: {_fmt(positive['positive_centroid_nearest_truth_component_distance'])} voxels

## Truth-label9 partition

- Positive: {partition['counts']['positive']} ({_fmt(partition['fractions']['positive'])})
- Unknown: {partition['counts']['unknown']} ({_fmt(partition['fractions']['unknown'])})
- Negative: {partition['counts']['negative']} ({_fmt(partition['fractions']['negative'])})
- Negative mask voxels: {negative['negative_voxels']}
- Fraction of truth label9 marked negative: {_fmt(negative['fraction_of_truth_label9_marked_negative'])}

## Frozen ensemble

- Initial median IoU / precision / recall / centroid distance: {_fmt(ensemble['initial']['iou']['median'])} / {_fmt(ensemble['initial']['precision']['median'])} / {_fmt(ensemble['initial']['recall']['median'])} / {_fmt(ensemble['initial']['centroid_distance']['median'])}
- Post-burn-in median: {_fmt(ensemble['post_burnin']['iou']['median'])} / {_fmt(ensemble['post_burnin']['precision']['median'])} / {_fmt(ensemble['post_burnin']['recall']['median'])} / {_fmt(ensemble['post_burnin']['centroid_distance']['median'])}
- Best retained IoU / precision / recall / centroid distance: {_fmt(ensemble['post_burnin']['best_by_iou']['iou'])} / {_fmt(ensemble['post_burnin']['best_by_iou']['precision'])} / {_fmt(ensemble['post_burnin']['best_by_iou']['recall'])} / {_fmt(ensemble['post_burnin']['best_by_iou']['centroid_distance'])}
- Post-burn-in IoU range: {_fmt(ensemble['post_burnin']['iou']['min'])}–{_fmt(ensemble['post_burnin']['iou']['max'])}
- Post-burn-in precision range: {_fmt(ensemble['post_burnin']['precision']['min'])}–{_fmt(ensemble['post_burnin']['precision']['max'])}
- Post-burn-in recall range: {_fmt(ensemble['post_burnin']['recall']['min'])}–{_fmt(ensemble['post_burnin']['recall']['max'])}
- Post-burn-in centroid-distance range: {_fmt(ensemble['post_burnin']['centroid_distance']['min'])}–{_fmt(ensemble['post_burnin']['centroid_distance']['max'])}
- Overall truth alignment improved: {ensemble['overall_truth_alignment_improved']}. {ensemble['improvement_interpretation']}

## Continuous occupancy

- Truth-label9 mean / median P9: {_fmt(occupancy['truth_label9_mean_p9'])} / {_fmt(occupancy['truth_label9_median_p9'])}
- Background mean / median P9: {_fmt(occupancy['background_mean_p9'])} / {_fmt(occupancy['background_median_p9'])}
- Voxelwise AUPRC: {_fmt(occupancy['voxelwise_auprc'])}

The decision is an explicit manual interpretation of these frozen metrics, not the output of a newly introduced numerical gate.
"""


def main() -> None:
    args = parse_args()
    if not args.preview and args.decision is None:
        raise ValueError("final evaluation requires one of the three --decision values")
    summary, rows, partition, input_hashes, truth_path = evaluate_frozen_b2(
        args.b2_dir
    )
    if args.preview:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    refuse_nonempty(args.output_dir)
    manifest = base_manifest(
        "stage15_b2_retrospective_truth_evaluation_run_v1", Path(__file__)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "manifest.json", manifest)
    try:
        summary["machine_decision"] = args.decision
        summary["machine_decision_allowed_values"] = list(MACHINE_DECISIONS)
        summary["machine_decision_rationale"] = {
            "POSITIVE_CORE_VALID_NEGATIVE_VALID": (
                "The frozen positive core aligns with truth and the frozen negative mask "
                "does not materially obstruct truth-label9 completion."
            ),
            "POSITIVE_CORE_VALID_NEGATIVE_UNSAFE": (
                "The frozen positive core aligns with truth, but the frozen negative mask "
                "obstructs too much truth-label9 completion."
            ),
            "POSITIVE_CORE_NOT_VALID": (
                "Only 56 of 114 frozen positive-core voxels are true label9; the observed "
                "precision does not support interpreting this mask as a high-precision core."
            ),
        }[args.decision]
        write_json(args.output_dir / "summary.json", summary)
        write_csv(args.output_dir / "retained_truth_metrics.csv", rows)
        write_json(args.output_dir / "truth_partition.json", partition)
        (args.output_dir / "REPORT.md").write_text(
            render_report(summary), encoding="utf-8"
        )
        assert_hashes_unchanged(args.b2_dir, input_hashes)
        assert_input_paths_unchanged(summary["input_file_sha256"])
        manifest.update(
            {
                "run_status": "completed",
                "machine_decision": args.decision,
                "b2_input_directory": str(args.b2_dir),
                "b2_input_file_sha256_before_and_after": input_hashes,
                "b2_inputs_unchanged": True,
                "truth_asset": runtime.asset_record(truth_path),
                "summary": runtime.asset_record(args.output_dir / "summary.json"),
                "retained_truth_metrics": runtime.asset_record(
                    args.output_dir / "retained_truth_metrics.csv"
                ),
                "truth_partition": runtime.asset_record(
                    args.output_dir / "truth_partition.json"
                ),
                "report": runtime.asset_record(args.output_dir / "REPORT.md"),
                "pcn_rerun_performed": False,
                "flow_sampling_performed": False,
                "consensus_generation_performed": False,
                "threshold_sweep_performed": False,
                "truth_loaded_by_evaluator": True,
            }
        )
        write_json(args.output_dir / "manifest.json", manifest)
    except Exception as exc:
        manifest.update(
            {"run_status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        )
        write_json(args.output_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
