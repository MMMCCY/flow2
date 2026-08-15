#!/usr/bin/env python3
"""Retrospective truth evaluation for the frozen Stage15-F P9 volume."""

from __future__ import annotations

import json
import csv
import statistics
import sys
from pathlib import Path
from typing import Mapping

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime
from guidance.inversion_score_probability import coarse_truth_occupancy_8
from guidance.seismic import tensor_sha256
from scripts.stage10.evaluate_bridge_information import binary_information_metrics
from scripts.stage15.common import (
    base_manifest,
    normalize_volume,
    read_json,
    refuse_nonempty,
    write_json,
)
from scripts.stage15.evaluate_coarse_binary_seismic_truth import coarse_metrics


EXPERIMENT_ROOT = PROJECT_DIR / "experiments/stage15_binary_seismic_consensus"
DEFAULT_OBSERVATION_DIR = EXPERIMENT_ROOT / "observations/cond_generation_0"
DEFAULT_INFERENCE_DIR = (
    EXPERIMENT_ROOT / "inversion_probability/cond_generation_0_8x8x8_v1"
)
DEFAULT_CALIBRATION_DIR = (
    EXPERIMENT_ROOT / "inversion_probability/calibration_n128_8x8x8_v1"
)
DEFAULT_OUTPUT = (
    EXPERIMENT_ROOT / "reports/inversion_score_probability_truth_evaluation_8x8x8_v2"
)


def parse_args() -> object:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--observation-dir", type=Path, default=DEFAULT_OBSERVATION_DIR)
    parser.add_argument("--inference-dir", type=Path, default=DEFAULT_INFERENCE_DIR)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def binary_metrics(prediction: torch.Tensor, truth: torch.Tensor, domain: torch.Tensor) -> dict[str, object]:
    predicted = prediction.bool() & domain.bool()
    target = truth.bool() & domain.bool()
    tp = int((predicted & target).sum())
    fp = int((predicted & ~target & domain.bool()).sum())
    fn = int((~predicted & target).sum())
    return {
        "positive_voxels": int(predicted.sum()),
        "true_positive_voxels": tp,
        "false_positive_voxels": fp,
        "false_negative_voxels": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
    }


def truth_partition(probability: torch.Tensor, truth: torch.Tensor) -> dict[str, object]:
    target = truth.bool()
    total = int(target.sum())
    counts = {
        "positive": int((target & (probability >= 0.8)).sum()),
        "unknown": int((target & (probability > 0.2) & (probability < 0.8)).sum()),
        "negative": int((target & (probability <= 0.2)).sum()),
    }
    return {
        "truth_label9_voxels": total,
        "counts": counts,
        "fractions": {name: value / total for name, value in counts.items()},
        "exactly_partitions_truth": sum(counts.values()) == total,
    }


def bounding_box(mask: torch.Tensor) -> list[list[int]] | None:
    coordinates = torch.nonzero(mask[0, 0].bool(), as_tuple=False)
    if not coordinates.numel():
        return None
    return [coordinates.amin(dim=0).tolist(), coordinates.amax(dim=0).tolist()]


def evaluate(
    observation_dir: Path, inference_dir: Path, calibration_dir: Path
) -> tuple[dict[str, object], dict[str, str]]:
    input_paths = [
        observation_dir / "manifest.json",
        observation_dir / "subsurface_mask.pt",
        inference_dir / "manifest.json",
        inference_dir / "coarse_inversion_score.pt",
        inference_dir / "inversion_score_probability_volume.pt",
        calibration_dir / "calibration_manifest.json",
        calibration_dir / "validation_summary.json",
        calibration_dir / "validation_metrics.csv",
        calibration_dir / "inversion_score_bin_edges.pt",
        calibration_dir / "inversion_score_probability_lookup.pt",
    ]
    hashes = {str(path.resolve()): runtime.file_sha256(path) for path in input_paths}
    inference_manifest = read_json(inference_dir / "manifest.json")
    if inference_manifest.get("run_status") != "completed":
        raise ValueError("Stage15-F held-out inference is incomplete")
    if inference_manifest.get("truth_loaded_by_runner") is not False:
        raise ValueError("Stage15-F held-out truth firewall was violated")
    calibration_manifest = read_json(calibration_dir / "calibration_manifest.json")
    if calibration_manifest.get("run_status") != "completed":
        raise ValueError("Stage15-F calibration is incomplete")
    observation_manifest = read_json(observation_dir / "manifest.json")
    truth_record = observation_manifest["phase1_assets"]["truth_model"]
    truth_path = Path(str(truth_record["path"]))
    if runtime.file_sha256(truth_path) != truth_record["sha256"]:
        raise ValueError("cond_generation_0 truth hash changed")
    hashes[str(truth_path.resolve())] = runtime.file_sha256(truth_path)

    truth = normalize_volume(runtime.load_tensor(truth_path), "true_model").long()
    subsurface = normalize_volume(
        runtime.load_tensor(observation_dir / "subsurface_mask.pt"), "subsurface_mask"
    ).bool()
    binary_truth = (truth == 9) & subsurface
    probability = normalize_volume(
        runtime.load_tensor(inference_dir / "inversion_score_probability_volume.pt"),
        "inversion_score_probability_volume",
    ).float()
    q_coarse = runtime.load_tensor(inference_dir / "coarse_inversion_score.pt").float()
    if q_coarse.shape != (1, 1, 8, 8, 8):
        raise ValueError("held-out inversion score must have shape [1,1,8,8,8]")
    if tensor_sha256(probability) != inference_manifest["probability_volume_tensor_sha256"]:
        raise ValueError("held-out probability tensor hash mismatch")
    information = binary_information_metrics(probability, binary_truth, subsurface)
    active = subsurface.bool()
    truth_scores = probability[binary_truth]
    background_scores = probability[active & ~binary_truth]
    positive = (probability >= 0.8) & active
    partition = truth_partition(probability, binary_truth)

    q_true, presence, support = coarse_truth_occupancy_8(binary_truth, subsurface)
    domain = support > 0
    raw_coarse = coarse_metrics(q_coarse, q_true, presence, domain)
    with (calibration_dir / "validation_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        positive_validation_rows = [
            row for row in csv.DictReader(handle) if int(row["label9_voxels"]) > 0
        ]
    validation_case_auprc = [float(row["auprc"]) for row in positive_validation_rows]
    validation_case_prevalence = [
        float(row["prevalence"]) for row in positive_validation_rows
    ]
    validation_case_deltas = [
        float(row["truth_mean_probability"]) - float(row["background_mean_probability"])
        for row in positive_validation_rows
    ]
    validation_positive_case_summary = {
        "case_count": len(positive_validation_rows),
        "auprc_median": statistics.median(validation_case_auprc),
        "auprc_above_own_prevalence_case_count": sum(
            auprc > prevalence
            for auprc, prevalence in zip(validation_case_auprc, validation_case_prevalence)
        ),
        "truth_mean_above_background_case_count": sum(
            delta > 0 for delta in validation_case_deltas
        ),
        "truth_minus_background_mean_probability": {
            "minimum": min(validation_case_deltas),
            "median": statistics.median(validation_case_deltas),
            "maximum": max(validation_case_deltas),
        },
        "cases": [
            {
                "root_seed": int(row["root_seed"]),
                "prevalence": float(row["prevalence"]),
                "auprc": float(row["auprc"]),
                "truth_minus_background_mean_probability": delta,
            }
            for row, delta in zip(positive_validation_rows, validation_case_deltas)
        ],
    }
    summary = {
        "schema": "stage15_f_inversion_score_probability_truth_evaluation_v1",
        "run_status": "completed",
        "calibration": {
            "case_count": int(calibration_manifest["full_parameters"]["calibration_case_count"]),
            "training_case_count": len(calibration_manifest["training_seeds"]),
            "validation_case_count": len(calibration_manifest["validation_seeds"]),
            "training_label9_positive_case_count": calibration_manifest[
                "training_label9_positive_case_count"
            ],
            "training_prevalence": calibration_manifest["training_prevalence"],
            "validation": read_json(calibration_dir / "validation_summary.json"),
            "validation_positive_cases": validation_positive_case_summary,
            "empty_quantile_bins_due_to_ties": calibration_manifest[
                "empty_bins_due_to_ties"
            ],
        },
        "heldout_probability": {
            **information,
            "truth_mean": float(truth_scores.mean()),
            "truth_median": float(truth_scores.median()),
            "background_mean": float(background_scores.mean()),
            "background_median": float(background_scores.median()),
            "subsurface_minimum": float(probability[active].min()),
            "subsurface_maximum": float(probability[active].max()),
            "positive_metrics_at_0p8": binary_metrics(positive, binary_truth, subsurface),
            "truth_partition_at_0p8_0p2": partition,
            "truth_bounding_box_xyz": bounding_box(binary_truth),
            "positive_bounding_box_xyz": bounding_box(positive),
        },
        "heldout_raw_coarse_inversion": raw_coarse,
        "historical_auprc": {
            "stage15_b2_occupancy": 0.057284505,
            "stage15_c_local_energy": 0.045394953,
        },
        "truth_loaded_only_by_evaluator": True,
        "flow_used": False,
        "threshold_sweep_performed": False,
        "input_file_sha256": hashes,
    }
    for path, expected in hashes.items():
        if runtime.file_sha256(Path(path)) != expected:
            raise RuntimeError(f"Stage15-F evaluation input changed: {path}")
    return summary, hashes


def render_report(summary: Mapping[str, object]) -> str:
    heldout = summary["heldout_probability"]
    validation = summary["calibration"]["validation"]
    validation_positive = summary["calibration"]["validation_positive_cases"]
    coarse = summary["heldout_raw_coarse_inversion"]
    positive = heldout["positive_metrics_at_0p8"]
    partition = heldout["truth_partition_at_0p8_0p2"]
    return f"""# Stage15-F — 8^3 inversion score to empirical P(label9)

This experiment uses one deterministic unregularized 8^3 inversion per synthetic case and one 64-bin empirical lookup. It uses no Flow, learned model, class balancing, or parameter sweep. The inversion score is not treated as a probability until after calibration.

## Calibration and validation

- Cases: {summary['calibration']['case_count']} ({summary['calibration']['training_case_count']} train / {summary['calibration']['validation_case_count']} validation)
- Train label9 prevalence: {summary['calibration']['training_prevalence']:.8g}
- Validation label9-positive cases: {validation['label9_positive_case_count']}/{validation['case_count']}
- Validation prevalence / AUPRC: {validation['prevalence']:.8g} / {validation['auprc']:.8g}
- Validation truth/background mean P9: {validation['truth_mean_probability']:.8g} / {validation['background_mean_probability']:.8g}
- Positive-case median AUPRC: {validation_positive['auprc_median']:.8g}
- Positive cases with AUPRC above own prevalence: {validation_positive['auprc_above_own_prevalence_case_count']}/{validation_positive['case_count']}
- Positive cases with truth mean P9 above background: {validation_positive['truth_mean_above_background_case_count']}/{validation_positive['case_count']}
- Empty quantile bins caused by tied inversion scores: {summary['calibration']['empty_quantile_bins_due_to_ties']}/64

## Retrospective cond_generation_0

- P9 AUPRC: {heldout['auprc']:.8g}
- Truth mean/median P9: {heldout['truth_mean']:.8g} / {heldout['truth_median']:.8g}
- Background mean/median P9: {heldout['background_mean']:.8g} / {heldout['background_median']:.8g}
- P9 range: {heldout['subsurface_minimum']:.8g} to {heldout['subsurface_maximum']:.8g}
- Positive >=0.8 voxels: {positive['positive_voxels']}
- Positive precision / recall / IoU: {positive['precision']} / {positive['recall']} / {positive['iou']}
- Truth positive/unknown/negative: {partition['counts']['positive']} / {partition['counts']['unknown']} / {partition['counts']['negative']}

Raw held-out coarse score localization: Pearson {coarse['pearson_correlation']:.8g}, Spearman {coarse['spearman_correlation']:.8g}, target/background mean {coarse['mean_q_pred_target_containing_cells']:.8g}/{coarse['mean_q_pred_background_only_cells']:.8g}, top-k {coarse['top_k']['overlap_count']}/{coarse['top_k']['k']}, centroid distance {coarse['centroid_distance_coarse_cells']:.8g} coarse cells.

Historical AUPRC: Stage15-B2 = 0.057284505; Stage15-C = 0.045394953.
"""


def main() -> None:
    args = parse_args()
    refuse_nonempty(args.output_dir)
    summary, hashes = evaluate(
        args.observation_dir, args.inference_dir, args.calibration_dir
    )
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "REPORT.md").write_text(render_report(summary), encoding="utf-8")
    manifest = base_manifest(
        "stage15_f_inversion_score_probability_truth_evaluation_run_v1", Path(__file__)
    )
    manifest.update(
        {
            "run_status": "completed",
            "truth_loaded_only_by_evaluator": True,
            "input_file_sha256_before_and_after": hashes,
        }
    )
    write_json(args.output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
