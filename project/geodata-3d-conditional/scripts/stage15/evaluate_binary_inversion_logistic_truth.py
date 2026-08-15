#!/usr/bin/env python3
"""Independent truth evaluator for the binary Stage15-G linear mapper."""

from __future__ import annotations

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
from guidance.seismic import tensor_sha256
from scripts.stage10.evaluate_bridge_information import binary_information_metrics
from scripts.stage15.common import (
    base_manifest,
    normalize_volume,
    read_json,
    refuse_nonempty,
    write_json,
)
from scripts.stage15.evaluate_inversion_score_probability_truth import (
    binary_metrics,
    truth_partition,
)


EXPERIMENT_ROOT = PROJECT_DIR / "experiments/stage15_binary_seismic_consensus"
DEFAULT_OBSERVATION_DIR = EXPERIMENT_ROOT / "observations/cond_generation_0"
DEFAULT_CALIBRATION_DIR = EXPERIMENT_ROOT / "binary_logistic/calibration_n128_8x8x8_v2"
DEFAULT_INFERENCE_DIR = EXPERIMENT_ROOT / "binary_logistic/cond_generation_0_8x8x8_v2"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "reports/binary_inversion_logistic_truth_evaluation_v2"


def parse_args() -> object:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--observation-dir", type=Path, default=DEFAULT_OBSERVATION_DIR)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--inference-dir", type=Path, default=DEFAULT_INFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def positive_validation_summary(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["label9_voxels"]) > 0]
    auprc = [float(row["auprc"]) for row in rows]
    prevalence = [float(row["prevalence"]) for row in rows]
    separation = [
        float(row["truth_mean_probability"]) - float(row["background_mean_probability"])
        for row in rows
    ]
    return {
        "case_count": len(rows),
        "median_auprc": statistics.median(auprc),
        "auprc_above_own_prevalence_case_count": sum(
            value > base for value, base in zip(auprc, prevalence)
        ),
        "positive_truth_background_separation_case_count": sum(value > 0 for value in separation),
        "median_truth_minus_background_probability": statistics.median(separation),
        "cases": [
            {
                "root_seed": int(row["root_seed"]),
                "prevalence": base,
                "auprc": value,
                "truth_minus_background_probability": delta,
            }
            for row, base, value, delta in zip(rows, prevalence, auprc, separation)
        ],
    }


def evaluate(
    observation_dir: Path, calibration_dir: Path, inference_dir: Path
) -> tuple[dict[str, object], dict[str, str]]:
    inputs = [
        observation_dir / "manifest.json",
        observation_dir / "subsurface_mask.pt",
        calibration_dir / "manifest.json",
        calibration_dir / "binary_logistic_checkpoint.pt",
        calibration_dir / "validation_metrics.csv",
        calibration_dir / "validation_summary.json",
        inference_dir / "manifest.json",
        inference_dir / "coarse_label9_probability.pt",
        inference_dir / "label9_probability_volume.pt",
    ]
    hashes = {str(path.resolve()): runtime.file_sha256(path) for path in inputs}
    calibration_manifest = read_json(calibration_dir / "manifest.json")
    inference_manifest = read_json(inference_dir / "manifest.json")
    if calibration_manifest.get("run_status") != "completed":
        raise ValueError("Stage15-G calibration is incomplete")
    if inference_manifest.get("run_status") != "completed":
        raise ValueError("Stage15-G held-out inference is incomplete")
    if inference_manifest.get("truth_loaded_by_runner") is not False:
        raise ValueError("Stage15-G truth firewall was violated")
    observation_manifest = read_json(observation_dir / "manifest.json")
    truth_record = observation_manifest["phase1_assets"]["truth_model"]
    truth_path = Path(str(truth_record["path"]))
    if runtime.file_sha256(truth_path) != truth_record["sha256"]:
        raise ValueError("cond_generation_0 truth changed")
    hashes[str(truth_path.resolve())] = runtime.file_sha256(truth_path)
    truth = normalize_volume(runtime.load_tensor(truth_path), "true_model").long()
    subsurface = normalize_volume(
        runtime.load_tensor(observation_dir / "subsurface_mask.pt"), "subsurface_mask"
    ).bool()
    binary_truth = (truth == 9) & subsurface
    probability = normalize_volume(
        runtime.load_tensor(inference_dir / "label9_probability_volume.pt"),
        "label9_probability_volume",
    ).float()
    if tensor_sha256(probability) != inference_manifest["probability_volume_tensor_sha256"]:
        raise ValueError("Stage15-G probability tensor hash mismatch")
    information = binary_information_metrics(probability, binary_truth, subsurface)
    truth_scores = probability[binary_truth]
    background_scores = probability[subsurface & ~binary_truth]
    positive = (probability >= 0.8) & subsurface
    summary = {
        "schema": "stage15_g_binary_inversion_logistic_truth_evaluation_v1",
        "run_status": "completed",
        "binary_target": "raw label9 versus every other subsurface voxel",
        "calibration": {
            "training_case_count": calibration_manifest["training_case_count"],
            "validation_case_count": calibration_manifest["validation_case_count"],
            "final_training_bce": calibration_manifest["final_training_bce"],
            "validation": read_json(calibration_dir / "validation_summary.json"),
            "positive_validation_cases": positive_validation_summary(
                calibration_dir / "validation_metrics.csv"
            ),
        },
        "heldout": {
            **information,
            "truth_mean_probability": float(truth_scores.mean()),
            "truth_median_probability": float(truth_scores.median()),
            "background_mean_probability": float(background_scores.mean()),
            "background_median_probability": float(background_scores.median()),
            "subsurface_probability_range": [
                float(probability[subsurface].min()),
                float(probability[subsurface].max()),
            ],
            "positive_metrics_at_0p8": binary_metrics(positive, binary_truth, subsurface),
            "truth_partition_at_0p8_0p2": truth_partition(probability, binary_truth),
        },
        "historical_auprc": {
            "stage15_b2": 0.057284505,
            "stage15_c": 0.045394953,
            "stage15_f_global_histogram": 0.08987236785890922,
        },
        "truth_loaded_only_by_evaluator": True,
        "flow_used": False,
        "parameter_sweep_performed": False,
        "input_file_sha256": hashes,
    }
    for path, expected in hashes.items():
        if runtime.file_sha256(Path(path)) != expected:
            raise RuntimeError(f"Stage15-G evaluation input changed: {path}")
    return summary, hashes


def render_report(summary: Mapping[str, object]) -> str:
    validation = summary["calibration"]["validation"]
    positive = summary["calibration"]["positive_validation_cases"]
    heldout = summary["heldout"]
    diagnostic = heldout["positive_metrics_at_0p8"]
    partition = heldout["truth_partition_at_0p8_0p2"]
    return f"""# Stage15-G — Binary case-relative inversion logistic mapper

The mapper is a single 4-to-1 linear layer. Its target is strictly binary: raw label9 versus every other subsurface voxel. It reuses frozen Stage15-F inversion scores and does not rerun seismic, inversion, or Flow.

## Validation

- Pooled prevalence / AUPRC: {validation['prevalence']:.8g} / {validation['auprc']:.8g}
- Truth/background mean P9: {validation['truth_mean_probability']:.8g} / {validation['background_mean_probability']:.8g}
- Positive-case median AUPRC: {positive['median_auprc']:.8g}
- Positive cases with AUPRC above own prevalence: {positive['auprc_above_own_prevalence_case_count']}/{positive['case_count']}
- Positive cases with truth P9 above background: {positive['positive_truth_background_separation_case_count']}/{positive['case_count']}

## Retrospective cond_generation_0

- AUPRC: {heldout['auprc']:.8g}
- Truth mean/median P9: {heldout['truth_mean_probability']:.8g} / {heldout['truth_median_probability']:.8g}
- Background mean/median P9: {heldout['background_mean_probability']:.8g} / {heldout['background_median_probability']:.8g}
- Probability range: {heldout['subsurface_probability_range'][0]:.8g} to {heldout['subsurface_probability_range'][1]:.8g}
- >=0.8 positive precision/recall/IoU: {diagnostic['precision']} / {diagnostic['recall']} / {diagnostic['iou']}
- Truth positive/unknown/negative: {partition['counts']['positive']} / {partition['counts']['unknown']} / {partition['counts']['negative']}

Historical AUPRC: B2 0.057284505; C 0.045394953; F global histogram 0.089872368.
"""


def main() -> None:
    args = parse_args()
    refuse_nonempty(args.output_dir)
    summary, hashes = evaluate(args.observation_dir, args.calibration_dir, args.inference_dir)
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "REPORT.md").write_text(render_report(summary), encoding="utf-8")
    manifest = base_manifest(
        "stage15_g_binary_inversion_logistic_truth_evaluation_run_v1", Path(__file__)
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
