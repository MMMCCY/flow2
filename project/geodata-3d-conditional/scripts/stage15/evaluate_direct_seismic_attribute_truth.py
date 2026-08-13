#!/usr/bin/env python3
"""Retrospective truth evaluation for the frozen Stage15-C direct probability map."""

from __future__ import annotations

import argparse
import json
import math
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
from scripts.stage10.evaluate_bridge_information import average_precision
from scripts.stage15.common import (
    base_manifest,
    normalize_volume,
    read_json,
    refuse_nonempty,
    write_json,
)
from scripts.stage15.evaluate_binary_flow_pcn_truth import binary_metrics, truth_partition


EXPERIMENT_ROOT = PROJECT_DIR / "experiments/stage15_binary_seismic_consensus"
DEFAULT_CALIBRATION = EXPERIMENT_ROOT / "direct_attribute/calibration_n128_v1"
DEFAULT_HELDOUT = EXPERIMENT_ROOT / "direct_attribute/cond_generation_0_v1"
DEFAULT_OBSERVATION = EXPERIMENT_ROOT / "observations/cond_generation_0"
DEFAULT_B3_SUMMARY = EXPERIMENT_ROOT / "reports/b2_truth_evaluation_v1/summary.json"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "reports/direct_attribute_truth_evaluation_v1"
FROZEN_B2_AUPRC = 0.057284505


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--heldout-dir", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--observation-dir", type=Path, default=DEFAULT_OBSERVATION)
    parser.add_argument("--b3-summary", type=Path, default=DEFAULT_B3_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def bbox(mask: torch.Tensor) -> dict[str, object] | None:
    points = torch.nonzero(mask.bool(), as_tuple=False)
    if points.numel() == 0:
        return None
    return {
        "voxel_count": int(points.shape[0]),
        "min_xyz": points.amin(dim=0).tolist(),
        "max_xyz": points.amax(dim=0).tolist(),
    }


def weighted_centroid(weights: torch.Tensor, domain: torch.Tensor) -> list[float] | None:
    selected = torch.where(domain.bool(), weights.float(), torch.zeros_like(weights.float()))
    total = selected.sum()
    if float(total) <= 0:
        return None
    coordinates = torch.stack(
        torch.meshgrid(
            *(torch.arange(size, dtype=torch.float32) for size in weights.shape),
            indexing="ij",
        ),
        dim=-1,
    )
    return ((coordinates * selected.unsqueeze(-1)).sum(dim=(0, 1, 2)) / total).tolist()


def euclidean(left: list[float] | None, right: list[float] | None) -> float:
    if left is None or right is None:
        return float("nan")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def evaluate(
    calibration_dir: Path,
    heldout_dir: Path,
    observation_dir: Path,
    b3_summary_path: Path,
) -> tuple[dict[str, object], dict[str, str]]:
    inputs = (
        calibration_dir / "calibration_manifest.json",
        calibration_dir / "attribute_bin_edges.pt",
        calibration_dir / "attribute_probability_lookup.pt",
        heldout_dir / "manifest.json",
        heldout_dir / "seismic_attribute_volume.pt",
        heldout_dir / "seismic_probability_volume.pt",
        observation_dir / "manifest.json",
        observation_dir / "subsurface_mask.pt",
        b3_summary_path,
    )
    hashes = {str(path.resolve()): runtime.file_sha256(path) for path in inputs}
    calibration_manifest = read_json(inputs[0])
    heldout_manifest = read_json(inputs[3])
    if calibration_manifest.get("run_status") != "completed":
        raise ValueError("Stage15-C calibration is not completed")
    if calibration_manifest.get("calibration_case_count") != 128:
        raise ValueError("Stage15-C calibration does not contain 128 cases")
    if calibration_manifest.get("cond_generation_0_excluded") is not True:
        raise ValueError("held-out case exclusion is not certified")
    if heldout_manifest.get("run_status") != "completed" or heldout_manifest.get("truth_loaded_by_runner") is not False:
        raise ValueError("held-out direct mapper did not preserve the truth firewall")

    attribute = normalize_volume(
        runtime.load_tensor(inputs[4]), "seismic_attribute_volume", torch.float32
    )
    probability = normalize_volume(
        runtime.load_tensor(inputs[5]), "seismic_probability_volume", torch.float32
    )
    subsurface = normalize_volume(runtime.load_tensor(inputs[7]), "subsurface_mask").bool()
    if not torch.isfinite(probability).all() or bool(((probability < 0) | (probability > 1)).any()):
        raise ValueError("held-out P9 volume must be finite in [0,1]")
    if bool((probability[~subsurface] != 0).any()):
        raise ValueError("held-out P9 must be zero outside subsurface")

    observation_manifest = read_json(inputs[6])
    truth_record = observation_manifest["phase1_assets"]["truth_model"]
    truth_path = Path(str(truth_record["path"]))
    if runtime.file_sha256(truth_path) != truth_record["sha256"]:
        raise ValueError("held-out truth asset hash changed")
    hashes[str(truth_path.resolve())] = runtime.file_sha256(truth_path)
    geology = normalize_volume(runtime.load_tensor(truth_path), "cond_generation_0_truth").long()
    binary_truth = (geology == 9) & subsurface

    positive = (probability >= 0.8) & subsurface
    negative = (probability <= 0.2) & subsurface
    unknown = subsurface & ~(positive | negative)
    positive_result = binary_metrics(
        positive[0, 0], binary_truth[0, 0], subsurface[0, 0]
    )
    positive_result["positive_voxels"] = positive_result.pop("predicted_voxels")
    partition = truth_partition(binary_truth, positive, unknown, negative, subsurface)

    domain_probability = probability[subsurface]
    truth_probability = probability[binary_truth]
    background_probability = probability[(~binary_truth) & subsurface]
    auprc = average_precision(domain_probability, binary_truth[subsurface])
    truth_mask = binary_truth[0, 0]
    probability_3d = probability[0, 0]
    domain_3d = subsurface[0, 0]
    truth_points = torch.nonzero(truth_mask, as_tuple=False).float()
    truth_center = truth_points.mean(dim=0).tolist() if truth_points.numel() else None
    probability_center = weighted_centroid(probability_3d, domain_3d)
    truth_box = bbox(truth_mask)
    positive_box = bbox(positive[0, 0])
    nonzero_box = bbox((probability_3d > 0) & domain_3d)
    maximum_p9 = float(domain_probability.max())
    maximum_mask = (probability_3d == maximum_p9) & domain_3d
    maximum_box = bbox(maximum_mask)
    in_truth_bbox_mass_fraction = float("nan")
    if truth_box is not None and float(probability_3d[domain_3d].sum()) > 0:
        lower = truth_box["min_xyz"]
        upper = truth_box["max_xyz"]
        box_mask = torch.zeros_like(domain_3d)
        box_mask[
            lower[0] : upper[0] + 1,
            lower[1] : upper[1] + 1,
            lower[2] : upper[2] + 1,
        ] = True
        in_truth_bbox_mass_fraction = float(
            probability_3d[box_mask & domain_3d].sum() / probability_3d[domain_3d].sum()
        )

    b3 = read_json(b3_summary_path)
    frozen_b2 = float(b3["occupancy"]["voxelwise_auprc"])
    if not math.isclose(frozen_b2, FROZEN_B2_AUPRC, rel_tol=0.0, abs_tol=5e-10):
        raise ValueError("frozen B2 AUPRC differs from the registered comparison value")
    summary: dict[str, object] = {
        "schema": "stage15_c_direct_attribute_truth_evaluation_v1",
        "run_status": "completed",
        "truth_role": "retrospective_evaluation_only",
        "calibration_case_count": 128,
        "attribute_definition": calibration_manifest["attribute_definition"],
        "time_to_depth": calibration_manifest["time_to_depth"],
        "voxelwise_auprc": auprc,
        "truth_background_separation": {
            "truth_label9_mean_p9": float(truth_probability.mean()),
            "truth_label9_median_p9": float(truth_probability.median()),
            "background_mean_p9": float(background_probability.mean()),
            "background_median_p9": float(background_probability.median()),
            "mean_difference_truth_minus_background": float(
                truth_probability.mean() - background_probability.mean()
            ),
            "median_difference_truth_minus_background": float(
                truth_probability.median() - background_probability.median()
            ),
        },
        "fixed_threshold_diagnostics": {
            "thresholds": {"positive": 0.8, "negative": 0.2},
            "positive": positive_result,
            "truth_partition": partition,
            "negative_voxels": int(negative.sum()),
            "unknown_voxels": int(unknown.sum()),
        },
        "spatial_localization": {
            "truth_label9_bbox": truth_box,
            "positive_p9_ge_0_8_bbox": positive_box,
            "nonzero_probability_bbox": nonzero_box,
            "maximum_p9": maximum_p9,
            "maximum_p9_plateau_bbox": maximum_box,
            "probability_weighted_centroid_xyz": probability_center,
            "truth_label9_centroid_xyz": truth_center,
            "probability_centroid_to_truth_centroid_distance": euclidean(
                probability_center, truth_center
            ),
            "probability_mass_fraction_inside_truth_bbox": in_truth_bbox_mass_fraction,
        },
        "comparison_to_frozen_b2": {
            "b2_occupancy_auprc": frozen_b2,
            "direct_attribute_auprc": auprc,
            "absolute_delta": auprc - frozen_b2,
            "ratio": auprc / frozen_b2,
            "direct_attribute_better_localization_by_auprc": auprc > frozen_b2,
        },
        "threshold_sweep_performed": False,
        "input_file_sha256": hashes,
    }
    for path, expected in hashes.items():
        if runtime.file_sha256(Path(path)) != expected:
            raise RuntimeError(f"Stage15-C evaluation input changed: {path}")
    return summary, hashes


def _fmt(value: object) -> str:
    return f"{float(value):.8g}"


def render_report(summary: Mapping[str, object]) -> str:
    separation = summary["truth_background_separation"]
    diagnostics = summary["fixed_threshold_diagnostics"]
    positive = diagnostics["positive"]
    partition = diagnostics["truth_partition"]
    comparison = summary["comparison_to_frozen_b2"]
    localization = summary["spatial_localization"]
    return f"""# Stage15-C — Direct binary seismic attribute to label9 probability

The truth-blind runner calibrated 64 fixed quantile bins from 128 newly seeded full StructuralGeo cases, then mapped the frozen `cond_generation_0` seismic observation without opening held-out truth. This separate evaluator opened truth retrospectively only after the probability volume was frozen.

## Fixed attribute

Centered local trace energy is the mean of squared seismic amplitudes over the 17-sample (128 ms) Ricker-scale window. Fixed zero padding retains the same denominator at record edges. Each trace is linearly resampled to 64 voxel centers using local surface datum, 50 m cells and constant 2500 m/s background velocity. No truth velocity, lateral filtering, second attribute, neural network or parameter sweep is used.

## Held-out probability metrics

- Voxelwise AUPRC: {_fmt(summary['voxelwise_auprc'])}
- Truth-label9 mean / median P9: {_fmt(separation['truth_label9_mean_p9'])} / {_fmt(separation['truth_label9_median_p9'])}
- Background mean / median P9: {_fmt(separation['background_mean_p9'])} / {_fmt(separation['background_median_p9'])}
- P9>=0.8 positive voxels / TP / FP: {positive['positive_voxels']} / {positive['true_positive_voxels']} / {positive['false_positive_voxels']}
- Positive precision / recall / IoU: {_fmt(positive['precision'])} / {_fmt(positive['recall'])} / {_fmt(positive['iou'])}
- Truth label9 in positive / unknown / negative: {partition['counts']['positive']} / {partition['counts']['unknown']} / {partition['counts']['negative']}
- Truth fractions in positive / unknown / negative: {_fmt(partition['fractions']['positive'])} / {_fmt(partition['fractions']['unknown'])} / {_fmt(partition['fractions']['negative'])}

## Spatial localization

- Probability-weighted centroid to truth centroid distance: {_fmt(localization['probability_centroid_to_truth_centroid_distance'])} voxels
- Probability mass inside the truth-label9 bounding box: {_fmt(localization['probability_mass_fraction_inside_truth_bbox'])}
- Maximum P9: {_fmt(localization['maximum_p9'])}

## Frozen B2 comparison

- B2 occupancy AUPRC: {_fmt(comparison['b2_occupancy_auprc'])}
- Direct attribute AUPRC: {_fmt(comparison['direct_attribute_auprc'])}
- Direct minus B2: {_fmt(comparison['absolute_delta'])}
- Direct attribute is better by voxelwise AUPRC: **{comparison['direct_attribute_better_localization_by_auprc']}**

The 0.8/0.2 masks are diagnostics only; no threshold sweep or downstream Flow guidance was run.
"""


def main() -> None:
    args = parse_args()
    refuse_nonempty(args.output_dir)
    summary, hashes = evaluate(
        args.calibration_dir, args.heldout_dir, args.observation_dir, args.b3_summary
    )
    args.output_dir.mkdir(parents=True)
    manifest = base_manifest(
        "stage15_c_direct_attribute_truth_evaluation_run_v1", Path(__file__)
    )
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "REPORT.md").write_text(render_report(summary), encoding="utf-8")
    manifest.update(
        {
            "run_status": "completed",
            "truth_loaded_by_evaluator": True,
            "truth_loaded_by_probability_runner": False,
            "input_file_sha256_before_and_after": hashes,
            "inputs_unchanged": True,
            "summary": runtime.asset_record(args.output_dir / "summary.json"),
            "report": runtime.asset_record(args.output_dir / "REPORT.md"),
            "flow_guidance_performed": False,
            "threshold_sweep_performed": False,
        }
    )
    write_json(args.output_dir / "manifest.json", manifest)
    for path, expected in hashes.items():
        if runtime.file_sha256(Path(path)) != expected:
            raise RuntimeError(f"Stage15-C evaluation input changed: {path}")


if __name__ == "__main__":
    main()
