#!/usr/bin/env python3
"""Retrospective truth-enabled evaluation for frozen Stage15 outputs only."""

from __future__ import annotations

import argparse
import itertools
import math
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime
from scripts.stage14.evaluate_gansim_style_geo_guidance import largest_component, sample_metrics
from scripts.stage15.common import base_manifest, normalize_volume, read_json, refuse_nonempty, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation-dir", type=Path, required=True)
    parser.add_argument("--inversion-dir", type=Path, required=True)
    parser.add_argument("--consensus-dir", type=Path, required=True)
    parser.add_argument("--flow-run-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _binary_metrics(predicted: torch.Tensor, actual: torch.Tensor, domain: torch.Tensor) -> dict[str, float]:
    predicted = predicted.bool() & domain
    actual = actual.bool() & domain
    intersection = int((predicted & actual).sum())
    predicted_count = int(predicted.sum())
    actual_count = int(actual.sum())
    union = int((predicted | actual).sum())
    return {
        "iou": intersection / union if union else float("nan"),
        "precision": intersection / predicted_count if predicted_count else float("nan"),
        "recall": intersection / actual_count if actual_count else float("nan"),
    }


def _centroid(mask: torch.Tensor) -> torch.Tensor | None:
    points = torch.nonzero(mask.bool(), as_tuple=False).float()
    return None if points.numel() == 0 else points.mean(dim=0)


def _centroid_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    a, b = _centroid(left), _centroid(right)
    return float("nan") if a is None or b is None else float(torch.linalg.vector_norm(a - b))


def _component_count(mask: torch.Tensor) -> int:
    _, count = ndimage.label(mask.detach().cpu().numpy(), structure=ndimage.generate_binary_structure(3, 1))
    return int(count)


def _major_component_recalls(predicted: torch.Tensor, actual: torch.Tensor, minimum_size: int = 20) -> list[float]:
    labels, count = ndimage.label(actual.detach().cpu().numpy(), structure=ndimage.generate_binary_structure(3, 1))
    predicted_np = predicted.detach().cpu().numpy().astype(bool)
    recalls: list[float] = []
    for component in range(1, count + 1):
        selected = labels == component
        size = int(selected.sum())
        if size >= minimum_size:
            recalls.append(float((predicted_np & selected).sum() / size))
    return recalls


def main() -> None:
    args = parse_args()
    refuse_nonempty(args.output_dir)
    manifest = base_manifest("stage15_retrospective_evaluation_v1", Path(__file__))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "manifest.json", manifest)
    try:
        observation_manifest = read_json(args.observation_dir / "manifest.json")
        inversion_manifest = read_json(args.inversion_dir / "manifest.json")
        consensus_manifest = read_json(args.consensus_dir / "consensus_manifest.json")
        if any(value.get("run_status") != "completed" for value in (observation_manifest, inversion_manifest, consensus_manifest)):
            raise ValueError("observation, inversion, and consensus must be frozen completed outputs")
        truth_record = observation_manifest["phase1_assets"]["truth_model"]
        truth_path = Path(str(truth_record["path"]))
        if runtime.file_sha256(truth_path) != truth_record["sha256"]:
            raise ValueError("authoritative truth hash changed")
        truth = normalize_volume(runtime.load_tensor(truth_path), "true_model").long()
        subsurface = normalize_volume(runtime.load_tensor(args.observation_dir / "subsurface_mask.pt"), "subsurface_mask").bool()
        condition_values = normalize_volume(runtime.load_tensor(args.observation_dir / "flow_condition_values.pt"), "flow_condition_values").long()
        condition_mask = normalize_volume(runtime.load_tensor(args.observation_dir / "flow_condition_mask.pt"), "flow_condition_mask").bool()
        binary_truth = (truth == 9) & subsurface
        positive = normalize_volume(runtime.load_tensor(args.consensus_dir / "positive_mask.pt"), "positive_mask").bool()
        negative = normalize_volume(runtime.load_tensor(args.consensus_dir / "negative_mask.pt"), "negative_mask").bool()
        confidence = positive | negative
        frequency = normalize_volume(runtime.load_tensor(args.consensus_dir / "occupancy_frequency.pt"), "occupancy_frequency", torch.float32)
        positive_metrics = _binary_metrics(positive, binary_truth, subsurface)
        predicted_confidence = positive
        correct_confidence = ((predicted_confidence == binary_truth) & confidence).sum()
        positive_errors = int((positive & ~binary_truth).sum())
        negative_errors = int((negative & binary_truth).sum())
        q_levels = torch.tensor([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
        frequency_values = frequency[subsurface]
        quantiles = torch.quantile(frequency_values, q_levels)
        histogram = torch.histogram(frequency_values, bins=10, range=(0.0, 1.0))
        consensus_metrics = {
            "positive_core": {
                **positive_metrics,
                "voxel_count": int(positive.sum()),
                "connected_components_6": _component_count(positive[0, 0]),
                "centroid_distance": _centroid_distance(positive[0, 0], binary_truth[0, 0]),
            },
            "high_confidence": {
                "accuracy": float(correct_confidence / confidence.sum()) if int(confidence.sum()) else float("nan"),
                "positive_error_count": positive_errors,
                "negative_error_count": negative_errors,
                "confidence_coverage": float(confidence.sum() / subsurface.sum()),
            },
            "frequency_quantiles": {str(float(q)): float(value) for q, value in zip(q_levels, quantiles)},
            "frequency_histogram_counts_10_equal_bins": histogram.hist.tolist(),
            "frequency_histogram_edges": histogram.bin_edges.tolist(),
            "thresholds_retained_without_retuning": consensus_manifest["thresholds"],
        }

        members = [normalize_volume(runtime.load_tensor(path), path.name).bool() for path in sorted(args.inversion_dir.glob("member_*/hard_binary.pt"))]
        disagreements = [float((a != b).float().mean()) for a, b in itertools.combinations(members, 2)]
        inversion_metrics = {
            "member_count": len(members),
            "unique_hard_model_count": len({runtime.file_sha256(path) for path in sorted(args.inversion_dir.glob("member_*/hard_binary.pt"))}),
            "mean_pairwise_disagreement": statistics.fmean(disagreements) if disagreements else 0.0,
            "condition_violations": [int(((member.float() != ((condition_values == 9).float())) & (condition_mask & subsurface)).sum()) for member in members],
        }

        flow_rows: list[dict[str, object]] = []
        if args.flow_run_dir is not None:
            flow_manifest = read_json(args.flow_run_dir / "run_manifest.json")
            if flow_manifest.get("run_status") != "completed" or flow_manifest.get("truth_loaded_by_runner") is not False:
                raise ValueError("Flow run must be completed behind the truth firewall")
            hidden = binary_truth & ~condition_mask
            largest_hidden = largest_component(hidden[0, 0])
            for sample_id in range(int(flow_manifest["n_samples"])):
                for arm in ("FLOW_ONLY", "FLOW_PLUS_BINARY_CONSENSUS"):
                    prediction = normalize_volume(runtime.load_tensor(args.flow_run_dir / arm / f"sample_{sample_id}.pt"), f"{arm}/{sample_id}").long()
                    row = sample_metrics(
                        prediction=prediction,
                        truth=truth,
                        condition_values=condition_values,
                        condition_mask=condition_mask,
                        subsurface_mask=subsurface,
                        hidden_label9_mask=hidden,
                        largest_hidden_component=largest_hidden,
                    )
                    predicted_target = (prediction == 9)[0, 0] & subsurface[0, 0]
                    major_recalls = _major_component_recalls(
                        predicted_target, binary_truth[0, 0]
                    )
                    row.update(
                        {
                            "arm": arm,
                            "sample_id": sample_id,
                            "label9_centroid_distance": _centroid_distance(predicted_target, binary_truth[0, 0]),
                            "label9_component_count_6": _component_count(predicted_target),
                            "false_bridge_voxels": int((predicted_target & ~binary_truth[0, 0]).sum()),
                            "component_merge_indicator": int(_component_count(predicted_target) < _component_count(binary_truth[0, 0])),
                            "major_component_count_min20": len(major_recalls),
                            "major_component_mean_recall": statistics.fmean(major_recalls) if major_recalls else float("nan"),
                            "major_component_min_recall": min(major_recalls) if major_recalls else float("nan"),
                        }
                    )
                    if int(row["condition_violation_count"]) != 0:
                        raise RuntimeError("retrospective Flow condition violation")
                    flow_rows.append(row)
            write_csv(args.output_dir / "flow_metrics.csv", flow_rows)

        report = {
            "consensus": consensus_metrics,
            "inversion_ensemble": inversion_metrics,
            "flow_samples": flow_rows,
            "truth_used_only_by_retrospective_evaluator": True,
            "thresholds_changed_after_truth": False,
        }
        write_json(args.output_dir / "metrics.json", report)
        manifest.update(
            {
                "run_status": "completed",
                "truth_asset": runtime.asset_record(truth_path),
                "observation_manifest": runtime.asset_record(args.observation_dir / "manifest.json"),
                "inversion_manifest": runtime.asset_record(args.inversion_dir / "manifest.json"),
                "consensus_manifest": runtime.asset_record(args.consensus_dir / "consensus_manifest.json"),
                "flow_manifest": None if args.flow_run_dir is None else runtime.asset_record(args.flow_run_dir / "run_manifest.json"),
                "metrics": runtime.asset_record(args.output_dir / "metrics.json"),
                "truth_loaded_by_retrospective_evaluator": True,
            }
        )
        write_json(args.output_dir / "manifest.json", manifest)
    except Exception as exc:
        manifest.update({"run_status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        write_json(args.output_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
