"""Evaluate target-label reconstruction metrics for saved geology ensembles."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import torch

from geology_io_utils import (
    find_sample_files,
    finite_stats,
    label_mask,
    load_sample_stack,
    load_volume,
    read_csv_rows,
    rows_by_sample_id,
    target_probability,
    target_component_stats,
    write_csv_rows,
    write_json,
)


TARGET_METRIC_FIELDS = [
    "sample_id",
    "path",
    "target_label",
    "target_iou",
    "target_precision",
    "target_recall",
    "target_f1",
    "target_volume",
    "predicted_target_volume",
    "target_volume_error",
    "target_volume_error_fraction",
    "target_centroid_distance",
    "target_connected_components",
    "largest_component_fraction",
    "ensemble_probability_overlap_inside_truth",
    "global_geo_misfit",
    "global_voxel_accuracy",
    "global_mean_iou",
]


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    value = numerator.float() / denominator.clamp_min(1).float()
    return torch.where(denominator > 0, value, torch.full_like(value, torch.nan))


def _centroid(mask: torch.Tensor) -> Optional[torch.Tensor]:
    coords = torch.nonzero(mask.detach().cpu().bool(), as_tuple=False)
    if coords.numel() == 0:
        return None
    return coords.float().mean(dim=0)


def _centroid_distance(predicted: torch.Tensor, target: torch.Tensor) -> float:
    predicted_centroid = _centroid(predicted)
    target_centroid = _centroid(target)
    if predicted_centroid is None or target_centroid is None:
        return float("nan")
    return float(torch.linalg.vector_norm(predicted_centroid - target_centroid).item())


def _to_float(row: Mapping[str, object], field: str) -> object:
    value = row.get(field, "")
    if value in ("", None):
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


def _threshold_metrics(probability: torch.Tensor, truth_mask: torch.Tensor, thresholds: Sequence[float]) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    truth = truth_mask.bool()
    for threshold in thresholds:
        predicted = probability >= float(threshold)
        intersection = (predicted & truth).sum().float()
        union = (predicted | truth).sum().float()
        predicted_count = predicted.sum().float()
        truth_count = truth.sum().float()
        key = f"{float(threshold):g}"
        metrics[key] = {
            "probability_iou_at_threshold": float((intersection / union.clamp_min(1)).item()) if union.item() > 0 else float("nan"),
            "probability_precision_at_threshold": float((intersection / predicted_count.clamp_min(1)).item()) if predicted_count.item() > 0 else float("nan"),
            "probability_recall_at_threshold": float((intersection / truth_count.clamp_min(1)).item()) if truth_count.item() > 0 else float("nan"),
            "predicted_voxels_at_threshold": float(predicted_count.item()),
        }
    return metrics


def save_probability_slices(probability: torch.Tensor, truth_mask: torch.Tensor, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prob = probability.detach().cpu().float()
    truth = truth_mask.detach().cpu().bool()
    z_values = [0, prob.shape[-1] // 3, (2 * prob.shape[-1]) // 3, prob.shape[-1] - 1]
    z_values = sorted(set(max(0, min(prob.shape[-1] - 1, int(value))) for value in z_values))
    figure, axes = plt.subplots(1, len(z_values), figsize=(4 * len(z_values), 4))
    if len(z_values) == 1:
        axes = [axes]
    for axis, z_index in zip(axes, z_values):
        image = axis.imshow(prob[:, :, z_index].T, origin="lower", cmap="magma", vmin=0, vmax=1)
        axis.contour(truth[:, :, z_index].float().T, levels=[0.5], colors="white", linewidths=0.7)
        axis.set_title(f"z={z_index}")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(image, ax=axes, shrink=0.75)
    figure.suptitle("Target-label ensemble probability slices")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def target_metric_records(
    realizations: torch.Tensor,
    truth_model: torch.Tensor,
    target_label: int,
    sample_records: Sequence[Mapping[str, object]],
    global_metric_rows: Optional[Sequence[Mapping[str, object]]] = None,
    thresholds: Sequence[float] = (0.05, 0.33, 0.62, 0.90),
) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    """Return per-sample and ensemble target-label metrics."""
    predicted_masks = label_mask(realizations, target_label)
    truth_mask = label_mask(truth_model, target_label)
    if truth_mask.shape[0] == 1 and predicted_masks.shape[0] > 1:
        truth_mask = truth_mask.expand(predicted_masks.shape[0], -1, -1, -1, -1)
    if truth_mask.shape != predicted_masks.shape:
        raise ValueError("truth_model and realizations must have matching spatial shape")

    global_by_id = rows_by_sample_id(global_metric_rows or [])
    probability = target_probability(realizations, target_label)
    probability_flat = probability.flatten()
    truth_single = truth_mask[:1].float()
    target_volume = truth_single.sum()
    ensemble_probability_overlap_inside_truth = float(
        (probability * truth_single).sum().div(target_volume.clamp_min(1)).item()
    )
    outside_count = (truth_single == 0).sum().clamp_min(1)
    probability_outside_mean = float(
        (probability * (truth_single == 0)).sum().div(outside_count).item()
    )
    entropy = -(probability_flat * torch.log(probability_flat.clamp_min(1e-8)) + (1 - probability_flat) * torch.log((1 - probability_flat).clamp_min(1e-8)))

    rows: List[Dict[str, object]] = []
    for index in range(predicted_masks.shape[0]):
        predicted = predicted_masks[index]
        target = truth_mask[index]
        intersection = (predicted & target).sum()
        union = (predicted | target).sum()
        predicted_count = predicted.sum()
        target_count = target.sum()
        iou = _safe_ratio(intersection, union)
        precision = _safe_ratio(intersection, predicted_count)
        recall = _safe_ratio(intersection, target_count)
        f1_denominator = precision + recall
        target_f1 = 2.0 * precision * recall / f1_denominator.clamp_min(1e-8)
        target_f1 = torch.where(
            torch.isfinite(precision) & torch.isfinite(recall) & (f1_denominator > 0),
            target_f1,
            torch.full_like(target_f1, torch.nan),
        )
        signed_volume_error = float((predicted_count - target_count).item())
        target_count_float = float(target_count.item())
        sample_id = int(sample_records[index]["sample_id"])
        global_row = global_by_id.get(sample_id, {})
        component_stats = target_component_stats(predicted)
        row = {
            "sample_id": sample_id,
            "path": sample_records[index]["path"],
            "target_label": int(target_label),
            "target_iou": float(iou.item()),
            "target_precision": float(precision.item()),
            "target_recall": float(recall.item()),
            "target_f1": float(target_f1.item()),
            "target_volume": target_count_float,
            "predicted_target_volume": float(predicted_count.item()),
            "target_volume_error": signed_volume_error,
            "target_volume_error_fraction": (
                signed_volume_error / target_count_float if target_count_float > 0 else float("nan")
            ),
            "target_centroid_distance": _centroid_distance(predicted[0], target[0]),
            "target_connected_components": component_stats["target_connected_components"],
            "largest_component_fraction": component_stats["largest_component_fraction"],
            "ensemble_probability_overlap_inside_truth": ensemble_probability_overlap_inside_truth,
            "global_geo_misfit": _to_float(global_row, "geo_misfit"),
            "global_voxel_accuracy": _to_float(global_row, "voxel_accuracy"),
            "global_mean_iou": _to_float(global_row, "mean_iou"),
        }
        rows.append(row)

    summary = {
        "target_label": int(target_label),
        "n_samples": len(rows),
        "target_volume": float(target_volume.item()),
        "ensemble_probability_overlap_inside_truth": ensemble_probability_overlap_inside_truth,
        "probability_outside_mean": probability_outside_mean,
        "probability_entropy_mean": float(entropy.mean().item()),
        "probability_threshold_metrics": _threshold_metrics(probability[0, 0], truth_single[0, 0].bool(), thresholds),
        "metrics": {
            field: finite_stats([float(row[field]) for row in rows if row[field] not in ("", None)])
            for field in (
                "target_iou",
                "target_precision",
                "target_recall",
                "target_f1",
                "target_volume_error_fraction",
                "target_centroid_distance",
                "largest_component_fraction",
            )
        },
        "description": (
            "Target-label metrics for saved categorical geology realizations. "
            "ensemble_probability_overlap_inside_truth is the mean ensemble "
            "target probability inside the truth target-label mask."
        ),
    }
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute target-label-specific dike/intrusion reconstruction metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--truth-model", type=Path, required=True)
    parser.add_argument("--target-label", type=int, required=True)
    parser.add_argument("--metrics-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-prefix", action="append", default=None)
    parser.add_argument("--threshold", type=float, action="append", default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_paths = find_sample_files(args.samples_dir, args.sample_prefix)
    realizations, sample_records = load_sample_stack(sample_paths, device=args.device)
    truth_model = load_volume(args.truth_model, device=args.device, single=True)
    global_rows = read_csv_rows(args.metrics_csv) if args.metrics_csv else None
    thresholds = args.threshold or [0.05, 0.33, 0.62, 0.90]
    rows, summary = target_metric_records(
        realizations=realizations,
        truth_model=truth_model,
        target_label=args.target_label,
        sample_records=sample_records,
        global_metric_rows=global_rows,
        thresholds=thresholds,
    )
    probability = target_probability(realizations, args.target_label)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(probability.cpu(), args.output_dir / "target_probability.pt")
    save_probability_slices(
        probability[0, 0],
        label_mask(truth_model, args.target_label)[0, 0],
        args.output_dir / "target_probability_slices.png",
    )
    summary["target_probability_path"] = str(args.output_dir / "target_probability.pt")
    summary["target_probability_slices"] = str(args.output_dir / "target_probability_slices.png")
    write_csv_rows(args.output_dir / "target_metrics.csv", rows, TARGET_METRIC_FIELDS)
    write_json(args.output_dir / "summary.json", summary)
    print(f"Saved target metrics: {args.output_dir / 'target_metrics.csv'}")
    print(f"Saved summary: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
