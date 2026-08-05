"""Hard-label and geometric evaluation for Phase-1 probability guidance."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence

import torch

from geophysics import mean_iou, voxel_accuracy
from geology_io_utils import finite_stats, target_component_stats
from inference_runtime import normalize_single_geology


PRIMARY_DELTA_FIELDS = (
    "global_voxel_accuracy",
    "global_mean_iou",
    "target_iou",
    "target_precision",
    "target_recall",
    "target_volume_error_fraction",
    "target_absolute_volume_error_fraction",
    "target_centroid_distance",
    "target_connected_components",
    "largest_component_fraction",
    "selected_roi_iou",
    "selected_roi_precision",
    "selected_roi_recall",
    "selected_absolute_volume_error_fraction",
    "selected_centroid_distance",
    "inside_roi_voxel_accuracy",
    "outside_roi_voxel_accuracy",
    "outside_roi_target_iou",
)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _centroid(mask: torch.Tensor) -> torch.Tensor | None:
    coordinates = torch.nonzero(mask.detach().cpu().bool(), as_tuple=False)
    if coordinates.numel() == 0:
        return None
    return coordinates.float().mean(dim=0)


def _centroid_distance(predicted: torch.Tensor, target: torch.Tensor) -> float:
    predicted_centroid = _centroid(predicted)
    target_centroid = _centroid(target)
    if predicted_centroid is None or target_centroid is None:
        return float("nan")
    return float(torch.linalg.vector_norm(predicted_centroid - target_centroid).item())


def sample_hard_metrics(
    prediction: torch.Tensor,
    truth_model: torch.Tensor,
    target_mask: torch.Tensor,
    roi_mask: torch.Tensor,
    condition_mask: torch.Tensor,
    target_label: int,
    sample_id: int,
    baseline_prediction: torch.Tensor | None = None,
) -> Dict[str, object]:
    """Return required global, target-label, ROI, condition, and paired metrics."""
    predicted = normalize_single_geology(prediction, "prediction").long()
    truth = normalize_single_geology(truth_model, "truth_model").long()
    selected = normalize_single_geology(target_mask, "target_mask").bool()
    roi = normalize_single_geology(roi_mask, "roi_mask").bool()
    condition = normalize_single_geology(condition_mask, "condition_mask").bool()
    if not (
        predicted.shape == truth.shape == selected.shape == roi.shape == condition.shape
    ):
        raise ValueError("prediction, truth, target, ROI, and condition shapes must match")

    predicted_target = predicted == int(target_label)
    truth_target = truth == int(target_label)
    intersection = int((predicted_target & truth_target).sum().item())
    union = int((predicted_target | truth_target).sum().item())
    predicted_count = int(predicted_target.sum().item())
    truth_count = int(truth_target.sum().item())

    roi_predicted = predicted_target & roi
    roi_intersection = int((roi_predicted & selected).sum().item())
    roi_union = int((roi_predicted | selected).sum().item())
    roi_predicted_count = int(roi_predicted.sum().item())
    selected_count = int(selected.sum().item())
    target_stats = target_component_stats(predicted_target)
    selected_stats = target_component_stats(roi_predicted)
    condition_violations = int(((predicted != truth) & condition).sum().item())
    outside_roi = ~roi
    valid_inside = roi & (truth != -1)
    valid_outside = outside_roi & (truth != -1)
    outside_predicted_target = predicted_target & outside_roi
    outside_truth_target = truth_target & outside_roi
    outside_intersection = int(
        (outside_predicted_target & outside_truth_target).sum().item()
    )
    outside_union = int(
        (outside_predicted_target | outside_truth_target).sum().item()
    )
    outside_predicted_count = int(outside_predicted_target.sum().item())
    outside_truth_count = int(outside_truth_target.sum().item())

    signed_volume_error = predicted_count - truth_count
    selected_volume_error = roi_predicted_count - selected_count
    row: Dict[str, object] = {
        "sample_id": int(sample_id),
        "global_voxel_accuracy": float(
            voxel_accuracy(predicted, truth, ignore_label=-1)[0].item()
        ),
        "global_mean_iou": float(mean_iou(predicted, truth, ignore_label=-1)[0].item()),
        "target_iou": _safe_ratio(intersection, union),
        "target_precision": _safe_ratio(intersection, predicted_count),
        "target_recall": _safe_ratio(intersection, truth_count),
        "target_volume": truth_count,
        "predicted_target_volume": predicted_count,
        "target_volume_error": signed_volume_error,
        "target_volume_error_fraction": _safe_ratio(signed_volume_error, truth_count),
        "target_absolute_volume_error": abs(signed_volume_error),
        "target_absolute_volume_error_fraction": _safe_ratio(
            abs(signed_volume_error),
            truth_count,
        ),
        "target_centroid_distance": _centroid_distance(
            predicted_target[0, 0], truth_target[0, 0]
        ),
        "target_connected_components": target_stats["target_connected_components"],
        "largest_component_fraction": target_stats["largest_component_fraction"],
        "selected_target_volume": selected_count,
        "predicted_target_volume_inside_roi": roi_predicted_count,
        "selected_volume_error": selected_volume_error,
        "selected_volume_error_fraction": _safe_ratio(
            selected_volume_error,
            selected_count,
        ),
        "selected_absolute_volume_error": abs(selected_volume_error),
        "selected_absolute_volume_error_fraction": _safe_ratio(
            abs(selected_volume_error),
            selected_count,
        ),
        "selected_roi_iou": _safe_ratio(roi_intersection, roi_union),
        "selected_roi_precision": _safe_ratio(roi_intersection, roi_predicted_count),
        "selected_roi_recall": _safe_ratio(roi_intersection, selected_count),
        "selected_centroid_distance": _centroid_distance(
            roi_predicted[0, 0], selected[0, 0]
        ),
        "selected_roi_connected_components": selected_stats[
            "target_connected_components"
        ],
        "selected_roi_largest_component_fraction": selected_stats[
            "largest_component_fraction"
        ],
        "inside_roi_voxel_accuracy": _safe_ratio(
            int(((predicted == truth) & valid_inside).sum().item()),
            int(valid_inside.sum().item()),
        ),
        "outside_roi_voxel_accuracy": _safe_ratio(
            int(((predicted == truth) & valid_outside).sum().item()),
            int(valid_outside.sum().item()),
        ),
        "outside_roi_target_iou": _safe_ratio(outside_intersection, outside_union),
        "outside_roi_target_precision": _safe_ratio(
            outside_intersection,
            outside_predicted_count,
        ),
        "outside_roi_target_recall": _safe_ratio(
            outside_intersection,
            outside_truth_count,
        ),
        "condition_violation_count": condition_violations,
    }

    if baseline_prediction is not None:
        baseline = normalize_single_geology(
            baseline_prediction,
            "baseline_prediction",
        ).long()
        if baseline.shape != predicted.shape:
            raise ValueError("baseline prediction shape must match guided prediction")
        changed = predicted != baseline
        row.update(
            {
                "paired_hard_change_count": int(changed.sum().item()),
                "paired_hard_change_fraction": float(changed.float().mean().item()),
                "paired_hard_change_inside_roi": int((changed & roi).sum().item()),
                "paired_hard_change_outside_roi": int((changed & ~roi).sum().item()),
                "paired_target_to_other_count": int(
                    ((baseline == int(target_label)) & (predicted != int(target_label)))
                    .sum()
                    .item()
                ),
                "paired_other_to_target_count": int(
                    ((baseline != int(target_label)) & (predicted == int(target_label)))
                    .sum()
                    .item()
                ),
            }
        )
    return row


def paired_metric_deltas(
    baseline: Mapping[str, object],
    guided: Mapping[str, object],
) -> Dict[str, object]:
    """Return guided-minus-baseline values for the primary hard metrics."""
    if int(baseline["sample_id"]) != int(guided["sample_id"]):
        raise ValueError("paired metric rows must have the same sample_id")
    deltas: Dict[str, object] = {"sample_id": int(guided["sample_id"])}
    for field in PRIMARY_DELTA_FIELDS:
        baseline_value = float(baseline[field])
        guided_value = float(guided[field])
        deltas[f"delta_{field}"] = guided_value - baseline_value
    return deltas


def class_transition_records(
    baseline_prediction: torch.Tensor,
    guided_prediction: torch.Tensor,
    sample_id: int,
) -> list[Dict[str, object]]:
    """Return a sparse hard-label transition matrix for one paired sample."""
    baseline = normalize_single_geology(
        baseline_prediction,
        "baseline_prediction",
    ).long()
    guided = normalize_single_geology(
        guided_prediction,
        "guided_prediction",
    ).long()
    if baseline.shape != guided.shape:
        raise ValueError("baseline and guided prediction shapes must match")
    pairs = torch.stack((baseline.flatten(), guided.flatten()), dim=1)
    unique, counts = torch.unique(pairs, dim=0, return_counts=True)
    rows: list[Dict[str, object]] = []
    for pair, count in zip(unique, counts):
        rows.append(
            {
                "sample_id": int(sample_id),
                "from_label": int(pair[0].item()),
                "to_label": int(pair[1].item()),
                "voxel_count": int(count.item()),
                "is_changed": bool(pair[0].item() != pair[1].item()),
            }
        )
    return rows


def ensemble_diversity_summary(
    realizations: torch.Tensor,
    target_mask: torch.Tensor,
    roi_mask: torch.Tensor,
    target_label: int,
    sample_hashes: Sequence[str] | None = None,
) -> Dict[str, object]:
    """Summarize ensemble disagreement, target coverage, and uniqueness."""
    if realizations.ndim == 4:
        samples = realizations.unsqueeze(1).long()
    elif realizations.ndim == 5 and realizations.shape[1] == 1:
        samples = realizations.long()
    else:
        raise ValueError("realizations must have shape [B,X,Y,Z] or [B,1,X,Y,Z]")
    selected = normalize_single_geology(target_mask, "target_mask").bool()
    roi = normalize_single_geology(roi_mask, "roi_mask").bool()
    if samples.shape[2:] != selected.shape[2:] or selected.shape != roi.shape:
        raise ValueError("ensemble, target, and ROI spatial shapes must match")

    target_occurrence = (samples == int(target_label)).float()
    probability = target_occurrence.mean(dim=0, keepdim=True)
    selected_count = selected.sum().clamp_min(1)
    outside_roi = ~roi
    outside_count = outside_roi.sum().clamp_min(1)
    selected_probability = float(
        (probability * selected).sum().div(selected_count).item()
    )
    selected_coverage = float(
        ((probability > 0) & selected).sum().float().div(selected_count).item()
    )
    outside_probability = float(
        (probability * outside_roi).sum().div(outside_count).item()
    )
    variance = target_occurrence.var(dim=0, unbiased=False, keepdim=True)

    pairwise_all: list[float] = []
    pairwise_roi: list[float] = []
    pairwise_outside: list[float] = []
    roi_count = roi.sum().clamp_min(1)
    for first in range(samples.shape[0]):
        for second in range(first + 1, samples.shape[0]):
            changed = samples[first : first + 1] != samples[second : second + 1]
            pairwise_all.append(float(changed.float().mean().item()))
            pairwise_roi.append(
                float((changed & roi).sum().float().div(roi_count).item())
            )
            pairwise_outside.append(
                float((changed & outside_roi).sum().float().div(outside_count).item())
            )

    hashes = list(sample_hashes or [])
    return {
        "n_samples": int(samples.shape[0]),
        "unique_decoded_samples": len(set(hashes)) if hashes else None,
        "mean_pairwise_hard_disagreement": (
            finite_stats(pairwise_all)["mean"] if pairwise_all else None
        ),
        "mean_pairwise_hard_disagreement_inside_roi": (
            finite_stats(pairwise_roi)["mean"] if pairwise_roi else None
        ),
        "mean_pairwise_hard_disagreement_outside_roi": (
            finite_stats(pairwise_outside)["mean"] if pairwise_outside else None
        ),
        "selected_target_mean_ensemble_probability": selected_probability,
        "selected_target_coverage_any_sample": selected_coverage,
        "outside_roi_target_mean_probability": outside_probability,
        "target_probability_variance_mean": float(variance.mean().item()),
    }


def summarize_rows(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    """Summarize every finite numeric metric in a list of per-sample rows."""
    numeric_fields: set[str] = set()
    for row in rows:
        for field, value in row.items():
            if field != "sample_id" and isinstance(value, (int, float)):
                numeric_fields.add(field)
    metrics: Dict[str, object] = {}
    for field in sorted(numeric_fields):
        values = [float(row[field]) for row in rows if field in row]
        finite = [value for value in values if math.isfinite(value)]
        metrics[field] = finite_stats(finite)
    return {"n_samples": len(rows), "metrics": metrics}
