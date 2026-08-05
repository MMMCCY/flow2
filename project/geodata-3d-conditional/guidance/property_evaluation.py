"""Hard-label and hard-property evaluation for Phase-2 guidance."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence

import torch

from geophysics import mean_iou
from geology_io_utils import connected_components_3d
from guidance.probability_evaluation import sample_hard_metrics
from guidance.property_volume import (
    hard_labels_to_properties,
    matched_multiscale_property_loss,
)
from inference_runtime import normalize_single_geology


PROPERTY_PRIMARY_FIELDS = (
    "global_voxel_accuracy",
    "global_mean_iou",
    "truth_present_mean_iou",
    "hard_property_loss",
    "hard_property_mae",
    "target_iou",
    "target_precision",
    "target_recall",
    "target_absolute_volume_error_fraction",
    "target_centroid_distance",
    "target_connected_components",
    "largest_component_fraction",
    "target_components_ge_20",
    "target_components_ge_100",
    "target_tiny_component_mass_fraction_le_5",
    "target_top4_component_mass_fraction",
    "inside_roi_voxel_accuracy",
    "outside_roi_voxel_accuracy",
)

COMPONENT_SIZE_THRESHOLDS = (5, 10, 20, 100)
COMPONENT_RANKS_RECORDED = 8


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")


def truth_present_mean_iou(
    prediction: torch.Tensor,
    truth_model: torch.Tensor,
) -> float:
    """Return mIoU over a fixed set of truth-present non-air classes.

    The historical ``global_mean_iou`` follows the union of classes present in
    each prediction and the truth. That is useful for penalizing hallucinated
    classes, but its denominator can change between a strict pair when only a
    few voxels of an otherwise absent class appear. This companion metric keeps
    the paired denominator fixed while absent-class hallucinations remain
    visible in the per-class table and historical global metric.
    """
    predicted = normalize_single_geology(prediction, "prediction").long()
    truth = normalize_single_geology(truth_model, "truth_model").long()
    if predicted.shape != truth.shape:
        raise ValueError("prediction and truth must have matching shapes")
    class_ids = sorted(
        int(value.item())
        for value in torch.unique(truth)
        if int(value.item()) != -1
    )
    return float(
        mean_iou(
            predicted,
            truth,
            class_ids=class_ids,
            ignore_label=-1,
        )[0].item()
    )


def size_stratified_component_metrics(mask: torch.Tensor) -> Dict[str, object]:
    """Return six-connected size diagnostics for one boolean 3-D mask."""
    value = mask.detach().cpu().bool()
    if value.ndim == 5:
        value = value[0, 0]
    elif value.ndim == 4:
        value = value[0]
    if value.ndim != 3:
        raise ValueError("component mask must resolve to [X, Y, Z]")
    sizes = sorted(
        (
            int(component["voxel_count"])
            for component in connected_components_3d(value)
        ),
        reverse=True,
    )
    total = int(value.sum().item())
    tiny_mass = sum(size for size in sizes if size <= 5)
    metrics: Dict[str, object] = {
        "target_tiny_component_mass_le_5": tiny_mass,
        "target_tiny_component_mass_fraction_le_5": _safe_ratio(
            tiny_mass,
            total,
        ),
        "target_top4_component_mass_fraction": _safe_ratio(
            sum(sizes[:4]),
            total,
        ),
        "target_top8_component_mass_fraction": _safe_ratio(
            sum(sizes[:8]),
            total,
        ),
    }
    for threshold in COMPONENT_SIZE_THRESHOLDS:
        metrics[f"target_components_ge_{threshold}"] = sum(
            size >= threshold for size in sizes
        )
    for rank in range(1, COMPONENT_RANKS_RECORDED + 1):
        metrics[f"target_component_{rank}_voxels"] = (
            sizes[rank - 1] if len(sizes) >= rank else 0
        )
    return metrics


def truth_component_recovery_rows(
    prediction: torch.Tensor,
    truth_model: torch.Tensor,
    target_label: int,
    sample_id: int,
) -> list[Dict[str, object]]:
    """Measure recovery of every truth target component independently."""
    predicted = normalize_single_geology(prediction, "prediction").long()
    truth = normalize_single_geology(truth_model, "truth_model").long()
    if predicted.shape != truth.shape:
        raise ValueError("prediction and truth must have matching shapes")
    predicted_target = (predicted == int(target_label))[0, 0].cpu()
    truth_target = (truth == int(target_label))[0, 0].cpu()
    components = sorted(
        connected_components_3d(truth_target),
        key=lambda component: int(component["voxel_count"]),
        reverse=True,
    )
    rows: list[Dict[str, object]] = []
    for rank, component in enumerate(components, start=1):
        coordinates = component["coords"].long()
        recovered = int(predicted_target[tuple(coordinates.T)].sum().item())
        truth_voxels = int(component["voxel_count"])
        rows.append(
            {
                "sample_id": int(sample_id),
                "truth_component_rank": rank,
                "truth_component_voxels": truth_voxels,
                "recovered_voxels": recovered,
                "recall": _safe_ratio(recovered, truth_voxels),
            }
        )
    return rows


def paired_truth_component_recovery_deltas(
    baseline_rows: Sequence[Mapping[str, object]],
    guided_rows: Sequence[Mapping[str, object]],
) -> list[Dict[str, object]]:
    """Return strict paired deltas for truth-component recovery rows."""
    baseline_index = {
        (int(row["sample_id"]), int(row["truth_component_rank"])): row
        for row in baseline_rows
    }
    deltas: list[Dict[str, object]] = []
    for guided in guided_rows:
        key = (
            int(guided["sample_id"]),
            int(guided["truth_component_rank"]),
        )
        if key not in baseline_index:
            raise ValueError(f"missing baseline truth-component row: {key}")
        baseline = baseline_index[key]
        if int(baseline["truth_component_voxels"]) != int(
            guided["truth_component_voxels"]
        ):
            raise ValueError(f"truth-component size differs within pair: {key}")
        deltas.append(
            {
                "sample_id": key[0],
                "truth_component_rank": key[1],
                "truth_component_voxels": int(
                    guided["truth_component_voxels"]
                ),
                "baseline_recovered_voxels": int(
                    baseline["recovered_voxels"]
                ),
                "guided_recovered_voxels": int(guided["recovered_voxels"]),
                "delta_recovered_voxels": int(guided["recovered_voxels"])
                - int(baseline["recovered_voxels"]),
                "baseline_recall": float(baseline["recall"]),
                "guided_recall": float(guided["recall"]),
                "delta_recall": float(guided["recall"])
                - float(baseline["recall"]),
            }
        )
    return deltas


def per_class_hard_metrics(
    prediction: torch.Tensor,
    truth_model: torch.Tensor,
    sample_id: int,
    class_ids: Sequence[int] | None = None,
) -> list[Dict[str, object]]:
    """Return one row per requested raw class, excluding air by default."""
    predicted = normalize_single_geology(prediction, "prediction").long()
    truth = normalize_single_geology(truth_model, "truth_model").long()
    if predicted.shape != truth.shape:
        raise ValueError("prediction and truth must have matching shapes")
    if class_ids is None:
        class_ids = sorted(
            int(value.item())
            for value in torch.unique(truth)
            if int(value.item()) != -1
        )
    rows: list[Dict[str, object]] = []
    for class_id in class_ids:
        truth_mask = truth == int(class_id)
        predicted_mask = predicted == int(class_id)
        truth_count = int(truth_mask.sum().item())
        predicted_count = int(predicted_mask.sum().item())
        intersection = int((truth_mask & predicted_mask).sum().item())
        union = int((truth_mask | predicted_mask).sum().item())
        signed_error = predicted_count - truth_count
        rows.append(
            {
                "sample_id": int(sample_id),
                "class_id": int(class_id),
                "truth_present": truth_count > 0,
                "truth_volume": truth_count,
                "predicted_volume": predicted_count,
                "volume_error": signed_error,
                "absolute_volume_error_fraction": _safe_ratio(
                    abs(signed_error),
                    truth_count,
                ),
                "iou": _safe_ratio(intersection, union),
                "precision": _safe_ratio(intersection, predicted_count),
                "recall": _safe_ratio(intersection, truth_count),
            }
        )
    return rows


def sample_property_hard_metrics(
    prediction: torch.Tensor,
    truth_model: torch.Tensor,
    condition_mask: torch.Tensor,
    target_mask: torch.Tensor,
    target_roi_mask: torch.Tensor,
    target_label: int,
    property_table: torch.Tensor,
    property_confidence: torch.Tensor,
    property_sigmas: Sequence[float],
    property_scale_weights: Sequence[float],
    property_channel_weights: torch.Tensor,
    sample_id: int,
    baseline_prediction: torch.Tensor | None = None,
) -> Dict[str, object]:
    """Return full geology, label-9 geometry, and hard-property diagnostics."""
    predicted = normalize_single_geology(prediction, "prediction").long()
    truth = normalize_single_geology(truth_model, "truth_model").long()
    row = sample_hard_metrics(
        prediction=predicted,
        truth_model=truth,
        target_mask=target_mask,
        roi_mask=target_roi_mask,
        condition_mask=condition_mask,
        target_label=target_label,
        sample_id=sample_id,
        baseline_prediction=baseline_prediction,
    )
    row["truth_present_mean_iou"] = truth_present_mean_iou(predicted, truth)
    row.update(size_stratified_component_metrics(predicted == int(target_label)))
    predicted_properties = hard_labels_to_properties(predicted, property_table)
    target_properties = hard_labels_to_properties(truth, property_table)
    hard_loss, hard_diagnostics = matched_multiscale_property_loss(
        predicted_properties,
        target_properties,
        property_confidence,
        sigmas=property_sigmas,
        scale_weights=property_scale_weights,
        channel_weights=property_channel_weights,
    )
    row["hard_property_loss"] = float(hard_loss.detach().cpu())
    row["hard_property_mae"] = float(
        hard_diagnostics["property_mae_mean"].detach().cpu()
    )
    row["property_confidence_fraction"] = float(
        hard_diagnostics["confidence_fraction"].detach().cpu()
    )
    if baseline_prediction is not None:
        baseline = normalize_single_geology(
            baseline_prediction,
            "baseline_prediction",
        ).long()
        confidence_mask = property_confidence.to(device=predicted.device).bool()
        if confidence_mask.shape[0] == 1 and predicted.shape[0] > 1:
            confidence_mask = confidence_mask.expand(
                predicted.shape[0], -1, -1, -1, -1
            )
        changed = predicted != baseline
        row["paired_hard_change_inside_property_confidence"] = int(
            (changed & confidence_mask).sum().item()
        )
        row["paired_hard_change_outside_property_confidence"] = int(
            (changed & ~confidence_mask).sum().item()
        )
    return row


def paired_property_metric_deltas(
    baseline: Mapping[str, object],
    guided: Mapping[str, object],
) -> Dict[str, object]:
    if int(baseline["sample_id"]) != int(guided["sample_id"]):
        raise ValueError("paired rows must have the same sample_id")
    row: Dict[str, object] = {"sample_id": int(guided["sample_id"])}
    for field in PROPERTY_PRIMARY_FIELDS:
        baseline_value = float(baseline[field])
        guided_value = float(guided[field])
        row[f"delta_{field}"] = guided_value - baseline_value
    return row


def paired_per_class_deltas(
    baseline_rows: Sequence[Mapping[str, object]],
    guided_rows: Sequence[Mapping[str, object]],
) -> list[Dict[str, object]]:
    baseline_index = {
        (int(row["sample_id"]), int(row["class_id"])): row
        for row in baseline_rows
    }
    deltas: list[Dict[str, object]] = []
    for guided in guided_rows:
        key = (int(guided["sample_id"]), int(guided["class_id"]))
        if key not in baseline_index:
            raise ValueError(f"missing baseline per-class row: {key}")
        baseline = baseline_index[key]
        row: Dict[str, object] = {
            "sample_id": key[0],
            "class_id": key[1],
            "truth_present": bool(guided["truth_present"]),
        }
        for field in (
            "iou",
            "precision",
            "recall",
            "predicted_volume",
            "absolute_volume_error_fraction",
        ):
            before = float(baseline[field])
            after = float(guided[field])
            row[f"baseline_{field}"] = before
            row[f"guided_{field}"] = after
            row[f"delta_{field}"] = after - before
        deltas.append(row)
    return deltas


def summarize_per_class_rows(
    rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    """Summarize finite per-class metrics without treating absent classes as zero."""
    class_ids = sorted({int(row["class_id"]) for row in rows})
    summary: Dict[str, object] = {"classes": {}}
    for class_id in class_ids:
        class_rows = [row for row in rows if int(row["class_id"]) == class_id]
        metrics: Dict[str, object] = {}
        for field in (
            "iou",
            "precision",
            "recall",
            "predicted_volume",
            "absolute_volume_error_fraction",
        ):
            values = [float(row[field]) for row in class_rows]
            finite = [value for value in values if math.isfinite(value)]
            metrics[field] = {
                "count": len(finite),
                "mean": sum(finite) / len(finite) if finite else None,
                "min": min(finite) if finite else None,
                "max": max(finite) if finite else None,
            }
        summary["classes"][str(class_id)] = {
            "truth_present": any(bool(row["truth_present"]) for row in class_rows),
            "metrics": metrics,
        }
    return summary
