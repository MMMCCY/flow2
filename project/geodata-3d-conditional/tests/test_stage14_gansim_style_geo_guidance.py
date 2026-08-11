from __future__ import annotations

import torch

from scripts.stage14.evaluate_gansim_style_geo_guidance import (
    largest_component,
    paired_deltas,
    sample_metrics,
)


def test_largest_component_uses_six_connectivity() -> None:
    mask = torch.zeros(4, 4, 4, dtype=torch.bool)
    mask[0, 0, 0] = True
    mask[1, 1, 1] = True
    mask[1, 1, 2] = True
    selected = largest_component(mask)
    assert int(selected.sum()) == 2
    assert selected[1, 1, 1]
    assert selected[1, 1, 2]


def test_hidden_metrics_and_condition_audit_are_domain_explicit() -> None:
    truth = torch.zeros(1, 1, 3, 3, 3, dtype=torch.long)
    truth[..., 0, 0, 0] = 9
    truth[..., 0, 0, 1] = 9
    condition_mask = torch.zeros_like(truth, dtype=torch.bool)
    condition_mask[..., 0, 0, 0] = True
    condition_values = torch.full_like(truth, -1)
    condition_values[condition_mask] = truth[condition_mask]
    subsurface = torch.ones_like(truth, dtype=torch.bool)
    hidden = (truth == 9) & ~condition_mask
    prediction = truth.clone()
    largest = hidden.clone()

    metrics = sample_metrics(
        prediction=prediction,
        truth=truth,
        condition_values=condition_values,
        condition_mask=condition_mask,
        subsurface_mask=subsurface,
        hidden_label9_mask=hidden,
        largest_hidden_component=largest,
    )
    assert metrics["label9_iou"] == 1.0
    assert metrics["hidden_label9_iou"] == 1.0
    assert metrics["hidden_label9_recall"] == 1.0
    assert metrics["largest_hidden_component_recall"] == 1.0
    assert metrics["condition_violation_count"] == 0


def test_paired_delta_is_guided_minus_baseline() -> None:
    baseline = {"case_id": "c", "sample_id": 0, "source_seed": 7}
    guided = dict(baseline)
    for index, field in enumerate(
        (
            "label9_iou",
            "label9_precision",
            "label9_recall",
            "hidden_label9_iou",
            "hidden_label9_precision",
            "hidden_label9_recall",
            "largest_hidden_component_recall",
            "truth_present_miou",
            "global_accuracy",
        )
    ):
        baseline[field] = index / 20
        guided[field] = baseline[field] + 0.05
    baseline["condition_violation_count"] = 0
    guided["condition_violation_count"] = 0
    delta = paired_deltas(baseline, guided)
    assert abs(delta["delta_hidden_label9_iou"] - 0.05) < 1e-12
    assert delta["baseline_condition_violation_count"] == 0
    assert delta["guided_condition_violation_count"] == 0
