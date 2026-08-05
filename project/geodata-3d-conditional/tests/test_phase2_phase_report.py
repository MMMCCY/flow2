from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.stage2.summarize_phase2a import sample_gate_audit


def _guided() -> dict[str, object]:
    return {
        "target_precision": 0.9,
        "target_recall": 0.5,
        "target_iou": 0.48,
        "predicted_target_volume": 5000,
        "target_volume": 8968,
        "target_tiny_component_mass_fraction_le_5": 0.05,
        "target_top8_component_mass_fraction": 0.88,
    }


def _delta() -> dict[str, object]:
    return {
        "delta_global_voxel_accuracy": 0.04,
        "delta_truth_present_mean_iou": 0.06,
        "delta_hard_property_loss": -0.9,
        "delta_target_iou": 0.45,
        "delta_target_precision": 0.8,
        "delta_target_recall": 0.45,
    }


def _class_rows(improved: int = 6) -> list[dict[str, object]]:
    return [
        {
            "class_id": class_id,
            "truth_present": True,
            "delta_iou": 0.01 if class_id < improved else -0.01,
        }
        for class_id in range(8)
    ]


def _component_rows() -> list[dict[str, object]]:
    return [
        {"truth_component_rank": rank, "guided_recall": recall}
        for rank, recall in enumerate((0.48, 0.54, 0.58, 0.37), start=1)
    ]


def test_phase2a_sample_gate_passes_recorded_operating_region() -> None:
    audit = sample_gate_audit(
        _guided(),
        _delta(),
        _class_rows(),
        _component_rows(),
        final_churn_fraction=0.009,
    )

    assert audit["passed"] is True
    assert audit["improved_truth_present_classes"] == 6
    assert audit["major_component_min_recall"] == 0.37


def test_phase2a_sample_gate_rejects_continuous_only_improvement() -> None:
    guided = _guided()
    guided["target_tiny_component_mass_fraction_le_5"] = 0.11
    audit = sample_gate_audit(
        guided,
        _delta(),
        _class_rows(improved=4),
        _component_rows(),
        final_churn_fraction=0.016,
    )

    assert audit["passed"] is False
    assert audit["checks"]["primary_directions"] is True
    assert audit["checks"]["majority_classes"] is False
    assert audit["checks"]["size_stratified_topology"] is False
    assert audit["checks"]["endpoint_churn"] is False
