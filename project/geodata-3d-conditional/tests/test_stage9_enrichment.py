from __future__ import annotations

from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from guidance.prior_ensemble import (
    discrimination_checks,
    next_action,
    support_checks,
)
from scripts.stage9.audit_prior_truth import (
    _best_of_n,
    _enrichment_and_correlations,
)


def _metric_rows(count=20):
    rows = []
    for index in range(count):
        quality = float(index) / (count - 1)
        rows.append(
            {
                "candidate_id": f"candidate_{index:06d}",
                "global_accuracy": quality,
                "truth_present_mean_iou": quality,
                "label9_iou": quality,
                "label9_precision": quality,
                "label9_recall": quality,
                "major_component_min_recall": quality,
                "major_component_mean_recall": quality,
                "condition_violations": 0,
                "support_pass": quality >= 0.8,
            }
        )
    return rows


def _ranking(order):
    return [
        {
            "candidate_id": f"candidate_{index:06d}",
            "hard_seismic_rmse": float(rank),
            "rank": rank + 1,
        }
        for rank, index in enumerate(order)
    ]


def test_support_gate_requires_all_frozen_thresholds_simultaneously():
    thresholds = {
        "label9_iou_minimum": 0.30,
        "label9_precision_minimum": 0.75,
        "label9_recall_minimum": 0.30,
        "major_component_min_recall_minimum": 0.25,
        "major_component_mean_recall_minimum": 0.40,
        "condition_violations_maximum": 0,
    }
    row = {
        "label9_iou": 0.30,
        "label9_precision": 0.75,
        "label9_recall": 0.30,
        "major_component_min_recall": 0.25,
        "major_component_mean_recall": 0.40,
        "condition_violations": 0,
    }
    assert all(support_checks(row, thresholds).values())
    row["label9_precision"] = 0.749999
    assert not all(support_checks(row, thresholds).values())


def test_enrichment_correlations_and_discrimination_gate():
    rows = _metric_rows()
    descending_quality = list(reversed(range(20)))
    ascending_quality = list(range(20))
    rankings = {
        "correct": _ranking(descending_quality),
        "zero": _ranking(ascending_quality),
        "shuffled_xy": _ranking(ascending_quality),
        "wrong_case": _ranking(ascending_quality),
    }
    enrichment, correlations = _enrichment_and_correlations(
        rows, rankings, (0.10, 0.05, 0.01)
    )
    verdict = discrimination_checks(correlations, enrichment)
    assert verdict["passed"] is True
    assert all(verdict["correct_target_spearman_strictly_negative"].values())
    top5 = [
        row
        for row in enrichment
        if row["observation"] == "correct" and row["subset"] == "top_5pct"
    ]
    assert all(row["count"] == 1 for row in top5)
    assert all(row["enrichment"] > 0 for row in top5)


def test_best_of_n_is_prefix_oracle_only_and_support_counted():
    rows = _metric_rows(count=4)
    result = _best_of_n(rows, (1, 4, 16))
    assert [row["N"] for row in result] == [1, 4]
    assert result[0]["deployable_selector"] is False
    assert result[0]["support_pass"] is False
    assert result[1]["support_pass"] is True
    assert result[1]["support_passing_candidate_count"] == 1


def test_machine_next_action_truth_table():
    assert next_action(True, True) == "STAGE9B_POSTERIOR_WEIGHTING"
    assert next_action(False, True) == "STAGE9C_ADAPTIVE_PROPOSAL_FEASIBILITY"
    assert next_action(True, False) == "STOP_REDESIGN_LIKELIHOOD_OR_PETROPHYSICS"
    assert next_action(False, False) == "STOP_REASSESS_FROZEN_INFERENCE_ROUTE"
