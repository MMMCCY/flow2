import csv
import json
from pathlib import Path

from scripts.stage8.analyze_stage8a_v2_r2 import _correlation_summary, _tree_sha256


PROJECT = Path(__file__).resolve().parents[1]
REPORT = PROJECT / "experiments/stage8_structured_posterior/reports/stage8a_v2_r2"


def test_score_alignment_sign_convention():
    rows = [
        {"canonical_center_score": score, "hard_delta_rmse_vs_parent": -score}
        for score in range(20)
    ]
    summary = _correlation_summary(rows)
    assert summary["spearman_rho_score_vs_hard_delta"] == -1.0
    assert summary["kendall_tau_score_vs_hard_delta"] == -1.0
    assert summary["higher_score_associated_with_lower_hard_rmse"] is True


def test_r2_outputs_preserve_frozen_sources_and_zero_hard_proposal_forwards():
    summary = json.loads((REPORT / "stage8a_v2_r2_summary.json").read_text())
    assert summary["primary_classification"] == "FIRST_ORDER_TO_FINITE_HARD_NONLINEARITY"
    assert summary["new_hard_proposal_forward_calls"] == 0
    assert summary["new_proposals_evaluated"] == 0
    assert summary["gradient_recomputation"]["all_recomputed_hashes_match_v2"] is True
    assert summary["gradient_recomputation"]["differentiable_forward_calls"] == 725
    assert summary["gradient_recomputation"]["backward_calls"] == 725
    assert summary["stage8a_v2_rerun"] is False
    assert summary["stage8b_run"] is False
    assert summary["stage8a_v3_implemented"] is False

    roots = {
        "stage8a_v1": PROJECT / "experiments/stage8_structured_posterior/runs/stage8a_v1",
        "stage8a_r1": PROJECT / "experiments/stage8_structured_posterior/reports/stage8a_r1",
        "stage8a_v2": PROJECT / "experiments/stage8_structured_posterior/runs/stage8a_v2",
    }
    for name, root in roots.items():
        digest, count = _tree_sha256(root)
        assert digest == summary["frozen_tree_hashes"][name]["tree_sha256"]
        assert count == summary["frozen_tree_hashes"][name]["file_count"]


def test_r2_center_actual_mask_and_finite_step_evidence():
    with (REPORT / "retrospective_center_rank.csv").open() as stream:
        centers = list(csv.DictReader(stream))
    assert {row["candidate_id"]: int(row["sensitivity_rank"]) for row in centers} == {
        "candidate_04": 5,
        "candidate_06": 1,
    }

    with (REPORT / "actual_mask_directional_derivative.csv").open() as stream:
        actual = list(csv.DictReader(stream))
    correct = [row for row in actual if row["observation_kind"] == "correct"]
    assert len(actual) == 8770
    assert len(correct) == 2240
    assert sum(row["actual_mask_predicts_improvement"] == "True" for row in correct) == 2106
    assert sum(row["actual_hard_improves_parent"] == "True" for row in correct) == 0
    assert all(row["truth_used"] == "False" for row in actual)
    assert all(int(row["condition_violations"]) == 0 for row in actual)

    with (REPORT / "candidate_04_06_finite_step_comparison.csv").open() as stream:
        finite = list(csv.DictReader(stream))
    assert {row["candidate_id"] for row in finite} == {"candidate_04", "candidate_06"}
    assert all(row["first_order_predicts_improvement"] == "True" for row in finite)
    assert all(row["finite_hard_step_improves"] == "True" for row in finite)
    assert all(row["sign_agreement"] == "True" for row in finite)
