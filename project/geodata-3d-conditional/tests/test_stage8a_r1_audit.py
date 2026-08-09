from scripts.stage8.analyze_stage8a_r1 import (
    DEFAULT_STAGE7_CONFIG,
    DEFAULT_STAGE7_TRACE,
    DEFAULT_STAGE8_RUN,
    PRIMARY_CATEGORY,
    _distribution,
    add_retrospective_truth_annotation,
    audit_stage8_correct_arms,
    freeze_stage7_library,
    oracle_postmortem_local_basin,
)


def test_linear_distribution_convention():
    distribution = _distribution([0.0, 10.0, 20.0])
    assert distribution["p05"] == 1.0
    assert distribution["median"] == 10.0
    assert distribution["p95"] == 19.0


def test_frozen_stage8_histories_are_complete_and_have_no_improving_proposal():
    proposals, arm_rows, move_rows, body_rows, arms = audit_stage8_correct_arms(
        DEFAULT_STAGE8_RUN
    )
    assert len(proposals) == 4 * 961
    assert len(arm_rows) == 4
    assert len(move_rows) == 4 * 6
    assert len(body_rows) == 4 * 960
    assert all(detail["noninitial_proposal_count"] == 960 for detail in arms.values())
    assert all(detail["proposal_delta_rmse_lt_zero_count"] == 0 for detail in arms.values())
    assert all(detail["zero_edit_proposal_count"] == 80 for detail in arms.values())


def test_stage7_loss_freeze_precedes_truth_and_has_monotonic_paths():
    frozen, barrier = freeze_stage7_library(DEFAULT_STAGE7_TRACE)
    assert len(frozen) == 1 + 12 + 66
    assert all("truth" not in key for row in frozen for key in row)
    assert barrier["loss_selected_solution_hard_rmse"] == 0.0
    assert barrier["monotonic_single_birth_path_exists"] is True
    assert barrier["single_birth_energy_barrier_required"] is False

    annotated, posthoc = add_retrospective_truth_annotation(frozen, DEFAULT_STAGE7_CONFIG)
    assert posthoc["loss_selected_solution_matches_truth"] is True
    assert sum(row["retrospective_is_exact_truth_state"] for row in annotated) == 1


def test_oracle_postmortem_uses_only_recorded_local_trace():
    frozen, barrier = freeze_stage7_library(DEFAULT_STAGE7_TRACE)
    del frozen
    rows, audit = oracle_postmortem_local_basin(
        DEFAULT_STAGE7_TRACE,
        barrier["loss_selected_solution_candidate_ids"],
        barrier["empty_hard_rmse"],
    )
    assert PRIMARY_CATEGORY == "RANDOM_BIRTH_BASIN_MISS"
    assert len(rows) == 62
    assert audit["better_than_empty_count"] == 62
    assert audit["inference_success"] is False
    assert audit["affects_stage8_gate"] is False
    assert audit["used_for_proposal_selection"] is False
    assert audit["used_for_hyperparameter_tuning"] is False
