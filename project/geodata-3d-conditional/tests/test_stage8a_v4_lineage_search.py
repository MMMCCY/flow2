import inspect

import torch

from guidance.structured_lineage_search import (
    ALLOCATION_RULE,
    lineage_preserving_structured_search,
)
from guidance.structured_posterior import (
    HardConditionProjector,
    StructuredBodySpec,
    StructuredState,
    materialize_state,
)
from guidance.structured_trust_region import scaled_body


class _FrozenTargetKernel:
    seed = 8202601

    def move_for(self, parent, *, proposal_index):
        return "birth"

    def propose(
        self, parent, *, generation, proposal_index, birth_center=None
    ):
        center = tuple(float(value) for value in birth_center)
        body = StructuredBodySpec(
            body_id=f"body_{generation:03d}_{proposal_index:03d}",
            center_x=center[0], center_y=center[1], center_z=center[2],
            size_x=6.0, size_y=6.0, size_z=6.0,
            orientation_deg=37.0, shape="ellipsoid", material_label=9,
        )
        return StructuredState(
            bodies=(*parent.bodies, body),
            state_id=f"g{generation:03d}_p{proposal_index:03d}_{parent.state_id}",
            parent_id=parent.state_id,
            proposal_move="birth",
        )


class _FrozenRanker:
    def __init__(self):
        self.results = []

    def rank(
        self, *, state, current_labels, current_predicted_seismic,
        observed_seismic, generation,
    ):
        return {
            "ranking_id": f"g{generation}_{state.state_id}",
            "ranked_centers": [
                {"center_xyz": [4, 4, 4], "score": 1.0 - index * 1e-6}
                for index in range(16)
            ],
        }

    def record_birth_result(self, ranking_id, record):
        self.results.append((ranking_id, dict(record)))

    def summary(self):
        return {
            "rankings": [],
            "birth_proposal_count": len(self.results),
            "truth_used": False,
        }


def _fixture():
    base = torch.zeros((1, 1, 8, 8, 8), dtype=torch.long)
    condition = torch.zeros_like(base, dtype=torch.bool)
    condition[..., 0, 0, 0] = True
    projector = HardConditionProjector(
        condition_values=base,
        condition_mask=condition,
        edit_mask=~condition,
    )
    full = StructuredBodySpec(
        body_id="target", center_x=4.0, center_y=4.0, center_z=4.0,
        size_x=6.0, size_y=6.0, size_z=6.0,
        orientation_deg=37.0, shape="ellipsoid", material_label=9,
    )

    def response(labels):
        return (labels == 9).sum().reshape(1).float()

    half, _ = materialize_state(
        StructuredState((scaled_body(full, 0.5),), "half", None, "fixture"),
        base_labels=base, projector=projector,
    )
    return base, projector, response, response(half)


def _run():
    base, projector, response, observation = _fixture()
    return lineage_preserving_structured_search(
        base_labels=base,
        projector=projector,
        observation=observation,
        hard_response=response,
        proposal_kernel=_FrozenTargetKernel(),
        beam_size=2,
        generations=1,
        proposals_per_parent=3,
        birth_center_ranker=_FrozenRanker(),
        scale_ladder=(0.25, 0.5, 0.75, 1.0),
    )


def test_v4_lineage_api_has_no_truth_and_allocation_rule_is_explicit():
    assert "truth" not in inspect.signature(
        lineage_preserving_structured_search
    ).parameters
    assert "immediately_following_slots" in ALLOCATION_RULE
    assert "skip_displaced_scheduled_proposals_without_replacement" in ALLOCATION_RULE


def test_v4_continues_before_beam_and_reuses_exact_slots():
    result = _run()
    audit = result["lineage_continuation"]
    assert result["forward_call_count"] == 7
    assert result["fixed_forward_call_budget"] == 7
    assert audit["slot_allocation"] == {
        "initial_empty": 1,
        "new_center": 2,
        "growth": 4,
        "nonbirth": 0,
        "total_hard_forward_calls": 7,
        "reallocated_existing_slots": 4,
    }
    assert audit["locally_improving_scale_0_25_seeds"] == 2
    assert audit["transition_attempts"] == {
        "0.25_to_0.50": 2,
        "0.50_to_0.75": 2,
        "0.75_to_1.00": 0,
    }
    assert audit["transition_successes"] == {
        "0.25_to_0.50": 2,
        "0.50_to_0.75": 0,
        "0.75_to_1.00": 0,
    }
    assert audit["displaced_scheduled_move_counts"] == {"birth": 4}
    assert audit["branch_final_global_beam_survival_count"] == 2
    for branch in audit["branches"]:
        assert branch["maximum_attained_scale"] == 0.5
        assert branch["failed_growth_scale"] == 0.75
        assert branch["termination"] == "first_non_improving_growth_step"
        rmses = [probe["hard_rmse"] for probe in branch["probes"]]
        assert rmses[1] < rmses[0]
        assert rmses[2] >= rmses[1]
        assert branch["final_candidate_state_id"] == branch["probes"][1][
            "child_state_id"
        ]
        assert branch["final_global_beam_survival"] is True
    assert all(row["condition_violations"] == 0 for row in result["trace"])
    assert all(row["truth_used_for_selection"] is False for row in result["trace"])


def test_v4_deterministic_replay_and_hard_categorical_states():
    left = _run()
    right = _run()
    assert left["best_state"] == right["best_state"]
    assert left["trace"] == right["trace"]
    assert left["lineage_continuation"] == right["lineage_continuation"]
    assert set(torch.unique(left["best_labels"]).tolist()) <= {0, 9}
