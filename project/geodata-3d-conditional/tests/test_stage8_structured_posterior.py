import inspect
from dataclasses import replace

import torch

from guidance.seismic import hard_labels_to_acoustic
from guidance.structured_birth_sensitivity import (
    DeterministicSensitivityBirthCenterRanker,
)
from guidance.structured_hard_inference import (
    StructuredModel as Stage7Model,
    StructuredObject as Stage7Object,
    materialize_model as stage7_materialize,
)
from guidance.structured_posterior import (
    HardConditionProjector,
    ProposalKernel,
    StructuredBodySpec,
    StructuredBounds,
    StructuredState,
    controlled_observations,
    inference_visible_audit,
    materialize_state,
    retrospective_hard_metrics,
    structured_search,
)
from guidance.structured_trust_region import (
    HardLossBirthContinuation,
    scaled_body,
    validate_scale_ladder,
)
from scripts.stage8.run_stage8 import _search_arm


def _bounds(maximum_body_count=2):
    return StructuredBounds(
        center_x=(2.0, 5.0), center_y=(2.0, 5.0), center_z=(2.0, 5.0),
        size_x=(1.5, 3.5), size_y=(1.5, 3.5), size_z=(1.5, 3.5),
        orientation_deg=(0.0, 180.0), shapes=("cuboid", "ellipsoid"),
        material_labels=(9,), maximum_body_count=maximum_body_count,
    )


def _projector():
    values = torch.zeros((1, 1, 8, 8, 8), dtype=torch.long)
    values[..., 7] = -1
    condition = torch.zeros_like(values, dtype=torch.bool)
    condition[..., 7] = True
    condition[..., 1, 1, :7] = True
    edit = ~condition
    return HardConditionProjector(
        condition_values=values, condition_mask=condition, edit_mask=edit
    )


def _body():
    return StructuredBodySpec(
        body_id="body", center_x=4.0, center_y=4.0, center_z=4.0,
        size_x=3.0, size_y=3.0, size_z=3.0, orientation_deg=15.0,
        shape="ellipsoid", material_label=9,
    )


def test_structured_body_and_state_serialization_roundtrip():
    state = StructuredState((_body(),), "state", None, "birth")
    assert StructuredState.from_record(state.record()) == state


def test_birth_death_and_maximum_count_bounds():
    kernel = ProposalKernel(_bounds(maximum_body_count=1), seed=4)
    empty = StructuredState((), "empty", None, "initial")
    born = kernel.propose(empty, generation=1, proposal_index=0)
    assert born.proposal_move == "birth" and len(born.bodies) == 1
    moves = {
        kernel.propose(born, generation=2, proposal_index=index).proposal_move
        for index in range(12)
    }
    assert "birth" not in moves
    assert "death" in moves


def test_translate_resize_rotation_stay_inside_registered_bounds():
    kernel = ProposalKernel(_bounds(), seed=7)
    state = StructuredState((_body(),), "parent", None, "fixture")
    proposals = [kernel.propose(state, generation=3, proposal_index=i) for i in range(60)]
    fields = {
        "translate": ("center_x", "center_y", "center_z"),
        "resize": ("size_x", "size_y", "size_z"),
        "rotate": ("orientation_deg",),
    }
    observed = set()
    for proposal in proposals:
        observed.add(proposal.proposal_move)
        for prefix, names in fields.items():
            if proposal.proposal_move == prefix:
                for name in names:
                    low, high = getattr(kernel.bounds, name)
                    assert low <= getattr(proposal.bodies[0], name) <= high
    assert set(fields) <= observed


def test_proposal_replay_is_deterministic():
    parent = StructuredState((_body(),), "parent", None, "fixture")
    left = ProposalKernel(_bounds(), seed=99).propose(parent, generation=5, proposal_index=11)
    right = ProposalKernel(_bounds(), seed=99).propose(parent, generation=5, proposal_index=11)
    assert left == right


def test_ranked_birth_replaces_only_center_random_draws():
    kernel = ProposalKernel(_bounds(), seed=99)
    parent = StructuredState((), "empty", None, "initial")
    uniform = kernel.propose(parent, generation=2, proposal_index=7)
    ranked = kernel.propose(
        parent, generation=2, proposal_index=7, birth_center=(2.0, 3.0, 4.0)
    )
    assert replace(
        ranked.bodies[0],
        center_x=uniform.bodies[0].center_x,
        center_y=uniform.bodies[0].center_y,
        center_z=uniform.bodies[0].center_z,
    ) == uniform.bodies[0]
    assert ranked.bodies[0].center_x == 2.0
    assert ranked.bodies[0].center_y == 3.0
    assert ranked.bodies[0].center_z == 4.0


def test_condition_projection_is_exact_and_edits_exclude_conditions():
    projector = _projector()
    base = projector.condition_values.clone()
    covering = StructuredBodySpec(
        body_id="cover", center_x=1.0, center_y=1.0, center_z=3.0,
        size_x=6.0, size_y=6.0, size_z=6.0, orientation_deg=0.0,
        shape="cuboid", material_label=9,
    )
    labels, audit = materialize_state(
        StructuredState((covering,), "s", None, "birth"),
        base_labels=base, projector=projector,
    )
    assert audit["condition_violations"] == 0
    assert torch.equal(labels[projector.condition_mask], base[projector.condition_mask])


def test_hard_property_mapping_matches_direct_table_indexing():
    labels = torch.tensor([[[[[-1, 0, 9]]]]])
    table = torch.arange(30, dtype=torch.float32).reshape(2, 15)
    acoustic = hard_labels_to_acoustic(labels, table)
    assert torch.equal(acoustic[0, :, 0, 0], table[:, labels[0, 0, 0, 0] + 1])


def test_stage7_hard_raster_fixture_is_preserved():
    projector = _projector()
    base = projector.condition_values.clone()
    body = _body()
    stage8, _ = materialize_state(
        StructuredState((body,), "s8", None, "birth"), base_labels=base, projector=projector
    )
    stage7_body = Stage7Object(
        object_id=body.body_id, presence=True, center_x=body.center_x,
        center_y=body.center_y, center_z=body.center_z, size_x=body.size_x,
        size_y=body.size_y, size_z=body.size_z,
        orientation_deg=body.orientation_deg, shape=body.shape,
        material_label=body.material_label, source_family="fixture",
    )
    stage7, report = stage7_materialize(
        Stage7Model((stage7_body,), "s7", None, "add"),
        baseline_labels=base, condition_mask=projector.condition_mask,
        air_start_z=7, allowed_material_labels=(9,),
    )
    assert report["valid"] is True
    assert torch.equal(stage8, stage7)


def test_control_observations_are_deterministic_and_value_preserving():
    correct = torch.arange(48, dtype=torch.float32).reshape(1, 1, 3, 4, 4)
    wrong = -correct
    first = controlled_observations(correct, wrong_case=wrong, shuffle_seed=8)
    second = controlled_observations(correct, wrong_case=wrong, shuffle_seed=8)
    assert set(first) == {"correct", "zero", "shuffled_xy", "wrong_case_observation"}
    assert all(torch.equal(first[key], second[key]) for key in first)
    assert torch.equal(
        first["shuffled_xy"].reshape(-1).sort().values,
        correct.reshape(-1).sort().values,
    )


def test_search_api_and_objective_have_no_truth_argument():
    assert "truth" not in inspect.signature(structured_search).parameters
    assert "truth" not in inspect.signature(inference_visible_audit).parameters
    assert "truth" not in inspect.signature(
        DeterministicSensitivityBirthCenterRanker.__init__
    ).parameters
    assert "truth" not in inspect.signature(
        DeterministicSensitivityBirthCenterRanker.rank
    ).parameters
    for target in (
        HardLossBirthContinuation.__init__,
        HardLossBirthContinuation.plan_new_center,
        HardLossBirthContinuation.plan_growth,
        HardLossBirthContinuation.record_result,
    ):
        assert "truth" not in inspect.signature(target).parameters


def test_v3_scale_ladder_is_frozen_strict_and_full_ending():
    assert validate_scale_ladder([0.25, 0.5, 0.75, 1.0]) == (
        0.25, 0.5, 0.75, 1.0
    )
    for invalid in ([0.5], [0.0, 1.0], [0.5, 0.5, 1.0], [0.25, 0.75]):
        try:
            validate_scale_ladder(invalid)
        except ValueError:
            pass
        else:  # pragma: no cover - explicit rejection table
            raise AssertionError(f"invalid ladder accepted: {invalid}")


def test_v3_scaled_bodies_are_nested_hard_categorical_and_preserve_geometry():
    values = torch.zeros((1, 1, 32, 32, 32), dtype=torch.long)
    condition = torch.zeros_like(values, dtype=torch.bool)
    condition[..., 0, 0, 0] = True
    projector = HardConditionProjector(
        condition_values=values,
        condition_mask=condition,
        edit_mask=~condition,
    )
    for shape in ("ellipsoid", "dike_hemisphere"):
        full = StructuredBodySpec(
            body_id=f"body_{shape}", center_x=15.25, center_y=14.75,
            center_z=16.0, size_x=14.0, size_y=10.0, size_z=12.0,
            orientation_deg=37.0, shape=shape, material_label=9,
        )
        masks = []
        for index, scale in enumerate((0.25, 0.5, 0.75, 1.0)):
            probe = scaled_body(full, scale)
            assert probe.center_x == full.center_x
            assert probe.center_y == full.center_y
            assert probe.center_z == full.center_z
            assert probe.orientation_deg == full.orientation_deg
            assert probe.shape == full.shape
            assert probe.material_label == full.material_label
            labels, audit = materialize_state(
                StructuredState((probe,), f"probe_{index}", None, "fixture"),
                base_labels=values, projector=projector,
            )
            assert audit["condition_violations"] == 0
            assert set(torch.unique(labels).tolist()) <= {0, 9}
            masks.append(labels == 9)
        assert all(not bool((left & ~right).any()) for left, right in zip(masks, masks[1:]))


def test_v3_continuation_requires_strict_hard_improvement_and_terminates_failure():
    controller = HardLossBirthContinuation(scale_ladder=(0.25, 0.5, 0.75, 1.0))
    parent = StructuredState((), "empty", None, "initial")
    full = StructuredState((_body(),), "full_child", "empty", "birth")
    child, guidance = controller.plan_new_center(
        parent=parent, full_child=full, generation=1, proposal_index=0,
        ranking_id="ranking", rank_index=0,
        center_score=2.0,
    )
    assert guidance["scale"] == 0.25
    assert child.bodies[0].size_x == _body().size_x * 0.25
    accepted = controller.record_result(
        child=child, hard_rmse=0.9, parent_rmse=1.0, empty_rmse=1.0,
        condition_violations=0,
    )
    assert accepted["continuation_authorized"] is True
    growth, growth_guidance = controller.plan_growth(
        parent=child, generation=2, proposal_index=0
    )
    assert growth_guidance["scale"] == 0.5
    assert growth.bodies[0].size_x == _body().size_x * 0.5
    rejected = controller.record_result(
        child=growth, hard_rmse=0.9, parent_rmse=0.9, empty_rmse=1.0,
        condition_violations=0,
    )
    assert rejected["continuation_authorized"] is False
    assert rejected["termination"] == "terminated_non_improving_hard_step"
    assert controller.can_grow(parent=growth, generation=3) is False


def _sensitivity_fixture(tmp_path):
    projector = _projector()
    base = projector.condition_values.clone()
    table = torch.ones((2, 15), dtype=torch.float32)
    table[0] = torch.linspace(1.0, 2.4, 15)
    table[1] = torch.linspace(0.5, 1.9, 15)
    subsurface = base != -1

    def forward(impedance, slowness, mask):
        return (impedance + 2.0 * slowness) * mask

    target_body = StructuredBodySpec(
        body_id="target", center_x=4.0, center_y=4.0, center_z=4.0,
        size_x=3.0, size_y=3.0, size_z=3.0, orientation_deg=0.0,
        shape="ellipsoid", material_label=9,
    )
    target, _ = materialize_state(
        StructuredState((target_body,), "target", None, "fixture"),
        base_labels=base, projector=projector,
    )
    response = lambda labels: forward(
        *hard_labels_to_acoustic(labels, table).split(1, dim=1), subsurface
    )
    observation = response(target)
    ranker = DeterministicSensitivityBirthCenterRanker(
        projector=projector, property_table=table, subsurface_mask=subsurface,
        seismic_forward=forward, bounds=_bounds(), target_label=9,
        canonical_size_xyz=(3.0, 3.0, 3.0), audit_dir=tmp_path,
        ranked_center_record_count=12,
    )
    return projector, base, observation, response, ranker


def test_multifield_sensitivity_ranker_is_deterministic_and_auditable(tmp_path):
    left = _sensitivity_fixture(tmp_path / "left")
    right = _sensitivity_fixture(tmp_path / "right")
    state = StructuredState((), "empty", None, "initial")
    left_ranking = left[4].rank(
        state=state, current_labels=left[1],
        current_predicted_seismic=left[3](left[1]), observed_seismic=left[2],
        generation=1,
    )
    right_ranking = right[4].rank(
        state=state, current_labels=right[1],
        current_predicted_seismic=right[3](right[1]), observed_seismic=right[2],
        generation=1,
    )
    assert left_ranking["ranked_centers"] == right_ranking["ranked_centers"]
    assert left_ranking["ranked_centers"][0]["center_xyz"] == [4, 4, 4]
    left_audit = left[4].summary()["rankings"][0]
    right_audit = right[4].summary()["rankings"][0]
    assert left_audit["sensitivity_map_sha256"] == right_audit["sensitivity_map_sha256"]
    assert left_audit["property_fields"] == ["impedance", "slowness"]
    assert set(left_audit["property_gradient_sha256"]) == {"impedance", "slowness"}
    assert left_audit["truth_fields_present"] is False
    assert left_audit["tie_breaking"].startswith("score_descending")


def test_sensitivity_guided_search_replays_and_keeps_hard_budget(tmp_path):
    left = _sensitivity_fixture(tmp_path / "left")
    right = _sensitivity_fixture(tmp_path / "right")
    kwargs = dict(
        base_labels=left[1], projector=left[0], observation=left[2],
        hard_response=left[3], proposal_kernel=ProposalKernel(_bounds(), seed=15),
        beam_size=2, generations=2, proposals_per_parent=3,
        birth_center_ranker=left[4],
    )
    first = structured_search(**kwargs)
    kwargs.update(
        base_labels=right[1], projector=right[0], observation=right[2],
        hard_response=right[3], proposal_kernel=ProposalKernel(_bounds(), seed=15),
        birth_center_ranker=right[4],
    )
    second = structured_search(**kwargs)
    assert first["forward_call_count"] == 13
    assert first["forward_call_count"] == first["fixed_forward_call_budget"]
    assert first["best_state"] == second["best_state"]
    assert first["trace"] == second["trace"]
    for result in (first, second):
        audit = result["birth_center_initializer"]
        assert audit["differentiable_forward_calls"] == audit["backward_calls"]
        assert audit["differentiable_forward_calls"] > 0
        assert audit["truth_used"] is False
        assert all(
            row["birth_guidance"]["truth_used"] is False
            for row in result["trace"] if "birth_guidance" in row
        )


def test_v3_trust_region_search_replays_and_reuses_exact_hard_slots(tmp_path):
    left = _sensitivity_fixture(tmp_path / "left")
    right = _sensitivity_fixture(tmp_path / "right")
    kwargs = dict(
        base_labels=left[1], projector=left[0], observation=left[2],
        hard_response=left[3], proposal_kernel=ProposalKernel(_bounds(), seed=15),
        beam_size=2, generations=3, proposals_per_parent=3,
        birth_center_ranker=left[4],
        birth_trust_region_controller=HardLossBirthContinuation(
            scale_ladder=(0.25, 0.5, 0.75, 1.0)
        ),
    )
    first = structured_search(**kwargs)
    kwargs.update(
        base_labels=right[1], projector=right[0], observation=right[2],
        hard_response=right[3], proposal_kernel=ProposalKernel(_bounds(), seed=15),
        birth_center_ranker=right[4],
        birth_trust_region_controller=HardLossBirthContinuation(
            scale_ladder=(0.25, 0.5, 0.75, 1.0)
        ),
    )
    second = structured_search(**kwargs)
    assert first["forward_call_count"] == 19
    assert first["fixed_forward_call_budget"] == 19
    assert first["best_state"] == second["best_state"]
    assert first["trace"] == second["trace"]
    assert first["birth_trust_region"] == second["birth_trust_region"]
    audit = first["birth_trust_region"]
    assert audit["probe_count"] == sum(
        "trust_region_probe" in row for row in first["trace"]
    )
    assert audit["new_center_probe_count"] + audit["growth_probe_count"] == audit["probe_count"]
    assert audit["truth_fields_present"] is False


def test_search_uses_fixed_budget_and_replays_selected_state():
    projector = _projector()
    base = projector.condition_values.clone()
    target, _ = materialize_state(
        StructuredState((_body(),), "target_fixture", None, "fixture"),
        base_labels=base, projector=projector,
    )
    response = lambda labels: (labels == 9).float()
    kwargs = dict(
        base_labels=base, projector=projector, observation=response(target),
        hard_response=response, proposal_kernel=ProposalKernel(_bounds(), seed=15),
        beam_size=3, generations=4, proposals_per_parent=6,
    )
    left = structured_search(**kwargs)
    kwargs["proposal_kernel"] = ProposalKernel(_bounds(), seed=15)
    right = structured_search(**kwargs)
    assert left["forward_call_count"] == 73
    assert left["forward_call_count"] == left["fixed_forward_call_budget"]
    assert left["best_state"] == right["best_state"]
    assert torch.equal(left["best_labels"], right["best_labels"])
    replay, _ = materialize_state(
        StructuredState.from_record(left["best_state"].record()),
        base_labels=base, projector=projector,
    )
    assert torch.equal(replay.cpu(), left["best_labels"])
    assert all(row["truth_used_for_selection"] is False for row in left["trace"])


def test_retrospective_metrics_are_explicitly_post_selection_only():
    projector = _projector()
    base = projector.condition_values.clone()
    selected, _ = materialize_state(
        StructuredState((_body(),), "selected", None, "fixture"),
        base_labels=base, projector=projector,
    )
    result = retrospective_hard_metrics(
        selected, truth_labels=selected.clone(), condition_mask=projector.condition_mask,
        target_label=9, base_labels=base,
    )
    assert result["retrospective_only"] is True
    assert result["used_for_selection"] is False
    assert result["concealed_target_iou"] == 1.0


def test_run_audit_writes_replay_state_trace_and_hashes(tmp_path):
    projector = _projector()
    base = projector.condition_values.clone()
    selected = _search_arm(
        arm_dir=tmp_path / "arm",
        base_labels=base,
        projector=projector,
        observation=torch.ones_like(base, dtype=torch.float32),
        response_fn=lambda labels: (labels == 9).float(),
        bounds=_bounds(),
        search_config={"beam_size": 2, "generations": 2, "proposals_per_parent": 3},
        proposal_seed=18,
    )
    assert (tmp_path / "arm/selection.json").is_file()
    assert (tmp_path / "arm/proposal_trace.json").is_file()
    assert len(selected["selected_labels_sha256"]) == 64
    assert len(selected["selected_response_sha256"]) == 64
    assert selected["selection_frozen_before_retrospective_evaluation"] is True
