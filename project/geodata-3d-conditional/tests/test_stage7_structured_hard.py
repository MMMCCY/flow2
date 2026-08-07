import torch

from guidance.structured_hard_inference import (
    StructuredObject,
    beam_evolutionary_search,
    materialize_model,
    StructuredModel,
)


def _object(name, center):
    return StructuredObject(
        object_id=name, presence=True,
        center_x=center[0], center_y=center[1], center_z=center[2],
        size_x=2.0, size_y=2.0, size_z=2.0,
        orientation_deg=0.0, shape="cuboid", material_label=9,
        source_family="test",
    )


def test_materialize_rejects_condition_intersection():
    baseline = torch.zeros((1, 1, 8, 8, 8), dtype=torch.long)
    condition = torch.zeros_like(baseline, dtype=torch.bool)
    condition[..., 3, 3, 3] = True
    model = StructuredModel((_object("bad", (3.0, 3.0, 3.0)),), "bad", None, "add")
    labels, report = materialize_model(
        model, baseline_labels=baseline, condition_mask=condition,
        air_start_z=7, allowed_material_labels=(0, 9),
    )
    assert labels is None
    assert "condition_intersection" in report["reasons"]


def test_beam_search_selects_only_by_hard_observation():
    baseline = torch.zeros((1, 1, 8, 8, 8), dtype=torch.long)
    condition = torch.zeros_like(baseline, dtype=torch.bool)
    library = [_object("wrong", (2.0, 2.0, 2.0)), _object("right", (5.0, 5.0, 3.0))]
    target_model = StructuredModel((library[1],), "truth_not_given_to_search", None, "test")
    target, _ = materialize_model(
        target_model, baseline_labels=baseline, condition_mask=condition,
        air_start_z=7, allowed_material_labels=(0, 9),
    )
    response = lambda labels: (labels == 9).float()
    result = beam_evolutionary_search(
        baseline_labels=baseline, condition_mask=condition, air_start_z=7,
        observation=response(target), hard_response=response,
        proposal_library=library, allowed_material_labels=(0, 9),
        kmax=1, beam_size=2, local_generations=0,
    )
    assert result["best_hard_rmse"] == 0.0
    assert result["best_model"].objects[0].object_id == "right"
    assert result["selection_used_truth"] is False
