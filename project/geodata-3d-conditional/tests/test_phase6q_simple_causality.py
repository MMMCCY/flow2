from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from guidance.simple_causality import (
    AnalyticObservationSuite,
    build_simple_causal_case,
    build_voxel_search_mask,
    controlled_observation,
    enumerate_hard_pairs,
    optimization_coefficients,
    optimize_candidate_logits,
    optimize_embedding_endpoint,
    optimize_hard_coordinates,
    optimize_voxel_logits,
    validate_embedding_endpoint_config,
    validate_simple_causality_config,
    validate_hard_coordinate_config,
    validate_voxel_reconstruction_config,
)


class _IdentitySeismic:
    def __call__(self, impedance, slowness, subsurface_mask):
        return impedance

    def reflectivity_spikes(self, impedance, slowness, subsurface_mask):
        return impedance


def _config() -> dict[str, object]:
    return {
        "schema": "phase6q_five_body_causality_v1",
        "id": "unit_test",
        "grid_shape": [16, 16, 16],
        "air_start_z": 14,
        "air_label": -1,
        "background_label": 0,
        "target_label": 9,
        "fixed_bodies": [
            {"id": "f0", "start": [1, 1, 3], "stop": [3, 3, 5], "well_xy": [2, 2]},
            {"id": "f1", "start": [5, 1, 5], "stop": [7, 3, 7], "well_xy": [6, 2]},
            {"id": "f2", "start": [9, 1, 7], "stop": [11, 3, 9], "well_xy": [10, 2]},
        ],
        "candidate_bodies": [
            {"id": "c0", "start": [1, 7, 3], "stop": [3, 9, 5]},
            {"id": "c1", "start": [5, 7, 5], "stop": [7, 9, 7]},
            {"id": "c2", "start": [9, 7, 7], "stop": [11, 9, 9]},
            {"id": "c3", "start": [13, 7, 9], "stop": [15, 9, 11]},
        ],
        "truth_candidate_indices": [1, 3],
        "observation_modes": ["property"],
        "blur_sigma_voxels": 1.0,
        "optimization": {
            "methods": ["soft", "ste_top2"],
            "updates": 60,
            "learning_rate": 0.3,
            "weight_decay": 0.0,
            "initial_logit": -2.0,
            "cardinality_weight": 0.01,
            "temperature_schedule": [{"temperature": 1.0, "steps": 60}],
            "hard_check_interval": 1,
        },
        "seismic_controls": ["correct", "zero", "shuffled_xy"],
        "shuffle_seed": 12,
        "enumeration_batch_size": 3,
        "inverse_crime": True,
        "measured_geophysics": False,
        "formal_training_authorized": False,
    }


def _suite(config: dict[str, object]) -> AnalyticObservationSuite:
    case = build_simple_causal_case(config)
    acoustic = torch.ones(2, 15)
    density = torch.ones(15)
    return AnalyticObservationSuite(
        case,
        acoustic_property_table=acoustic,
        density_table=density,
        seismic_operator=_IdentitySeismic(),
        gravity_operator=object(),
        blur_sigma_voxels=1.0,
    )


def _voxel_config() -> dict[str, object]:
    return {
        "schema": "phase6q_voxel_reconstruction_v1",
        "id": "unit_voxel",
        "search_region": {"start": [0, 5, 0], "stop": [16, 16, 13]},
        "observation_modes": ["property"],
        "methods": ["soft_voxel", "ste_voxel"],
        "updates": 50,
        "learning_rate": 0.5,
        "weight_decay": 0.0,
        "initial_logit": -4.0,
        "gradient_clip_norm": 10.0,
        "temperature_schedule": [{"temperature": 1.0, "steps": 50}],
        "hard_check_interval": 1,
        "seismic_controls": ["correct", "zero", "shuffled_xy"],
        "shuffle_seed": 12,
        "regularization": {"volume": 0.0, "smoothness": 0.0, "truth_roi": False},
        "hard_selection": "minimum_hard_physics_loss_only",
        "formal_training_authorized": False,
    }


def _hard_coordinate_config() -> dict[str, object]:
    return {
        "schema": "phase6q_hard_coordinate_v1",
        "id": "unit_hard",
        "observation_modes": ["property"],
        "seismic_controls": ["correct", "zero", "shuffled_xy"],
        "max_iterations": 10,
        "proposal_flip_counts": [4, 8, 16],
        "improvement_tolerance": 1e-10,
        "allow_additions": True,
        "allow_removals": True,
        "selection": "minimum_hard_physics_rmse_only",
        "regularization": {"volume": 0.0, "smoothness": 0.0, "truth_roi": False},
        "formal_training_authorized": False,
    }


def _embedding_endpoint_config() -> dict[str, object]:
    return {
        "schema": "phase6q_embedding_endpoint_v1",
        "id": "unit_embedding",
        "search_region": {"start": [0, 5, 0], "stop": [16, 16, 13]},
        "observation_modes": ["property"],
        "methods": ["soft_embedding", "ste_embedding_rock"],
        "updates": 50,
        "learning_rate": 0.3,
        "weight_decay": 0.0,
        "gradient_clip_norm": 10.0,
        "temperature_schedule": [{"temperature": 0.2, "steps": 50}],
        "hard_check_interval": 1,
        "max_state_norm_to_embedding_norm": 4.0,
        "seismic_controls": ["correct", "zero", "shuffled_xy"],
        "shuffle_seed": 12,
        "hard_selection": "minimum_hard_physics_loss_only",
        "regularization": {"volume": 0.0, "smoothness": 0.0, "truth_roi": False},
        "flow_unet_loaded": False,
        "formal_training_authorized": False,
    }


def test_simple_case_is_exactly_three_drilled_plus_two_hidden_bodies() -> None:
    config = _config()
    resolved = validate_simple_causality_config(config)
    assert resolved["truth_candidate_indices"] == [1, 3]
    case = build_simple_causal_case(config)
    assert case.validation["truth_baseline_difference_voxels"] == 16
    assert case.validation["candidate_condition_overlap_voxels"] == 0
    assert case.validation["truth_condition_mismatches"] == 0
    assert [row["fixed_target_voxels"] for row in case.validation["well_reports"]] == [2, 2, 2]
    assert all(row["candidate_target_voxels"] == 0 for row in case.validation["well_reports"])
    assert int((case.truth_labels == 9).sum()) == 40
    assert int((case.baseline_labels == 9).sum()) == 24


def test_validation_rejects_body_overlap_and_training_authorization() -> None:
    overlap = _config()
    overlap["candidate_bodies"][0]["start"] = [1, 1, 3]
    overlap["candidate_bodies"][0]["stop"] = [3, 3, 5]
    with pytest.raises(ValueError, match="disjoint"):
        validate_simple_causality_config(overlap)
    training = _config()
    training["formal_training_authorized"] = True
    with pytest.raises(ValueError, match="forbid formal training"):
        validate_simple_causality_config(training)


def test_hard_property_enumeration_uniquely_ranks_truth_pair() -> None:
    suite = _suite(_config())
    result = enumerate_hard_pairs(suite, "property", batch_size=3)
    assert result["candidate_pair_count"] == 6
    assert result["truth_pair_rank"] == 1
    assert result["truth_pair_rmse"] == 0.0
    assert result["near_numerical_zero_count"] == 1
    assert result["second_best_nontruth_rmse"] > 0


@pytest.mark.parametrize("method", ["soft", "ste_top2"])
def test_candidate_optimizer_recovers_two_hidden_bodies_on_direct_property(method: str) -> None:
    config = _config()
    suite = _suite(config)
    result = optimize_candidate_logits(
        suite,
        "property",
        control="correct",
        method=method,
        temperatures=[1.0] * 60,
        learning_rate=0.3,
        weight_decay=0.0,
        initial_logit=-2.0,
        cardinality_weight=0.01,
        hard_check_interval=1,
        shuffle_seed=12,
    )
    assert result["best_metrics"]["selected_indices"] == [1, 3]
    assert result["best_metrics"]["hard_rmse"] == 0.0
    assert result["best_metrics"]["body_precision"] == 1.0
    assert result["best_metrics"]["body_recall"] == 1.0
    if method == "ste_top2":
        assert result["trace"][0]["selected_count"] == 2


def test_ste_forward_is_hard_top_two_but_preserves_soft_gradient() -> None:
    logits = torch.tensor([[0.1, 0.8, -0.2, 0.4]], requires_grad=True)
    forward, probabilities, hard = optimization_coefficients(logits, 1.0, "ste_top2")
    assert torch.equal(forward.detach(), hard)
    assert int(hard.sum()) == 2
    assert torch.nonzero(hard[0], as_tuple=False).flatten().tolist() == [1, 3]
    forward.sum().backward()
    assert logits.grad is not None
    assert torch.all(logits.grad > 0)
    assert torch.all((probabilities > 0) & (probabilities < 1))


def test_xy_shuffle_is_deterministic_and_not_identity() -> None:
    values = torch.arange(1 * 1 * 8 * 7 * 3).reshape(1, 1, 8, 7, 3).float()
    first = controlled_observation(values, "shuffled_xy", shuffle_seed=99)
    second = controlled_observation(values, "shuffled_xy", shuffle_seed=99)
    assert torch.equal(first, second)
    assert not torch.equal(first, values)
    assert torch.equal(first.flatten().sort().values, values.flatten().sort().values)


def test_voxel_search_is_broad_unconditioned_and_contains_hidden_truth() -> None:
    case = build_simple_causal_case(_config())
    resolved = validate_voxel_reconstruction_config(
        _voxel_config(), grid_shape=case.truth_labels.shape[2:]
    )
    assert resolved["updates"] == 50
    search, report = build_voxel_search_mask(case, _voxel_config())
    assert report["condition_overlap_voxels"] == 0
    assert report["fixed_target_overlap_voxels"] == 0
    assert report["hidden_truth_inside_search_voxels"] == 16
    assert int(search.sum()) > 16


@pytest.mark.parametrize("method", ["soft_voxel", "ste_voxel"])
def test_free_voxel_optimizer_recovers_hidden_truth_on_direct_property(method: str) -> None:
    config = _config()
    case = build_simple_causal_case(config)
    suite = _suite(config)
    search, _ = build_voxel_search_mask(case, _voxel_config())
    result = optimize_voxel_logits(
        suite,
        "property",
        search_mask=search,
        control="correct",
        method=method,
        temperatures=[1.0] * 50,
        learning_rate=0.5,
        weight_decay=0.0,
        initial_logit=-4.0,
        gradient_clip_norm=10.0,
        hard_check_interval=1,
        shuffle_seed=12,
    )
    best = result["best_metrics"]
    assert best["hard_rmse"] == 0.0
    assert best["hidden_iou"] == 1.0
    assert best["hidden_body_0_recall"] == 1.0
    assert best["hidden_body_1_recall"] == 1.0


def test_monotone_hard_coordinate_solver_recovers_direct_property() -> None:
    resolved = validate_hard_coordinate_config(_hard_coordinate_config())
    assert resolved["proposal_flip_counts"] == [4, 8, 16]
    config = _config()
    case = build_simple_causal_case(config)
    suite = _suite(config)
    search, _ = build_voxel_search_mask(case, _voxel_config())
    result = optimize_hard_coordinates(
        suite,
        "property",
        search_mask=search,
        control="correct",
        max_iterations=10,
        proposal_flip_counts=[4, 8, 16],
        improvement_tolerance=1e-10,
        shuffle_seed=12,
    )
    final = result["final_metrics"]
    assert final["hard_rmse"] == 0.0
    assert final["hidden_iou"] == 1.0
    losses = [row["hard_rmse"] for row in result["trace"]]
    assert all(right <= left for left, right in zip(losses, losses[1:]))


@pytest.mark.parametrize(
    "method",
    [
        "soft_embedding",
        "ste_embedding_rock",
        "soft_embedding_binary",
        "ste_embedding_binary",
    ],
)
def test_checkpoint_embedding_endpoint_recovers_direct_property(method: str) -> None:
    resolved = validate_embedding_endpoint_config(
        _embedding_endpoint_config(), grid_shape=[16, 16, 16]
    )
    assert resolved["flow_unet_loaded"] is False
    config = _config()
    case = build_simple_causal_case(config)
    suite = _suite(config)
    search, _ = build_voxel_search_mask(case, _voxel_config())
    result = optimize_embedding_endpoint(
        suite,
        "property",
        search_mask=search,
        embedding_weight=torch.eye(15),
        control="correct",
        method=method,
        temperatures=[0.2] * 50,
        learning_rate=0.3,
        weight_decay=0.0,
        gradient_clip_norm=10.0,
        hard_check_interval=1,
        max_state_norm_to_embedding_norm=4.0,
        shuffle_seed=12,
    )
    best = result["best_metrics"]
    assert best["hard_rmse"] == 0.0
    assert best["hidden_iou"] == 1.0
    assert best["hidden_body_0_recall"] == 1.0
    assert best["hidden_body_1_recall"] == 1.0


def test_all_class_soft_field_accepts_exact_one_hot_air_above_surface() -> None:
    suite = _suite(_config())
    labels = suite.case.baseline_labels
    probabilities = torch.nn.functional.one_hot(
        (labels[:, 0] + 1).long(), num_classes=15
    ).permute(0, 4, 1, 2, 3).float()
    field = suite.field_from_probabilities(probabilities, "seismic")
    assert field.shape == labels.shape
    assert torch.isfinite(field).all()
