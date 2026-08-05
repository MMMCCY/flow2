from __future__ import annotations

import json
from pathlib import Path
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from guidance.physics_attainment import (
    field_attainment_diagnostics,
    optimize_endpoint_state,
    project_and_clip_state,
)
from scripts.stage6.run_physics_attainment_limit import validate_config
from scripts.stage6.run_physics_guidance_ladder import validate_ladder_config


def test_frozen_phase6p_config_matches_implementation() -> None:
    path = (
        PROJECT_DIR
        / "experiments/stage6_geo_adapter/configs/physics_attainment_seismic_endpoint_v1.json"
    )
    config = json.loads(path.read_text(encoding="utf-8"))
    resolved = validate_config(config)
    assert resolved["optimization_steps"] == 200
    assert resolved["sampling_seed"] == 42
    assert resolved["sampling_steps"] == 32
    assert resolved["high_attainment_lower"] == 0.8
    config["allow_hyperparameter_sweep_on_case"] = True
    try:
        validate_config(config)
    except ValueError as error:
        assert "allow_hyperparameter_sweep_on_case" in str(error)
    else:
        raise AssertionError("legacy Phase-6P case unexpectedly allowed a sweep")


def test_frozen_phase6p_extreme_ladder_matches_implementation() -> None:
    path = (
        PROJECT_DIR
        / "experiments/stage6_geo_adapter/configs/physics_attainment_seismic_trajectory_ladder_v1.json"
    )
    config = json.loads(path.read_text(encoding="utf-8"))
    resolved = validate_ladder_config(config)
    assert [level["alpha"] for level in resolved["levels"]] == [
        0.25,
        0.5,
        1.0,
        2.0,
        4.0,
    ]
    assert resolved["seed"] == 42
    assert resolved["n_steps"] == 32
    config["levels"][1]["max_guidance_ratio"] = 0.25
    try:
        validate_ladder_config(config)
    except ValueError as error:
        assert "alpha=cap" in str(error)
    else:
        raise AssertionError("Phase-6P ladder unexpectedly accepted a mismatched cap")


def test_field_attainment_reports_magnitude_and_alignment() -> None:
    observed = torch.ones(1, 1, 2, 2, 2)
    baseline = torch.zeros_like(observed)
    candidate = torch.full_like(observed, 0.75)
    diagnostics = field_attainment_diagnostics(
        observed, baseline, candidate, torch.ones_like(observed)
    )
    assert abs(float(diagnostics["baseline_rmse"]) - 1.0) < 1e-7
    assert abs(float(diagnostics["candidate_rmse"]) - 0.25) < 1e-7
    assert abs(float(diagnostics["attainment"]) - 0.75) < 1e-7
    assert (
        abs(
            float(diagnostics["update_to_required_residual_norm_ratio"])
            - 0.75
        )
        < 1e-7
    )
    assert abs(float(diagnostics["update_residual_cosine"]) - 1.0) < 1e-7
    assert diagnostics["candidate_closer_to_observation_than_baseline"] is True
    assert (
        diagnostics["candidate_closer_to_observation_than_to_baseline"] is True
    )


def test_project_and_clip_preserves_exact_conditions() -> None:
    state = torch.tensor([[[[[10.0, 3.0, 0.0]]]]])
    embedded = torch.tensor([[[[[7.0, 0.0, 0.0]]]]])
    condition = torch.tensor([[[[[True, False, False]]]]])
    projected = project_and_clip_state(
        state, embedded, condition, max_voxel_norm=2.0
    )
    assert float(projected[..., 0]) == 7.0
    assert abs(float(projected[..., 1]) - 2.0) < 1e-7
    assert float(projected[..., 2]) == 0.0


def test_endpoint_optimizer_lowers_hard_loss_and_preserves_conditions() -> None:
    initial = torch.zeros(1, 1, 1, 1, 4)
    embedded = torch.zeros_like(initial)
    condition = torch.zeros(1, 1, 1, 1, 4, dtype=torch.bool)
    condition[..., 0] = True
    target = torch.tensor([[[[[0.0, 1.0, -1.0, 0.5]]]]])

    def soft_loss(state: torch.Tensor, temperature: float):
        del temperature
        loss = (state - target).square().mean()
        return loss, {"toy_rmse": torch.sqrt(loss)}

    def hard_evaluate(state: torch.Tensor):
        loss = float((state - target).square().mean().detach())
        return {"hard_loss": loss}, {"decoded": state.detach().round()}

    result = optimize_endpoint_state(
        initial_state=initial,
        embedded_conditions=embedded,
        condition_mask=condition,
        soft_loss=soft_loss,
        hard_evaluate=hard_evaluate,
        temperature_schedule=[{"temperature": 0.2, "steps": 30}],
        learning_rate=0.15,
        weight_decay=0.0,
        gradient_clip_norm=10.0,
        hard_check_interval=2,
        max_voxel_norm=3.0,
    )
    assert float(result["best_metrics"]["hard_loss"]) < 0.01
    assert float(result["best_metrics"]["hard_loss"]) < float(
        result["initial_metrics"]["hard_loss"]
    )
    assert int(result["best_step"]) > 0
    assert torch.equal(result["best_state"][..., 0], embedded.cpu()[..., 0])
    checked = [
        float(row["hard_loss"])
        for row in result["trace"]
        if bool(row["hard_checked"])
    ]
    assert abs(min(checked) - float(result["best_metrics"]["hard_loss"])) < 1e-12
