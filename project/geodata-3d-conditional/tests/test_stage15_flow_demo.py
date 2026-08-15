from __future__ import annotations

import json
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
from scripts.stage15.run_flow_demo import evidence_transform
from guidance.probability_volume import (
    COARSE_OCCUPANCY_LOSS_MODE,
    probability_volume_loss,
)


def test_evidence_transform_is_fixed_max_scaling_without_thresholding() -> None:
    probability = torch.tensor([0.0, 0.1, 0.2]).reshape(1, 1, 1, 1, 3)
    support = torch.tensor([False, True, True]).reshape_as(probability)
    result = evidence_transform(probability, support)
    assert torch.equal(result, torch.tensor([0.0, 0.5, 1.0]).reshape_as(result))


def test_flow_demo_is_disclosed_posthoc_fixed_seed_screen() -> None:
    config = json.loads((PROJECT_DIR / "experiments/stage15_binary_seismic_consensus/configs/flow_demo_coarse_occupancy_v1.json").read_text())
    assert len(config["source_seeds"]) == 8
    assert config["fine_voxel_repeat_used_in_loss"] is False
    assert config["hard_dice_core_used"] is False
    assert config["probability_loss_mode"] == "binary_coarse_occupancy_bce_v1"
    assert config["dice_weight"] == 0.0
    assert config["parameter_sweep"] is False
    assert "coarse" in config["scientific_role"]


def test_flow_runner_is_truth_blind() -> None:
    source = (PROJECT_DIR / "scripts/stage15/run_flow_demo.py").read_text()
    assert "true_model" not in source
    assert "phase1_assets" not in source
    assert "selection_is_truth" not in source


def test_coarse_occupancy_loss_accepts_8_cubed_binary_target_and_backpropagates() -> None:
    torch.manual_seed(7)
    state = torch.randn((1, 3, 64, 64, 64), requires_grad=True)
    embeddings = torch.randn((11, 3))
    target = torch.rand((1, 1, 8, 8, 8)) * 0.25
    roi = torch.ones((1, 1, 64, 64, 64), dtype=torch.bool)
    loss, diagnostics = probability_volume_loss(
        state,
        embeddings,
        target,
        roi,
        target_label=9,
        tau=0.5,
        bce_weight=1.0,
        dice_weight=0.0,
        loss_mode=COARSE_OCCUPANCY_LOSS_MODE,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert float(state.grad.norm()) > 0
    assert diagnostics["probability_dice_loss"] == 0
