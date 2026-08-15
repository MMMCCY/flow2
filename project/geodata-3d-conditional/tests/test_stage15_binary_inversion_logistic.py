from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_DIR = Path(__file__).resolve().parents[1]

from guidance.binary_inversion_logistic import (
    FEATURE_NAMES,
    binary_inversion_features,
    coarse_support_count_8,
    weighted_mean_std,
    within_case_percentile,
)


def test_percentile_is_case_relative_and_tie_aware() -> None:
    q = torch.zeros((1, 1, 8, 8, 8))
    domain = torch.zeros_like(q, dtype=torch.bool)
    domain.reshape(-1)[:4] = True
    q.reshape(-1)[:4] = torch.tensor([0.1, 0.2, 0.2, 0.9])
    result = within_case_percentile(q, domain).reshape(-1)[:4]
    assert torch.allclose(result, torch.tensor([0.25, 0.75, 0.75, 1.0]))


def test_binary_features_have_only_four_frozen_channels() -> None:
    q = torch.linspace(0, 1, 512).reshape(1, 1, 8, 8, 8)
    support = torch.ones_like(q, dtype=torch.long)
    features = binary_inversion_features(q, support)
    assert features.shape == (8, 8, 8, 4)
    assert FEATURE_NAMES == (
        "raw_q",
        "within_case_percentile",
        "vertical_contrast",
        "depth",
    )


def test_coarse_support_counts_exactly_cover_subsurface() -> None:
    support = torch.zeros((1, 1, 64, 64, 64), dtype=torch.bool)
    support[..., :40] = True
    counts = coarse_support_count_8(support)
    assert int(counts.sum()) == int(support.sum())
    assert torch.equal(counts[..., :5], torch.full_like(counts[..., :5], 8**3))
    assert int(counts[..., 5:].sum()) == 0


def test_soft_occupancy_weighted_bce_equals_explicit_binary_voxels() -> None:
    logits = torch.tensor([0.3, -0.7])
    positive = torch.tensor([2.0, 1.0])
    support = torch.tensor([4.0, 2.0])
    occupancy = positive / support
    compressed = (
        F.binary_cross_entropy_with_logits(logits, occupancy, reduction="none") * support
    ).sum() / support.sum()
    explicit_logits = torch.tensor([0.3, 0.3, 0.3, 0.3, -0.7, -0.7])
    explicit_target = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    explicit = F.binary_cross_entropy_with_logits(explicit_logits, explicit_target)
    assert torch.allclose(compressed, explicit)


def test_training_standardization_uses_voxel_weights() -> None:
    features = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0]])
    mean, std = weighted_mean_std(features, torch.tensor([3.0, 1.0]))
    assert torch.allclose(mean, torch.tensor([0.25, 0.5, 0.75, 1.0]))
    assert bool((std > 0).all())


def test_stage15_g_is_binary_linear_and_has_no_sweep() -> None:
    config = json.loads(
        (
            PROJECT_DIR
            / "experiments/stage15_binary_seismic_consensus/configs/binary_inversion_logistic_8x8x8_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert config["target"] == "binary_raw_label9_vs_every_other_subsurface_voxel"
    assert config["model"] == "single_linear_layer_4_to_1"
    assert config["feature_names"] == list(FEATURE_NAMES)
    assert config["class_balancing"] is False
    assert config["parameter_sweep"] is False
    assert config["flow_used"] is False
    assert config["seismic_forward_rerun"] is False
    assert config["seismic_inversion_rerun"] is False


def test_stage15_g_heldout_runner_does_not_name_hidden_truth_assets() -> None:
    source = (PROJECT_DIR / "scripts/stage15/run_binary_inversion_logistic.py").read_text(
        encoding="utf-8"
    )
    assert "true_model" not in source
    assert "phase1_assets" not in source
    assert "load_checkpoint" not in source
    assert "projected_fixed_euler" not in source
