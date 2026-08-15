from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]

from guidance.binary_seismic_inversion import (
    BinaryAcousticProperties,
    binary_occupancy_to_acoustic,
)
from guidance.coarse_binary_seismic import (
    coarse_truth_occupancy,
    upsample_coarse_occupancy,
)
from scripts.stage15.evaluate_coarse_binary_seismic_truth import (
    coarse_metrics,
    top_k_localization,
)


def test_4_cubed_upsampling_repeats_each_cell_to_16_cubed_fine_voxels() -> None:
    q = torch.zeros((1, 1, 4, 4, 4))
    q[0, 0, 1, 2, 3] = 0.75
    fine = upsample_coarse_occupancy(q)
    assert fine.shape == (1, 1, 64, 64, 64)
    selected = fine[0, 0, 16:32, 32:48, 48:64]
    assert torch.equal(selected, torch.full((16, 16, 16), 0.75))
    assert int((fine == 0.75).sum()) == 16**3


def test_coarse_acoustic_variable_never_modifies_air() -> None:
    q = upsample_coarse_occupancy(torch.full((1, 1, 4, 4, 4), 0.25))
    support = torch.ones_like(q, dtype=torch.bool)
    support[..., 56:] = False
    properties = BinaryAcousticProperties(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    impedance, slowness = binary_occupancy_to_acoustic(q, support, properties)
    assert torch.equal(impedance[~support], torch.full_like(impedance[~support], 2.0))
    assert torch.equal(slowness[~support], torch.full_like(slowness[~support], 0.5))


def test_coarse_truth_is_subsurface_restricted_unthresholded_block_mean() -> None:
    truth = torch.zeros((1, 1, 64, 64, 64), dtype=torch.bool)
    support = torch.zeros_like(truth)
    support[..., :32] = True
    truth[0, 0, 0:16, 0:16, 0:8] = True
    q_true, presence, counts = coarse_truth_occupancy(truth, support)
    assert q_true[0, 0, 0, 0, 0] == 0.5
    assert bool(presence[0, 0, 0, 0, 0])
    assert counts[0, 0, 0, 0, 0] == 16**3
    assert counts[0, 0, 0, 0, 2] == 0


def test_top_k_uses_number_of_truth_containing_cells() -> None:
    q = torch.tensor([0.9, 0.8, 0.2, 0.1]).reshape(1, 1, 1, 1, 4)
    presence = torch.tensor([True, False, True, False]).reshape_as(q)
    domain = torch.ones_like(presence)
    result = top_k_localization(q, presence, domain)
    assert result["k"] == 2
    assert result["overlap_count"] == 1
    assert result["overlap_fraction"] == 0.5


def test_perfect_coarse_ranking_has_perfect_localization_metrics() -> None:
    q_true = torch.arange(64, dtype=torch.float32).reshape(1, 1, 4, 4, 4) / 63
    presence = q_true > 0.5
    domain = torch.ones_like(presence)
    metrics = coarse_metrics(q_true, q_true, presence, domain)
    assert metrics["pearson_correlation"] == pytest.approx(1.0)
    assert metrics["spearman_correlation"] == pytest.approx(1.0)
    assert metrics["coarse_presence_auprc"] == 1.0
    assert metrics["top_k"]["overlap_fraction"] == 1.0
    assert metrics["centroid_distance_coarse_cells"] == 0.0


def test_stage15_e_config_is_frozen_without_regularization_or_sweep() -> None:
    path = (
        PROJECT_DIR
        / "experiments/stage15_binary_seismic_consensus/configs/coarse_binary_seismic_inversion_4x4x4_v1.json"
    )
    config = json.loads(path.read_text(encoding="utf-8"))
    assert config["coarse_grid_shape"] == [4, 4, 4]
    assert config["fine_voxels_per_coarse_cell"] == [16, 16, 16]
    assert len(config["inversion_seeds"]) == len(set(config["inversion_seeds"])) == 8
    assert config["regularization"] is None
    assert config["parameter_sweep"] is False
    assert config["flow_used"] is False


def test_stage15_e_runner_is_truth_and_flow_blind() -> None:
    source = (
        PROJECT_DIR / "scripts/stage15/run_coarse_binary_seismic_inversion.py"
    ).read_text(encoding="utf-8")
    assert "true_model" not in source
    assert "phase1_assets" not in source
    assert "load_checkpoint" not in source
    assert "load_flow" not in source
    assert "projected_fixed_euler" not in source
