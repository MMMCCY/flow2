from __future__ import annotations

import json
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]

from guidance.seismic_attribute_probability import (
    apply_probability_lookup,
    depth_resample_local_energy,
    fit_empirical_probability_lookup,
    local_seismic_energy,
)
from scripts.stage15.run_direct_seismic_attribute_probability import (
    build_attribute,
    contiguous_subsurface_from_geology,
)


def test_local_energy_is_centered_squared_trace_without_lateral_mixing() -> None:
    seismic = torch.zeros((1, 1, 2, 1, 5))
    seismic[0, 0, 0, 0, 2] = 3.0
    seismic[0, 0, 1, 0, 2] = 4.0
    energy = local_seismic_energy(seismic, 3)
    assert torch.equal(energy[0, 0, 0, 0], torch.tensor([0.0, 3.0, 3.0, 3.0, 0.0]))
    assert torch.allclose(
        energy[0, 0, 1, 0], torch.tensor([0.0, 16 / 3, 16 / 3, 16 / 3, 0.0])
    )


def test_fixed_velocity_time_to_depth_uses_local_surface_voxel_centers() -> None:
    energy = torch.arange(10, dtype=torch.float32).view(1, 1, 1, 1, 10)
    subsurface = torch.tensor([[[[[True, True, True, False]]]]])
    attribute = depth_resample_local_energy(
        energy,
        subsurface,
        sample_interval_ms=1.0,
        vertical_cell_size_m=1.0,
        background_velocity_m_s=1000.0,
    )
    # Cell TWT is 2 ms: top/middle/bottom voxel centers are 1/3/5 ms.
    assert torch.equal(attribute.flatten(), torch.tensor([5.0, 3.0, 1.0, 0.0]))


def test_runner_attribute_uses_frozen_time_to_depth_config_key() -> None:
    seismic = torch.arange(10, dtype=torch.float32).view(1, 1, 1, 1, 10)
    subsurface = torch.tensor([[[[[True, True, True, False]]]]])
    protocol = {
        "attribute": {"window_num_samples": 1},
        "time_to_depth": {
            "vertical_cell_size_m": 1.0,
            "background_velocity_m_s": 1000.0,
        },
    }
    parameters = {"time_sampling": {"sample_interval_ms": 1.0}}
    result = build_attribute(
        seismic, subsurface, protocol=protocol, seismic_parameters=parameters
    )
    assert result.shape == subsurface.shape


def test_calibration_subsurface_fills_internal_air_below_local_surface() -> None:
    geology = torch.tensor([[[[[0, -1, 9, -1]]]]])
    support = contiguous_subsurface_from_geology(geology)
    assert torch.equal(support.flatten(), torch.tensor([True, True, True, False]))


def test_laplace_lookup_preserves_natural_counts_without_balancing() -> None:
    values = torch.tensor([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
    labels = torch.tensor([False, False, True, False, True, True])
    edges = torch.tensor([0.0, 0.5, 1.0])
    probability, total, positive = fit_empirical_probability_lookup(values, labels, edges)
    assert total.tolist() == [3, 3]
    assert positive.tolist() == [1, 2]
    assert torch.allclose(probability, torch.tensor([2 / 5, 3 / 5]))


def test_probability_lookup_stays_continuous_and_zero_outside_subsurface() -> None:
    attribute = torch.tensor([[[[[0.1, 0.7, 0.9]]]]])
    subsurface = torch.tensor([[[[[True, True, False]]]]])
    probability = apply_probability_lookup(
        attribute,
        subsurface,
        torch.tensor([0.0, 0.5, 1.0]),
        torch.tensor([0.25, 0.75]),
    )
    assert torch.equal(probability.flatten(), torch.tensor([0.25, 0.75, 0.0]))


def test_stage15_c_protocol_freezes_128_unique_new_seeds_and_fixed_settings() -> None:
    path = (
        PROJECT_DIR
        / "experiments/stage15_binary_seismic_consensus/configs/direct_seismic_attribute_probability_v1.json"
    )
    config = json.loads(path.read_text(encoding="utf-8"))
    seeds = config["calibration_seeds"]
    assert len(seeds) == len(set(seeds)) == 128
    assert seeds == list(range(15150000, 15150128))
    assert config["attribute"]["window_num_samples"] == 17
    assert config["calibration"]["quantile_bin_count"] == 64
    assert config["calibration"]["class_balancing"] is False
    assert config["calibration_subsurface_policy"] == "topmost_non_air_column_fill_below_v1"
    assert config["diagnostic_thresholds"] == {"positive": 0.8, "negative": 0.2}


def test_stage15_c_runner_is_heldout_truth_blind_and_evaluator_owns_truth() -> None:
    runner = (
        PROJECT_DIR / "scripts/stage15/run_direct_seismic_attribute_probability.py"
    ).read_text(encoding="utf-8")
    evaluator = (
        PROJECT_DIR / "scripts/stage15/evaluate_direct_seismic_attribute_truth.py"
    ).read_text(encoding="utf-8")
    assert "true_model.pt" not in runner
    assert "truth_restricted" not in runner
    assert "phase1_assets" not in runner
    assert "phase1_assets" in evaluator
    assert "cond_generation_0_truth" in evaluator
