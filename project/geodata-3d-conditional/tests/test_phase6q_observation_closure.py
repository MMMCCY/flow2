from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from guidance.observation_closure_audit import (
    closure_metrics,
    epsilon_safe_attainment,
    response_geometry,
)


def test_truth_observation_closure_is_exact() -> None:
    observation = torch.arange(24, dtype=torch.float64).reshape(1, 1, 2, 3, 4)
    result = closure_metrics(observation, observation.clone())
    assert result["exact_tensor_equal"] is True
    assert result["truth_loss"] == 0.0
    assert result["relative_difference_l2"] == 0.0
    assert result["truth_observation_hash"] == result["recomputed_truth_observation_hash"]


def test_baseline_truth_response_separation_reports_raw_and_normalized_metrics() -> None:
    truth = torch.tensor([0.0, 1.0, 2.0, 3.0])
    baseline = torch.tensor([0.0, 0.5, 1.0, 1.5])
    result = response_geometry(baseline, truth)
    assert result["raw_rmse"] > 0
    assert result["response_l2_distance"] > 0
    assert result["normalized_mse_by_truth_energy"] > 0
    assert result["response_cosine"] == pytest.approx(1.0)


def test_soft_and_hard_attainment_denominators_are_reference_consistent() -> None:
    assert epsilon_safe_attainment(4.0, 2.0, 0.0) == pytest.approx(0.5)
    assert epsilon_safe_attainment(10.0, 5.0, 2.0) == pytest.approx(0.625)
    assert torch.isnan(torch.tensor(epsilon_safe_attainment(1.0, 1.0, 1.0)))
