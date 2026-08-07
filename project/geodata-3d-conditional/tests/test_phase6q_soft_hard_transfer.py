from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from guidance.soft_hard_transfer_audit import (
    paired_attainment,
    projection_erasure_fraction,
    response_distance_geometry,
    spatial_energy_fractions,
)


def test_response_closer_classification_and_direction() -> None:
    baseline = torch.tensor([0.0, 0.0])
    truth = torch.tensor([2.0, 0.0])
    result = response_distance_geometry(
        baseline=baseline, guided=torch.tensor([1.5, 0.0]), truth=truth
    )
    assert result["closer_to"] == "truth"
    assert result["truth_direction_fraction"] == pytest.approx(0.75)


def test_attainment_never_mixes_soft_and_hard_denominators() -> None:
    result = paired_attainment(
        soft_baseline_loss=8.0,
        guided_soft_loss=4.0,
        soft_truth_loss=0.0,
        hard_baseline_loss=20.0,
        guided_hard_loss=15.0,
        hard_truth_loss=0.0,
    )
    assert result == {"soft_attainment": 0.5, "hard_attainment": 0.25}


def test_spatial_energy_and_projection_erasure() -> None:
    value = torch.tensor([[[[[1.0, 3.0]]]]])
    regions = {
        "left": torch.tensor([[[[[True, False]]]]]),
        "right": torch.tensor([[[[[False, True]]]]]),
    }
    result = spatial_energy_fractions(value, regions)
    assert result["left_energy_fraction"] == pytest.approx(0.1)
    assert result["right_energy_fraction"] == pytest.approx(0.9)
    assert projection_erasure_fraction(
        loss_before=10.0, loss_pre_projection=6.0, loss_post_projection=8.0
    ) == pytest.approx(0.5)
