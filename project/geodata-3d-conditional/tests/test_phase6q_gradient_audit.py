from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from guidance.causality_gradient_audit import (
    cosine_decode_categories,
    finite_difference_directional_audit,
    response_truth_direction_fraction,
    update_semantics_row,
)


def test_finite_difference_and_negative_gradient_descent() -> None:
    state = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    direction = torch.tensor([0.3, 0.4, -0.2], dtype=torch.float64)
    objective = lambda value: (value.square()).sum()
    summary, rows, gradient = finite_difference_directional_audit(
        name="quadratic",
        objective=objective,
        state=state,
        direction=direction,
        epsilons=[1e-2, 1e-3, 1e-4, 1e-5],
        negative_gradient_step_norm=1e-4,
    )
    assert summary["all_sign_match"] is True
    assert summary["best_relative_error"] < 1e-10
    assert summary["negative_gradient_local_descent"] is True
    assert torch.equal(gradient, 2.0 * state)
    assert len(rows) == 4


def test_cosine_decoder_matches_expected_argmax_and_tie_policy() -> None:
    embeddings = torch.eye(3, dtype=torch.float64)
    state = torch.tensor([[[[[0.5]]], [[[0.5]]], [[[0.0]]]]], dtype=torch.float64)
    categories = cosine_decode_categories(state, embeddings)
    assert categories.item() == 0


def test_truth_direction_projection_separates_parallel_and_orthogonal_motion() -> None:
    baseline = torch.tensor([0.0, 0.0])
    truth = torch.tensor([2.0, 0.0])
    fraction, orthogonal = response_truth_direction_fraction(
        baseline, torch.tensor([1.0, 1.0]), truth
    )
    assert fraction == pytest.approx(0.5)
    assert orthogonal == pytest.approx(0.5)


def test_actual_negative_update_sign_lowers_quadratic() -> None:
    state = torch.tensor([1.0, -1.0], dtype=torch.float64)
    objective = lambda value: value.square().mean()
    response = lambda value: value
    hard_loss = lambda value: float(value.square().mean())
    row = update_semantics_row(
        name="negative_gradient",
        state=state,
        update=torch.tensor([-0.1, 0.1], dtype=torch.float64),
        objective=objective,
        soft_response=response,
        hard_loss=hard_loss,
        baseline_response=state,
        truth_response=torch.zeros_like(state),
    )
    assert row["soft_loss_improved"] is True
    assert row["hard_loss_after"] < row["hard_loss_before"]
