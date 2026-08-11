from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

import pytest
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from guidance.prior_ensemble import rank_scores
from guidance.structured_posterior import controlled_observations
from scripts.stage9 import audit_prior_truth as truth_auditor
from scripts.stage9.audit_prior_ranking import (
    compute_rankings,
    validate_completed_pool,
)
from scripts.stage9.run_prior_ensemble import load_inference_case


def test_inference_and_ranking_public_apis_cannot_receive_truth():
    for target in (load_inference_case, compute_rankings, rank_scores):
        assert "truth" not in inspect.signature(target).parameters


def test_ranking_is_deterministic_with_candidate_id_tie_break():
    scores = {
        "candidate_000002": {name: 0.5 for name in ("correct", "zero", "shuffled_xy", "wrong_case")},
        "candidate_000001": {name: 0.5 for name in ("correct", "zero", "shuffled_xy", "wrong_case")},
        "candidate_000000": {name: 0.1 for name in ("correct", "zero", "shuffled_xy", "wrong_case")},
    }
    rankings = rank_scores(scores)
    for rows in rankings.values():
        assert [row["candidate_id"] for row in rows] == [
            "candidate_000000",
            "candidate_000001",
            "candidate_000002",
        ]
        assert [row["rank"] for row in rows] == [1, 2, 3]


def test_changing_truth_cannot_change_inference_ranking():
    scores = {
        f"candidate_{index:06d}": {
            "correct": float(index),
            "zero": float(3 - index),
            "shuffled_xy": float(index % 2),
            "wrong_case": float(index + 1),
        }
        for index in range(4)
    }
    arbitrary_truth_a = torch.zeros(4, 4, 4)
    arbitrary_truth_b = torch.full((4, 4, 4), 9)
    first = rank_scores(scores)
    del arbitrary_truth_a
    second = rank_scores(scores)
    del arbitrary_truth_b
    assert first == second


def test_control_observations_include_exact_four_arms_and_replay():
    correct = torch.arange(48, dtype=torch.float32).reshape(1, 1, 3, 4, 4)
    wrong = -correct
    first = controlled_observations(correct, wrong_case=wrong, shuffle_seed=77)
    second = controlled_observations(correct, wrong_case=wrong, shuffle_seed=77)
    normalized = {
        "wrong_case" if key == "wrong_case_observation" else key: value
        for key, value in first.items()
    }
    assert set(normalized) == {"correct", "zero", "shuffled_xy", "wrong_case"}
    assert all(torch.equal(first[key], second[key]) for key in first)
    assert torch.count_nonzero(first["zero"]) == 0
    assert torch.equal(first["wrong_case_observation"], wrong)


def test_incomplete_pool_is_refused(tmp_path: Path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema": "stage9a_inference_visible_candidate_pool_v1", "status": "running"}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        validate_completed_pool(tmp_path)


def test_truth_loader_is_unreachable_when_freeze_validation_fails(monkeypatch):
    opened = False

    def fail_validation(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("ranking incomplete")

    def forbidden_open(*args, **kwargs):
        del args, kwargs
        nonlocal opened
        opened = True
        raise AssertionError("truth must not open")

    monkeypatch.setattr(truth_auditor, "validate_frozen_pool_and_rankings", fail_validation)
    monkeypatch.setattr(truth_auditor, "load_retrospective_case", forbidden_open)
    with pytest.raises(RuntimeError, match="ranking incomplete"):
        truth_auditor.validate_then_load_retrospective(
            Path("pool"), Path("ranking"), Path("retrospective")
        )
    assert opened is False
