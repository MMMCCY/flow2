from __future__ import annotations

import inspect
import csv
import json
from pathlib import Path

import pytest
import torch

from guidance.geophysical_probability_bridge import (
    hard_condition_violation_count,
    scalar_gaussian_sample_bridge,
    shuffle_xy_probability,
    stable_logsumexp,
    validate_grid_alignment,
    validate_probabilities,
)
from scripts.stage10 import build_probability_bridge as builder
from scripts.stage10.common import load_stage10_inference_case
from scripts.stage10.common import load_frozen_config, validate_bridge_collection
from scripts.stage10.evaluate_bridge_information import average_precision, roc_auc
from guidance.prior_ensemble import load_tensor_gzip, source_seed
from guidance.seismic import tensor_sha256


def _class_model() -> dict[str, object]:
    return {
        "raw_labels": [-1, 0, 9],
        "log_impedance_mean": [-4.0, 0.0, 3.0],
        "log_impedance_sigma": [0.2, 0.4, 0.3],
        "class_prior": [0.2, 0.5, 0.3],
    }


def test_logsumexp_is_stable_for_extreme_finite_values() -> None:
    values = torch.tensor([[[-10000.0], [10000.0], [9999.0]]], dtype=torch.float64)
    actual = stable_logsumexp(values, dim=1)
    expected = torch.logsumexp(values, dim=1, keepdim=True)
    assert torch.equal(actual, expected)
    assert torch.isfinite(actual).all()


def test_scalar_gaussian_bridge_is_normalized_and_deterministic() -> None:
    samples = torch.tensor(
        [
            [[[[0.0, 3.0], [2.8, -4.0]]]],
            [[[[0.1, 2.9], [3.1, -4.1]]]],
        ],
        dtype=torch.float32,
    )
    first, first_entropy = scalar_gaussian_sample_bridge(samples, _class_model())
    second, second_entropy = scalar_gaussian_sample_bridge(samples, _class_model())
    assert torch.equal(first, second)
    assert torch.equal(first_entropy, second_entropy)
    checks = validate_probabilities(first)
    assert checks["normalization_max_error"] < 2e-6
    assert torch.isfinite(first).all()
    assert torch.isfinite(first_entropy).all()
    assert first[0, 2, 0, 0, 1] > first[0, 1, 0, 0, 1]
    assert first[0, 1, 0, 0, 0] > first[0, 2, 0, 0, 0]


def test_scalar_gaussian_bridge_rejects_nonfinite_samples() -> None:
    samples = torch.zeros((2, 1, 2, 2, 2))
    samples[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN/Inf"):
        scalar_gaussian_sample_bridge(samples, _class_model())


def test_shuffle_xy_is_deterministic_and_histogram_preserving() -> None:
    generator = torch.Generator().manual_seed(10)
    logits = torch.randn((1, 3, 4, 5, 2), generator=generator)
    probability = torch.softmax(logits, dim=1)
    first, first_permutation = shuffle_xy_probability(probability, seed=44)
    second, second_permutation = shuffle_xy_probability(probability, seed=44)
    different, _ = shuffle_xy_probability(probability, seed=45)
    assert torch.equal(first, second)
    assert torch.equal(first_permutation, second_permutation)
    assert not torch.equal(first, different)
    assert torch.equal(
        first.reshape(-1).sort().values,
        probability.reshape(-1).sort().values,
    )


def test_axis_grid_agreement_rejects_transpose() -> None:
    expected = torch.zeros((1, 1, 3, 4, 5))
    validate_grid_alignment(expected, expected.clone(), expected_shape=(3, 4, 5))
    with pytest.raises(ValueError, match="does not match"):
        validate_grid_alignment(expected.transpose(-1, -2), expected_shape=(3, 4, 5))


def test_hard_condition_preservation_counter() -> None:
    values = torch.tensor([[[[[0, 9], [2, -1]]]]])
    mask = torch.tensor([[[[[True, True], [False, True]]]]])
    labels = values.expand(3, -1, -1, -1, -1).clone()
    labels[:, :, :, 1, 0] = 9
    assert hard_condition_violation_count(labels, values, mask) == 0
    labels[1, 0, 0, 0, 1] = 0
    assert hard_condition_violation_count(labels, values, mask) == 1


def test_binary_rank_metrics_handle_ties() -> None:
    targets = torch.tensor([0, 1, 0, 1], dtype=torch.bool)
    constant = torch.full((4,), 0.5)
    assert average_precision(constant, targets) == pytest.approx(0.5)
    assert roc_auc(constant, targets) == pytest.approx(0.5)
    perfect = torch.tensor([0.1, 0.8, 0.2, 0.9])
    assert average_precision(perfect, targets) == pytest.approx(1.0)
    assert roc_auc(perfect, targets) == pytest.approx(1.0)


def test_inference_stage_signatures_cannot_accept_truth_files() -> None:
    functions = (
        scalar_gaussian_sample_bridge,
        load_stage10_inference_case,
        builder._load_fixed_prior_members,
        builder._build_case,
    )
    forbidden = {"truth", "truth_model", "truth_path", "truth_file", "truth_property"}
    for function in functions:
        parameters = set(inspect.signature(function).parameters)
        assert parameters.isdisjoint(forbidden), (function.__name__, parameters & forbidden)


def test_frozen_bridge_outputs_are_finite_normalized_and_condition_safe() -> None:
    config = load_frozen_config()
    bridges = validate_bridge_collection(config)
    root = Path(__file__).resolve().parents[1] / "experiments/stage10_geophysical_probability_bridge"
    for case_id, (manifest, tensors) in bridges.items():
        validate_probabilities(tensors["probability_all_classes"])
        for tensor in tensors.values():
            assert torch.isfinite(tensor).all(), case_id
        assert all(
            int(record["candidate_index"]) in range(100, 112)
            for record in manifest["candidate_records"]
        )
        with (root / "bridge" / case_id / "member_inversion_metrics.csv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 12
        assert all(int(row["condition_violations"]) == 0 for row in rows)


def test_stage10_b0_registration_matches_canonical_stage9_flow_only_asset() -> None:
    """Regression anchor for B0 even though the failed A gate forbids running B."""
    project = Path(__file__).resolve().parents[1]
    repository = project.parents[1]
    stage10 = json.loads(
        (project / "experiments/stage10_geophysical_probability_bridge/configs/frozen_experiment_config.json").read_text()
    )
    seed_bank = json.loads(
        (project / "experiments/stage10_geophysical_probability_bridge/configs/flow_seed_bank.json").read_text()
    )
    stage9 = json.loads(
        (project / "experiments/stage9_flow_prior_posterior/configs/stage9a_prior_support_v1.json").read_text()
    )
    for field in ("checkpoint", "checkpoint_sha256", "integrator", "n_euler_steps", "condition_projection", "hard_decode"):
        assert stage10[field] == stage9[field]
    case_id = "native_seed20260901"
    assert seed_bank["cases"][case_id]["pilot"][0] == source_seed(
        stage9, case_index=0, candidate_index=0, mode="formal"
    )
    pool = project / "experiments/stage9_flow_prior_posterior/runs/stage9a_prior_support_v1/formal" / case_id / "pool"
    manifest = json.loads((pool / "manifest.json").read_text())
    first_chunk = manifest["model_chunks"][0]
    models = load_tensor_gzip(pool / first_chunk["path"], expected=first_chunk)
    with (pool / "candidate_pool.csv").open("r", encoding="utf-8", newline="") as stream:
        first_row = next(csv.DictReader(stream))
    assert int(first_row["source_seed"]) == seed_bank["cases"][case_id]["pilot"][0]
    assert tensor_sha256(models[0:1]) == first_row["hard_model_sha256"]
