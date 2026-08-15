from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]

from guidance.inversion_score_probability import (
    coarse_truth_occupancy_8,
    upsample_inversion_score,
)
from guidance.seismic_attribute_probability import (
    apply_probability_lookup,
    fit_empirical_probability_lookup,
    quantile_bin_edges,
)
from scripts.stage15.evaluate_inversion_score_probability_truth import (
    binary_metrics,
    truth_partition,
)
from scripts.stage15.run_inversion_score_probability import (
    invert_observation,
    validate_protocol,
)


def test_8_cubed_score_repeats_each_cell_to_8_cubed_fine_voxels() -> None:
    coarse = torch.zeros((1, 1, 8, 8, 8))
    coarse[0, 0, 1, 2, 3] = 0.75
    fine = upsample_inversion_score(coarse)
    assert fine.shape == (1, 1, 64, 64, 64)
    assert torch.equal(fine[0, 0, 8:16, 16:24, 24:32], torch.full((8, 8, 8), 0.75))
    assert int((fine == 0.75).sum()) == 8**3


def test_8_cubed_truth_occupancy_is_subsurface_restricted() -> None:
    truth = torch.zeros((1, 1, 64, 64, 64), dtype=torch.bool)
    support = torch.zeros_like(truth)
    support[..., :16] = True
    truth[0, 0, 0:8, 0:8, 0:4] = True
    occupancy, presence, counts = coarse_truth_occupancy_8(truth, support)
    assert occupancy[0, 0, 0, 0, 0] == 0.5
    assert bool(presence[0, 0, 0, 0, 0])
    assert counts[0, 0, 0, 0, 0] == 8**3
    assert counts[0, 0, 0, 0, 2] == 0


def test_empirical_lookup_maps_higher_inversion_score_to_higher_p9() -> None:
    score = torch.tensor([0.1, 0.2, 0.8, 0.9])
    label = torch.tensor([False, False, True, True])
    edges = quantile_bin_edges(score, 2)
    lookup, totals, positives = fit_empirical_probability_lookup(score, label, edges)
    volume = score.reshape(1, 1, 1, 1, 4)
    probability = apply_probability_lookup(volume, torch.ones_like(volume, dtype=torch.bool), edges, lookup)
    assert totals.tolist() == [2, 2]
    assert positives.tolist() == [0, 2]
    assert probability[0, 0, 0, 0, 0] < probability[0, 0, 0, 0, -1]


def test_truth_partition_and_positive_metrics_are_exact() -> None:
    probability = torch.tensor([0.9, 0.5, 0.1, 0.9]).reshape(1, 1, 1, 1, 4)
    truth = torch.tensor([True, True, True, False]).reshape_as(probability)
    domain = torch.ones_like(truth)
    partition = truth_partition(probability, truth)
    assert partition["counts"] == {"positive": 1, "unknown": 1, "negative": 1}
    assert partition["exactly_partitions_truth"]
    metrics = binary_metrics(probability >= 0.8, truth, domain)
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == pytest.approx(1 / 3)
    assert metrics["iou"] == 0.25


def test_stage15_f_protocol_freezes_simple_model_level_split() -> None:
    path = (
        PROJECT_DIR
        / "experiments/stage15_binary_seismic_consensus/configs/inversion_score_probability_8x8x8_v1.json"
    )
    config = json.loads(path.read_text(encoding="utf-8"))
    seeds = validate_protocol(config)
    assert seeds[:2] == [15180000, 15180001]
    assert seeds[-1] == 15180127
    assert len(seeds[:96]) == 96
    assert len(seeds[96:]) == 32
    assert config["regularization"] is None
    assert config["parameter_sweep"] is False
    assert config["flow_used"] is False


def test_heldout_inversion_api_cannot_receive_truth_or_flow() -> None:
    signature = inspect.signature(invert_observation)
    assert set(signature.parameters) == {
        "observed",
        "subsurface",
        "operator",
        "properties",
        "learning_rate",
        "iterations",
    }
    source = inspect.getsource(invert_observation)
    assert "truth" not in source
    assert "flow" not in source.lower()
    assert "checkpoint" not in source.lower()
