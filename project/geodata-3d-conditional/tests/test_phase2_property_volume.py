from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from guidance.property_volume import (
    PROPERTY_CONFIG_SCHEMA,
    gaussian_blur_property_channels,
    hard_labels_to_properties,
    matched_multiscale_property_loss,
    probabilities_to_expected_properties,
    property_table_from_config,
    property_volume_loss,
)


def _config() -> dict[str, object]:
    return {
        "schema": PROPERTY_CONFIG_SCHEMA,
        "description": "small explicit Phase-2 test codebook",
        "channels": [
            {
                "name": "density",
                "unit": "relative",
                "weight": 2.0,
                "values": {"-1": 0.0, "0": 1.0, "1": 3.0},
            },
            {
                "name": "susceptibility",
                "unit": "relative",
                "weight": 1.0,
                "values": {"-1": 0.0, "0": 4.0, "1": 2.0},
            },
        ],
    }


def test_property_config_requires_complete_raw_label_coverage() -> None:
    table, weights, metadata = property_table_from_config(_config(), num_categories=3)

    assert table.tolist() == [[0.0, 1.0, 3.0], [0.0, 4.0, 2.0]]
    assert weights.tolist() == pytest.approx([2 / 3, 1 / 3])
    assert metadata["raw_label_range"] == [-1, 1]
    assert metadata["truth_derived"] is True
    assert metadata["is_measured_geophysics"] is False

    invalid = _config()
    invalid["channels"] = [
        {"name": "density", "values": {"-1": 0.0, "1": 3.0}}
    ]
    with pytest.raises(ValueError, match="label coverage mismatch"):
        property_table_from_config(invalid, num_categories=3)


def test_label9_contrast_ablation_config_is_complete_and_explicit() -> None:
    config_path = (
        PROJECT_DIR
        / "experiments"
        / "stage2_property"
        / "configs"
        / "ideal_density_susceptibility_label9_contrast_v1.json"
    )
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    table, weights, metadata = property_table_from_config(
        config,
        num_categories=15,
    )

    assert table.shape == (2, 15)
    assert weights.tolist() == pytest.approx([0.5, 0.5])
    assert metadata["raw_label_range"] == [-1, 13]
    assert metadata["truth_derived"] is True
    assert metadata["is_measured_geophysics"] is False
    # Raw label k occupies embedding/property-table column k + 1.
    assert table[1, 9 + 1].item() == pytest.approx(0.1)
    other_rock_values = torch.cat((table[1, 1:10], table[1, 11:]))
    assert table[1, 10] > 18 * other_rock_values.max()


def test_hard_and_soft_property_mapping_use_raw_label_offset() -> None:
    table, _, _ = property_table_from_config(_config(), num_categories=3)
    labels = torch.tensor([-1, 0, 1]).reshape(1, 1, 1, 1, 3)
    hard = hard_labels_to_properties(labels, table)
    probabilities = torch.nn.functional.one_hot(
        (labels[:, 0] + 1).long(),
        num_classes=3,
    ).permute(0, 4, 1, 2, 3).float()
    expected = probabilities_to_expected_properties(probabilities, table)

    assert hard.shape == (1, 2, 1, 1, 3)
    assert torch.equal(hard, expected)
    assert hard[0, 0, 0, 0].tolist() == [0.0, 1.0, 3.0]
    assert hard[0, 1, 0, 0].tolist() == [0.0, 4.0, 2.0]


def test_multichannel_gaussian_blur_does_not_mix_properties() -> None:
    volume = torch.zeros((1, 2, 7, 7, 7))
    volume[:, 0, 3, 3, 3] = 1.0
    blurred = gaussian_blur_property_channels(volume, sigma=1.0)

    assert blurred[:, 0].max() > 0
    assert torch.count_nonzero(blurred[:, 1]) == 0
    assert blurred[:, 0].sum() == pytest.approx(1.0, rel=1e-5)


def test_matched_multiscale_loss_is_zero_for_exact_target_and_unit_invariant() -> None:
    target = torch.zeros((1, 2, 7, 7, 7))
    target[:, 0, 1:4, 2:5, 3:6] = 2.0
    target[:, 1, 3:6, 1:4, 2:5] = 5.0
    predicted = target.clone()
    confidence = torch.ones((1, 1, 7, 7, 7))

    exact, _ = matched_multiscale_property_loss(
        predicted,
        target,
        confidence,
        sigmas=(0.0, 1.0),
        scale_weights=(0.6, 0.4),
    )
    perturbed = predicted.clone()
    perturbed[:, :, 0:2] += 0.4
    base, _ = matched_multiscale_property_loss(
        perturbed,
        target,
        confidence,
        sigmas=(0.0, 1.0),
        scale_weights=(0.6, 0.4),
    )
    rescaled, _ = matched_multiscale_property_loss(
        perturbed * 10.0,
        target * 10.0,
        confidence,
        sigmas=(0.0, 1.0),
        scale_weights=(0.6, 0.4),
    )

    assert exact == pytest.approx(0.0, abs=1e-8)
    assert base > 0
    assert rescaled == pytest.approx(float(base), rel=1e-5)


def test_property_loss_has_finite_gradient_and_respects_zero_confidence_at_full_scale() -> None:
    embedding = torch.eye(3)
    state = torch.randn((1, 3, 4, 4, 4), requires_grad=True)
    table, channel_weights, _ = property_table_from_config(_config(), num_categories=3)
    truth = torch.zeros((1, 1, 4, 4, 4), dtype=torch.long)
    truth[:, :, 2:] = 1
    target = hard_labels_to_properties(truth, table)
    confidence = torch.ones((1, 1, 4, 4, 4))
    confidence[:, :, 0, 0, 0] = 0.0

    loss, diagnostics = property_volume_loss(
        state,
        embedding,
        target,
        table,
        confidence,
        tau=0.25,
        sigmas=(0.0,),
        scale_weights=(1.0,),
        channel_weights=channel_weights,
    )
    gradient = torch.autograd.grad(loss, state)[0]

    assert torch.isfinite(loss)
    assert loss > 0
    assert torch.isfinite(gradient).all()
    assert gradient.norm() > 0
    assert torch.count_nonzero(gradient[:, :, 0, 0, 0]) == 0
    assert diagnostics["property_mae_mean"] > 0
