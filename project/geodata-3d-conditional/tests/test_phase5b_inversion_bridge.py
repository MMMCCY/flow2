import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import inference_runtime as runtime
from guidance.inversion_bridge import (
    log_impedance_property_table,
    posterior_spread_confidence,
    property_config_from_table,
)
from guidance.probability_volume import tensor_sha256
from guidance.property_volume import property_table_from_config
from scripts.stage2.run_property_guidance import (
    _load_external_property_assets,
    validate_args,
)


def test_log_impedance_config_round_trip() -> None:
    acoustic = torch.tensor(
        [[1.0, 10.0, 100.0], [1.0, 0.5, 0.25]], dtype=torch.float32
    )
    table = log_impedance_property_table(acoustic).float()
    config = property_config_from_table(table, description="unit")
    parsed, weights, metadata = property_table_from_config(config, 3)
    assert torch.equal(parsed, table)
    assert weights.tolist() == [1.0]
    assert metadata["channel_names"] == ["log_acoustic_impedance"]


def test_spread_confidence_is_truth_blind_bounded_and_condition_zero() -> None:
    spread = torch.tensor([[[[[0.1, 0.2, 0.4, 0.0]]]]])
    subsurface = torch.tensor([[[[[True, True, True, False]]]]])
    condition = torch.tensor([[[[[True, False, False, True]]]]])
    confidence, metadata = posterior_spread_confidence(
        spread, subsurface, condition
    )
    # Active positive values are [0.2, 0.4], whose lower median is 0.2.
    assert metadata["spread_reference_median"] == pytest.approx(0.2)
    assert confidence.flatten().tolist() == pytest.approx(
        [0.0, 0.5, 0.2, 0.0]
    )
    assert metadata["active_voxels"] == 2


def _write_external_assets(directory: Path, *, violate_condition: bool = False) -> None:
    directory.mkdir()
    values = {
        "property_table.pt": torch.tensor([[0.0, 1.0, 2.0]]),
        "target_properties.pt": torch.tensor([[[[[1.0]], [[1.5]]]]]),
        "property_confidence.pt": torch.tensor(
            [[[[[1.0 if violate_condition else 0.0]], [[0.5]]]]]
        ),
        "condition_mask.pt": torch.tensor([[[[[True]], [[False]]]]]),
    }
    records = {}
    for filename, value in values.items():
        path = directory / filename
        torch.save(value, path)
        records[filename] = {
            "sha256": runtime.file_sha256(path),
            "tensor_sha256": tensor_sha256(value),
        }
    manifest = {
        "schema": "phase5b_inversion_property_assets_v1",
        "status": "complete",
        "truth_geology_loaded": False,
        "truth_acoustic_loaded": False,
        "truth_metrics_used_for_construction": False,
        "phase5a_pass_bit_used_as_stop_gate": True,
        "generated_tensors": records,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_external_asset_loader_validates_hashes_shapes_and_conditions(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid"
    _write_external_assets(valid)
    condition = torch.tensor([[[[[True]], [[False]]]]])
    target, confidence, manifest = _load_external_property_assets(
        valid,
        property_table=torch.tensor([[0.0, 1.0, 2.0]]),
        condition_mask=condition,
        expected_shape=(1, 1, 2, 1, 1),
    )
    assert target.shape == (1, 1, 2, 1, 1)
    assert confidence.sum() == pytest.approx(0.5)
    assert manifest["truth_metrics_used_for_construction"] is False

    invalid = tmp_path / "invalid"
    _write_external_assets(invalid, violate_condition=True)
    with pytest.raises(ValueError, match="zero at hard conditions"):
        _load_external_property_assets(
            invalid,
            property_table=torch.tensor([[0.0, 1.0, 2.0]]),
            condition_mask=condition,
            expected_shape=(1, 1, 2, 1, 1),
        )


def test_phase5b_runner_requires_external_assets_and_confidence() -> None:
    common = dict(
        model_weights="ema",
        n_samples=1,
        n_steps=32,
        alpha=0.0,
        baseline_dir=None,
        max_guidance_ratio=0.25,
        grad_clip_norm=1.0,
        guidance_start=0.25,
        target_roi_radius=6,
        output_dir=Path("unused"),
        experiment_stage="phase5b_inversion_property_bridge_v1",
        external_property_dir=None,
        confidence_mode="external_posterior_spread_v1",
    )
    with pytest.raises(ValueError, match="external-property-dir"):
        validate_args(SimpleNamespace(**common))
    common["external_property_dir"] = Path("bridge")
    common["confidence_mode"] = "unconditioned_nonair_v1"
    with pytest.raises(ValueError, match="external_posterior_spread_v1"):
        validate_args(SimpleNamespace(**common))
