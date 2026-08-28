"""Focused CPU regressions for frozen Stage18 evidence semantics."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
for root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from guidance.binary_seismic_inversion import BinaryAcousticProperties
from guidance.probability_volume import tensor_sha256
from scripts.stage15.common import refuse_nonempty
from scripts.stage17.common import read_csv
from scripts.stage18.common import (
    NEW_ARM,
    hard_seismic_metrics,
    load_stage18_references,
    validate_stage18_config,
)


CONFIG_PATH = (
    PROJECT_DIR
    / "experiments/stage18_evidence_semantics/configs/continuous_property_target_v1.json"
)
RUNNER_PATH = PROJECT_DIR / "scripts/stage18/run_continuous_property_target.py"


def _config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_frozen_config_changes_target_only_and_keeps_stage17_confidence() -> None:
    config = _config()
    validate_stage18_config(config)
    assert config["authorized_arm"] == NEW_ARM
    assert config["target_properties_policy"] == "binary_impedance_score_no_threshold"
    assert config["confidence_policy"] == (
        "binary_impedance_score_times_free_subsurface_fixed_from_stage17c"
    )
    assert config["spatial_confidence_weighting_held_fixed"] is True
    assert config["thresholding"] is False
    assert config["score_rescaling"] is False


def test_config_rejects_reexpanded_low_score_background_weight() -> None:
    config = _config()
    config["confidence_policy"] = "free_subsurface"
    config["spatial_confidence_weighting_held_fixed"] = False
    with pytest.raises(ValueError, match="confidence_policy|spatial_confidence"):
        validate_stage18_config(config)


def test_runner_constructs_score_target_and_score_weighted_confidence_without_threshold() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "target_properties = score.to" in source
    assert "confidence = (score * free_subsurface_cpu.float()).to" in source
    # No comparison may turn the evidence score into a binary target/mask.
    score_thresholds = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        names = {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}
        if "score" in names:
            score_thresholds.append(node)
    assert score_thresholds == []


def test_stage17_reference_hashes_and_initial_noise_match() -> None:
    stage17, _, records = load_stage18_references(_config())
    assert len(stage17["cases"]) == 5
    assert len(records) == 30
    expected = {
        42: "5b3e3cc13e70f1477679559ea82781e02fc58130469ff39d5c71b5db647f9eea",
        142: "9e3fe8095b7f2705dfa4353f4606f4b4cb9d6a00c3ed5cbb0d450674e650e9cb",
        242: "e6f10761b6eee07b0a77a58fc4f2170c9a107a156a7d8db89b83ebc4b1ac2c49",
    }
    for seed, digest in expected.items():
        generator = torch.Generator(device="cpu").manual_seed(seed)
        initial = torch.randn(1, 15, 64, 64, 64, generator=generator)
        assert tensor_sha256(initial) == digest
        assert {
            row["initial_noise_sha256"]
            for row in records
            if int(row["source_seed"]) == seed
        } == {digest}


def test_nonempty_output_is_refused(tmp_path: Path) -> None:
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "existing.txt").write_text("immutable", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        refuse_nonempty(output)


def test_binary_hard_seismic_uses_decoded_label9_occupancy() -> None:
    properties = BinaryAcousticProperties(
        air_density=1.0,
        air_velocity=1.0,
        background_density=2.0,
        background_velocity=3.0,
        target_density=4.0,
        target_velocity=5.0,
    )
    decoded = torch.full((1, 1, 64, 64, 64), 4, dtype=torch.long)
    decoded[..., 0, 0, 0] = 9
    support = torch.ones_like(decoded, dtype=torch.bool)

    class ImpedanceOperator:
        def __call__(self, impedance, slowness, mask):
            return impedance

    observed = torch.full(decoded.shape, 6.0)
    observed[..., 0, 0, 0] = 20.0
    metrics = hard_seismic_metrics(
        decoded,
        support,
        observed,
        properties,
        ImpedanceOperator(),
        torch.device("cpu"),
    )
    assert metrics["hard_seismic_mse"] == pytest.approx(0.0)
    assert metrics["hard_seismic_rmse"] == pytest.approx(0.0)


def test_soft_and_hard_mass_share_exact_free_subsurface_mask() -> None:
    support = torch.tensor([1, 1, 1, 0], dtype=torch.bool).view(1, 1, 1, 1, 4)
    condition = torch.tensor([0, 1, 0, 0], dtype=torch.bool).view(1, 1, 1, 1, 4)
    free = support & ~condition
    soft = torch.tensor([0.25, 0.9, 0.75, 1.0]).view(1, 1, 1, 1, 4)
    decoded = torch.tensor([9, 9, 4, 9]).view(1, 1, 1, 1, 4)
    assert float((soft * free.float()).sum()) == pytest.approx(1.0)
    assert int(((decoded == 9) & free).sum()) == 1


def test_stage18_runner_truth_firewall_is_static() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "truth_model=" in source  # sampler API receives observed condition values only
    assert "tensors[\"truth\"]" not in source
    assert "truth_loaded_by_runner\": False" in source
