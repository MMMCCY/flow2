from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.stage2.run_property_guidance import PHASE2_PAIR_FIELDS
from scripts.stage2.visualize_property_guidance import _validate_phase2_pair


def _write_pair(root: Path) -> tuple[Path, Path]:
    baseline_dir = root / "baseline"
    guided_dir = root / "guided"
    baseline_dir.mkdir()
    guided_dir.mkdir()
    baseline = {field: f"same-{field}" for field in PHASE2_PAIR_FIELDS}
    baseline.update(
        {
            "alpha": 0.0,
            "max_post_projection_condition_violations": 0,
            "n_samples": 1,
            "run_status": "completed",
            "samples_written": 1,
            "target_label": 9,
        }
    )
    guided = dict(baseline)
    guided.update(
        {
            "alpha": 0.25,
            "pairing_validation": {"paired": True},
        }
    )
    for directory, config in ((baseline_dir, baseline), (guided_dir, guided)):
        (directory / "config.json").write_text(
            json.dumps(config),
            encoding="utf-8",
        )
        torch.save(torch.zeros((1, 1, 2, 2, 2)), directory / "sample_0.pt")
    return baseline_dir, guided_dir


def test_phase2_visualization_requires_a_completed_strict_pair(tmp_path: Path) -> None:
    baseline_dir, guided_dir = _write_pair(tmp_path)

    baseline, guided = _validate_phase2_pair(baseline_dir, guided_dir, 9)

    assert baseline["alpha"] == 0.0
    assert guided["alpha"] == 0.25


def test_phase2_visualization_rejects_changed_property_target(tmp_path: Path) -> None:
    baseline_dir, guided_dir = _write_pair(tmp_path)
    config_path = guided_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["target_properties_sha256"] = "different"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="target_properties_sha256"):
        _validate_phase2_pair(baseline_dir, guided_dir, 9)
