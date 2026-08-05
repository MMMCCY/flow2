from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.stage1.summarize_phase1b import component_size_summary


def test_component_report_separates_major_bodies_from_tiny_fragments() -> None:
    mask = torch.zeros((1, 1, 12, 12, 12), dtype=torch.bool)
    mask[:, :, 1:4, 1:4, 1:4] = True  # 27 voxels
    mask[:, :, 6:8, 6:8, 6:8] = True  # 8 voxels
    mask[:, :, 10, 10, 10] = True  # one-voxel fragment

    summary = component_size_summary(mask)

    assert summary["component_count"] == 3
    assert summary["components_ge_20"] == 1
    assert summary["components_ge_5"] == 2
    assert summary["top_component_sizes"][:3] == [27, 8, 1]
    assert summary["tiny_component_mass_le_5"] == 1
    assert summary["tiny_component_mass_fraction_le_5"] == pytest.approx(1 / 36)

