from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.stage1.visualize_probability_guidance import (
    _cutaway_keep,
    _image_grid,
    paired_change_masks,
)


def test_cutaway_removes_one_quadrant_and_preserves_shape() -> None:
    keep = _cutaway_keep((8, 8, 8), cut_fraction=0.5)

    assert keep.shape == (8, 8, 8)
    assert keep.dtype == np.bool_
    assert int(keep.sum()) == 384
    assert keep[1, 1, 1]
    assert not keep[6, 1, 1]
    assert keep[6, 6, 1]


def test_vtk_cell_order_preserves_label_counts_and_coordinates() -> None:
    pytest.importorskip("pyvista")
    values = torch.zeros((4, 5, 6), dtype=torch.int16).numpy()
    values[1:3, 2:5, 3:6] = 9
    values[3, 1, 4] = 7
    grid = _image_grid(values, "label", values)

    restored = np.asarray(grid.cell_data["label"]).reshape(values.shape, order="F")
    assert grid.n_cells == values.size
    assert int((restored == 9).sum()) == int((values == 9).sum())
    assert restored[3, 1, 4] == 7
    assert np.array_equal(restored, values)


def test_paired_change_masks_are_auditable_and_roi_limited() -> None:
    baseline = np.zeros((3, 3, 3), dtype=np.int64)
    guided = baseline.copy()
    target = np.zeros_like(baseline, dtype=bool)
    roi = np.zeros_like(target)
    roi[:2] = True
    target[0, 0, 0] = True
    target[0, 0, 1] = True
    baseline[0, 0, 1] = 9
    guided[0, 0, 0] = 9  # recovered
    guided[0, 0, 1] = 0  # lost
    guided[1, 0, 0] = 9  # false addition inside ROI
    guided[2, 0, 0] = 9  # target-related change outside ROI

    masks = paired_change_masks(baseline, guided, target, roi, target_label=9)

    assert int(masks["correct_target_recovered"].sum()) == 1
    assert int(masks["correct_target_lost"].sum()) == 1
    assert int(masks["false_target_added"].sum()) == 1
    assert int(masks["false_target_removed"].sum()) == 0
    assert int(masks["all_hard_changes_inside_roi"].sum()) == 3
    assert int(masks["target_related_changes"].sum()) == 3
