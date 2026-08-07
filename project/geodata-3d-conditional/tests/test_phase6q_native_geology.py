from __future__ import annotations
from pathlib import Path
import sys
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
STRUCTURALGEO_SRC = PROJECT_DIR.parents[1] / "StructuralGeo-main" / "src"
for path in (PROJECT_DIR, STRUCTURALGEO_SRC):
    if str(path) not in sys.path: sys.path.insert(0, str(path))

from guidance.native_geology_audit import build_structuralgeo_native_case, connected_target_statistics


def test_native_case_has_three_drilled_two_hidden_event_masks() -> None:
    case, metadata = build_structuralgeo_native_case(seed=20260807)
    assert case.body_masks.shape == (5, 64, 64, 64)
    assert metadata["event_roles"] == ["drilled"] * 3 + ["hidden"] * 2
    assert all(int(mask.sum()) > 0 for mask in case.body_masks)
    assert not bool((case.body_masks[3:] & case.condition_mask[0, 0]).any())
    assert set(torch.unique(case.truth_labels).tolist()) == {-1, 0, 9}


def test_connected_statistics_reports_label_frequency() -> None:
    labels = torch.zeros((1, 1, 4, 4, 4), dtype=torch.long)
    labels[..., 0, 0, 0] = 9; labels[..., 3, 3, 3] = 9
    stats = connected_target_statistics(labels, target_label=9, condition_mask=torch.zeros_like(labels, dtype=torch.bool))
    assert stats["component_count_6"] == 2
    assert stats["unconditioned_component_count"] == 2
    assert stats["raw_label_frequency"]["9"] == 2
