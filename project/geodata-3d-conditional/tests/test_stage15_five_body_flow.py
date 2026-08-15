from __future__ import annotations

import json
from pathlib import Path

from scripts.stage15.build_five_body_flow_case import body_masks


ROOT = Path(__file__).resolve().parents[1] / "experiments/stage15_five_body_flow"


def test_frozen_five_body_geometry_is_equal_disjoint_and_all_label9() -> None:
    protocol = json.loads((ROOT / "configs/frozen_protocol_v1.json").read_text())
    masks, bodies = body_masks(protocol)
    assert masks.shape == (5, 64, 64, 64)
    assert [int(mask.sum()) for mask in masks] == [640] * 5
    assert int(masks.sum(0).max()) == 1
    assert [body["role"] for body in bodies].count("drilled") == 3
    assert [body["role"] for body in bodies].count("hidden") == 2
    assert protocol["target_label"] == 9
