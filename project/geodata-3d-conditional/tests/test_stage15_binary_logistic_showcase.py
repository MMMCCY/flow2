from __future__ import annotations

from scripts.stage15.build_binary_logistic_showcase import select_showcase


def test_showcase_selection_uses_largest_absolute_auprc_gain() -> None:
    rows = [
        {"root_seed": "1", "label9_voxels": "10", "prevalence": "0.01", "auprc": "0.04"},
        {"root_seed": "2", "label9_voxels": "20", "prevalence": "0.10", "auprc": "0.15"},
        {"root_seed": "3", "label9_voxels": "0", "prevalence": "0", "auprc": ""},
    ]
    assert select_showcase(rows)["root_seed"] == "2"
