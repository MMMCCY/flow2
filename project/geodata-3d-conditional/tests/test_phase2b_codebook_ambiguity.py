from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from guidance.property_volume import (
    property_codebook_diagnostics,
    property_table_from_config,
)
from scripts.stage2.run_property_guidance import PHASE2_EXPERIMENT_STAGES
from scripts.stage2.summarize_phase2b_n4_bracket import (
    classify_n4_level,
    n4_candidate_selection,
)
from scripts.stage2.summarize_phase2b_n4_fallback import (
    fallback_decision,
    validate_bracket_prerequisite,
)
from scripts.stage2.summarize_phase2b_screen import (
    EXPERIMENT_STAGE,
    _anchor_regression,
    load_manifest,
    promotion_recommendation,
)


CONFIG_DIR = (
    PROJECT_DIR
    / "experiments/stage2_property/configs/phase2b_codebook_ambiguity_v1"
)
MANIFEST_PATH = CONFIG_DIR / "sweep_manifest.json"


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _table(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    table, weights, _ = property_table_from_config(
        _read_json(path),
        num_categories=15,
    )
    return table, weights


def test_phase2b_manifest_freezes_order_stage_and_contrast() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    levels = manifest["levels"]

    assert EXPERIMENT_STAGE in PHASE2_EXPERIMENT_STAGES
    assert [level["id"] for level in levels] == [
        "distinct_c100_anchor",
        "paired_c100",
        "paired_c025",
        "paired_c010",
        "paired_c004_overlap",
    ]
    assert [level["label9_susceptibility"] for level in levels] == pytest.approx(
        [0.1, 0.1, 0.025, 0.01, 0.004]
    )
    assert manifest["fixed_settings"] == {
        "alpha": 0.25,
        "max_guidance_ratio": 0.25,
        "n_samples": 1,
        "n_steps": 32,
        "seed": 42,
        "target_label": 9,
    }


def test_phase2b_ambiguous_sweep_changes_only_label9_susceptibility() -> None:
    filenames = (
        "paired_c100_v1.json",
        "paired_c025_v1.json",
        "paired_c010_v1.json",
        "paired_c004_overlap_v1.json",
    )
    tables = [_table(CONFIG_DIR / filename)[0] for filename in filenames]

    for table in tables[1:]:
        assert torch.equal(table[0], tables[0][0])
        non_target = torch.ones(15, dtype=torch.bool)
        non_target[9 + 1] = False
        assert torch.equal(table[1, non_target], tables[0][1, non_target])
    assert [float(table[1, 9 + 1]) for table in tables] == pytest.approx(
        [0.1, 0.025, 0.01, 0.004]
    )


def test_phase2b_final_level_is_exact_label6_label9_collision() -> None:
    table, weights = _table(CONFIG_DIR / "paired_c004_overlap_v1.json")
    diagnostics = property_codebook_diagnostics(
        table,
        weights,
        target_raw_label=9,
    )

    assert torch.equal(table[:, 6 + 1], table[:, 9 + 1])
    assert diagnostics["target_exact_property_group"] == [6, 9]
    assert diagnostics["target_nearest_raw_labels"] == [6]
    assert diagnostics["target_nearest_range_normalized_distance"] == pytest.approx(0.0)


def test_phase2b_high_contrast_keeps_label9_distinct_despite_density_overlap() -> None:
    table, weights = _table(CONFIG_DIR / "paired_c100_v1.json")
    diagnostics = property_codebook_diagnostics(
        table,
        weights,
        target_raw_label=9,
    )

    assert table[0, 6 + 1] == pytest.approx(table[0, 9 + 1])
    assert diagnostics["target_exact_property_group"] == [9]
    assert 9 not in diagnostics["target_nearest_raw_labels"]
    assert diagnostics["target_nearest_range_normalized_distance"] > 0


def test_phase2b_promotion_rule_selects_most_degraded_pass_and_neighbor() -> None:
    levels = load_manifest(MANIFEST_PATH)["levels"]
    rows = [
        {
            "level_id": level["id"],
            "screen_gate_pass": level["id"]
            in {"distinct_c100_anchor", "paired_c100", "paired_c025"},
        }
        for level in levels
    ]

    result = promotion_recommendation(levels, rows, anchor_regression_pass=True)

    assert result["status"] == "candidate_identified_not_confirmed"
    assert result["selected_level"] == "paired_c025"
    assert result["promote_levels"] == ["paired_c025", "paired_c010"]


def test_phase2b_promotion_rule_brackets_exact_collision_with_previous_level() -> None:
    levels = load_manifest(MANIFEST_PATH)["levels"]
    rows = [
        {"level_id": level["id"], "screen_gate_pass": True}
        for level in levels
    ]

    result = promotion_recommendation(levels, rows, anchor_regression_pass=True)

    assert result["selected_level"] == "paired_c004_overlap"
    assert result["promote_levels"] == ["paired_c004_overlap", "paired_c010"]


def test_phase2b_anchor_regression_allows_only_tiny_guided_repeat_difference(
    tmp_path: Path,
) -> None:
    pair_root = tmp_path / "pair"
    reference_root = tmp_path / "reference"
    for root in (pair_root, reference_root):
        (root / "baseline").mkdir(parents=True)
        (root / "alpha025").mkdir()
    baseline = torch.zeros(1000, dtype=torch.long)
    reference_guided = torch.zeros_like(baseline)
    repeated_guided = reference_guided.clone()
    repeated_guided[0] = 9
    torch.save(baseline, pair_root / "baseline/sample_0.pt")
    torch.save(repeated_guided, pair_root / "alpha025/sample_0.pt")
    torch.save(baseline, reference_root / "baseline/sample_0.pt")
    torch.save(reference_guided, reference_root / "alpha025/sample_0.pt")
    metrics = {
        "global_voxel_accuracy": 0.63,
        "truth_present_mean_iou": 0.34,
        "target_iou": 0.48,
        "target_precision": 0.90,
        "target_recall": 0.51,
    }
    with (reference_root / "alpha025/sample_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", *metrics])
        writer.writeheader()
        writer.writerow({"sample_id": 0, **metrics})

    result = _anchor_regression(pair_root, reference_root, {0: metrics})

    assert result["passed"] is True
    assert result["baseline_hard_mismatch_voxels"] == 0
    assert result["guided_hard_disagreement_fraction"] == pytest.approx(0.001)


@pytest.mark.parametrize(
    ("pair_gates", "diversity", "expected"),
    [
        ([True, True, True, True], True, "confirmed_seed42_pass"),
        ([True, True, True, True], False, "diversity_gate_failure"),
        ([False, False, False, False], True, "confirmed_seed42_failure"),
        ([True, False, True, False], True, "transition_region"),
    ],
)
def test_phase2b_n4_level_classification_is_frozen(
    pair_gates: list[bool],
    diversity: bool,
    expected: str,
) -> None:
    assert classify_n4_level(pair_gates, diversity) == expected


def test_phase2b_n4_selection_promotes_most_degraded_confirmed_level() -> None:
    levels = load_manifest(MANIFEST_PATH)["levels"]
    rows = [
        {
            "level_id": "paired_c025",
            "classification": "confirmed_seed42_pass",
        },
        {
            "level_id": "paired_c010",
            "classification": "confirmed_seed42_failure",
        },
    ]

    selection = n4_candidate_selection(levels, rows)

    assert selection["candidate_level"] == "paired_c025"
    assert selection["adjacent_lower_level"] == "paired_c010"
    assert selection["adjacent_lower_classification"] == "confirmed_seed42_failure"
    assert selection["promote_to_multiseed"] is True


def test_phase2b_fallback_requires_the_completed_nonpromoting_bracket() -> None:
    summary = {
        "levels": [
            {
                "level_id": "paired_c025",
                "classification": "transition_region",
            },
            {
                "level_id": "paired_c010",
                "classification": "confirmed_seed42_failure",
            },
        ],
        "selection": {"promote_to_multiseed": False},
    }

    assert validate_bracket_prerequisite(summary) == {
        "paired_c025": "transition_region",
        "paired_c010": "confirmed_seed42_failure",
    }


@pytest.mark.parametrize(
    ("classification", "promote"),
    [
        ("confirmed_seed42_pass", True),
        ("transition_region", False),
        ("confirmed_seed42_failure", False),
        ("diversity_gate_failure", False),
    ],
)
def test_phase2b_fallback_only_promotes_confirmed_pass(
    classification: str,
    promote: bool,
) -> None:
    result = fallback_decision({"classification": classification})

    assert result["promote_to_multiseed"] is promote
    assert result["candidate_level"] == ("paired_c100" if promote else None)
