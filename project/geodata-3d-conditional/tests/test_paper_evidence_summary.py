"""Integrity tests for the machine-grounded paper evidence package."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path

from scripts.paper_figures import build_paper_evidence_summary as builder


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
SUMMARY_PATH = PROJECT_DIR / "paper/figure_data/paper_evidence_summary.json"
EVIDENCE_CSV = PROJECT_DIR / "paper/figure_data/paper_evidence_summary.csv"
TABLE_MD = PROJECT_DIR / "paper/tables/table01_evidence_summary.md"
TABLE_TEX = PROJECT_DIR / "paper/tables/table01_evidence_summary.tex"
TABLE_CSV = PROJECT_DIR / "paper/tables/table01_evidence_summary.csv"
FACT_SHEET = PROJECT_DIR / "paper/PAPER_RESULT_FACTS.md"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary() -> dict[str, object]:
    payload = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _metric_index(summary: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        metric["metric_id"]: metric
        for experiment in summary["experiments"]
        for metric in experiment["primary_metrics"]
    }


def _assert_finite(value: object) -> None:
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for child in value.values():
            _assert_finite(child)
    elif isinstance(value, list):
        for child in value:
            _assert_finite(child)


def test_every_metric_and_headline_has_existing_hashed_source() -> None:
    summary = _summary()
    _assert_finite(summary)
    metric_index = _metric_index(summary)
    assert len(metric_index) >= 50
    for experiment in summary["experiments"]:
        assert experiment["source_files"]
        for record in experiment["source_files"]:
            path = REPOSITORY_ROOT / record["path"]
            assert path.is_file()
            assert _sha256(path) == record["sha256"]
        for metric in experiment["primary_metrics"]:
            source = metric["source"]
            assert source["path"]
            assert source["json_key"]
            path = REPOSITORY_ROOT / source["path"]
            assert path.is_file()
            assert _sha256(path) == source["sha256"]
    for metric in summary["headline_metrics"]:
        assert metric["metric_id"] in metric_index
        assert metric["source"]["path"]
        assert metric["source"]["json_key"]


def test_required_scientific_roles_and_boundaries_are_explicit() -> None:
    summary = _summary()
    experiments = {row["experiment_id"]: row for row in summary["experiments"]}
    required_fields = {
        "experiment_id",
        "display_name",
        "evidence_type",
        "inference_method",
        "n_cases",
        "n_pairs_or_samples",
        "truth_visible_during_inference",
        "observation_visible",
        "selection_rule",
        "primary_metrics",
        "machine_decision",
        "scientific_interpretation",
        "source_files",
    }
    assert len(experiments) == 9
    assert all(required_fields <= set(experiment) for experiment in experiments.values())
    assert "truth-derived" in experiments["phase1_oracle_probability"]["evidence_type"]
    assert experiments["phase1_oracle_probability"]["truth_visible_during_inference"] is True
    assert "upper bound" in experiments["phase2a_ideal_property"]["evidence_type"]
    assert experiments["phase2a_ideal_property"]["truth_visible_during_inference"] is True
    assert "bounded structured" in experiments["stage7_structured_hard_seismic"]["evidence_type"]
    assert experiments["stage7_structured_hard_seismic"]["truth_visible_during_inference"] is False
    stage9_oracle = next(
        metric
        for metric in experiments["stage9a_unrestricted_flow_prior"]["primary_metrics"]
        if metric["metric_id"] == "s9_oracle_best_label9_iou"
    )
    assert stage9_oracle["scientific_role"] == "retrospective upper bound"
    assert "Not a deployable selector" in stage9_oracle["qualification"]
    assert "synthetic" in experiments["stage12b_fullgeo_probability_bridge"]["evidence_type"]
    assert "synthetic" in experiments["stage14_probability_guided_flow"]["evidence_type"]
    boundaries = summary["scientific_boundaries"]
    assert boundaries["stage7_is_structured_inference_not_cfm_posterior"] is True
    assert boundaries["stage7_truth_used_for_selection"] is False
    assert boundaries["stage14_is_paired_synthetic_not_measured_field_test"] is True
    assert boundaries["cross_protocol_head_to_head_claim"] is False


def test_phase1_and_phase2_required_metric_inventory_is_complete() -> None:
    summary = _summary()
    experiments = {row["experiment_id"]: row for row in summary["experiments"]}
    p1_ids = {metric["metric_id"] for metric in experiments["phase1_oracle_probability"]["primary_metrics"]}
    p2_ids = {metric["metric_id"] for metric in experiments["phase2a_ideal_property"]["primary_metrics"]}
    assert {
        "p1_global_accuracy",
        "p1_global_miou",
        "p1_label9_iou",
        "p1_label9_precision",
        "p1_label9_recall",
        "p1_centroid_distance",
        "p1_roi_iou",
        "p1_condition_violations",
        "p1_paired_runs",
    } <= p1_ids
    assert {
        "p2_global_accuracy",
        "p2_dynamic_miou",
        "p2_truth_present_miou",
        "p2_hard_property_loss",
        "p2_label9_iou",
        "p2_label9_precision",
        "p2_label9_recall",
        "p2_centroid_distance",
        "p2_condition_violations",
        "p2_paired_runs",
    } <= p2_ids


def test_fullgeo_cohort_metric_maps_every_case_to_its_manifest() -> None:
    metric = _metric_index(_summary())["fullgeo_label9_voxel_counts"]
    expected_ids = [f"fullgeo_case{index:02d}" for index in range(1, 6)]
    assert list(metric["value"]) == expected_ids
    assert list(metric["source_by_case"]) == expected_ids
    for case_id, source in metric["source_by_case"].items():
        path = REPOSITORY_ROOT / source["path"]
        assert path.is_file()
        assert _sha256(path) == source["sha256"]
        assert source["json_key"] == "raw_label_counts.9"


def test_frozen_headline_metrics_match_authoritative_values() -> None:
    metric = _metric_index(_summary())
    assert metric["p1_label9_iou"]["value"] == {
        "baseline": 0.03144480434529132,
        "guided": 0.8098611425842716,
    }
    assert metric["p2_label9_iou"]["value"] == {
        "baseline": 0.03144480434529132,
        "guided": 0.4807524758898813,
    }
    assert metric["s7_hidden_iou_range"]["value"] == [
        0.9141716566866267,
        0.9874266554903605,
    ]
    assert metric["s7_correct_rank_first"]["value"] == 3
    assert metric["s9_unique_hard_models"]["value"] == 3072
    assert metric["s9_support_pass_count"]["value"] == 0
    assert metric["s9_discrimination_pass_count"]["value"] == 0
    assert metric["s9_oracle_best_label9_iou"]["value"] == {
        "native_seed20260901": 0.11701541850220264,
        "native_seed20260902": 0.13477816017114688,
        "native_seed20260903": 0.12891236306729265,
    }
    assert metric["s12_diagonal_mean_auprc"]["value"] == 0.04176687121111301
    assert metric["s12_off_diagonal_mean_auprc"]["value"] == 0.04187063309937007
    assert metric["s14_overall_median_hidden_iou_delta"]["value"] == -0.031507776294942384
    assert metric["s14_positive_case_count"]["value"] == 1


def test_generated_artifacts_do_not_drift_from_builder() -> None:
    expected = builder.render_all()
    assert set(expected) == {
        SUMMARY_PATH,
        EVIDENCE_CSV,
        TABLE_TEX,
        TABLE_MD,
        TABLE_CSV,
        FACT_SHEET,
    }
    for path, content in expected.items():
        assert path.is_file()
        assert path.read_text(encoding="utf-8") == content


def test_evidence_csv_round_trips_all_json_metrics() -> None:
    summary = _summary()
    metric = _metric_index(summary)
    rows = list(csv.DictReader(io.StringIO(EVIDENCE_CSV.read_text(encoding="utf-8"))))
    assert len(rows) == len(metric)
    assert {row["metric_id"] for row in rows} == set(metric)
    for row in rows:
        expected = metric[row["metric_id"]]
        assert json.loads(row["value_json"]) == expected["value"]
        assert row["source_path"] == expected["source"]["path"]
        assert row["source_key"] == expected["source"]["json_key"]


def test_table_values_are_generated_from_json_and_have_integrity_caption() -> None:
    summary = _summary()
    table = summary["table1"]
    md = TABLE_MD.read_text(encoding="utf-8")
    tex = TABLE_TEX.read_text(encoding="utf-8")
    assert builder.CAPTION_INTEGRITY_NOTE in table["caption"]
    assert builder.CAPTION_INTEGRITY_NOTE in md
    assert builder.CAPTION_INTEGRITY_NOTE in tex
    assert md == builder.render_table_markdown(table)
    assert tex == builder.render_table_latex(table)
    assert TABLE_CSV.read_text(encoding="utf-8") == builder.render_table_csv(table)
    table_ids = {metric_id for row in table["rows"] for metric_id in row["source_metric_ids"]}
    assert table_ids <= set(_metric_index(summary))
    assert "Label-9 IoU 0.031 -> 0.810" in md
    assert "Support 0/3; discrimination 0/3" in md
    assert "Median change in hidden IoU -0.0315; positive 1/5" in md


def test_prohibited_claims_are_absent_from_table_caption_and_templates() -> None:
    prohibited = (
        "CFM seismic inversion recovers hidden geology with IoU 0.987",
        "Phase 1 IoU 0.81 is a measured-geophysics result",
        "The method provides a calibrated Bayesian posterior",
        "Stage 7 truth was used for candidate selection",
        "Current results demonstrate field-data generalization",
    )
    templates = "\n".join(
        [
            _summary()["table1"]["caption"],
            TABLE_MD.read_text(encoding="utf-8"),
            TABLE_TEX.read_text(encoding="utf-8"),
        ]
    )
    assert all(claim not in templates for claim in prohibited)


def test_figure_links_cover_every_existing_paper_manifest() -> None:
    summary = _summary()
    linked = {row["manifest_path"] for row in summary["figure_links"]}
    existing = {
        str(path.relative_to(REPOSITORY_ROOT))
        for path in (PROJECT_DIR / "paper/manifests").glob("*.json")
    }
    assert linked == existing
    for row in summary["figure_links"]:
        path = REPOSITORY_ROOT / row["manifest_path"]
        assert _sha256(path) == row["manifest_sha256"]
