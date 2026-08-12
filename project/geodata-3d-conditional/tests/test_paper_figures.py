"""Lightweight regression tests for the paper-figure provenance/QC helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from scripts.paper_figures.style import (
    CAMERA,
    CAMERA_PRESETS,
    LABEL9_COLOR,
    LABEL_COLORS,
    OBSERVATION_COLOR,
    robust_symmetric_limit,
    target_metrics,
    write_deterministic_npz,
)


def _hash(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_target_metrics_use_hard_label9() -> None:
    truth = np.array([[[9, 9], [0, 0]]])
    prediction = np.array([[[9, 0], [9, 0]]])
    metrics = target_metrics(truth, prediction)
    assert metrics == {"IoU9": 1 / 3, "Precision9": 1 / 2, "Recall9": 1 / 2}


def test_sparse_residual_limit_uses_one_pooled_nonzero_population() -> None:
    left = np.array([0.0, 0.0, -1.0, 2.0])
    right = np.array([0.0, 3.0, -4.0, 0.0])
    limit = robust_symmetric_limit((left, right), 50.0, ignore_zeros=True)
    assert limit == 2.5


def test_npz_writer_is_byte_deterministic(tmp_path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    arrays = {"z": np.arange(5, dtype=np.int16), "a": np.eye(3, dtype=np.float32)}
    write_deterministic_npz(first, **arrays)
    write_deterministic_npz(second, **arrays)
    assert _hash(first) == _hash(second)
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == ["a.npy", "z.npy"]
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())


def test_visual_language_reserves_label9_and_observation_colors() -> None:
    assert LABEL_COLORS[9] == LABEL9_COLOR
    assert LABEL9_COLOR != OBSERVATION_COLOR
    assert CAMERA["parallel_projection"] is True


PROJECT_DIR = Path(__file__).resolve().parents[1]
FRAMEWORK_DATA = PROJECT_DIR / "paper/figure_data/figure01_joint_framework.json"
FRAMEWORK_MANIFEST = PROJECT_DIR / "paper/manifests/figure01_joint_framework.json"
JOINT_MANIFEST = PROJECT_DIR / "paper/manifests/figure03_joint_inference.json"
EVIDENCE_DATA = PROJECT_DIR / "paper/figure_data/figure04_evidence_hierarchy.json"
EVIDENCE_MANIFEST = PROJECT_DIR / "paper/manifests/figure04_evidence_hierarchy.json"
FULLGEO_DATA = PROJECT_DIR / "paper/figure_data/fullgeo_3d_benchmark.json"
FULLGEO_MANIFEST = PROJECT_DIR / "paper/manifests/fullgeo_3d_benchmark.json"
GALLERY_MANIFEST = PROJECT_DIR / "paper/manifests/3d_gallery.json"


def _read_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _assert_output_records_valid(records: list[dict[str, object]]) -> None:
    assert records
    for record in records:
        path = PROJECT_DIR.parent.parent / str(record["path"])
        assert path.is_file()
        assert path.stat().st_size > 0
        assert _hash(path) == record["sha256"]


def test_joint_framework_marks_frozen_flow_and_evidence_roles() -> None:
    data = _read_manifest(FRAMEWORK_DATA)
    manifest = _read_manifest(FRAMEWORK_MANIFEST)
    assert data["boundaries"]["inference"] == "frozen conditional Flow; no checkpoint update"
    assert "continuous embedding\n+ ODE" in data["main_pipeline"]
    assert "hard categorical\ndecode" in data["main_pipeline"]
    assert [row["level"] for row in data["information_hierarchy"]] == ["I", "II", "III"]
    assert [row["role"] for row in data["information_hierarchy"]] == [
        "truth-derived oracle upper bound",
        "ideal/noiseless upper bound",
        "truth-blind observation",
    ]
    assert data["reciprocal_concept"]["bayesian_posterior_claim"] is False
    assert manifest["scientific_boundaries"] == {
        "frozen_flow_explicit": True,
        "checkpoint_update": False,
        "hard_decode_explicit": True,
        "final_bayesian_posterior_claim": False,
        "research_target_marked_as_goal": True,
    }
    _assert_output_records_valid(manifest["outputs"])


def test_evidence_hierarchy_uses_exact_frozen_facts_and_roles() -> None:
    data = _read_manifest(EVIDENCE_DATA)
    manifest = _read_manifest(EVIDENCE_MANIFEST)
    facts = data["facts"]
    assert facts["sparse_flow_label9_iou"] == 0.03144480434529132
    assert facts["oracle_probability_label9_iou"] == 0.8098611425842716
    assert facts["ideal_property_label9_iou"] == 0.4807524758898813
    assert facts["stage7_hidden_iou_range"] == [0.9141716566866267, 0.9874266554903605]
    assert facts["stage7_correct_rank_first_count"] == 3
    assert facts["stage9a_unique_hard_models"] == 3072
    assert facts["stage9a_support_case_count"] == 0
    assert facts["stage9a_discrimination_case_count"] == 0
    assert facts["stage12b_diagonal_mean_auprc"] == 0.04176687121111301
    assert facts["stage12b_off_diagonal_mean_auprc"] == 0.04187063309937007
    assert facts["stage12b_diagonal_row_maximum_count"] == 1
    assert facts["stage14_median_hidden_iou_delta"] == -0.031507776294942384
    assert facts["stage14_positive_case_count"] == 1
    assert data["cross_protocol_paired_comparison"] is False
    assert [node["role"] for node in data["nodes"]] == [
        "REFERENCE",
        "ORACLE",
        "UPPER BOUND",
        "ACQUISITION\nDOMAIN",
        "BOUNDED\nSPACE",
        "TRUTH-BLIND",
        "BRIDGE ONLY",
        "PAIRED\nSYNTHETIC",
    ]
    assert manifest["scientific_boundaries"]["stage14_measured_field_test"] is False
    _assert_output_records_valid(manifest["outputs"])


def test_fullgeo_benchmark_has_exact_five_frozen_cases() -> None:
    data = _read_manifest(FULLGEO_DATA)
    manifest = _read_manifest(FULLGEO_MANIFEST)
    expected_ids = [f"fullgeo_case{index:02d}" for index in range(1, 6)]
    expected_counts = [120, 16083, 1736, 13293, 4111]
    assert data["case_ids"] == expected_ids
    assert manifest["case_ids"] == expected_ids
    assert [case["label9_voxel_count"] for case in data["cases"]] == expected_counts
    assert all(case["well_count"] == 9 for case in data["cases"])
    assert manifest["quality_control"]["registered_cases_exactly_once"] == expected_ids
    assert manifest["camera"]["name"] == "perspective_iso"
    assert manifest["camera"]["parameters"] == CAMERA_PRESETS["perspective_iso"]
    assert manifest["camera"]["identical_for_all_cases_and_rows"] is True
    assert manifest["target"]["raw_label"] == 9
    assert len(manifest["source_tensor_hashes"]) == 5
    assert len(manifest["outputs"]) == 6
    _assert_output_records_valid(manifest["outputs"])


def test_new_figure_pngs_are_600_dpi_and_print_ready() -> None:
    manifests = (
        _read_manifest(FRAMEWORK_MANIFEST),
        _read_manifest(EVIDENCE_MANIFEST),
        _read_manifest(FULLGEO_MANIFEST),
    )
    png_records = [
        row
        for manifest in manifests
        for row in manifest["outputs"]
        if Path(row["path"]).suffix == ".png"
    ]
    assert len(png_records) == 4
    for record in png_records:
        path = PROJECT_DIR.parent.parent / str(record["path"])
        with Image.open(path) as image:
            assert image.width >= 4000
            assert image.height >= 1200
            dpi = image.info.get("dpi")
            assert dpi is not None
            assert all(abs(float(value) - 600.0) <= 1.0 for value in dpi)


def test_joint_figure_expected_outputs_and_hashes_exist() -> None:
    manifest = _read_manifest(JOINT_MANIFEST)
    assert manifest["figure_id"] == "figure03_joint_inference"
    assert manifest["case_ids"][0] == "native_seed20260809"
    _assert_output_records_valid(manifest["outputs"])
    suffixes = {Path(row["path"]).suffix for row in manifest["outputs"]}
    assert suffixes == {".pdf", ".svg", ".png"}


def test_joint_figure_frozen_metrics_and_truth_firewall() -> None:
    manifest = _read_manifest(JOINT_MANIFEST)
    metrics = manifest["metrics"]
    stage7 = metrics["stage7"]
    assert stage7["baseline_hard_seismic_rmse"] == 0.010582241229712963
    assert stage7["selected_hard_seismic_rmse"] == 0.00431425217539072
    assert stage7["baseline_hidden"]["hidden_iou"] == 0.0
    assert stage7["selected_hidden"]["hidden_iou"] == 0.9874266554903605
    assert stage7["selected_hidden"]["hidden_precision"] == 0.9940928270042194
    assert stage7["selected_hidden"]["hidden_recall"] == 0.9932546374367622
    assert stage7["correct_observation_rank_first"] == 3
    assert metrics["stage9a"] == {
        "discrimination_cases": 0,
        "n_cases": 3,
        "support_cases": 0,
        "unique_hard_models": 3072,
    }
    assert metrics["stage14"]["overall_paired_median_hidden_label9_iou_delta"] == -0.031507776294942384
    assert metrics["stage14"]["positive_case_count"] == 1
    assert manifest["scientific_boundaries"]["stage7_is_cfm_posterior"] is False
    assert manifest["scientific_boundaries"]["stage9a_truth_oracle_candidate_displayed"] is False
    assert manifest["scientific_boundaries"]["stage14_is_measured_geophysics"] is False
    assert "hard observed seismic RMSE only" in manifest["selection_criterion"]


def test_gallery_registered_cases_samples_and_camera_consistency() -> None:
    manifest = _read_manifest(GALLERY_MANIFEST)
    assert manifest["case_ids"]["full_structuralgeo"] == [
        "fullgeo_case01",
        "fullgeo_case02",
        "fullgeo_case03",
        "fullgeo_case04",
        "fullgeo_case05",
    ]
    assert manifest["case_ids"]["hero"] == "native_seed20260809"
    assert manifest["sample_ids"]["ensemble_sample_ids"] == [0, 1, 2, 3]
    assert manifest["sample_ids"]["ensemble_source_seeds"] == [9301000, 9301001, 9301002, 9301003]
    assert "no truth or seismic ranking" in manifest["sample_selection"]
    assert manifest["camera"]["identical_within_each_comparison"] is True
    assert set(manifest["camera"]["hero_views"]) == {
        "perspective_iso",
        "perspective_oblique",
        "top_oblique",
    }
    _assert_output_records_valid(manifest["outputs"])


def test_joint_and_gallery_png_resolution_is_print_ready() -> None:
    manifests = (_read_manifest(JOINT_MANIFEST), _read_manifest(GALLERY_MANIFEST))
    png_records = [
        row
        for manifest in manifests
        for row in manifest["outputs"]
        if Path(row["path"]).suffix == ".png"
    ]
    assert len(png_records) == 7
    for record in png_records:
        path = PROJECT_DIR.parent.parent / str(record["path"])
        with Image.open(path) as image:
            assert image.width >= 4000
            assert image.height >= 1200
            dpi = image.info.get("dpi")
            assert dpi is not None
            assert all(abs(float(value) - 600.0) <= 1.0 for value in dpi)
