from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from analyze_dike_observability import observability_analysis, observability_summary, save_observability_artifacts
from compare_gravity_residuals import save_residual_comparison
from create_density_config import build_density_config
from create_susceptibility_config import build_susceptibility_config
from evaluate_target_feature import target_metric_records
from geophysics import (
    GravityGradientForward,
    LithologyPropertyMap,
    MagneticTMIForward,
    SimpleGravityForward,
)
from geology_io_utils import (
    find_sample_files,
    load_density_config,
    load_sample_stack,
    load_susceptibility_config,
    property_map_from_susceptibility_config,
    target_probability,
    write_json,
)
from guided_geophysical_sampling import multi_physics_guidance_loss
from make_dike_guidance_demo import _resolve_target_label, _select_residual_sample_id
from screen_dike_demo_candidates import _classify, parse_candidate
from select_dike_demo_case import candidate_records
from visualize_truth_model_labels import label_summary
from visualize_dike_ensemble import save_ensemble_figures


def test_load_sample_stack_normalizes_common_shapes(tmp_path: Path) -> None:
    torch.save(torch.zeros((4, 5, 6), dtype=torch.long), tmp_path / "sample_0.pt")
    torch.save(torch.ones((1, 4, 5, 6), dtype=torch.long), tmp_path / "sample_1.pt")

    samples, records = load_sample_stack(find_sample_files(tmp_path))

    assert samples.shape == (2, 1, 4, 5, 6)
    assert [record["sample_id"] for record in records] == [0, 1]


def test_target_metric_records_identifies_exact_and_missing_target() -> None:
    truth = torch.ones((1, 1, 6, 6, 6), dtype=torch.long)
    truth[:, :, 2:4, 2:4, 1:5] = 7
    exact = truth[0]
    missing = torch.ones_like(exact)
    samples = torch.stack([exact, missing], dim=0)
    records = [
        {"sample_id": 0, "path": "sample_0.pt", "stack_index": 0},
        {"sample_id": 1, "path": "sample_1.pt", "stack_index": 1},
    ]

    rows, summary = target_metric_records(samples, truth, 7, records)

    assert rows[0]["target_iou"] == 1.0
    assert rows[0]["target_f1"] == 1.0
    assert rows[1]["target_recall"] == 0.0
    assert summary["target_label"] == 7
    assert "0.33" in summary["probability_threshold_metrics"]
    assert target_probability(samples, 7).shape == (1, 1, 6, 6, 6)


def test_observability_summary_detects_proxy_change() -> None:
    truth = torch.ones((1, 1, 8, 8, 6), dtype=torch.long)
    truth[:, :, 3:5, 2:6, 1:5] = 10

    summary = observability_summary(
        truth_model=truth,
        target_label=10,
        replacement_label=1,
        kernel_size=3,
    )

    assert summary["target_volume"] > 0
    assert "recommended_for_demo" in summary
    assert summary["lightweight_gravity_proxy_delta_peak_abs"] > 0


def test_density_config_roundtrip_preserves_manual_target_label(tmp_path: Path) -> None:
    truth = torch.ones((1, 1, 5, 5, 5), dtype=torch.long)
    truth[:, :, 2:4, 2:3, 1:4] = 10

    config = build_density_config(truth, target_label=10, target_density=3.5)
    write_json(tmp_path / "density_config.json", config)
    loaded = load_density_config(tmp_path / "density_config.json")

    assert loaded["target_label"] == 10
    assert loaded["densities"][10] == 3.5
    assert loaded["target_to_nearest_non_target_density_contrast"] > 0


def test_susceptibility_config_roundtrip_preserves_manual_target_label(tmp_path: Path) -> None:
    truth = torch.ones((1, 1, 5, 5, 5), dtype=torch.long)
    truth[:, :, 2:4, 2:3, 1:4] = 10

    config = build_susceptibility_config(
        truth,
        target_label=10,
        target_susceptibility=7.5,
    )
    write_json(tmp_path / "susceptibility_config.json", config)
    loaded = load_susceptibility_config(tmp_path / "susceptibility_config.json")
    property_map = property_map_from_susceptibility_config(loaded)

    assert loaded["target_label"] == 10
    assert loaded["susceptibilities"][10] == 7.5
    assert property_map.properties[10] == 7.5


def test_magnetic_forward_detects_susceptibility_contrast() -> None:
    susceptibility = torch.zeros((1, 1, 7, 7, 5), dtype=torch.float32)
    susceptibility[:, :, 3:5, 2:5, 1:4] = 5.0

    field = MagneticTMIForward(kernel_size=3)(susceptibility)

    assert field.shape == (1, 1, 7, 7)
    assert torch.isfinite(field).all()
    assert field.abs().max() > 0


def test_gravity_gradient_forward_detects_density_contrast() -> None:
    density = torch.zeros((1, 1, 7, 7, 5), dtype=torch.float32)
    density[:, :, 3:5, 2:5, 1:4] = 2.0

    field = GravityGradientForward(kernel_size=3)(density)

    assert field.shape == (1, 1, 7, 7)
    assert torch.isfinite(field).all()
    assert field.abs().max() > 0


def test_multi_physics_guidance_loss_supports_magnetic_and_gradient_gradients() -> None:
    embedding_weight = torch.eye(3, dtype=torch.float32)
    x = torch.randn((1, 3, 5, 5, 4), dtype=torch.float32, requires_grad=True)
    lithology = torch.zeros((1, 1, 5, 5, 4), dtype=torch.long)
    lithology[:, :, 2:4, 2:4, 1:3] = 1
    density_map = LithologyPropertyMap(properties={-1: 0.0, 0: 0.1, 1: 2.0})
    susceptibility_map = LithologyPropertyMap(properties={-1: 0.0, 0: 0.01, 1: 5.0})
    gravity_forward = SimpleGravityForward(kernel_size=3)
    gravity_gradient_forward = GravityGradientForward(kernel_size=3)
    magnetic_forward = MagneticTMIForward(kernel_size=3)
    observed_gravity = gravity_forward(density_map(lithology)).detach()
    observed_gradient = gravity_gradient_forward(density_map(lithology)).detach()
    observed_magnetic = magnetic_forward(susceptibility_map(lithology)).detach()

    loss, diagnostics = multi_physics_guidance_loss(
        x=x,
        embedding_weight=embedding_weight,
        property_map=density_map,
        forward_model=gravity_forward,
        observed_gravity=observed_gravity,
        susceptibility_map=susceptibility_map,
        magnetic_forward_model=magnetic_forward,
        observed_magnetic=observed_magnetic,
        gravity_gradient_forward_model=gravity_gradient_forward,
        observed_gravity_gradient=observed_gradient,
        physics_mode="joint",
        gravity_weight=0.0,
        magnetic_weight=1.0,
        gravity_gradient_weight=0.25,
        tau=0.5,
    )
    grad = torch.autograd.grad(loss, x)[0]

    assert torch.isfinite(loss)
    assert torch.isfinite(grad).all()
    assert diagnostics["magnetic_loss"] >= 0
    assert diagnostics["gravity_gradient_loss"] >= 0


def test_target_label_resolution_prefers_manual_density_config() -> None:
    label, source = _resolve_target_label(
        target_label=None,
        density_config={"target_label": 10, "densities": {10: 3.5}},
        allow_auto_target_selection=False,
    )

    assert label == 10
    assert source == "density_config.target_label"


def test_target_label_resolution_rejects_conflict() -> None:
    try:
        _resolve_target_label(
            target_label=7,
            density_config={"target_label": 10, "densities": {10: 3.5}},
            allow_auto_target_selection=False,
        )
    except SystemExit as exc:
        assert "conflicts" in str(exc)
    else:
        raise AssertionError("expected conflicting target labels to fail")


def test_candidate_screening_classifies_main_demo_candidate() -> None:
    candidate = parse_candidate("case_a:7")
    args = type(
        "Args",
        (),
        {
            "min_target_voxels": 1000,
            "min_borehole_hits": 1,
            "min_observability_delta": 0.01,
            "min_prior_volume_ratio": 0.2,
            "max_prior_volume_ratio": 2.0,
            "min_geo_improvement": 0.02,
            "geology_tolerance": 0.02,
            "min_target_iou_improvement": 0.001,
            "min_target_recall_improvement": 0.001,
            "min_centroid_improvement": 0.5,
        },
    )()
    row = {
        "observability_normalized_delta": 0.5,
        "baseline_target_volume_ratio": 0.8,
        "geo_misfit_improvement": 0.1,
        "target_iou_improvement": 0.002,
        "target_recall_improvement": 0.0,
        "target_centroid_distance_improvement": 0.0,
        "truth_target_volume": 5000,
        "borehole_target_hits": 3,
        "baseline_voxel_accuracy_mean": 0.5,
        "guided_voxel_accuracy_mean": 0.5,
        "baseline_mean_iou_mean": 0.2,
        "guided_mean_iou_mean": 0.2,
        "baseline_borehole_consistency_mean": 1.0,
        "guided_borehole_consistency_mean": 0.999,
    }

    recommendation, _ = _classify(row, args)

    assert candidate.case_name == "case_a"
    assert candidate.target_label == 7
    assert recommendation == "main_demo_candidate"


def test_manual_truth_label_summary_reports_borehole_hits() -> None:
    truth = torch.ones((1, 1, 5, 5, 5), dtype=torch.long)
    truth[:, :, 1:4, 2:3, 1:4] = 10
    boreholes = torch.full_like(truth, -1)
    boreholes[:, :, 2, 2, 2] = 10

    rows = label_summary(truth, boreholes)
    target_row = next(row for row in rows if row["label"] == 10)

    assert target_row["borehole_hits"] == 1
    assert target_row["target_connected_components"] == 1


def test_observability_artifacts_are_saved(tmp_path: Path) -> None:
    truth = torch.ones((1, 1, 8, 8, 6), dtype=torch.long)
    truth[:, :, 3:5, 2:6, 1:5] = 10
    summary, truth_gravity, removed, delta, target = observability_analysis(
        truth_model=truth,
        target_label=10,
        replacement_label=1,
        kernel_size=3,
    )

    paths = save_observability_artifacts(tmp_path, truth_gravity, removed, delta, target)

    assert summary["recommended_for_demo"] is True
    assert (tmp_path / "truth_gravity.png").exists()
    assert (tmp_path / "removed_target_gravity.png").exists()
    assert (tmp_path / "delta_gravity.png").exists()
    assert (tmp_path / "target_mask_slices.png").exists()
    assert set(paths) == {"truth_gravity", "removed_target_gravity", "delta_gravity", "target_mask_slices"}


def test_candidate_records_selects_thin_observable_component() -> None:
    truth = torch.ones((1, 1, 10, 10, 8), dtype=torch.long)
    truth[:, :, 2:8, 4:6, 1:7] = 10
    boreholes = torch.full_like(truth, -1)
    boreholes[:, :, 2, 4, 1:7] = truth[:, :, 2, 4, 1:7]

    candidates = candidate_records(
        truth_model=truth,
        boreholes=boreholes,
        min_voxels=8,
        kernel_size=3,
    )

    assert candidates
    assert candidates[0]["target_label"] == 10
    assert "density_contrast" in candidates[0]
    assert candidates[0]["lightweight_gravity_proxy_observability"] > 0


def test_residual_individual_images_are_saved(tmp_path: Path) -> None:
    truth = torch.ones((1, 1, 8, 8, 6), dtype=torch.long)
    truth[:, :, 3:5, 2:6, 1:5] = 10
    baseline = truth.clone()
    baseline[:, :, 3:5, 2:6, 1:3] = 1
    guided = truth.clone()
    observed = torch.zeros((1, 1, 8, 8))

    summary = save_residual_comparison(
        baseline_sample=baseline,
        guided_sample=guided,
        observed_gravity=observed,
        output_dir=tmp_path,
        sample_id=7,
        kernel_size=3,
    )

    assert (tmp_path / "gravity_proxy_residual_comparison.png").exists()
    assert (tmp_path / "observed_gravity.png").exists()
    assert (tmp_path / "baseline_predicted_gravity_sample_7.png").exists()
    assert (tmp_path / "guided_predicted_gravity_sample_7.png").exists()
    assert (tmp_path / "baseline_residual_sample_7.png").exists()
    assert (tmp_path / "guided_residual_sample_7.png").exists()
    assert (tmp_path / "residual_difference_sample_7.png").exists()
    assert "residual_rms_reduction" in summary


def test_unpaired_visualization_does_not_save_changed_voxels(tmp_path: Path) -> None:
    truth = torch.ones((1, 1, 5, 5, 5), dtype=torch.long)
    truth[:, :, 1:4, 2:3, 1:4] = 7
    baseline = truth.clone()
    guided = truth.clone()
    guided[:, :, 3:4, 2:3, 1:4] = 1
    records = [{"sample_id": 3, "path": "sample_3.pt", "stack_index": 0}]

    summary = save_ensemble_figures(
        baseline=baseline,
        baseline_records=records,
        guided=guided,
        guided_records=records,
        truth_model=truth,
        target_label=7,
        output_dir=tmp_path,
        sample_ids=[3],
        paired_by_seed=False,
    )

    assert not list(tmp_path.glob("changed_voxels_3d_sample_*.png"))
    assert "changed_voxels_3d_sample_3" not in summary["figures"]


def test_make_demo_uses_selected_sample_id_not_hardcoded_zero() -> None:
    rows = [
        {"role": "failure_case", "sample_id": "0"},
        {"role": "max_geo_improvement_preserve_geology", "sample_id": "7"},
    ]

    assert _select_residual_sample_id(rows) == 7
