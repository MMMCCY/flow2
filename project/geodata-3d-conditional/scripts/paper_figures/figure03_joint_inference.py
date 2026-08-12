#!/usr/bin/env python3
"""Generate the paper Figure 3 and its frozen-result 3-D geology gallery."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
STRUCTURALGEO_SRC = REPOSITORY_ROOT / "StructuralGeo-main/src"
for path in (PROJECT_DIR, STRUCTURALGEO_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from guidance.native_geology_audit import build_structuralgeo_native_case
from scripts.paper_figures.style import (
    CAMERA_PRESETS,
    DOUBLE_COLUMN_MM,
    FIGURE_DATA_DIR,
    FIGURES_DIR,
    LABEL9_COLOR,
    MANIFESTS_DIR,
    OBSERVATION_COLOR,
    RESIDUAL_CMAP,
    SEISMIC_CMAP,
    configure_matplotlib,
    ensure_output_dirs,
    generation_record,
    load_volume,
    mm_to_inches,
    output_records,
    panel_label,
    read_json,
    render_categorical_volume_3d,
    render_label_comparison_3d,
    render_label_frequency_3d,
    render_sparse_constraints_3d,
    save_figure,
    sha256,
    show_render,
    source_record,
    validate_same_shape,
    write_json,
)


SCRIPT_PATH = Path(__file__).resolve()
STYLE_PATH = SCRIPT_PATH.with_name("style.py")
FIGURE_ID = "figure03_joint_inference"
GALLERY_ID = "3d_gallery"
HERO_CASE_ID = "native_seed20260809"
HERO_CASE_SEED = 20260809
TARGET_LABEL = 9
SLICE_Y = 42
FULLGEO_CASE_IDS = tuple(f"fullgeo_case{index:02d}" for index in range(1, 6))
SPARSE_CASE_ID = "fullgeo_case02"
ENSEMBLE_CASE_ID = "fullgeo_case01"
ENSEMBLE_SOURCE_SEEDS = (9301000, 9301001, 9301002, 9301003)

STAGE7_DIR = PROJECT_DIR / "experiments/stage6_inference_causality/reports/stage7_v1_final_v2"
STAGE7_SUMMARY = STAGE7_DIR / "stage7_summary.json"
STAGE7_REPORT = STAGE7_DIR / "STAGE7_REPORT.md"
STAGE7_BASELINE = STAGE7_DIR / f"states/{HERO_CASE_ID}/zero/best_labels.pt"
STAGE7_SELECTED = STAGE7_DIR / f"states/{HERO_CASE_ID}/correct/best_labels.pt"
STAGE7_HISTORY = STAGE7_DIR / f"states/{HERO_CASE_ID}/correct/selected_event_history.json"
OLD_FIGURE_DATA = FIGURE_DATA_DIR / "figure03_seismic_structured.npz"
NATIVE_BUILDER = PROJECT_DIR / "guidance/native_geology_audit.py"

STAGE9_DIR = PROJECT_DIR / "experiments/stage9_flow_prior_posterior/reports/stage9a_prior_support_v1"
STAGE9_SUMMARY = STAGE9_DIR / "summary.json"
STAGE9_REPORT = STAGE9_DIR / "STAGE9A_REPORT.md"

STAGE14_DIR = PROJECT_DIR / "experiments/stage14_gansim_style_geo_guidance"
STAGE14_SUMMARY = STAGE14_DIR / "reports/pilot_v1/summary.json"
STAGE14_REPORT = STAGE14_DIR / "reports/pilot_v1/STAGE14_REPORT.md"
STAGE14_CASE_MANIFEST = STAGE14_DIR / f"runs/pilot_v1/cases/{ENSEMBLE_CASE_ID}/manifest.json"

FULLGEO_DIR = PROJECT_DIR / "experiments/full_structuralgeo_benchmark"
FULLGEO_REPORT = FULLGEO_DIR / "FULL_STRUCTURALGEO_BENCHMARK_BUILD_REPORT.md"

INK = "#25313A"
MUTED = "#647078"
PASS = "#2F7D5B"
LIMIT = "#A34A3F"
PALE_GREEN = "#EAF3EE"
PALE_RED = "#F7ECE9"


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(json.dumps(list(value.shape)).encode("utf-8"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _created_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _case_summary(summary: Mapping[str, object]) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    cases = [row for row in summary["native_replicas"] if row["case_id"] == HERO_CASE_ID]
    if len(cases) != 1:
        raise ValueError(f"expected one frozen Stage-7 case {HERO_CASE_ID}")
    case = cases[0]
    correct = [row for row in case["arms"] if row["optimized_by"] == "correct"]
    baseline = [row for row in case["arms"] if row["optimized_by"] == "zero"]
    if len(correct) != 1 or len(baseline) != 1:
        raise ValueError("missing frozen Stage-7 correct or zero arm")
    return case, correct[0], baseline[0]


def _hidden_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth = np.asarray(truth, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    tp = int((truth & prediction).sum())
    fp = int((~truth & prediction).sum())
    fn = int((truth & ~prediction).sum())
    return {
        "hidden_iou": tp / max(tp + fp + fn, 1),
        "hidden_precision": tp / max(tp + fp, 1),
        "hidden_recall": tp / max(tp + fn, 1),
    }


def _load_stage7() -> dict[str, object]:
    summary = read_json(STAGE7_SUMMARY)
    if summary.get("status") != "completed" or summary.get("truth_used_for_selection") is not False:
        raise ValueError("Stage-7 summary is not the completed truth-blind result")
    case, selected_row, baseline_row = _case_summary(summary)
    if selected_row.get("selection_criterion") != "minimum hard observed seismic RMSE only":
        raise ValueError("Stage-7 selector criterion changed")
    if selected_row.get("truth_used_for_selection") is not False:
        raise ValueError("Stage-7 selected row records truth use")

    native_case, native_metadata = build_structuralgeo_native_case(seed=HERO_CASE_SEED)
    if native_metadata["body_voxel_counts"] != case["native_truth_metadata"]["body_voxel_counts"]:
        raise ValueError("rebuilt hero truth differs from frozen Stage-7 metadata")
    truth = native_case.truth_labels[0, 0].cpu().numpy().astype(np.int16)
    hidden_truth = native_case.body_masks[3:].any(dim=0).cpu().numpy().astype(bool)
    baseline = load_volume(STAGE7_BASELINE, dtype=np.int16)
    selected = load_volume(STAGE7_SELECTED, dtype=np.int16)
    validate_same_shape(
        {"truth": truth, "hidden truth": hidden_truth, "baseline": baseline, "selected": selected}
    )
    condition_mask = native_case.condition_mask[0, 0].cpu().numpy().astype(bool)
    if np.any(selected[condition_mask] != truth[condition_mask]):
        raise ValueError("saved Stage-7 selection violates hard conditions")

    domain = np.zeros_like(hidden_truth, dtype=bool)
    domain[4:60, 28:56, 8:52] = True
    baseline_hidden = (baseline == TARGET_LABEL) & domain
    selected_hidden = (selected == TARGET_LABEL) & domain
    selected_metrics = _hidden_metrics(hidden_truth, selected_hidden)
    baseline_metrics = _hidden_metrics(hidden_truth, baseline_hidden)
    for computed, saved in (
        ("hidden_iou", "hidden_target_iou"),
        ("hidden_precision", "hidden_target_precision"),
        ("hidden_recall", "hidden_target_recall"),
    ):
        if abs(selected_metrics[computed] - float(selected_row[saved])) > 5e-8:
            raise ValueError(f"Stage-7 {computed} does not match frozen report")

    with np.load(OLD_FIGURE_DATA, allow_pickle=False) as data:
        sections = {
            name: np.asarray(data[name])
            for name in (
                "observed_section",
                "baseline_predicted_section",
                "selected_predicted_section",
                "baseline_residual_section",
                "selected_residual_section",
            )
        }
        full_rmse = np.asarray(data["full_volume_rmse"], dtype=np.float64)
        amplitude_limit = float(np.asarray(data["amplitude_limit"]))
        residual_limit = float(np.asarray(data["residual_limit"]))
    baseline_rmse = float(baseline_row["hard_correct_observation_rmse"])
    selected_rmse = float(selected_row["hard_correct_observation_rmse"])
    if not np.allclose(full_rmse, [baseline_rmse, selected_rmse], rtol=0, atol=2e-8):
        raise ValueError("frozen Figure-3 seismic data disagree with Stage-7 summary")
    return {
        "summary": summary,
        "case": case,
        "selected_row": selected_row,
        "baseline_row": baseline_row,
        "truth": truth,
        "baseline": baseline,
        "selected": selected,
        "hidden_truth": hidden_truth,
        "baseline_hidden": baseline_hidden,
        "selected_hidden": selected_hidden,
        "selected_metrics": selected_metrics,
        "baseline_metrics": baseline_metrics,
        "baseline_rmse": baseline_rmse,
        "selected_rmse": selected_rmse,
        "sections": sections,
        "amplitude_limit": amplitude_limit,
        "residual_limit": residual_limit,
        "well_xy": tuple(tuple(int(v) for v in point) for point in native_case.well_xy),
        "condition_mask": condition_mask,
    }


def _load_limits() -> dict[str, object]:
    stage9 = read_json(STAGE9_SUMMARY)
    if stage9.get("status") != "complete":
        raise ValueError("Stage-9A summary is not complete")
    unique = sum(int(case["ensemble"]["unique_hard_model_count"]) for case in stage9["cases"])
    support_cases = sum(bool(case["SUPPORT_PASS"]) for case in stage9["cases"])
    discrimination_cases = sum(bool(case["DISCRIMINATION_PASS"]) for case in stage9["cases"])
    if unique != 3072 or support_cases != 0 or discrimination_cases != 0:
        raise ValueError("Stage-9A frozen aggregate facts changed")
    if stage9["SUPPORT_PASS"] is not False or stage9["DISCRIMINATION_PASS"] is not False:
        raise ValueError("Stage-9A frozen verdict changed")

    stage14 = read_json(STAGE14_SUMMARY)
    if stage14.get("status") != "complete_stop_no_further_experiments":
        raise ValueError("Stage-14 summary is not the frozen terminal result")
    delta = float(stage14["overall_paired_median_hidden_label9_iou_delta"])
    positive = int(stage14["positive_case_count"])
    if abs(delta - (-0.031507776294942384)) > 1e-15 or positive != 1:
        raise ValueError("Stage-14 frozen aggregate facts changed")
    return {
        "stage9": stage9,
        "stage14": stage14,
        "stage9_unique_models": unique,
        "stage9_support_cases": support_cases,
        "stage9_discrimination_cases": discrimination_cases,
        "stage14_delta": delta,
        "stage14_positive_cases": positive,
    }


def _fullgeo_case_paths(case_id: str) -> dict[str, Path]:
    root = FULLGEO_DIR / "cases" / case_id
    return {
        "manifest": root / "manifest.json",
        "truth": root / "truth/true_model.pt",
        "hidden": root / "truth/hidden_label9_mask.pt",
        "condition_values": root / "condition/condition_values.pt",
        "condition_mask": root / "condition/condition_mask.pt",
        "surface_mask": root / "condition/surface_mask.pt",
        "well_xy": root / "condition/well_xy.json",
    }


def _load_well_xy(path: Path) -> tuple[tuple[int, int], ...]:
    payload = read_json(path)
    values = payload.get("well_xy")
    if not isinstance(values, list) or len(values) != 9:
        raise ValueError(f"unexpected well layout: {path}")
    return tuple(tuple(int(v) for v in point) for point in values)


def _load_fullgeo() -> dict[str, dict[str, object]]:
    cases: dict[str, dict[str, object]] = {}
    for case_id in FULLGEO_CASE_IDS:
        paths = _fullgeo_case_paths(case_id)
        manifest = read_json(paths["manifest"])
        if manifest.get("case_id") != case_id:
            raise ValueError(f"Full StructuralGeo case mismatch: {case_id}")
        truth = load_volume(paths["truth"], dtype=np.int16)
        hidden = load_volume(paths["hidden"]).astype(bool)
        condition_values = load_volume(paths["condition_values"], dtype=np.int16)
        condition_mask = load_volume(paths["condition_mask"]).astype(bool)
        surface_mask = load_volume(paths["surface_mask"]).astype(bool)
        validate_same_shape(
            {
                "truth": truth,
                "hidden": hidden,
                "condition values": condition_values,
                "condition mask": condition_mask,
                "surface mask": surface_mask,
            }
        )
        if not np.array_equal(condition_values[condition_mask], truth[condition_mask]):
            raise ValueError(f"condition values disagree with truth: {case_id}")
        if not np.array_equal(hidden, (truth == TARGET_LABEL) & ~condition_mask):
            raise ValueError(f"hidden label-9 mask semantics changed: {case_id}")
        cases[case_id] = {
            "paths": paths,
            "manifest": manifest,
            "truth": truth,
            "hidden": hidden,
            "condition_values": condition_values,
            "condition_mask": condition_mask,
            "surface_mask": surface_mask,
            "well_xy": _load_well_xy(paths["well_xy"]),
        }
    return cases


def _load_ensemble(case: Mapping[str, object]) -> dict[str, object]:
    manifest = read_json(STAGE14_CASE_MANIFEST)
    if manifest.get("case_id") != ENSEMBLE_CASE_ID or manifest.get("truth_loaded_by_flow_runner") is not False:
        raise ValueError("unexpected Stage-14 ensemble provenance")
    sample_rows = manifest.get("samples", [])
    recorded_seeds = tuple(int(row["source_seed"]) for row in sample_rows)
    if recorded_seeds != ENSEMBLE_SOURCE_SEEDS:
        raise ValueError("registered ensemble source seeds changed")
    samples = []
    paths = []
    condition_values = np.asarray(case["condition_values"])
    condition_mask = np.asarray(case["condition_mask"], dtype=bool)
    for row, seed in zip(sample_rows, ENSEMBLE_SOURCE_SEEDS):
        path = STAGE14_DIR / f"runs/pilot_v1/cases/{ENSEMBLE_CASE_ID}/BASELINE/source_seed_{seed}.pt"
        sample = load_volume(path, dtype=np.int16)
        if int(row["baseline_condition_violations"]) != 0:
            raise ValueError(f"saved ensemble member records a condition violation: {seed}")
        if not np.array_equal(sample[condition_mask], condition_values[condition_mask]):
            raise ValueError(f"ensemble member violates exact conditions: {seed}")
        samples.append(sample)
        paths.append(path)
    frequency = np.mean(np.stack([sample == TARGET_LABEL for sample in samples]), axis=0)
    return {
        "manifest": manifest,
        "samples": samples,
        "paths": paths,
        "frequency": frequency,
        "sample_ids": [int(row["sample_id"]) for row in sample_rows],
        "source_seeds": list(ENSEMBLE_SOURCE_SEEDS),
    }


def _plot_seismic(
    ax,
    section: np.ndarray,
    *,
    limit: float,
    cmap,
    title: str,
    panel: str,
    ylabel: bool,
):
    image = ax.imshow(
        section.T,
        origin="upper",
        extent=(-0.5, section.shape[0] - 0.5, section.shape[-1] * 8.0, 0.0),
        cmap=cmap,
        vmin=-limit,
        vmax=limit,
        aspect="auto",
        interpolation="nearest",
        rasterized=True,
    )
    ax.set_title(title, pad=1.5, fontsize=6.9)
    ax.set_xticks((0, 32, 63))
    ax.set_yticks((0, 1200, 2400))
    ax.set_xlabel("Trace $x$", labelpad=0.5, fontsize=6.2)
    if ylabel:
        ax.set_ylabel("TWT [ms]", labelpad=1.0, fontsize=6.2)
    else:
        ax.set_yticklabels([])
    panel_label(ax, panel)
    return image


def _metric_block(ax, y: float, title: str, lines: Sequence[str], *, positive: bool) -> None:
    face = PALE_GREEN if positive else PALE_RED
    edge = PASS if positive else LIMIT
    patch = FancyBboxPatch(
        (0.03, y - 0.235),
        0.94,
        0.215,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor=face,
        edgecolor=edge,
        linewidth=0.75,
    )
    ax.add_patch(patch)
    ax.add_patch(
        Rectangle((0.03, y - 0.235), 0.022, 0.215, transform=ax.transAxes, color=edge, linewidth=0)
    )
    ax.text(0.075, y - 0.045, title, transform=ax.transAxes, va="top", ha="left", fontsize=7.1, fontweight="bold", color=INK)
    ax.text(0.075, y - 0.096, "\n".join(lines), transform=ax.transAxes, va="top", ha="left", fontsize=6.45, linespacing=1.18, color=INK)


def _make_main_figure(stage7: Mapping[str, object], limits: Mapping[str, object]) -> dict[str, str]:
    camera = CAMERA_PRESETS["perspective_iso"]
    geology_images = (
        render_categorical_volume_3d(stage7["truth"], camera=camera, context_opacity=0.27, borehole_xy=stage7["well_xy"], condition_mask=stage7["condition_mask"]),
        render_categorical_volume_3d(stage7["baseline"], camera=camera, context_opacity=0.27, borehole_xy=stage7["well_xy"], condition_mask=stage7["condition_mask"]),
        render_categorical_volume_3d(stage7["selected"], camera=camera, context_opacity=0.27, borehole_xy=stage7["well_xy"], condition_mask=stage7["condition_mask"]),
        render_label_comparison_3d(stage7["hidden_truth"], stage7["selected_hidden"], camera=camera),
    )
    fig = plt.figure(figsize=(mm_to_inches(DOUBLE_COLUMN_MM), 5.45))
    outer = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.38, 1.25, 0.64),
        left=0.016,
        right=0.992,
        top=0.91,
        bottom=0.075,
        wspace=0.13,
    )

    geology = outer[0].subgridspec(2, 2, wspace=0.02, hspace=0.12)
    geology_titles = (
        "Ground-truth geology$^{*}$",
        "Concealed baseline geology",
        "Hard-seismic selected geology",
        "Selected body + truth outline$^{*}$",
    )
    for index, (image, title) in enumerate(zip(geology_images, geology_titles)):
        ax = fig.add_subplot(geology[index // 2, index % 2])
        show_render(ax, image)
        ax.set_title(title, pad=0.5, fontsize=6.8)
        panel_label(ax, f"A{index + 1}")
    fig.text(0.022, 0.977, "A  3-D geological recovery", ha="left", va="top", fontsize=8.1, fontweight="bold", color=INK)

    seismic = outer[1].subgridspec(3, 3, height_ratios=(1.0, 1.0, 0.055), wspace=0.08, hspace=0.28)
    sections = stage7["sections"]
    amp = _plot_seismic(
        fig.add_subplot(seismic[0, 0]), sections["observed_section"], limit=stage7["amplitude_limit"],
        cmap=SEISMIC_CMAP, title=f"Observed ($y={SLICE_Y}$)", panel="B1", ylabel=True,
    )
    _plot_seismic(
        fig.add_subplot(seismic[0, 1]), sections["baseline_predicted_section"], limit=stage7["amplitude_limit"],
        cmap=SEISMIC_CMAP, title=f"Baseline predicted\nRMSE={stage7['baseline_rmse']:.5f}", panel="B2", ylabel=False,
    )
    _plot_seismic(
        fig.add_subplot(seismic[0, 2]), sections["selected_predicted_section"], limit=stage7["amplitude_limit"],
        cmap=SEISMIC_CMAP, title=f"Selected predicted\nRMSE={stage7['selected_rmse']:.5f}", panel="B3", ylabel=False,
    )
    residual = _plot_seismic(
        fig.add_subplot(seismic[1, 0]), sections["baseline_residual_section"], limit=stage7["residual_limit"],
        cmap=RESIDUAL_CMAP, title="Baseline residual", panel="B4", ylabel=True,
    )
    _plot_seismic(
        fig.add_subplot(seismic[1, 1]), sections["selected_residual_section"], limit=stage7["residual_limit"],
        cmap=RESIDUAL_CMAP, title="Selected residual", panel="B5", ylabel=False,
    )
    note_ax = fig.add_subplot(seismic[1, 2])
    note_ax.axis("off")
    note_ax.text(
        0.03,
        0.94,
        "Selector\nfull $64\\times64\\times320$\nhard seismic volume\n\nDisplay\nfixed $y=42$ slice\nshared scales",
        transform=note_ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.35,
        color=MUTED,
        linespacing=1.12,
    )
    cax1 = fig.add_subplot(seismic[2, 0])
    fig.colorbar(amp, cax=cax1, orientation="horizontal").ax.tick_params(length=1.5, labelsize=5.5, pad=0.5)
    cax2 = fig.add_subplot(seismic[2, 1])
    fig.colorbar(residual, cax=cax2, orientation="horizontal").ax.tick_params(length=1.5, labelsize=5.5, pad=0.5)
    fig.add_subplot(seismic[2, 2]).axis("off")
    fig.text(0.482, 0.977, "B  Seismic consistency", ha="left", va="top", fontsize=8.1, fontweight="bold", color=INK)

    summary_ax = fig.add_subplot(outer[2])
    summary_ax.axis("off")
    panel_label(summary_ax, "C")
    summary_ax.text(0.12, 0.985, "Inference limits", transform=summary_ax.transAxes, ha="left", va="top", fontsize=8.0, fontweight="bold", color=INK)
    stage7_range = [
        float(next(arm for arm in case["arms"] if arm["optimized_by"] == "correct")["hidden_target_iou"])
        for case in stage7["summary"]["native_replicas"]
    ]
    _metric_block(
        summary_ax,
        0.91,
        "Structured hard inference",
        (
            f"correct rank-first: {sum(bool(case['correct_optimized_is_best_against_correct']) for case in stage7['summary']['native_replicas'])}/{len(stage7_range)}",
            f"hidden IoU: {min(stage7_range):.3f}–{max(stage7_range):.3f}",
        ),
        positive=True,
    )
    _metric_block(
        summary_ax,
        0.62,
        "Frozen Flow prior",
        (f"unique models: {limits['stage9_unique_models']}", "support: 0/3", "discrimination: 0/3"),
        positive=False,
    )
    _metric_block(
        summary_ax,
        0.30,
        "Geo-probability $\\rightarrow$ Flow",
        (f"median $\\Delta$ hidden IoU: {limits['stage14_delta']:.4f}", f"positive cases: {limits['stage14_positive_cases']}/5"),
        positive=False,
    )
    summary_ax.text(
        0.04,
        0.015,
        "$^{*}$ retrospective truth\nselector: hard seismic RMSE only",
        transform=summary_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.5,
        color=MUTED,
        linespacing=1.12,
    )
    return save_figure(fig, FIGURES_DIR / FIGURE_ID, title="Structured hard inference and frozen-Flow limits")


def _save_image_montage(
    images: Sequence[np.ndarray],
    titles: Sequence[str],
    stem: Path,
    *,
    ncols: int,
    title: str,
    panel_prefix: str = "",
) -> dict[str, str]:
    nrows = int(np.ceil(len(images) / ncols))
    fig = plt.figure(figsize=(mm_to_inches(DOUBLE_COLUMN_MM), 2.38 * nrows))
    grid = fig.add_gridspec(nrows, ncols, left=0.018, right=0.992, top=0.80, bottom=0.025, wspace=0.025, hspace=0.09)
    for index in range(nrows * ncols):
        ax = fig.add_subplot(grid[index // ncols, index % ncols])
        if index >= len(images):
            ax.axis("off")
            continue
        show_render(ax, images[index])
        ax.set_title(titles[index], pad=0.5, fontsize=7.2)
        if panel_prefix:
            panel_label(ax, f"{panel_prefix}{index + 1}")
    fig.suptitle(title, y=0.972, fontsize=9.0, fontweight="bold", color=INK)
    return save_figure(fig, stem, title=title)


def _make_gallery(
    stage7: Mapping[str, object],
    fullgeo: Mapping[str, Mapping[str, object]],
    ensemble: Mapping[str, object],
) -> dict[str, dict[str, str]]:
    gallery_dir = FIGURES_DIR / "3d_gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, str]] = {}
    for camera_name in ("perspective_iso", "perspective_oblique", "top_oblique"):
        camera = CAMERA_PRESETS[camera_name]
        images = [
            render_categorical_volume_3d(
                stage7[key],
                camera=camera,
                context_opacity=0.30,
                borehole_xy=stage7["well_xy"],
                condition_mask=stage7["condition_mask"],
            )
            for key in ("truth", "baseline", "selected")
        ]
        outputs[f"stage7_{camera_name}"] = _save_image_montage(
            images,
            ("Truth$^{*}$", "Concealed baseline", "Hard-seismic selected"),
            gallery_dir / f"stage7_{camera_name}",
            ncols=3,
            title=f"Stage 7 hero — {camera_name.replace('_', ' ')}",
            panel_prefix="H",
        )

    montage_camera = CAMERA_PRESETS["perspective_iso"]
    montage_images = [
        render_categorical_volume_3d(fullgeo[case_id]["truth"], camera=montage_camera, context_opacity=0.31)
        for case_id in FULLGEO_CASE_IDS
    ]
    outputs["full_structuralgeo_montage"] = _save_image_montage(
        montage_images,
        FULLGEO_CASE_IDS,
        gallery_dir / "full_structuralgeo_montage",
        ncols=5,
        title="Prospectively registered Full StructuralGeo benchmark",
        panel_prefix="G",
    )

    sparse = fullgeo[SPARSE_CASE_ID]
    sparse_image = render_sparse_constraints_3d(
        sparse["truth"],
        sparse["condition_values"],
        sparse["surface_mask"],
        sparse["hidden"],
        well_xy=sparse["well_xy"],
        camera=CAMERA_PRESETS["perspective_oblique"],
    )
    context_image = render_categorical_volume_3d(
        sparse["truth"], camera=CAMERA_PRESETS["perspective_oblique"], context_opacity=0.28
    )
    outputs["sparse_constraints_hidden_target"] = _save_image_montage(
        (context_image, sparse_image),
        ("Full 3-D truth context$^{*}$", "Surface + 9 wells vs hidden label 9$^{*}$"),
        gallery_dir / "sparse_constraints_hidden_target",
        ncols=2,
        title=f"Sparse geological constraints versus hidden deep target — {SPARSE_CASE_ID}",
        panel_prefix="S",
    )

    ensemble_images = [
        render_categorical_volume_3d(
            sample,
            camera=montage_camera,
            context_opacity=0.29,
            borehole_xy=fullgeo[ENSEMBLE_CASE_ID]["well_xy"],
            condition_mask=fullgeo[ENSEMBLE_CASE_ID]["condition_mask"],
        )
        for sample in ensemble["samples"]
    ]
    ensemble_images.append(render_label_frequency_3d(ensemble["frequency"], camera=montage_camera))
    ensemble_titles = [f"sample {sample_id} / seed {seed}" for sample_id, seed in zip(ensemble["sample_ids"], ensemble["source_seeds"])]
    ensemble_titles.append("Label-9 occurrence frequency")
    outputs["condition_exact_cfm_ensemble"] = _save_image_montage(
        ensemble_images,
        ensemble_titles,
        gallery_dir / "condition_exact_cfm_ensemble",
        ncols=5,
        title=f"Condition-exact frozen-CFM ensemble — {ENSEMBLE_CASE_ID}",
        panel_prefix="E",
    )
    return outputs


def _flatten_gallery_outputs(outputs: Mapping[str, Mapping[str, str]]) -> dict[str, str]:
    return {f"{name}_{kind}": path for name, group in outputs.items() for kind, path in group.items()}


def _tensor_record(path: Path, array: np.ndarray, role: str) -> dict[str, object]:
    record = source_record(path, role)
    record["tensor_content_sha256"] = _array_sha256(array)
    record["shape"] = list(array.shape)
    record["dtype"] = str(array.dtype)
    return record


def generate() -> dict[str, object]:
    configure_matplotlib()
    ensure_output_dirs()
    stage7 = _load_stage7()
    limits = _load_limits()
    fullgeo = _load_fullgeo()
    ensemble = _load_ensemble(fullgeo[ENSEMBLE_CASE_ID])

    stage7_iou_range = [
        float(next(arm for arm in case["arms"] if arm["optimized_by"] == "correct")["hidden_target_iou"])
        for case in stage7["summary"]["native_replicas"]
    ]
    stage7_rank_first_count = sum(
        bool(case["correct_optimized_is_best_against_correct"])
        for case in stage7["summary"]["native_replicas"]
    )
    figure_data = {
        "schema": "figure03_joint_inference_data_v1",
        "case_id": HERO_CASE_ID,
        "slice_indices": {"seismic_y": SLICE_Y},
        "metrics": {
            "stage7": {
                "baseline_hard_seismic_rmse": stage7["baseline_rmse"],
                "selected_hard_seismic_rmse": stage7["selected_rmse"],
                "baseline_hidden": stage7["baseline_metrics"],
                "selected_hidden": stage7["selected_metrics"],
                "native_hidden_iou_range": [min(stage7_iou_range), max(stage7_iou_range)],
                "correct_observation_rank_first": stage7_rank_first_count,
                "n_native_cases": len(stage7_iou_range),
            },
            "stage9a": {
                "unique_hard_models": limits["stage9_unique_models"],
                "support_cases": limits["stage9_support_cases"],
                "discrimination_cases": limits["stage9_discrimination_cases"],
                "n_cases": 3,
            },
            "stage14": {
                "overall_paired_median_hidden_label9_iou_delta": limits["stage14_delta"],
                "positive_case_count": limits["stage14_positive_cases"],
                "n_cases": 5,
            },
        },
        "display_scales": {
            "seismic_shared_symmetric_limit": stage7["amplitude_limit"],
            "residual_shared_symmetric_limit": stage7["residual_limit"],
            "policy": "frozen pooled 99.5th-percentile limits from figure03_seismic_structured.npz",
        },
        "selection": {
            "criterion": "minimum hard observed seismic RMSE only",
            "truth_used_for_selection": False,
            "truth_metrics_role": "retrospective evaluation only",
        },
    }
    data_path = FIGURE_DATA_DIR / f"{FIGURE_ID}.json"
    write_json(data_path, figure_data)

    main_outputs = _make_main_figure(stage7, limits)
    gallery_outputs = _make_gallery(stage7, fullgeo, ensemble)

    generation = generation_record(SCRIPT_PATH)
    generation["generated_at_utc"] = _created_at()
    main_sources = (
        (STAGE7_SUMMARY, "authoritative Stage-7 machine summary"),
        (STAGE7_REPORT, "authoritative Stage-7 report"),
        (STAGE7_BASELINE, "Stage-7 concealed baseline categorical tensor"),
        (STAGE7_SELECTED, "Stage-7 hard-seismic-selected categorical tensor"),
        (STAGE7_HISTORY, "Stage-7 selected structured-object history"),
        (OLD_FIGURE_DATA, "frozen exact Stage-7 seismic display data"),
        (NATIVE_BUILDER, "frozen deterministic hero truth builder"),
        (STAGE9_SUMMARY, "authoritative Stage-9A machine summary"),
        (STAGE9_REPORT, "authoritative Stage-9A report"),
        (STAGE14_SUMMARY, "authoritative Stage-14 machine summary"),
        (STAGE14_REPORT, "authoritative Stage-14 report"),
        (STYLE_PATH, "shared paper style and reusable 3-D renderer"),
        (data_path, "exact Figure-3 plotted metrics and display policy"),
    )
    main_manifest = {
        "schema_version": "paper_figure_manifest_v2",
        "figure_id": FIGURE_ID,
        "git_head": generation["git_head"],
        "generation_timestamp_utc": generation["generated_at_utc"],
        "source_files": [source_record(path, role) for path, role in main_sources],
        "source_tensor_hashes": {
            "rebuilt_truth": {
                "tensor_content_sha256": _array_sha256(stage7["truth"]),
                "reconstruction_rule": "build_structuralgeo_native_case(seed=20260809).truth_labels[0,0]",
                "truth_visibility": "retrospective display only",
            },
            "baseline": _tensor_record(STAGE7_BASELINE, stage7["baseline"], "concealed baseline hard labels"),
            "selected": _tensor_record(STAGE7_SELECTED, stage7["selected"], "hard-seismic-selected hard labels"),
            "hidden_truth": {
                "tensor_content_sha256": _array_sha256(stage7["hidden_truth"].astype(np.uint8)),
                "reconstruction_rule": "native_case.body_masks[3:].any(dim=0)",
                "truth_visibility": "retrospective display/evaluation only",
            },
        },
        "case_ids": [HERO_CASE_ID, *[case["case_id"] for case in limits["stage9"]["cases"]], *FULLGEO_CASE_IDS],
        "sample_ids": [],
        "camera": {"main": CAMERA_PRESETS["perspective_iso"], "camera_consistent_within_panel_A": True},
        "slice_indices": {"seismic_y": SLICE_Y},
        "categorical_color_map": generation["style"]["label_colors"],
        "metrics_source_paths": [
            {"metric": "Stage7 hero hard seismic RMSE and hidden metrics", "source_path": str(STAGE7_SUMMARY.relative_to(REPOSITORY_ROOT)), "json_key": "native_replicas[case_id=native_seed20260809].arms[optimized_by=zero|correct]"},
            {"metric": "Stage7 hidden IoU range and correct rank-first 3/3", "source_path": str(STAGE7_SUMMARY.relative_to(REPOSITORY_ROOT)), "json_key": "native_replicas[*].arms[optimized_by=correct].hidden_target_iou; correct_optimized_is_best_against_correct"},
            {"metric": "Stage9A 3072 unique models, support 0/3, discrimination 0/3", "source_path": str(STAGE9_SUMMARY.relative_to(REPOSITORY_ROOT)), "json_key": "cases[*].ensemble.unique_hard_model_count; cases[*].SUPPORT_PASS; cases[*].DISCRIMINATION_PASS"},
            {"metric": "Stage14 paired median delta and positive cases", "source_path": str(STAGE14_SUMMARY.relative_to(REPOSITORY_ROOT)), "json_key": "overall_paired_median_hidden_label9_iou_delta; positive_case_count"},
        ],
        "truth_visibility_status": {
            "A1_ground_truth": "retrospective synthetic-benchmark display",
            "A2_baseline": "inference output / truth-blind concealed construction",
            "A3_selected": "truth-blind hard-seismic selection",
            "A4_truth_outline": "retrospective geometry evaluation",
            "B_observation": "synthetic inference-visible hard seismic generated once from truth",
            "C_stage9a": "aggregate inference-visible and retrospective gate facts; oracle candidate not displayed",
            "C_stage14": "paired experiment metrics; not measured field geophysics",
        },
        "selection_criterion": "Stage 7 candidate selection = minimum hard observed seismic RMSE only",
        "truth_metrics": "retrospective evaluation only; never used for proposal selection, ranking, or tuning",
        "scientific_boundaries": {
            "stage7_is_cfm_posterior": False,
            "stage7_description": "bounded structured hard-geophysics inference",
            "stage9a_truth_oracle_candidate_displayed": False,
            "stage14_is_measured_geophysics": False,
        },
        "metrics": figure_data["metrics"],
        "generation": generation,
        "outputs": output_records(main_outputs),
        "quality_control": {
            "condition_violations": 0,
            "shared_seismic_scale": True,
            "shared_residual_scale": True,
            "camera_consistency": True,
            "stage7_metrics_match_frozen_summary": True,
            "stage9a_counts_match_frozen_summary": True,
            "stage14_summary_matches_frozen_report": True,
        },
    }
    main_manifest_path = MANIFESTS_DIR / f"{FIGURE_ID}.json"
    write_json(main_manifest_path, main_manifest)

    gallery_source_records = [
        source_record(STAGE7_SUMMARY, "authoritative Stage-7 machine summary"),
        source_record(STAGE7_BASELINE, "Stage-7 concealed baseline categorical tensor"),
        source_record(STAGE7_SELECTED, "Stage-7 hard-seismic-selected categorical tensor"),
        source_record(NATIVE_BUILDER, "frozen deterministic hero truth builder"),
        source_record(STYLE_PATH, "shared paper style and reusable 3-D renderer"),
        source_record(FULLGEO_REPORT, "Full StructuralGeo build report"),
    ]
    tensor_records = [
        {
            "role": "Stage-7 rebuilt hero truth",
            "tensor_content_sha256": _array_sha256(stage7["truth"]),
            "shape": list(stage7["truth"].shape),
            "dtype": str(stage7["truth"].dtype),
            "reconstruction_rule": "build_structuralgeo_native_case(seed=20260809).truth_labels[0,0]",
        },
        _tensor_record(STAGE7_BASELINE, stage7["baseline"], "Stage-7 concealed baseline"),
        _tensor_record(STAGE7_SELECTED, stage7["selected"], "Stage-7 hard-seismic-selected geology"),
    ]
    for case_id in FULLGEO_CASE_IDS:
        case = fullgeo[case_id]
        gallery_source_records.extend(
            source_record(case["paths"][name], f"{case_id} {name}")
            for name in ("manifest", "truth", "hidden", "condition_values", "condition_mask", "surface_mask", "well_xy")
        )
        tensor_records.append(_tensor_record(case["paths"]["truth"], case["truth"], f"{case_id} truth"))
    gallery_source_records.append(source_record(STAGE14_CASE_MANIFEST, "frozen ensemble case manifest"))
    for path, sample in zip(ensemble["paths"], ensemble["samples"]):
        tensor_records.append(_tensor_record(path, sample, "condition-exact frozen-CFM ensemble member"))
    flat_gallery_outputs = _flatten_gallery_outputs(gallery_outputs)
    gallery_manifest = {
        "schema_version": "paper_3d_gallery_manifest_v1",
        "gallery_id": GALLERY_ID,
        "git_head": generation["git_head"],
        "generation_timestamp_utc": generation["generated_at_utc"],
        "source_files": gallery_source_records,
        "source_tensor_hashes": tensor_records,
        "case_ids": {"hero": HERO_CASE_ID, "full_structuralgeo": list(FULLGEO_CASE_IDS), "sparse_constraints": SPARSE_CASE_ID, "ensemble": ENSEMBLE_CASE_ID},
        "sample_ids": {"ensemble_sample_ids": ensemble["sample_ids"], "ensemble_source_seeds": ensemble["source_seeds"]},
        "sample_selection": "all four preregistered Stage-14 BASELINE source seeds for the first frozen protocol case; no truth or seismic ranking",
        "camera": {
            "hero_views": {name: CAMERA_PRESETS[name] for name in ("perspective_iso", "perspective_oblique", "top_oblique")},
            "fullgeo_montage": CAMERA_PRESETS["perspective_iso"],
            "sparse_constraints": CAMERA_PRESETS["perspective_oblique"],
            "ensemble": CAMERA_PRESETS["perspective_iso"],
            "identical_within_each_comparison": True,
        },
        "categorical_color_map": generation["style"]["label_colors"],
        "truth_visibility_status": {
            "hero_truth": "retrospective display only",
            "fullgeo_montage": "registered benchmark truth display",
            "sparse_constraints": "truth/hidden target explanatory display",
            "ensemble_members": "truth-blind frozen-CFM baseline outputs",
            "ensemble_frequency": "derived only from the four displayed ensemble members",
        },
        "selection_criterion": "no gallery member was selected or ranked by truth, seismic fit, or visual attractiveness",
        "outputs": output_records(flat_gallery_outputs),
        "quality_control": {
            "registered_fullgeo_cases_exactly_once": list(FULLGEO_CASE_IDS),
            "ensemble_condition_violations": 0,
            "ensemble_member_count": len(ensemble["samples"]),
            "camera_consistency": True,
            "all_outputs_pdf_svg_png": True,
        },
        "generation": generation,
    }
    gallery_manifest_path = MANIFESTS_DIR / "3d_gallery.json"
    write_json(gallery_manifest_path, gallery_manifest)
    return {
        "figure": FIGURE_ID,
        "outputs": main_outputs,
        "manifest": str(main_manifest_path.relative_to(PROJECT_DIR)),
        "gallery_outputs": gallery_outputs,
        "gallery_manifest": str(gallery_manifest_path.relative_to(PROJECT_DIR)),
    }


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    parse_args()
    print(generate())


if __name__ == "__main__":
    main()
