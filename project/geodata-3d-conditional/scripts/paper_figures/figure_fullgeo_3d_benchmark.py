#!/usr/bin/env python3
"""Render the five preregistered Full StructuralGeo benchmark truths in 3-D."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.paper_figures.style import (
    CAMERA_PRESETS,
    DOUBLE_COLUMN_MM,
    FIGURE_DATA_DIR,
    FIGURES_DIR,
    LABEL9_COLOR,
    LABEL_COLORS,
    MANIFESTS_DIR,
    OBSERVATION_COLOR,
    configure_matplotlib,
    ensure_output_dirs,
    generation_record,
    load_volume,
    mm_to_inches,
    output_records,
    read_json,
    render_categorical_volume_3d,
    render_target_only_3d,
    save_figure,
    sha256,
    show_render,
    source_record,
    write_json,
)


SCRIPT_PATH = Path(__file__).resolve()
STYLE_PATH = SCRIPT_PATH.with_name("style.py")
FIGURE_ID = "fullgeo_3d_benchmark"
PRESENTATION_ID = "fullgeo_3d_benchmark_presentation"
TARGET_LABEL = 9
CASE_IDS = tuple(f"fullgeo_case{index:02d}" for index in range(1, 6))
BENCHMARK_DIR = PROJECT_DIR / "experiments/full_structuralgeo_benchmark"
BUILD_REPORT = BENCHMARK_DIR / "FULL_STRUCTURALGEO_BENCHMARK_BUILD_REPORT.md"
CAMERA_NAME = "perspective_iso"
CAMERA = CAMERA_PRESETS[CAMERA_NAME]

INK = "#25313A"
MUTED = "#65727A"


def _case_paths(case_id: str) -> dict[str, Path]:
    root = BENCHMARK_DIR / "cases" / case_id
    return {
        "manifest": root / "manifest.json",
        "truth": root / "truth/true_model.pt",
        "well_xy": root / "condition/well_xy.json",
    }


def _load_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for case_id in CASE_IDS:
        paths = _case_paths(case_id)
        manifest = read_json(paths["manifest"])
        wells = read_json(paths["well_xy"])
        truth = load_volume(paths["truth"], dtype=np.int16)
        if manifest.get("case_id") != case_id:
            raise ValueError(f"case-id mismatch for {case_id}")
        if list(truth.shape) != [64, 64, 64] or manifest.get("resolution") != [64, 64, 64]:
            raise ValueError(f"unexpected resolution for {case_id}")
        count = int(np.count_nonzero(truth == TARGET_LABEL))
        if count != int(manifest["raw_label_counts"][str(TARGET_LABEL)]):
            raise ValueError(f"label-9 count differs from frozen manifest for {case_id}")
        expected_file_hash = manifest["saved_file_sha256"]["truth/true_model.pt"]
        if sha256(paths["truth"]) != expected_file_hash:
            raise ValueError(f"truth file hash differs from frozen manifest for {case_id}")
        well_xy = tuple(tuple(int(value) for value in point) for point in wells["well_xy"])
        if len(well_xy) != 9 or wells.get("truth_independent") is not True:
            raise ValueError(f"invalid preregistered well layout for {case_id}")
        cases.append(
            {
                "case_id": case_id,
                "truth": truth,
                "label9_count": count,
                "root_seed": int(manifest["root_seed"]),
                "well_xy": well_xy,
                "paths": paths,
                "manifest": manifest,
            }
        )
    if [case["case_id"] for case in cases] != list(CASE_IDS):
        raise ValueError("benchmark cases are not the five preregistered cases in fixed order")
    return cases


def _render_cases(cases: list[dict[str, object]]) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    context_images: list[np.ndarray] = []
    target_images: list[np.ndarray] = []
    presentation_images: list[np.ndarray] = []
    for case in cases:
        truth = case["truth"]
        wells = case["well_xy"]
        context_images.append(
            render_categorical_volume_3d(
                truth,
                borehole_xy=wells,
                camera=CAMERA,
                context_opacity=0.24,
                target_opacity=0.98,
                window_size=(900, 820),
            )
        )
        target_images.append(
            render_target_only_3d(
                truth == TARGET_LABEL,
                camera=CAMERA,
                window_size=(900, 820),
            )
        )
        presentation_images.append(
            render_categorical_volume_3d(
                truth,
                borehole_xy=wells,
                camera=CAMERA,
                context_opacity=0.42,
                target_opacity=1.0,
                window_size=(1000, 900),
            )
        )
    return context_images, target_images, presentation_images


def _publication_figure(
    cases: list[dict[str, object]],
    context_images: list[np.ndarray],
    target_images: list[np.ndarray],
) -> dict[str, str]:
    fig = plt.figure(figsize=(mm_to_inches(DOUBLE_COLUMN_MM), 4.63))
    grid = fig.add_gridspec(2, 5, left=0.018, right=0.995, bottom=0.085, top=0.875, wspace=0.015, hspace=0.105)
    for column, case in enumerate(cases):
        top = fig.add_subplot(grid[0, column])
        bottom = fig.add_subplot(grid[1, column])
        show_render(top, context_images[column])
        show_render(bottom, target_images[column])
        top.set_title(
            f"Case {column + 1}\nseed {case['root_seed']}",
            fontsize=7.2,
            fontweight="bold",
            color=INK,
            pad=1.0,
        )
        bottom.text(
            0.5,
            -0.055,
            f"label 9: {case['label9_count']:,} voxels",
            transform=bottom.transAxes,
            ha="center",
            va="top",
            fontsize=6.35,
            color=INK,
        )
    fig.text(0.019, 0.963, "Full StructuralGeo benchmark: five prospectively registered 3-D cases", fontsize=9.5, fontweight="bold", color=INK, va="top")
    fig.text(0.019, 0.908, "a  Full categorical context + fixed nine-well layout", fontsize=6.8, color=MUTED, va="top")
    fig.text(0.019, 0.492, "b  Target-only geometry (raw label 9)", fontsize=6.8, color=MUTED, va="top")
    handles = [
        Patch(facecolor=LABEL9_COLOR, edgecolor="none", label="label 9 target"),
        Patch(facecolor=OBSERVATION_COLOR, edgecolor="none", label="nine fixed wells"),
        Patch(facecolor="#87929A", edgecolor="none", alpha=0.45, label="other raw categorical labels"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.50, 0.012), ncol=3, frameon=False, fontsize=6.4, handlelength=1.4)
    return save_figure(fig, FIGURES_DIR / FIGURE_ID, title="Full StructuralGeo five-case 3-D benchmark")


def _presentation_figure(cases: list[dict[str, object]], images: list[np.ndarray]) -> dict[str, str]:
    fig = plt.figure(figsize=(13.333, 4.35))
    grid = fig.add_gridspec(1, 5, left=0.018, right=0.992, bottom=0.13, top=0.82, wspace=0.015)
    for column, (case, image) in enumerate(zip(cases, images)):
        ax = fig.add_subplot(grid[0, column])
        show_render(ax, image)
        ax.set_title(f"Case {column + 1}  ·  {case['label9_count']:,} target voxels", fontsize=12.0, fontweight="bold", color=INK, pad=4.0)
    fig.text(0.02, 0.965, "Full StructuralGeo benchmark", fontsize=22, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.885, "Five frozen 64³ truths · identical camera · raw label 9 highlighted · nine preregistered wells", fontsize=12.5, color=MUTED, va="top")
    handles = [
        Patch(facecolor=LABEL9_COLOR, edgecolor="none", label="label 9 target"),
        Patch(facecolor=OBSERVATION_COLOR, edgecolor="none", label="fixed wells"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.50, 0.015), ncol=2, frameon=False, fontsize=10.5)
    return save_figure(fig, FIGURES_DIR / PRESENTATION_ID, title="Full StructuralGeo benchmark presentation montage")


def generate() -> dict[str, object]:
    configure_matplotlib()
    ensure_output_dirs()
    cases = _load_cases()
    context_images, target_images, presentation_images = _render_cases(cases)

    data = {
        "schema": "fullgeo_3d_benchmark_data_v1",
        "case_ids": list(CASE_IDS),
        "case_selection": "all five prospectively registered benchmark cases, fixed order; no replacement or visual selection",
        "target_label": TARGET_LABEL,
        "target_semantics": "raw StructuralGeo label 9; demonstration target lithology",
        "resolution": [64, 64, 64],
        "camera_name": CAMERA_NAME,
        "camera": CAMERA,
        "categorical_color_map": {str(key): value for key, value in LABEL_COLORS.items()},
        "cases": [
            {
                "case_id": case["case_id"],
                "root_seed": case["root_seed"],
                "label9_voxel_count": case["label9_count"],
                "well_count": len(case["well_xy"]),
            }
            for case in cases
        ],
    }
    data_path = FIGURE_DATA_DIR / f"{FIGURE_ID}.json"
    write_json(data_path, data)
    publication_outputs = _publication_figure(cases, context_images, target_images)
    presentation_outputs = _presentation_figure(cases, presentation_images)
    flat_outputs = {**publication_outputs, **{f"presentation_{kind}": value for kind, value in presentation_outputs.items()}}

    source_files = [
        source_record(BUILD_REPORT, "authoritative benchmark construction report"),
        source_record(STYLE_PATH, "shared paper visual language and 3-D renderer"),
        source_record(data_path, "exact plotted case inventory and display policy"),
    ]
    tensor_hashes = []
    for case in cases:
        paths = case["paths"]
        source_files.extend(
            [
                source_record(paths["manifest"], f"{case['case_id']} frozen manifest"),
                source_record(paths["truth"], f"{case['case_id']} truth tensor"),
                source_record(paths["well_xy"], f"{case['case_id']} preregistered well layout"),
            ]
        )
        frozen_manifest = case["manifest"]
        tensor_hashes.append(
            {
                "case_id": case["case_id"],
                "path": str(paths["truth"].relative_to(REPOSITORY_ROOT)),
                "saved_file_sha256": frozen_manifest["saved_file_sha256"]["truth/true_model.pt"],
                "tensor_content_sha256": frozen_manifest["tensor_content_hashes"]["truth/true_model.pt"],
                "shape": list(case["truth"].shape),
                "dtype": str(case["truth"].dtype),
                "truth_visibility": "retrospective registered benchmark truth display",
            }
        )

    generation = generation_record(SCRIPT_PATH)
    manifest = {
        "schema_version": "paper_3d_benchmark_manifest_v1",
        "figure_id": FIGURE_ID,
        "source_files": source_files,
        "metric_extraction_keys": [
            {
                "metric": "raw label-9 voxel count",
                "source_path": "experiments/full_structuralgeo_benchmark/cases/<case_id>/manifest.json",
                "json_key": "raw_label_counts.9",
                "validation": "recounted directly from each true_model.pt",
            }
        ],
        "case_ids": list(CASE_IDS),
        "sample_ids": [],
        "source_tensor_hashes": tensor_hashes,
        "camera": {"name": CAMERA_NAME, "parameters": CAMERA, "identical_for_all_cases_and_rows": True},
        "categorical_color_map": generation["style"]["label_colors"],
        "target": {"raw_label": 9, "color": LABEL9_COLOR, "meaning": "demonstration target lithology; raw labels 10–13 are not merged"},
        "truth_visibility_status": {
            "all_models": "retrospective display of prospectively registered synthetic benchmark truths",
            "selection_firewall": "no downstream metric, Flow output, seismic result, or visual attractiveness used for case selection",
            "well_layout": "inference-visible, truth-independent, prospectively fixed",
        },
        "case_selection": data["case_selection"],
        "generation": generation,
        "outputs": output_records(flat_outputs),
        "quality_control": {
            "registered_cases_exactly_once": list(CASE_IDS),
            "fixed_camera": True,
            "fixed_categorical_palette": True,
            "truth_file_hashes_match_frozen_manifests": True,
            "label9_counts_recomputed": True,
            "publication_and_presentation_variants": True,
        },
    }
    manifest_path = MANIFESTS_DIR / f"{FIGURE_ID}.json"
    write_json(manifest_path, manifest)
    return {
        "figure": FIGURE_ID,
        "outputs": publication_outputs,
        "presentation_outputs": presentation_outputs,
        "manifest": str(manifest_path.relative_to(PROJECT_DIR)),
    }


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    parse_args()
    print(generate())


if __name__ == "__main__":
    main()
