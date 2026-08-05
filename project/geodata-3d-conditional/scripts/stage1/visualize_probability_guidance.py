#!/usr/bin/env python3
"""Render paper-style Phase-1 truth/baseline/guided 3-D comparisons.

The renderer uses voxel surfaces and a fixed L-shaped cutaway inspired by the
figures in ``Synthetic Geology: Structural Geology Meets Deep Learning``. It
also exports VTK image data for interactive inspection in PyVista or ParaView.

The ensemble panels are empirical occurrence probabilities calculated from
saved hard-label realizations. They are not per-sample soft decoder outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for import_root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import inference_runtime as runtime


# Stable categorical colors shared by all panels. Raw labels are -1..13.
LABEL_COLORS = {
    -1: "#ffffff",
    0: "#6d4c41",
    1: "#8c1d40",
    2: "#08306b",
    3: "#2171b5",
    4: "#22a7c7",
    5: "#16b89a",
    6: "#63c74d",
    7: "#b8de29",
    8: "#f3e55b",
    9: "#f29e2e",
    10: "#e76818",
    11: "#d73027",
    12: "#b2182b",
    13: "#7b3294",
}
TRUTH_TARGET_COLOR = "#f2c14e"
BASELINE_TARGET_COLOR = "#2878b5"
GUIDED_TARGET_COLOR = "#2a9d55"


def _import_pyvista():
    try:
        import pyvista as pv
    except ImportError as error:
        raise RuntimeError(
            "PyVista/VTK is required for paper-style 3-D rendering. "
            "Install pyvista and vtk in the active environment."
        ) from error
    pv.OFF_SCREEN = True
    return pv


def _load_json(path: Path) -> Dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def _load_volume(path: Path) -> torch.Tensor:
    tensor = runtime.normalize_single_geology(
        runtime.load_tensor(path, map_location="cpu"), str(path)
    )
    return tensor[0, 0].long().cpu()


def _sample_id(path: Path) -> int:
    match = re.fullmatch(r"sample_(\d+)\.pt", path.name)
    if match is None:
        raise ValueError(f"not a Phase-1 sample filename: {path}")
    return int(match.group(1))


def _sample_paths(directory: Path) -> Dict[int, Path]:
    paths = {_sample_id(path): path for path in directory.glob("sample_*.pt")}
    if not paths:
        raise FileNotFoundError(f"no sample_*.pt files found in {directory}")
    return dict(sorted(paths.items()))


def _read_csv_rows(path: Path) -> list[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _metric_rows_by_id(path: Path) -> Dict[int, Dict[str, str]]:
    return {int(row["sample_id"]): row for row in _read_csv_rows(path)}


def _choose_sample_id(guided_dir: Path, requested: int | None) -> int:
    """Choose a representative, preferring the largest paired ROI-IoU gain."""
    available = _sample_paths(guided_dir)
    if requested is not None:
        if requested not in available:
            raise ValueError(
                f"sample {requested} is absent; available ids: {sorted(available)}"
            )
        return requested
    delta_path = guided_dir / "paired_deltas.csv"
    if delta_path.is_file():
        rows = _read_csv_rows(delta_path)
        if rows and "delta_selected_roi_iou" in rows[0]:
            return int(
                max(rows, key=lambda row: float(row["delta_selected_roi_iou"]))[
                    "sample_id"
                ]
            )
    return min(available)


def _validate_pair(
    baseline_dir: Path,
    guided_dir: Path,
    target_label: int,
) -> tuple[Dict[str, object], Dict[str, object]]:
    """Reject figures that could imply an invalid paired comparison.

    The runner already validates all protocol fields before sampling. Here we
    verify that the completed artifacts still carry that verdict and that the
    saved target/noise/sample identities agree.
    """
    baseline = _load_json(baseline_dir / "config.json")
    guided = _load_json(guided_dir / "config.json")
    if baseline.get("run_status") != "completed" or guided.get("run_status") != "completed":
        raise ValueError("baseline and guided runs must both be completed")
    pairing = guided.get("pairing_validation")
    if not isinstance(pairing, dict) or pairing.get("paired") is not True:
        raise ValueError("guided config does not record strict pairing")
    if int(baseline["target_label"]) != target_label or int(guided["target_label"]) != target_label:
        raise ValueError("requested target label differs from run configs")
    if float(baseline["alpha"]) != 0.0 or float(guided["alpha"]) <= 0.0:
        raise ValueError("expected alpha-zero baseline and positive-alpha guided run")
    if int(baseline.get("max_post_projection_condition_violations", -1)) != 0:
        raise ValueError("baseline contains post-projection condition violations")
    if int(guided.get("max_post_projection_condition_violations", -1)) != 0:
        raise ValueError("guided run contains post-projection condition violations")

    identity_fields = (
        "checkpoint_sha256",
        "model_weight_source",
        "ema_applied",
        "truth_model_sha256",
        "boreholes_sha256",
        "target_mask_sha256",
        "target_probability_sha256",
        "roi_mask_sha256",
        "initial_noise_policy",
        "initial_noise_sha256",
        "integrator",
        "n_steps",
        "seed",
    )
    for field in identity_fields:
        if baseline.get(field) != guided.get(field):
            raise ValueError(f"strict-pair identity differs for field {field!r}")
    if set(_sample_paths(baseline_dir)) != set(_sample_paths(guided_dir)):
        raise ValueError("baseline and guided sample ids differ")
    return baseline, guided


def _label_color(label: int) -> str:
    return LABEL_COLORS.get(int(label), "#808080")


def _image_grid(array: np.ndarray, scalar_name: str, values: np.ndarray):
    """Create cell-centered image data without transposing voxel axes."""
    pv = _import_pyvista()
    array = np.asarray(array)
    grid = pv.ImageData(
        dimensions=tuple(int(value) + 1 for value in array.shape),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
    )
    grid.cell_data[scalar_name] = np.asarray(values).reshape(array.shape).ravel(order="F")
    return grid


def _binary_surface(mask: np.ndarray):
    values = np.asarray(mask, dtype=np.uint8)
    if not values.any():
        return None
    grid = _image_grid(values, "mask", values)
    selected = grid.threshold(0.5, scalars="mask", preference="cell")
    return selected.extract_surface(algorithm="dataset_surface")


def _cutaway_keep(shape: Sequence[int], cut_fraction: float) -> np.ndarray:
    """Return an L-shaped keep mask exposing two internal vertical faces."""
    x, y, _ = np.indices(tuple(int(value) for value in shape))
    cut_x = float(shape[0]) * float(cut_fraction)
    cut_y = float(shape[1]) * float(cut_fraction)
    return ~((x >= cut_x) & (y < cut_y))


def _add_categorical_volume(
    plotter,
    volume: np.ndarray,
    cut_fraction: float,
    show_legend: bool,
) -> list[int]:
    keep = _cutaway_keep(volume.shape, cut_fraction)
    labels = sorted(int(value) for value in np.unique(volume) if int(value) != -1)
    legend = []
    for label in labels:
        surface = _binary_surface((volume == label) & keep)
        if surface is None:
            continue
        color = _label_color(label)
        plotter.add_mesh(
            surface,
            color=color,
            smooth_shading=False,
            show_edges=False,
            opacity=1.0,
            label=f"label {label}",
        )
        legend.append((f"label {label}", color))
    if show_legend and legend:
        plotter.add_legend(
            legend,
            loc="upper right",
            size=(0.19, min(0.055 * len(legend), 0.52)),
            bcolor="white",
            border=True,
            face="rectangle",
        )
    return labels


def _add_target_geometry(
    plotter,
    prediction_mask: np.ndarray,
    truth_mask: np.ndarray,
    color: str,
    show_truth_body: bool,
) -> None:
    if show_truth_body:
        truth_surface = _binary_surface(truth_mask)
        if truth_surface is not None:
            plotter.add_mesh(
                truth_surface,
                color=TRUTH_TARGET_COLOR,
                opacity=0.17,
                show_edges=False,
                smooth_shading=False,
                label="oracle target",
            )
    prediction_surface = _binary_surface(prediction_mask)
    if prediction_surface is not None:
        plotter.add_mesh(
            prediction_surface,
            color=color,
            opacity=0.92,
            show_edges=False,
            smooth_shading=False,
            label="decoded target",
        )


def _condition_column_indices(boreholes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observations_per_column = (np.asarray(boreholes) != -1).sum(axis=2)
    return (
        np.argwhere(observations_per_column == 1),
        np.argwhere(observations_per_column > 1),
    )


def _add_boreholes(plotter, boreholes: np.ndarray, target_label: int) -> None:
    pv = _import_pyvista()
    surface_columns, borehole_columns = _condition_column_indices(boreholes)
    if surface_columns.size:
        surface_points = []
        for x_index, y_index in surface_columns:
            z_index = int(np.flatnonzero(boreholes[x_index, y_index] != -1)[0])
            surface_points.append(
                (float(x_index) + 0.5, float(y_index) + 0.5, float(z_index) + 0.5)
            )
        plotter.add_points(
            np.asarray(surface_points),
            color="#5b5b5b",
            opacity=0.2,
            point_size=2.0,
            render_points_as_spheres=False,
        )
    for x_index, y_index in borehole_columns:
        observed_z = np.flatnonzero(boreholes[x_index, y_index] != -1)
        if observed_z.size == 0:
            continue
        line = pv.Line(
            (float(x_index) + 0.5, float(y_index) + 0.5, float(observed_z.min())),
            (float(x_index) + 0.5, float(y_index) + 0.5, float(observed_z.max() + 1)),
        )
        plotter.add_mesh(line, color="#222222", line_width=2.0, opacity=0.62)
    hits = np.argwhere(boreholes == int(target_label)).astype(float) + 0.5
    if hits.size:
        plotter.add_points(
            hits,
            color="#d62728",
            point_size=9,
            render_points_as_spheres=True,
            label="target borehole hits",
        )


def _set_camera(plotter, shape: Sequence[int]) -> None:
    center = np.asarray(shape, dtype=float) / 2.0
    plotter.camera_position = [
        tuple(center + np.asarray((1.72, -1.35, 1.38)) * max(shape)),
        tuple(center),
        (0.0, 0.0, 1.0),
    ]
    plotter.camera.parallel_projection = True
    plotter.camera.zoom(1.1)


def _decorate_panel(plotter, title: str, shape: Sequence[int]) -> None:
    plotter.set_background("white")
    plotter.add_text(title, position="upper_left", font_size=12, color="#111111")
    plotter.add_bounding_box(color="#a8a8a8", line_width=1.0)
    plotter.add_axes(line_width=2, color="#333333")
    _set_camera(plotter, shape)


def _metric_text(row: Mapping[str, str] | None) -> str:
    if row is None:
        return ""
    return (
        f"IoU={float(row['selected_roi_iou']):.3f}, "
        f"R={float(row['selected_roi_recall']):.3f}, "
        f"CC={int(float(row['selected_roi_connected_components']))}"
    )


def render_triplet(
    truth: np.ndarray,
    baseline: np.ndarray,
    guided: np.ndarray,
    target_mask: np.ndarray,
    roi_mask: np.ndarray,
    boreholes: np.ndarray,
    target_label: int,
    sample_id: int,
    alpha: float,
    baseline_metrics: Mapping[str, str] | None,
    guided_metrics: Mapping[str, str] | None,
    cut_fraction: float,
    path: Path,
) -> None:
    del roi_mask
    pv = _import_pyvista()
    plotter = pv.Plotter(
        shape=(2, 3), off_screen=True, window_size=(2100, 1320), border=False
    )
    plotter.enable_anti_aliasing("ssaa")
    volumes = (
        (truth, "(a) Truth categorical model"),
        (baseline, f"(b) Paired baseline - sample {sample_id}"),
        (guided, f"(c) Guided - sample {sample_id}, alpha={alpha:g}"),
    )
    for column, (volume, title) in enumerate(volumes):
        plotter.subplot(0, column)
        _add_categorical_volume(plotter, volume, cut_fraction, show_legend=column == 0)
        _decorate_panel(plotter, title, truth.shape)

    prediction_masks = (
        target_mask,
        baseline == int(target_label),
        guided == int(target_label),
    )
    titles = (
        "(d) Oracle target geometry",
        f"(e) Baseline target - {_metric_text(baseline_metrics)}",
        f"(f) Guided target - {_metric_text(guided_metrics)}",
    )
    colors = (TRUTH_TARGET_COLOR, BASELINE_TARGET_COLOR, GUIDED_TARGET_COLOR)
    for column, (mask, title, color) in enumerate(zip(prediction_masks, titles, colors)):
        plotter.subplot(1, column)
        _add_target_geometry(
            plotter,
            prediction_mask=mask,
            truth_mask=target_mask,
            color=color,
            show_truth_body=column > 0,
        )
        _add_boreholes(plotter, boreholes, target_label)
        legend = [("oracle target", TRUTH_TARGET_COLOR)]
        if column > 0:
            legend.append(("decoded target", color))
        legend.append(("target borehole hits", "#d62728"))
        plotter.add_legend(
            legend,
            loc="upper right",
            size=(0.23, 0.18),
            bcolor="white",
            border=False,
        )
        _decorate_panel(plotter, title, truth.shape)
    plotter.screenshot(path, return_img=True)
    plotter.close()


def paired_change_masks(
    baseline: np.ndarray,
    guided: np.ndarray,
    target_mask: np.ndarray,
    roi_mask: np.ndarray,
    target_label: int,
) -> Dict[str, np.ndarray]:
    baseline_target = baseline == int(target_label)
    guided_target = guided == int(target_label)
    return {
        "correct_target_recovered": (~baseline_target) & guided_target & target_mask,
        "correct_target_lost": baseline_target & (~guided_target) & target_mask,
        "false_target_added": (~baseline_target) & guided_target & (~target_mask) & roi_mask,
        "false_target_removed": baseline_target & (~guided_target) & (~target_mask) & roi_mask,
        "all_hard_changes_inside_roi": (baseline != guided) & roi_mask,
        "target_related_changes": (baseline_target != guided_target) & roi_mask,
    }


def render_paired_changes(
    baseline: np.ndarray,
    guided: np.ndarray,
    target_mask: np.ndarray,
    roi_mask: np.ndarray,
    target_label: int,
    sample_id: int,
    path: Path,
) -> Dict[str, int]:
    pv = _import_pyvista()
    masks = paired_change_masks(baseline, guided, target_mask, roi_mask, target_label)
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    panels = (
        (
            "(a) Correct target changes",
            (("correct_target_recovered", "recovered", "#1a9850"),
             ("correct_target_lost", "lost", "#d73027")),
        ),
        (
            "(b) False target changes",
            (("false_target_added", "false added", "#fdae61"),
             ("false_target_removed", "false removed", "#4575b4")),
        ),
        (
            "(c) All paired hard-label changes",
            (("all_hard_changes_inside_roi", "all changed", "#777777"),
             ("target_related_changes", "target changes", "#984ea3")),
        ),
    )
    plotter = pv.Plotter(
        shape=(1, 3), off_screen=True, window_size=(2000, 690), border=False
    )
    plotter.enable_anti_aliasing("ssaa")
    for column, (title, entries) in enumerate(panels):
        plotter.subplot(0, column)
        truth_surface = _binary_surface(target_mask)
        if truth_surface is not None:
            plotter.add_mesh(
                truth_surface,
                color=TRUTH_TARGET_COLOR,
                opacity=0.12,
                show_edges=False,
            )
        legend = [("oracle target", TRUTH_TARGET_COLOR)]
        for key, label, color in entries:
            surface = _binary_surface(masks[key])
            if surface is not None:
                plotter.add_mesh(surface, color=color, opacity=0.92, show_edges=False)
            legend.append((f"{label} ({counts[key]})", color))
        plotter.add_legend(
            legend,
            loc="upper right",
            size=(0.31, 0.19),
            bcolor="white",
            border=False,
        )
        _decorate_panel(plotter, f"{title} - sample {sample_id}", target_mask.shape)
    plotter.screenshot(path, return_img=True)
    plotter.close()
    return counts


def _ensemble_probability(paths: Mapping[int, Path], target_label: int) -> np.ndarray:
    masks = [(_load_volume(path).numpy() == int(target_label)) for path in paths.values()]
    return np.stack(masks, axis=0).mean(axis=0)


def render_probability_isosurfaces(
    baseline_probability: np.ndarray,
    guided_probability: np.ndarray,
    target_mask: np.ndarray,
    target_label: int,
    thresholds: Sequence[float],
    path: Path,
) -> None:
    pv = _import_pyvista()
    plotter = pv.Plotter(
        shape=(1, 3), off_screen=True, window_size=(2000, 690), border=False
    )
    plotter.enable_anti_aliasing("ssaa")
    threshold_colors = ("#fee8a8", "#f5a742", "#c93312", "#7f0000")
    for column, (probability, title) in enumerate(
        (
            (None, "(a) Oracle target"),
            (baseline_probability, "(b) Baseline ensemble"),
            (guided_probability, "(c) Guided ensemble"),
        )
    ):
        plotter.subplot(0, column)
        if column == 0:
            _add_target_geometry(
                plotter,
                prediction_mask=target_mask,
                truth_mask=target_mask,
                color=TRUTH_TARGET_COLOR,
                show_truth_body=False,
            )
            legend = [("oracle target", TRUTH_TARGET_COLOR)]
        else:
            truth_surface = _binary_surface(target_mask)
            if truth_surface is not None:
                plotter.add_mesh(truth_surface, color=TRUTH_TARGET_COLOR, opacity=0.1)
            legend = [("oracle target", TRUTH_TARGET_COLOR)]
            for index, threshold in enumerate(thresholds):
                surface = _binary_surface(probability >= float(threshold))
                color = threshold_colors[min(index, len(threshold_colors) - 1)]
                if surface is not None:
                    plotter.add_mesh(
                        surface,
                        color=color,
                        opacity=min(0.25 + 0.2 * index, 0.9),
                        show_edges=False,
                    )
                legend.append((f"P(label {target_label}) >= {threshold:g}", color))
        plotter.add_legend(
            legend,
            loc="upper right",
            size=(0.29, 0.07 + 0.34 * (column > 0)),
            bcolor="white",
            border=False,
        )
        _decorate_panel(plotter, title, target_mask.shape)
    plotter.screenshot(path, return_img=True)
    plotter.close()


def export_vtk_volumes(
    output_dir: Path,
    truth: np.ndarray,
    baseline: np.ndarray,
    guided: np.ndarray,
    target_mask: np.ndarray,
    roi_mask: np.ndarray,
    boreholes: np.ndarray,
    target_label: int,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = (("truth", truth), ("baseline", baseline), ("guided", guided))
    paths: Dict[str, str] = {}
    for name, volume in arrays:
        grid = _image_grid(volume, "label", volume.astype(np.int16))
        grid.cell_data["target_mask"] = target_mask.astype(np.uint8).ravel(order="F")
        grid.cell_data["target_roi"] = roi_mask.astype(np.uint8).ravel(order="F")
        grid.cell_data["is_target_label"] = (volume == target_label).astype(np.uint8).ravel(order="F")
        grid.cell_data["borehole_observed"] = (boreholes != -1).astype(np.uint8).ravel(order="F")
        if name != "truth":
            grid.cell_data["differs_from_truth"] = (volume != truth).astype(np.uint8).ravel(order="F")
        if name == "guided":
            grid.cell_data["differs_from_baseline"] = (guided != baseline).astype(np.uint8).ravel(order="F")
        path = output_dir / f"{name}.vti"
        grid.save(path)
        paths[name] = str(path)
    return paths


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output directory is not empty; pass --overwrite to replace named figures: {path}"
        )
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render PyVista Phase-1 truth/baseline/guided figures and VTK volumes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--truth-model", type=Path, required=True)
    parser.add_argument("--boreholes", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--guided-dir", type=Path, required=True)
    parser.add_argument("--target-label", type=int, default=9)
    parser.add_argument("--sample-id", type=int, default=None)
    parser.add_argument("--cut-fraction", type=float, default=0.52)
    parser.add_argument(
        "--probability-threshold",
        type=float,
        action="append",
        default=None,
        help="Repeat for ensemble probability surfaces; defaults to 0.25, 0.5, 0.75.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.cut_fraction < 1.0:
        raise ValueError("cut_fraction must be in (0,1)")
    thresholds = tuple(args.probability_threshold or (0.25, 0.5, 0.75))
    if any(not 0.0 < value <= 1.0 for value in thresholds):
        raise ValueError("probability thresholds must be in (0,1]")
    thresholds = tuple(sorted(set(float(value) for value in thresholds)))
    _prepare_output_dir(args.output_dir, args.overwrite)

    baseline_config, guided_config = _validate_pair(
        args.baseline_dir, args.guided_dir, args.target_label
    )
    sample_id = _choose_sample_id(args.guided_dir, args.sample_id)
    truth = _load_volume(args.truth_model).numpy()
    boreholes = _load_volume(args.boreholes).numpy()
    baseline_paths = _sample_paths(args.baseline_dir)
    guided_paths = _sample_paths(args.guided_dir)
    baseline = _load_volume(baseline_paths[sample_id]).numpy()
    guided = _load_volume(guided_paths[sample_id]).numpy()
    target_mask = _load_volume(args.guided_dir / "target_mask.pt").numpy().astype(bool)
    roi_mask = _load_volume(args.guided_dir / "target_roi_mask.pt").numpy().astype(bool)
    if len({truth.shape, boreholes.shape, baseline.shape, guided.shape, target_mask.shape, roi_mask.shape}) != 1:
        raise ValueError("truth, boreholes, samples, target mask, and ROI shapes must match")

    baseline_metrics = _metric_rows_by_id(args.baseline_dir / "sample_metrics.csv").get(sample_id)
    guided_metrics = _metric_rows_by_id(args.guided_dir / "sample_metrics.csv").get(sample_id)
    triplet_path = args.output_dir / "truth_baseline_guided_3d.png"
    changes_path = args.output_dir / "paired_changes_3d.png"
    probability_path = args.output_dir / "ensemble_probability_isosurfaces_3d.png"

    print("[1/4] Rendering categorical and target-geometry comparison", flush=True)
    render_triplet(
        truth=truth,
        baseline=baseline,
        guided=guided,
        target_mask=target_mask,
        roi_mask=roi_mask,
        boreholes=boreholes,
        target_label=args.target_label,
        sample_id=sample_id,
        alpha=float(guided_config["alpha"]),
        baseline_metrics=baseline_metrics,
        guided_metrics=guided_metrics,
        cut_fraction=args.cut_fraction,
        path=triplet_path,
    )
    print("[2/4] Rendering strict paired-change audit", flush=True)
    changes = render_paired_changes(
        baseline=baseline,
        guided=guided,
        target_mask=target_mask,
        roi_mask=roi_mask,
        target_label=args.target_label,
        sample_id=sample_id,
        path=changes_path,
    )
    print("[3/4] Rendering ensemble probability surfaces", flush=True)
    baseline_probability = _ensemble_probability(baseline_paths, args.target_label)
    guided_probability = _ensemble_probability(guided_paths, args.target_label)
    render_probability_isosurfaces(
        baseline_probability,
        guided_probability,
        target_mask,
        args.target_label,
        thresholds,
        probability_path,
    )
    print("[4/4] Exporting VTK image volumes and manifest", flush=True)
    vtk_paths = export_vtk_volumes(
        output_dir=args.output_dir / "vtk",
        truth=truth,
        baseline=baseline,
        guided=guided,
        target_mask=target_mask,
        roi_mask=roi_mask,
        boreholes=boreholes,
        target_label=args.target_label,
    )
    manifest = {
        "description": (
            "Paper-style PyVista visualization of a strict Phase-1 paired "
            "probability-guidance result. The probability target is truth-derived."
        ),
        "ensemble_probability_definition": (
            "Empirical per-voxel occurrence frequency of the target label across "
            "saved hard-label realizations."
        ),
        "truth_model": str(args.truth_model),
        "boreholes": str(args.boreholes),
        "baseline_dir": str(args.baseline_dir),
        "guided_dir": str(args.guided_dir),
        "target_label": int(args.target_label),
        "sample_id": int(sample_id),
        "sample_selection": (
            "explicit --sample-id"
            if args.sample_id is not None
            else "maximum delta_selected_roi_iou from paired_deltas.csv"
        ),
        "component_mode": guided_config.get("component_mode"),
        "component_rank": guided_config.get("component_rank"),
        "alpha": float(guided_config["alpha"]),
        "strict_pairing": True,
        "baseline_protocol_version": baseline_config.get(
            "phase1_protocol_version", baseline_config.get("protocol_version")
        ),
        "guided_protocol_version": guided_config.get(
            "phase1_protocol_version", guided_config.get("protocol_version")
        ),
        "condition_violations": 0,
        "cut_fraction": float(args.cut_fraction),
        "probability_thresholds": list(thresholds),
        "ensemble_size": len(guided_paths),
        "label_colors": {str(key): value for key, value in LABEL_COLORS.items()},
        "baseline_metrics": baseline_metrics,
        "guided_metrics": guided_metrics,
        "paired_change_counts": changes,
        "figures": {
            "truth_baseline_guided_3d": str(triplet_path),
            "paired_changes_3d": str(changes_path),
            "ensemble_probability_isosurfaces_3d": str(probability_path),
        },
        "vtk_volumes": vtk_paths,
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f"Rendered Phase-1 figures and VTK volumes: {args.output_dir}")


if __name__ == "__main__":
    main()
