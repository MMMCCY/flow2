"""Shared, deterministic visual language for the flow2 paper figures.

This module deliberately contains no experiment logic.  It provides fixed
paper dimensions, categorical colours, provenance helpers, lightweight QC,
and PyVista renderers whose camera is shared by every geology panel.
"""

from __future__ import annotations

from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Mapping, Sequence
import zipfile

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
PAPER_DIR = PROJECT_DIR / "paper"
FIGURES_DIR = PAPER_DIR / "figures"
FIGURE_DATA_DIR = PAPER_DIR / "figure_data"
MANIFESTS_DIR = PAPER_DIR / "manifests"

SINGLE_COLUMN_MM = 85.0
DOUBLE_COLUMN_MM = 178.0
FONT_FAMILY = "STIXGeneral"
FONT_SIZE_PT = 8.0

# Stable, colour-vision-aware categorical colours.  Raw labels are -1..13.
# Air is rendered as background.  Label 9 and observations have reserved hues.
LABEL9_COLOR = "#D95F02"
OBSERVATION_COLOR = "#168A89"
TRUTH_OUTLINE_COLOR = "#2B2B2B"
LABEL_COLORS: dict[int, str] = {
    -1: "#FFFFFF",
    0: "#7A6652",
    1: "#8C4E85",
    2: "#3B6FB6",
    3: "#5A91C8",
    4: "#56B4B8",
    5: "#4DA66D",
    6: "#86B84A",
    7: "#C5C55A",
    8: "#E5C55C",
    9: LABEL9_COLOR,
    10: "#C94C4C",
    11: "#A33E63",
    12: "#7B4FA3",
    13: "#5E4B8B",
}

SEISMIC_CMAP = LinearSegmentedColormap.from_list(
    "flow2_seismic",
    ("#1E3A5F", "#75A7C7", "#F7F7F5", "#D98979", "#8D2535"),
    N=256,
)
RESIDUAL_CMAP = LinearSegmentedColormap.from_list(
    "flow2_residual",
    ("#263D73", "#87AED0", "#FAFAF8", "#E79A78", "#8C2D3B"),
    N=256,
)

CAMERA = {
    "position_direction": [1.72, -1.35, 1.38],
    "focal_point_fraction": [0.50, 0.50, 0.46],
    "view_up": [0.0, 0.0, 1.0],
    "parallel_projection": True,
    "zoom": 1.14,
    "cut_fraction": 0.52,
}


def configure_matplotlib() -> None:
    """Apply the single paper style and stable vector-output settings."""
    mpl.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": FONT_SIZE_PT,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#4A4A4A",
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "flow2-paper-figures-v1",
            "mathtext.fontset": "stix",
        }
    )


def mm_to_inches(value: float) -> float:
    return float(value) / 25.4


def ensure_output_dirs() -> None:
    for path in (FIGURES_DIR, FIGURE_DATA_DIR, MANIFESTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def project_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path.resolve()


def repository_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def require_file(path: Path, role: str = "source artifact") -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing {role}: {path}")
    return path


def read_json(path: Path) -> dict[str, object]:
    path = require_file(path, "JSON artifact")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def read_csv_row(path: Path, sample_id: int) -> dict[str, str]:
    path = require_file(path, "CSV artifact")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["sample_id"]) == sample_id]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one sample_id={sample_id} row in {path}")
    return rows[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with require_file(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_record(path: Path, role: str) -> dict[str, object]:
    path = require_file(path, role)
    try:
        recorded_path = str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        recorded_path = str(path)
    return {
        "role": role,
        "path": recorded_path,
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()


def generation_record(script_path: Path) -> dict[str, object]:
    return {
        "script": source_record(script_path, "generation script"),
        "git_head": git_head(),
        "python": sys.version.split()[0],
        "matplotlib": mpl.__version__,
        "camera": CAMERA,
        "style": {
            "font_family": FONT_FAMILY,
            "base_font_size_pt": FONT_SIZE_PT,
            "single_column_mm": SINGLE_COLUMN_MM,
            "double_column_mm": DOUBLE_COLUMN_MM,
            "label_colors": {str(key): value for key, value in LABEL_COLORS.items()},
            "label9_color": LABEL9_COLOR,
            "observation_color": OBSERVATION_COLOR,
            "categorical_colormap_policy": "fixed discrete raw-label palette",
            "seismic_colormap_policy": "zero-centered perceptually balanced diverging",
        },
    }


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")


def write_deterministic_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write an NPZ with fixed member timestamps and ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def load_volume(path: Path, *, dtype: np.dtype | None = None) -> np.ndarray:
    value = torch.load(require_file(path, "tensor artifact"), map_location="cpu", weights_only=True)
    if not torch.is_tensor(value):
        raise TypeError(f"expected tensor in {path}")
    while value.ndim > 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 3:
        raise ValueError(f"expected a single 3-D volume in {path}, got {tuple(value.shape)}")
    array = value.detach().cpu().numpy()
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    validate_array(path.name, array, ndim=3)
    return array


def validate_array(name: str, array: np.ndarray, *, ndim: int | None = None) -> None:
    array = np.asarray(array)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name}: expected {ndim} dimensions, got {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name}: empty array")
    if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
        raise ValueError(f"{name}: NaN or Inf detected")


def validate_same_shape(named_arrays: Mapping[str, np.ndarray]) -> tuple[int, ...]:
    shapes = {name: tuple(np.asarray(value).shape) for name, value in named_arrays.items()}
    unique = set(shapes.values())
    if len(unique) != 1:
        raise ValueError(f"inconsistent shapes: {shapes}")
    return next(iter(unique))


def target_metrics(truth: np.ndarray, prediction: np.ndarray, target_label: int = 9) -> dict[str, float]:
    validate_same_shape({"truth": truth, "prediction": prediction})
    target = np.asarray(truth) == int(target_label)
    predicted = np.asarray(prediction) == int(target_label)
    tp = int(np.logical_and(target, predicted).sum())
    fp = int(np.logical_and(~target, predicted).sum())
    fn = int(np.logical_and(target, ~predicted).sum())
    return {
        "IoU9": tp / max(tp + fp + fn, 1),
        "Precision9": tp / max(tp + fp, 1),
        "Recall9": tp / max(tp + fn, 1),
    }


def assert_metrics_match(
    truth: np.ndarray,
    prediction: np.ndarray,
    row: Mapping[str, str],
    *,
    tolerance: float = 5e-8,
) -> dict[str, float]:
    computed = target_metrics(truth, prediction)
    saved = {
        "IoU9": float(row["target_iou"]),
        "Precision9": float(row["target_precision"]),
        "Recall9": float(row["target_recall"]),
    }
    for name in computed:
        if abs(computed[name] - saved[name]) > tolerance:
            raise ValueError(
                f"wrong metric/candidate for {name}: computed={computed[name]} saved={saved[name]}"
            )
    return computed


def robust_symmetric_limit(
    arrays: Iterable[np.ndarray],
    percentile: float = 99.5,
    *,
    ignore_zeros: bool = False,
) -> float:
    values = [np.abs(np.asarray(value, dtype=np.float64)).ravel() for value in arrays]
    if not values:
        raise ValueError("at least one array is required")
    population = np.concatenate(values)
    if ignore_zeros:
        population = population[population > 0]
        if population.size == 0:
            raise ValueError("robust-limit population contains no nonzero samples")
    limit = float(np.percentile(population, percentile))
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError(f"invalid robust symmetric limit: {limit}")
    return limit


def panel_label(ax, label: str, *, color: str = "#171717") -> None:
    ax.text(
        0.015,
        0.985,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color=color,
        zorder=20,
        bbox={"boxstyle": "square,pad=0.10", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )


@contextmanager
def reproducible_environment():
    old = os.environ.get("SOURCE_DATE_EPOCH")
    os.environ["SOURCE_DATE_EPOCH"] = "946684800"  # 2000-01-01 UTC
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("SOURCE_DATE_EPOCH", None)
        else:
            os.environ["SOURCE_DATE_EPOCH"] = old


def save_figure(fig, stem: Path, *, title: str) -> dict[str, str]:
    """Save PDF/SVG plus a true 600-dpi PNG with fixed metadata."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = {suffix: stem.with_suffix(f".{suffix}") for suffix in ("pdf", "svg", "png")}
    fixed_time = datetime(2000, 1, 1, tzinfo=timezone.utc)
    with reproducible_environment():
        fig.savefig(
            paths["pdf"],
            dpi=600,
            metadata={
                "Title": title,
                "Author": "flow2 scientific visualization pipeline",
                "Creator": "Matplotlib",
                "CreationDate": fixed_time,
                "ModDate": fixed_time,
            },
        )
        fig.savefig(
            paths["svg"],
            dpi=600,
            metadata={"Title": title, "Creator": "Matplotlib", "Date": "2000-01-01T00:00:00Z"},
        )
        fig.savefig(
            paths["png"],
            dpi=600,
            metadata={"Title": title, "Software": "flow2 paper figures"},
        )
    plt.close(fig)
    return {name: str(path.relative_to(PROJECT_DIR)) for name, path in paths.items()}


def _import_pyvista():
    try:
        import pyvista as pv
    except ImportError as error:
        raise RuntimeError("PyVista and VTK are required for 3-D paper panels") from error
    pv.OFF_SCREEN = True
    return pv


def _image_grid(array: np.ndarray, scalar_name: str = "value"):
    pv = _import_pyvista()
    array = np.asarray(array)
    grid = pv.ImageData(
        dimensions=tuple(int(value) + 1 for value in array.shape),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
    )
    grid.cell_data[scalar_name] = array.ravel(order="F")
    return grid


def binary_surface(mask: np.ndarray):
    values = np.asarray(mask, dtype=np.uint8)
    validate_array("binary surface", values, ndim=3)
    if not values.any():
        return None
    grid = _image_grid(values, "mask")
    selected = grid.threshold(0.5, scalars="mask", preference="cell")
    return selected.extract_surface(algorithm="dataset_surface")


def cutaway_mask(shape: Sequence[int], fraction: float | None = None) -> np.ndarray:
    fraction = CAMERA["cut_fraction"] if fraction is None else float(fraction)
    x, y, _ = np.indices(tuple(int(value) for value in shape))
    return ~((x >= shape[0] * fraction) & (y < shape[1] * fraction))


def _set_camera(plotter, shape: Sequence[int]) -> None:
    size = float(max(shape))
    focal = np.asarray(shape, dtype=float) * np.asarray(CAMERA["focal_point_fraction"])
    position = focal + np.asarray(CAMERA["position_direction"], dtype=float) * size
    plotter.camera_position = [tuple(position), tuple(focal), tuple(CAMERA["view_up"])]
    plotter.camera.parallel_projection = bool(CAMERA["parallel_projection"])
    # VTK's unrendered default parallel scale is 1.0.  Set it explicitly so an
    # off-screen first frame contains the complete 64^3 domain on every host.
    plotter.camera.parallel_scale = 0.70 * size
    plotter.camera.zoom(float(CAMERA["zoom"]))


def _new_plotter(window_size: tuple[int, int] = (1000, 860)):
    pv = _import_pyvista()
    plotter = pv.Plotter(off_screen=True, window_size=window_size, border=False)
    plotter.set_background("white")
    plotter.enable_anti_aliasing("ssaa")
    return plotter


def _add_surface(plotter, surface, color: str, opacity: float = 1.0) -> None:
    if surface is None:
        return
    plotter.add_mesh(
        surface,
        color=color,
        opacity=float(opacity),
        smooth_shading=False,
        show_edges=False,
        ambient=0.34,
        diffuse=0.66,
        specular=0.0,
    )


def _add_wells(plotter, well_xy: Sequence[Sequence[int]], *, top_z: float = 56.0) -> None:
    pv = _import_pyvista()
    for x, y in well_xy:
        line = pv.Line((float(x) + 0.5, float(y) + 0.5, 0.0), (float(x) + 0.5, float(y) + 0.5, top_z))
        _add_surface(plotter, line.tube(radius=0.26), OBSERVATION_COLOR, 0.96)


def render_categorical_panel(
    volume: np.ndarray,
    *,
    borehole_xy: Sequence[Sequence[int]] = (),
    window_size: tuple[int, int] = (1000, 860),
) -> np.ndarray:
    volume = np.asarray(volume)
    validate_array("categorical volume", volume, ndim=3)
    plotter = _new_plotter(window_size)
    keep = cutaway_mask(volume.shape)
    labels = sorted(int(value) for value in np.unique(volume) if int(value) != -1)
    # Background first, then target last to keep the emphasis consistent.
    labels = [value for value in labels if value != 9] + ([9] if 9 in labels else [])
    for label in labels:
        _add_surface(plotter, binary_surface((volume == label) & keep), LABEL_COLORS.get(label, "#777777"))
    if borehole_xy:
        _add_wells(plotter, borehole_xy)
    _set_camera(plotter, volume.shape)
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return np.asarray(image)[..., :3]


def render_conditions_panel(
    truth: np.ndarray,
    boreholes: np.ndarray,
    well_xy: Sequence[Sequence[int]],
    *,
    target_label: int = 9,
    window_size: tuple[int, int] = (1000, 860),
) -> np.ndarray:
    validate_same_shape({"truth": truth, "boreholes": boreholes})
    pv = _import_pyvista()
    plotter = _new_plotter(window_size)
    surface = pv.Plane(
        center=(truth.shape[0] / 2, truth.shape[1] / 2, 56.0),
        direction=(0.0, 0.0, 1.0),
        i_size=float(truth.shape[0]),
        j_size=float(truth.shape[1]),
    )
    _add_surface(plotter, surface, "#D7D2C7", 0.55)
    _add_wells(plotter, well_xy)
    target_hits = np.argwhere(np.asarray(boreholes) == int(target_label)).astype(float) + 0.5
    if target_hits.size:
        plotter.add_points(
            target_hits,
            color=LABEL9_COLOR,
            point_size=13,
            render_points_as_spheres=True,
        )
    _set_camera(plotter, truth.shape)
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return np.asarray(image)[..., :3]


def render_target_panel(
    target_mask: np.ndarray,
    *,
    well_xy: Sequence[Sequence[int]] = (),
    ghost_mask: np.ndarray | None = None,
    window_size: tuple[int, int] = (1000, 860),
) -> np.ndarray:
    target_mask = np.asarray(target_mask, dtype=bool)
    validate_array("target mask", target_mask, ndim=3)
    plotter = _new_plotter(window_size)
    if ghost_mask is not None:
        validate_same_shape({"target": target_mask, "ghost": ghost_mask})
        _add_surface(plotter, binary_surface(np.asarray(ghost_mask, dtype=bool)), TRUTH_OUTLINE_COLOR, 0.11)
    _add_surface(plotter, binary_surface(target_mask), LABEL9_COLOR, 0.98)
    if well_xy:
        _add_wells(plotter, well_xy)
    _set_camera(plotter, target_mask.shape)
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return np.asarray(image)[..., :3]


def render_acquisition_panel(
    hidden_truth: np.ndarray,
    *,
    well_xy: Sequence[Sequence[int]],
    window_size: tuple[int, int] = (1000, 860),
) -> np.ndarray:
    hidden_truth = np.asarray(hidden_truth, dtype=bool)
    validate_array("hidden truth", hidden_truth, ndim=3)
    pv = _import_pyvista()
    plotter = _new_plotter(window_size)
    surface = pv.Plane(
        center=(hidden_truth.shape[0] / 2, hidden_truth.shape[1] / 2, 56.0),
        direction=(0.0, 0.0, 1.0),
        i_size=float(hidden_truth.shape[0]),
        j_size=float(hidden_truth.shape[1]),
    )
    _add_surface(plotter, surface, "#D7D2C7", 0.32)
    _add_surface(plotter, binary_surface(hidden_truth), LABEL9_COLOR, 0.28)
    _add_wells(plotter, well_xy)
    _set_camera(plotter, hidden_truth.shape)
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return np.asarray(image)[..., :3]


def trim_render(image: np.ndarray, *, pad: int = 8, threshold: int = 252) -> np.ndarray:
    """Remove deterministic all-white margins from an off-screen render."""
    image = np.asarray(image)
    mask = np.any(image[..., :3] < int(threshold), axis=-1)
    if not mask.any():
        return image
    rows, cols = np.nonzero(mask)
    y0, y1 = max(int(rows.min()) - pad, 0), min(int(rows.max()) + pad + 1, image.shape[0])
    x0, x1 = max(int(cols.min()) - pad, 0), min(int(cols.max()) + pad + 1, image.shape[1])
    return image[y0:y1, x0:x1]


def show_render(ax, image: np.ndarray) -> None:
    ax.imshow(trim_render(image), interpolation="nearest", rasterized=True)
    ax.set_axis_off()


def output_records(outputs: Mapping[str, str]) -> list[dict[str, object]]:
    return [source_record(PROJECT_DIR / value, f"figure output ({kind})") for kind, value in outputs.items()]
