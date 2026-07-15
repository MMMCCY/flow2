"""Render two decoded 64^3 tensors in a side-by-side 3D view.

This script is focused on inspecting model outputs from
`project/geodata-3d-unconditional/samples/...`.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path("/home/mcy/Geoflow")
STRUCTURALGEO_SRC = REPO_ROOT / "StructuralGeo-main" / "src"
FLOWTRAIN_PROJECT = (
    REPO_ROOT
    / "flowtrain_stochastic_interpolation-main"
    / "project"
    / "geodata-3d-unconditional"
)

sys.path.insert(0, str(STRUCTURALGEO_SRC))
sys.path.insert(0, str(FLOWTRAIN_PROJECT))

import pyvista as pv
import torch

from geogen.model import GeoModel
import geogen.plot as geovis

DEFAULT_DECODED_DIR = (
    REPO_ROOT
    / "flowtrain_stochastic_interpolation-main"
    / "project"
    / "geodata-3d-unconditional"
    / "samples"
    / "cat-embeddings-18d-normed-64cubed"
)
DEFAULT_DECODED_A = DEFAULT_DECODED_DIR / "decoded_s100_0.pt"
DEFAULT_DECODED_B = DEFAULT_DECODED_DIR / "decoded_s100_1.pt"
DEFAULT_OUT_DIR = REPO_ROOT / "repro_samples" / "decoded_3d_views"
DEFAULT_BOUNDS = ((-1920, 1920), (-1920, 1920), (-1920, 1920))

def load_tensor(path: Path, shift_back: bool = False) -> torch.Tensor:
    """Load a decoded tensor and optionally shift labels from 0..14 to -1..13."""
    if not path.exists():
        raise FileNotFoundError(f"Tensor file not found: {path}")

    tensor = torch.load(path, map_location="cpu")
    if shift_back:
        tensor = tensor - 1
    print(
        "loaded",
        path,
        "shape=",
        tuple(tensor.shape),
        "unique=",
        sorted(torch.unique(tensor).tolist()),
    )
    return tensor


def tensor_to_model(tensor: torch.Tensor) -> GeoModel:
    """Convert a saved tensor to a GeoModel for plotting."""
    if tensor.dim() == 5 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.dim() == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return GeoModel.from_tensor(data_tensor=tensor, bounds=DEFAULT_BOUNDS)


def save_side_by_side(models: list[GeoModel], labels: list[str], out_path: Path) -> None:
    """Render two 3D volumes side-by-side and save a screenshot."""
    if len(models) != 2:
        raise ValueError("Expected exactly two models.")

    plotter = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(1800, 900))
    for i, model in enumerate(models):
        plotter.subplot(0, i)
        geovis.volview(
            model,
            plotter=plotter,
            show_bounds=True,
            clim=(-1.5, 13.5),
            geowords=True,
        )
        plotter.add_text(labels[i], position="upper_left", font_size=12)

    plotter.link_views()
    plotter.camera_position = "iso"
    plotter.camera.zoom(1.1)
    plotter.screenshot(str(out_path), transparent_background=False)
    plotter.close()
    print("saved", out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display two decoded tensors in 3D and save a side-by-side screenshot."
    )
    parser.add_argument(
        "--decoded-a",
        type=Path,
        default=DEFAULT_DECODED_A,
        help="Path to the first decoded tensor.",
    )
    parser.add_argument(
        "--decoded-b",
        type=Path,
        default=DEFAULT_DECODED_B,
        help="Path to the second decoded tensor.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for screenshots.",
    )
    parser.add_argument(
        "--shift-back",
        action="store_true",
        help="Shift labels by -1 (0..14 -> -1..13) before plotting.",
    )
    parser.add_argument(
        "--output-name",
        default="decoded_pair_3d.png",
        help="Output image file name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tensor_a = load_tensor(args.decoded_a, shift_back=args.shift_back)
    tensor_b = load_tensor(args.decoded_b, shift_back=args.shift_back)

    model_a = tensor_to_model(tensor_a)
    model_b = tensor_to_model(tensor_b)

    save_side_by_side(
        [model_a, model_b],
        [args.decoded_a.stem, args.decoded_b.stem],
        args.out_dir / args.output_name,
    )


if __name__ == "__main__":
    pv.OFF_SCREEN = True
    main()
