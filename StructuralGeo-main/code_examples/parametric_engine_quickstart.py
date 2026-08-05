"""Quickstart for the parameter driven GeoGen extension.

Run from this folder or from the project root:
    python StructuralGeo-main/code_examples/parametric_engine_quickstart.py

Use ``--no-show`` when you only want to verify model generation without opening
the PyVista window.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from geogen.engine import (
    DikePlaneSpec,
    FaultSpec,
    FoldSpec,
    GeoModelSpec,
    IntrusionSpec,
    ParametricGeoEngine,
    SedimentationSpec,
    Uniform,
)


def build_demo_spec(resolution=(128,128,128)):
    """Create a demo with a fixed intrusion anchor and randomized surrounding structure."""
    return GeoModelSpec(
        name="anchored_intrusion_demo",
        resolution=resolution,
        seed=2026,
        events=[
            SedimentationSpec(depth=1600, layer_thickness=(120, 360)),
            FoldSpec(
                strike=(0, 180),
                dip=(70, 100),
                period=(4500, 9000),
                amplitude=(250, 650),
                fourier_harmonics=(3, 5),
            ),
            FaultSpec(
                strike=(80, 140),
                dip=(60, 85),
                rake=(75, 110),
                amplitude=(150, 550),
            ),
            IntrusionSpec(
                origin=(0, 0, -350),
                kind="hemisphere",
                diameter=Uniform(900, 1600),
                height=Uniform(250, 600),
                value=9,
                clip=False,
            ),
            DikePlaneSpec(
                origin=(800, -600, -250),
                strike=(0, 360),
                dip=(82, 96),
                width=(80, 180),
                length=(3500, 7000),
            ),
        ],
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and visualize a parameterized 3D GeoModel.")
    parser.add_argument(
        "--view",
        choices=("vol", "orthslice", "nslice", "oneslice", "categorical", "transformation"),
        default="vol",
        help="PyVista view to open.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Build the model and print its history without opening a visualization window.",
    )
    parser.add_argument(
        "--all-views",
        action="store_true",
        help="Open the same family of PyVista views demonstrated in quickstart.py.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        help="X/Y resolution. Z resolution is half this value.",
    )
    return parser.parse_args()


def print_model_summary(model):
    data_grid = model.get_data_grid()
    values = model.data.copy()
    values[np.isnan(values)] = model.EMPTY_VALUE
    filled_values = sorted(set(values.astype(int).tolist()))
    print(f"Parametric GeoModel data grid shape: {data_grid.shape}")
    print(f"Rock values present: {filled_values}")
    print(model)
    print(model.get_history_string(unpacked=True))


def show_quickstart_views(engine, model):
    engine.visualize(model, view="vol", show_bounds=True).show()
    engine.visualize(model, view="orthslice").show()
    engine.visualize(model, view="nslice", n=10).show()
    engine.visualize(model, view="oneslice").show()
    engine.visualize(model, view="categorical").show()


def main():
    args = parse_args()
    engine = ParametricGeoEngine()
    spec = build_demo_spec()

    model = engine.generate(spec)

    print_model_summary(model)

    if not args.no_show:
        if args.all_views:
            show_quickstart_views(engine, model)
        else:
            view_kwargs = {"show_bounds": True} if args.view == "vol" else {}
            plotter = engine.visualize(model, view=args.view, **view_kwargs)
            plotter.show()


if __name__ == "__main__":
    main()
