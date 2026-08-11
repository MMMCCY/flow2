#!/usr/bin/env python3
"""Generate Figure 1: vector framework and geophysical evidence pathway."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.paper_figures.style import (
    DOUBLE_COLUMN_MM,
    FIGURE_DATA_DIR,
    FIGURES_DIR,
    LABEL9_COLOR,
    MANIFESTS_DIR,
    OBSERVATION_COLOR,
    configure_matplotlib,
    ensure_output_dirs,
    generation_record,
    mm_to_inches,
    output_records,
    save_figure,
    source_record,
    write_json,
)


SCRIPT_PATH = Path(__file__).resolve()
FIGURE_ID = "figure01_framework"

INK = "#24313A"
MUTED = "#66727A"
FLOW = "#3969AC"
LATENT = "#7A5195"
EVIDENCE = "#2E8B78"
PHYSICS = "#B4552D"
PALE = "#F4F5F3"


def _box(ax, center, text, *, width=1.42, height=0.70, face=PALE, edge=INK, fontsize=7.6):
    x, y = center
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=0.8,
        facecolor=face,
        edgecolor=edge,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=INK, linespacing=1.05)
    return patch


def _arrow(ax, start, end, *, color=MUTED, width=0.9, style="-|>", mutation=8, connection="arc3"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation,
        linewidth=width,
        color=color,
        connectionstyle=connection,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(patch)
    return patch


def _cube(ax, center, *, scale=0.22, face="#D9D0BF", target=False):
    x, y = center
    dx, dy = scale, scale * 0.58
    front = Polygon(
        [(x - dx, y - dy), (x + dx, y - dy), (x + dx, y + dy), (x - dx, y + dy)],
        closed=True,
        facecolor=face,
        edgecolor=INK,
        linewidth=0.65,
    )
    top = Polygon(
        [(x - dx, y + dy), (x, y + 2 * dy), (x + 2 * dx, y + 2 * dy), (x + dx, y + dy)],
        closed=True,
        facecolor="#EEEAE2",
        edgecolor=INK,
        linewidth=0.65,
    )
    side = Polygon(
        [(x + dx, y - dy), (x + 2 * dx, y), (x + 2 * dx, y + 2 * dy), (x + dx, y + dy)],
        closed=True,
        facecolor="#B9AE9B",
        edgecolor=INK,
        linewidth=0.65,
    )
    for patch in (front, top, side):
        ax.add_patch(patch)
    if target:
        ax.add_patch(
            Polygon(
                [(x - 0.11, y - 0.07), (x + 0.05, y - 0.15), (x + 0.17, y + 0.10), (x, y + 0.17)],
                closed=True,
                facecolor=LABEL9_COLOR,
                edgecolor="none",
                alpha=0.95,
            )
        )


def _latent_glyph(ax, center):
    x, y = center
    t = np.linspace(-1, 1, 80)
    ax.plot(x + 0.28 * t, y + 0.12 * np.sin(np.pi * t), color=LATENT, lw=1.5)
    for value in (-0.75, -0.25, 0.25, 0.75):
        ax.plot(x + 0.28 * value, y + 0.12 * np.sin(np.pi * value), "o", ms=2.6, color=LATENT)


def _seismic_glyph(ax, center, color=PHYSICS):
    x, y = center
    t = np.linspace(-1, 1, 120)
    envelope = np.exp(-3.5 * t**2)
    ax.plot(x + 0.29 * t, y + 0.13 * envelope * np.sin(8 * np.pi * t), color=color, lw=1.0)


def generate() -> dict[str, object]:
    configure_matplotlib()
    ensure_output_dirs()
    fig = plt.figure(figsize=(mm_to_inches(DOUBLE_COLUMN_MM), 4.42))
    ax = fig.add_axes([0.025, 0.04, 0.95, 0.93])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.35)
    ax.axis("off")

    ax.text(0.08, 6.13, "(a)", fontsize=8.7, fontweight="bold", va="top")
    ax.text(0.42, 6.13, "Conditional geological generation", fontsize=9.2, fontweight="bold", va="top", color=INK)

    centers = [(0.80, 4.72), (2.25, 4.72), (3.70, 4.72), (5.15, 4.72), (6.60, 4.72), (8.05, 4.72), (9.40, 4.72)]
    labels = (
        "StructuralGeo",
        "3-D categorical\nmodel",
        "surface + borehole\nobservations",
        "frozen conditional\nFlow",
        "continuous embedding\n+ ODE",
        "hard categorical\ndecode",
        "geological\nensemble",
    )
    colours = ("#EDF1EB", "#F4EFE6", "#E6F2EF", "#E7EEF7", "#EEE8F4", "#F4EFE6", "#F2E9E2")
    widths = (1.18, 1.20, 1.28, 1.28, 1.35, 1.20, 1.05)
    for center, label, colour, width in zip(centers, labels, colours, widths):
        _box(ax, center, label, width=width, face=colour)
    for left, right in zip(centers[:-1], centers[1:]):
        _arrow(ax, (left[0] + 0.65, left[1]), (right[0] - 0.65, right[1]))
    _cube(ax, (2.10, 5.27), scale=0.12, target=True)
    _cube(ax, (9.20, 5.25), scale=0.10, target=True)
    _cube(ax, (9.45, 5.18), scale=0.10, face="#D7DDE6", target=True)
    _cube(ax, (9.70, 5.24), scale=0.10, face="#E3D7C6", target=True)
    _latent_glyph(ax, (6.60, 5.21))
    ax.plot([3.46, 3.46], [5.02, 5.34], color=OBSERVATION_COLOR, lw=1.2)
    ax.plot([3.66, 3.66], [5.02, 5.34], color=OBSERVATION_COLOR, lw=1.2)
    ax.plot([3.86, 3.86], [5.02, 5.34], color=OBSERVATION_COLOR, lw=1.2)

    ax.text(0.08, 3.62, "(b)", fontsize=8.7, fontweight="bold", va="top")
    ax.text(
        0.42,
        3.62,
        "Additional inference-time evidence",
        fontsize=9.2,
        fontweight="bold",
        va="top",
        color=INK,
    )
    evidence_centers = [(2.10, 2.95), (4.05, 2.95), (6.08, 2.95)]
    evidence_labels = ("probability\nevidence", "property\nevidence", "geophysical\nobservation")
    for center, label in zip(evidence_centers, evidence_labels):
        _box(ax, center, label, width=1.42, height=0.67, face="#E7F2EF", edge=EVIDENCE)
        _arrow(ax, (center[0], 3.30), (5.15, 4.35), color=EVIDENCE, width=0.85, connection="arc3,rad=-0.12")
    ax.text(3.70, 4.20, "native hard conditioning", color=OBSERVATION_COLOR, ha="center", fontsize=6.9)
    ax.text(4.08, 2.48, "inference-time guidance / control", color=EVIDENCE, ha="center", fontsize=7.2)

    ax.text(0.08, 1.96, "(c)", fontsize=8.7, fontweight="bold", va="top")
    ax.text(0.42, 1.96, "Geophysical likelihood branch", fontsize=9.2, fontweight="bold", va="top", color=INK)
    geo_centers = [(1.42, 0.92), (3.18, 0.92), (4.95, 0.92), (6.72, 0.92), (8.55, 0.92)]
    geo_labels = (
        "geology",
        "petrophysical\nmapping",
        "forward operator\n$\\mathcal{F}$",
        "predicted seismic",
        "compare with\nobserved seismic",
    )
    geo_colours = ("#F4EFE6", "#F3EBE6", "#F3EBE6", "#F3EBE6", "#E7F2EF")
    for center, label, colour in zip(geo_centers, geo_labels, geo_colours):
        _box(ax, center, label, width=1.38, height=0.68, face=colour, edge=PHYSICS if center[0] < 8 else EVIDENCE)
    for left, right in zip(geo_centers[:-1], geo_centers[1:]):
        _arrow(ax, (left[0] + 0.72, left[1]), (right[0] - 0.72, right[1]), color=PHYSICS)
    _cube(ax, (1.32, 1.42), scale=0.11, target=True)
    _seismic_glyph(ax, (6.72, 1.41), PHYSICS)
    _seismic_glyph(ax, (8.40, 1.41), EVIDENCE)
    _seismic_glyph(ax, (8.72, 1.41), PHYSICS)

    # The likelihood closes the loop into the frozen inference path.
    _arrow(ax, (9.18, 1.26), (5.55, 4.34), color=PHYSICS, width=1.05, connection="arc3,rad=0.25")
    ax.text(8.72, 2.22, "observation mismatch", color=PHYSICS, fontsize=7.2, ha="center")

    data = {
        "schema": "flow2_framework_figure_data_v1",
        "main_pipeline": list(labels),
        "additional_evidence": list(evidence_labels),
        "geophysical_branch": list(geo_labels),
        "scientific_boundary": {
            "flow": "frozen conditional Flow",
            "decode": "hard categorical",
            "native_conditioning": "surface + borehole observations",
            "additional_evidence_role": "inference-time guidance / control",
            "geophysical_comparison": "predicted versus observed seismic",
        },
    }
    data_path = FIGURE_DATA_DIR / f"{FIGURE_ID}.json"
    write_json(data_path, data)
    outputs = save_figure(fig, FIGURES_DIR / FIGURE_ID, title="Flow2 framework")
    manifest = {
        "schema": "paper_figure_manifest_v1",
        "figure_id": FIGURE_ID,
        "title": "Framework",
        "source_experiment": "method schematic; no experimental screenshot or result values",
        "source_artifacts": [source_record(data_path, "figure schematic data")],
        "case_id": None,
        "candidate_or_sample_ids": [],
        "metrics_shown": [],
        "oracle_panels": [],
        "truth_blind_panels": [],
        "generation": generation_record(SCRIPT_PATH),
        "outputs": output_records(outputs),
        "quality_control": {
            "vector_native": True,
            "raster_panels": False,
            "long_sentences_inside_figure": False,
            "background": "white",
        },
    }
    manifest_path = MANIFESTS_DIR / f"{FIGURE_ID}.json"
    write_json(manifest_path, manifest)
    return {"figure": FIGURE_ID, "outputs": outputs, "manifest": str(manifest_path.relative_to(PROJECT_DIR))}


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    parse_args()
    print(generate())


if __name__ == "__main__":
    main()
