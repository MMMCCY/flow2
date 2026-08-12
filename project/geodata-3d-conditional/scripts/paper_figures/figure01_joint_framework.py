#!/usr/bin/env python3
"""Generate Figure 1: reciprocal joint-modeling framework and evidence hierarchy."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.paper_figures.style import (
    DOUBLE_COLUMN_MM,
    FIGURE_DATA_DIR,
    FIGURES_DIR,
    MANIFESTS_DIR,
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
STYLE_PATH = SCRIPT_PATH.with_name("style.py")
FIGURE_ID = "figure01_joint_framework"
RESEARCH_GOAL = PROJECT_DIR / "docs/RESEARCH_GOAL.md"
EXPERIMENT_PROTOCOL = PROJECT_DIR / "docs/EXPERIMENT_PROTOCOL.md"

INK = "#24313A"
MUTED = "#647078"
FLOW = "#3969AC"
EVIDENCE = "#238878"
PHYSICS = "#B4552D"
ORACLE = "#B57A16"
TRAIN_FILL = "#F1F4EF"
INFER_FILL = "#EEF2F8"
EVAL_FILL = "#F6F0E9"


def _arrow(ax, start, end, *, color=MUTED, width=0.9, connection="arc3", style="-|>"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=8,
        linewidth=width,
        color=color,
        connectionstyle=connection,
        shrinkA=2,
        shrinkB=2,
        zorder=2,
    )
    ax.add_patch(patch)
    return patch


def _box(
    ax,
    center,
    text,
    *,
    width=1.22,
    height=0.66,
    face="white",
    edge=INK,
    fontsize=7.0,
    weight="normal",
):
    x, y = center
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.065",
        facecolor=face,
        edgecolor=edge,
        linewidth=0.78,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        fontweight=weight,
        linespacing=1.07,
        zorder=4,
    )
    return patch


def _group(ax, x0, x1, title, subtitle, face, edge):
    ax.add_patch(
        FancyBboxPatch(
            (x0, 4.55),
            x1 - x0,
            2.18,
            boxstyle="round,pad=0.025,rounding_size=0.07",
            facecolor=face,
            edgecolor=edge,
            linewidth=0.8,
            zorder=0,
        )
    )
    ax.text(x0 + 0.12, 6.54, title, ha="left", va="top", fontsize=8.2, fontweight="bold", color=edge)
    ax.text(x0 + 0.12, 6.25, subtitle, ha="left", va="top", fontsize=6.05, color=MUTED)


def _evidence_card(ax, x, level, title, subtitle, *, oracle):
    edge = ORACLE if oracle else EVIDENCE
    face = "#FBF4E5" if oracle else "#EAF4F2"
    ax.add_patch(
        FancyBboxPatch(
            (x - 0.94, 2.61),
            1.88,
            1.10,
            boxstyle="round,pad=0.025,rounding_size=0.06",
            facecolor=face,
            edgecolor=edge,
            linewidth=0.8,
        )
    )
    ax.text(x - 0.80, 3.56, level, ha="left", va="top", fontsize=6.3, fontweight="bold", color=edge)
    ax.text(x, 3.24, title, ha="center", va="center", fontsize=6.75, fontweight="bold", color=INK, linespacing=1.0)
    ax.text(x, 2.83, subtitle, ha="center", va="center", fontsize=5.65, color=MUTED, linespacing=1.03)


def generate() -> dict[str, object]:
    configure_matplotlib()
    ensure_output_dirs()
    fig = plt.figure(figsize=(mm_to_inches(DOUBLE_COLUMN_MM), 4.85))
    ax = fig.add_axes([0.018, 0.035, 0.968, 0.945])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.7)
    ax.axis("off")

    ax.text(0.08, 7.56, "Reciprocal geological–geophysical joint modeling", fontsize=9.7, fontweight="bold", color=INK, va="top")
    ax.text(11.90, 7.54, "conceptual framework — not a claim of a completed Bayesian posterior", fontsize=6.5, color=MUTED, ha="right", va="top")

    _group(ax, 0.12, 4.08, "A  Training context", "StructuralGeo + sparse conditioning", TRAIN_FILL, "#56725A")
    _group(ax, 4.18, 10.18, "B  Inference", "frozen Flow · no checkpoint update", INFER_FILL, FLOW)
    _group(ax, 10.28, 11.88, "C  Evaluation", "hard outputs", EVAL_FILL, PHYSICS)

    centers = ((0.78, 5.48), (2.02, 5.48), (3.35, 5.48), (4.92, 5.48), (6.45, 5.48), (7.95, 5.48), (9.42, 5.48), (11.08, 5.48))
    labels = (
        "StructuralGeo\nhistories",
        "3-D categorical\ngeology",
        "surface + 9\nboreholes",
        "frozen conditional\nFlow",
        "continuous embedding\n+ ODE",
        "hard categorical\ndecode",
        "geological\nensemble",
        "hard geology\n+ physical\nconsistency",
    )
    faces = (TRAIN_FILL, "#F5EFE5", "#E8F3F0", "#E8EEF8", "#EFEAF5", "#F5EFE5", "#F3EDE6", EVAL_FILL)
    edges = ("#56725A", INK, EVIDENCE, FLOW, "#755293", INK, INK, PHYSICS)
    widths = (1.05, 1.10, 1.15, 1.30, 1.38, 1.20, 1.10, 1.30)
    for center, label, face, edge, width in zip(centers, labels, faces, edges, widths):
        _box(ax, center, label, width=width, face=face, edge=edge, fontsize=6.55, weight="bold" if center[0] in (4.92, 11.08) else "normal")
    for left, right in zip(centers[:-1], centers[1:]):
        _arrow(ax, (left[0] + widths[centers.index(left)] / 2 + 0.02, 5.48), (right[0] - widths[centers.index(right)] / 2 - 0.02, 5.48))

    ax.text(0.18, 4.28, "D  Information hierarchy entering inference", fontsize=8.5, fontweight="bold", color=INK, va="top")
    _evidence_card(ax, 3.45, "LEVEL I", "3-D probability\nevidence", "truth-derived oracle\ncontrollability upper bound", oracle=True)
    _evidence_card(ax, 5.72, "LEVEL II", "3-D property\nevidence", "ideal, full-resolution, noiseless\nupper bound", oracle=True)
    _evidence_card(ax, 7.99, "LEVEL III", "acquisition-domain\nseismic", "truth-blind synthetic observation\nspatially broad, band-limited", oracle=False)
    for x, target_x, curve in ((3.45, 4.75, -0.16), (5.72, 6.25, -0.05), (7.99, 6.85, 0.15)):
        _arrow(ax, (x, 3.67), (target_x, 4.84), color=ORACLE if x < 7 else EVIDENCE, width=0.95, connection=f"arc3,rad={curve}")
    ax.text(9.70, 3.15, "distinct protocols /\ninference mechanisms", ha="left", va="center", fontsize=5.85, color=MUTED)

    ax.text(0.18, 2.30, "E  Geophysical likelihood and reciprocal constraints", fontsize=8.5, fontweight="bold", color=INK, va="top")
    geo_centers = ((0.92, 1.22), (2.42, 1.22), (3.92, 1.22), (5.42, 1.22), (6.92, 1.22))
    geo_labels = ("hard geology", "petrophysical\nmapping", "forward operator\n$\\mathcal{F}$", "predicted\nseismic", "observed-seismic\ncomparison")
    for center, label in zip(geo_centers, geo_labels):
        _box(ax, center, label, width=1.18, height=0.62, face="#F6EEE9", edge=PHYSICS, fontsize=6.45)
    for left, right in zip(geo_centers[:-1], geo_centers[1:]):
        _arrow(ax, (left[0] + 0.61, 1.22), (right[0] - 0.61, 1.22), color=PHYSICS)

    _arrow(ax, (7.48, 1.49), (6.45, 4.80), color=EVIDENCE, width=1.0, connection="arc3,rad=-0.12")
    ax.text(7.62, 2.02, "Geophysics provides\nspatially broad evidence", fontsize=6.25, color=EVIDENCE, ha="center")
    _arrow(ax, (5.04, 4.83), (9.12, 1.20), color=FLOW, width=1.0, connection="arc3,rad=-0.14")
    ax.text(9.30, 2.02, "Geological prior constrains\nadmissible models", fontsize=6.25, color=FLOW, ha="center")

    ax.add_patch(Rectangle((8.10, 0.32), 3.70, 0.88, facecolor="#F2F5F3", edgecolor="#9AA5A0", linewidth=0.65))
    ax.text(9.95, 0.97, "Research target", ha="center", va="center", fontsize=6.5, fontweight="bold", color=MUTED)
    ax.text(9.95, 0.65, "condition-exact · observation-consistent\ngeologically plausible ensemble", ha="center", va="center", fontsize=6.15, color=INK, linespacing=1.05)

    data = {
        "schema": "figure01_joint_framework_data_v1",
        "main_pipeline": list(labels),
        "boundaries": {
            "training": "StructuralGeo + surface/borehole conditioning",
            "inference": "frozen conditional Flow; no checkpoint update",
            "evaluation": "hard categorical geology + geophysical consistency",
        },
        "information_hierarchy": [
            {"level": "I", "evidence": "3-D probability evidence", "role": "truth-derived oracle upper bound"},
            {"level": "II", "evidence": "3-D property evidence", "role": "ideal/noiseless upper bound"},
            {"level": "III", "evidence": "acquisition-domain seismic", "role": "truth-blind observation"},
        ],
        "reciprocal_concept": {
            "geological_prior": "constrains admissible models",
            "geophysics": "provides spatially broad evidence beyond sparse geology",
            "bayesian_posterior_claim": False,
        },
        "research_target": "condition-exact, observation-consistent, geologically plausible ensemble",
    }
    data_path = FIGURE_DATA_DIR / f"{FIGURE_ID}.json"
    write_json(data_path, data)
    outputs = save_figure(fig, FIGURES_DIR / FIGURE_ID, title="Reciprocal geological-geophysical joint modeling framework")
    manifest = {
        "schema_version": "paper_figure_manifest_v2",
        "figure_id": FIGURE_ID,
        "source_experiment": "method and research-goal schematic; no cross-protocol metric comparison",
        "source_files": [
            source_record(RESEARCH_GOAL, "authoritative research goal"),
            source_record(EXPERIMENT_PROTOCOL, "strict inference/evaluation protocol"),
            source_record(STYLE_PATH, "shared paper visual language"),
            source_record(data_path, "exact schematic content"),
        ],
        "metric_extraction_keys": [],
        "case_ids": [],
        "model_hashes": [],
        "camera": None,
        "color_map": generation_record(SCRIPT_PATH)["style"]["label_colors"],
        "label_meanings": {"label9": "demonstration target lithology; not a universal dike semantic"},
        "truth_visibility": "no truth tensor displayed",
        "oracle_vs_inference_visible": data["information_hierarchy"],
        "scientific_boundaries": {
            "frozen_flow_explicit": True,
            "checkpoint_update": False,
            "hard_decode_explicit": True,
            "final_bayesian_posterior_claim": False,
            "research_target_marked_as_goal": True,
        },
        "generation": generation_record(SCRIPT_PATH),
        "outputs": output_records(outputs),
        "quality_control": {"vector_native": True, "white_background": True, "distinct_evidence_protocols_explicit": True},
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
