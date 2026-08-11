#!/usr/bin/env python3
"""Generate Figure 4: a protocol-aware information hierarchy matrix."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


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
    read_json,
    save_figure,
    source_record,
    write_json,
)


SCRIPT_PATH = Path(__file__).resolve()
FIGURE_ID = "figure04_information_hierarchy"
P1_SUMMARY = PROJECT_DIR / "experiments/stage1_probability/reports/phase1b_v4_12pair/summary.json"
P2_SUMMARY = PROJECT_DIR / "experiments/stage2_property/reports/phase2a_v1_12pair/summary.json"
S7_SUMMARY = PROJECT_DIR / "experiments/stage6_inference_causality/reports/stage7_v1_final_v2/stage7_summary.json"
S9_SUMMARY = PROJECT_DIR / "experiments/stage9_flow_prior_posterior/reports/stage9a_prior_support_v1/summary.json"
P1_REPORT = PROJECT_DIR / "docs/PHASE1_REPORT.md"
P2_REPORT = PROJECT_DIR / "docs/PHASE2A_REPORT.md"
S7_REPORT = PROJECT_DIR / "experiments/stage6_inference_causality/reports/stage7_v1_final_v2/STAGE7_REPORT.md"
S9_REPORT = PROJECT_DIR / "experiments/stage9_flow_prior_posterior/reports/stage9a_prior_support_v1/STAGE9A_REPORT.md"

INK = "#25313A"
MUTED = "#617078"
GRID = "#D4D9D7"
ORACLE = "#B86B00"
BLIND = "#2B6CA3"
PASS = "#2F7D5B"
FAIL = "#A63C3C"
ROW_FILLS = ("#FBF7EC", "#F9F5EA", "#EFF5F8", "#F8EEEE")


def _cell(ax, x0, y0, width, height, text, *, face, align="left", color=INK, weight="normal", fontsize=7.35):
    ax.add_patch(Rectangle((x0, y0), width, height, facecolor=face, edgecolor=GRID, linewidth=0.65))
    x = x0 + (0.06 * width if align == "left" else width / 2)
    ax.text(
        x,
        y0 + height / 2,
        text,
        ha=align,
        va="center",
        fontsize=fontsize,
        color=color,
        fontweight=weight,
        linespacing=1.16,
    )


def _badge(ax, x, y, text, color):
    width = 0.88 if len(text) < 9 else 1.15
    patch = FancyBboxPatch(
        (x - width / 2, y - 0.16),
        width,
        0.32,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor="white",
        edgecolor=color,
        linewidth=0.75,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=6.35, color=color, fontweight="bold")


def generate() -> dict[str, object]:
    configure_matplotlib()
    ensure_output_dirs()
    p1 = read_json(P1_SUMMARY)
    p2 = read_json(P2_SUMMARY)
    s7 = read_json(S7_SUMMARY)
    s9 = read_json(S9_SUMMARY)
    if p1.get("strict_pairing_validated") is not True or int(p1.get("n_pairs", 0)) != 12:
        raise ValueError("unexpected Phase-1 authoritative summary")
    if int(p2.get("n_pairs", 0)) != 12 or p2["gates"]["all_pair_gates_pass"] is not True:
        raise ValueError("unexpected Phase-2 authoritative summary")
    if s7.get("truth_used_for_selection") is not False or s7.get("status") != "completed":
        raise ValueError("unexpected Stage-7 selector provenance")
    if s9.get("status") != "complete":
        raise ValueError("unexpected Stage-9A status")

    p1_iou = p1["metrics"]["target_iou"]
    p2_iou = p2["metrics"]["target_iou"]
    s7_correct = [
        next(arm for arm in case["arms"] if arm["optimized_by"] == "correct")
        for case in s7["native_replicas"]
    ]
    s7_iou_range = (min(float(row["hidden_target_iou"]) for row in s7_correct), max(float(row["hidden_target_iou"]) for row in s7_correct))
    unique_models = sum(int(case["ensemble"]["unique_hard_model_count"]) for case in s9["cases"])
    support_count = sum(int(case["support_passing_candidate_count"]) for case in s9["cases"])
    if unique_models != 3072 or support_count != 0:
        raise ValueError("Stage-9A terminal facts differ from the registered report")
    if s9["SUPPORT_PASS"] is not False or s9["DISCRIMINATION_PASS"] is not False:
        raise ValueError("Stage-9A terminal verdict changed")

    rows = [
        {
            "stage": "Phase 1\nprobability",
            "badge": ("ORACLE", ORACLE),
            "representation": "$P(\\mathrm{label}\\,9)$\n3-D voxel field",
            "specificity": "voxel-resolved\nclass-targeted",
            "ambiguity": "low by construction\ntruth-aligned target",
            "outcome": "12/12 paired gains\nIoU$_9$ {:.3f} $\\rightarrow$ {:.3f}".format(float(p1_iou["baseline"]["mean"]), float(p1_iou["guided"]["mean"])),
            "outcome_color": PASS,
        },
        {
            "stage": "Phase 2\nproperty",
            "badge": ("ORACLE", ORACLE),
            "representation": "density + susceptibility\n3-D property fields",
            "specificity": "voxel-resolved\nall-class mapping",
            "ambiguity": "codebook-dependent\ncontrast sensitive",
            "outcome": "12/12 paired gains\nIoU$_9$ {:.3f} $\\rightarrow$ {:.3f}".format(float(p2_iou["baseline"]["mean"]), float(p2_iou["guided"]["mean"])),
            "outcome_color": PASS,
        },
        {
            "stage": "Stage 7\nstructured seismic",
            "badge": ("TRUTH-BLIND", BLIND),
            "representation": "full $x$-$y$-time\nseismic volume",
            "specificity": "depth-localized,\nband-limited response",
            "ambiguity": "petrophysical +\nwavelet non-uniqueness",
            "outcome": "correct arm ranks first 3/3\nhidden IoU {:.3f}–{:.3f}".format(*s7_iou_range),
            "outcome_color": PASS,
        },
        {
            "stage": "Stage 9A\nunrestricted Flow + seismic",
            "badge": ("TRUTH-BLIND", BLIND),
            "representation": "full $x$-$y$-time seismic\n+ frozen Flow pool",
            "specificity": "global likelihood;\nweak target enrichment",
            "ambiguity": "prior support +\nlikelihood alignment",
            "outcome": "3072 unique hard models\n0 full-support candidates\nSUPPORT=FAIL\nDISCRIMINATION=FAIL",
            "outcome_color": FAIL,
        },
    ]

    fig = plt.figure(figsize=(mm_to_inches(DOUBLE_COLUMN_MM), 3.78))
    ax = fig.add_axes([0.018, 0.04, 0.968, 0.93])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.3)
    ax.axis("off")
    ax.text(0.02, 5.18, "(a)", fontsize=8.7, fontweight="bold", va="top")
    ax.text(0.42, 5.18, "Information hierarchy across distinct protocols", fontsize=9.4, fontweight="bold", va="top", color=INK)
    ax.text(9.96, 5.17, "schematic — magnitudes are not cross-protocol benchmarks", fontsize=6.9, ha="right", va="top", color=MUTED)

    x_edges = (0.0, 1.85, 4.0, 5.95, 7.80, 10.0)
    headers = ("Protocol", "Evidence representation", "Spatial specificity", "Categorical ambiguity", "Hard reconstruction outcome")
    header_y, header_h = 4.52, 0.48
    for index, header in enumerate(headers):
        _cell(ax, x_edges[index], header_y, x_edges[index + 1] - x_edges[index], header_h, header, face="#E8ECEB", align="center", weight="bold", fontsize=7.15)
    row_h = 1.05
    for row_index, row in enumerate(rows):
        y0 = header_y - (row_index + 1) * row_h
        fill = ROW_FILLS[row_index]
        _cell(ax, x_edges[0], y0, x_edges[1] - x_edges[0], row_h, row["stage"], face=fill, align="left", weight="bold", fontsize=7.55)
        _badge(ax, (x_edges[0] + x_edges[1]) / 2, y0 + 0.19, row["badge"][0], row["badge"][1])
        _cell(ax, x_edges[1], y0, x_edges[2] - x_edges[1], row_h, row["representation"], face=fill)
        _cell(ax, x_edges[2], y0, x_edges[3] - x_edges[2], row_h, row["specificity"], face=fill)
        _cell(ax, x_edges[3], y0, x_edges[4] - x_edges[3], row_h, row["ambiguity"], face=fill)
        _cell(
            ax,
            x_edges[4],
            y0,
            x_edges[5] - x_edges[4],
            row_h,
            row["outcome"],
            face=fill,
            color=row["outcome_color"],
            weight="bold" if row_index == 3 else "normal",
            fontsize=6.95 if row_index == 3 else 7.25,
        )
    ax.text(
        0.02,
        0.10,
        "Oracle fields test controllability; structured and unrestricted seismic rows test different truth-blind inference spaces.",
        fontsize=7.0,
        color=MUTED,
        va="bottom",
    )

    data = {
        "schema": "information_hierarchy_figure_data_v1",
        "cross_protocol_benchmark": False,
        "rows": rows,
        "facts": {
            "phase1_pairs": 12,
            "phase1_mean_label9_iou": [float(p1_iou["baseline"]["mean"]), float(p1_iou["guided"]["mean"])],
            "phase2_pairs": 12,
            "phase2_mean_label9_iou": [float(p2_iou["baseline"]["mean"]), float(p2_iou["guided"]["mean"])],
            "stage7_native_correct_rank_first_count": sum(bool(case["correct_optimized_is_best_against_correct"]) for case in s7["native_replicas"]),
            "stage7_hidden_iou_range": list(s7_iou_range),
            "stage9a_unique_hard_models": unique_models,
            "stage9a_full_support_candidates": support_count,
            "stage9a_support_pass": False,
            "stage9a_discrimination_pass": False,
        },
    }
    data_path = FIGURE_DATA_DIR / f"{FIGURE_ID}.json"
    write_json(data_path, data)
    outputs = save_figure(fig, FIGURES_DIR / FIGURE_ID, title="Information hierarchy")
    source_paths = (
        (P1_SUMMARY, "authoritative Phase-1 machine summary"),
        (P1_REPORT, "authoritative Phase-1 human report"),
        (P2_SUMMARY, "authoritative Phase-2a machine summary"),
        (P2_REPORT, "authoritative Phase-2a human report"),
        (S7_SUMMARY, "authoritative Stage-7 final summary"),
        (S7_REPORT, "authoritative Stage-7 final report"),
        (S9_SUMMARY, "authoritative Stage-9A machine summary"),
        (S9_REPORT, "authoritative Stage-9A report"),
        (data_path, "exact plotted matrix data"),
    )
    manifest = {
        "schema": "paper_figure_manifest_v1",
        "figure_id": FIGURE_ID,
        "title": "Information hierarchy",
        "source_experiment": ["Phase1", "Phase2a", "Stage7", "Stage9A"],
        "source_artifacts": [source_record(path, role) for path, role in source_paths],
        "case_id": "multiple distinct protocols; no shared-case benchmark implied",
        "candidate_or_sample_ids": [],
        "metrics_shown": data["facts"],
        "oracle_panels": ["Phase-1 probability row", "Phase-2 property row"],
        "truth_blind_panels": ["Stage-7 structured seismic row", "Stage-9A unrestricted Flow + seismic row"],
        "stage9a_limitations_visible": [
            "3072 unique hard models",
            "0 full-support candidates",
            "SUPPORT=FAIL",
            "DISCRIMINATION=FAIL",
        ],
        "generation": generation_record(SCRIPT_PATH),
        "outputs": output_records(outputs),
        "quality_control": {
            "vector_native": True,
            "cross_protocol_numeric_bar_chart": False,
            "stage9a_failure_not_hidden": True,
            "oracle_and_truth_blind_roles_explicit": True,
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
