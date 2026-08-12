#!/usr/bin/env python3
"""Generate Figure 4: evidence hierarchy, observability, and inference boundaries."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


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
STYLE_PATH = SCRIPT_PATH.with_name("style.py")
FIGURE_ID = "figure04_evidence_hierarchy"

P1_SUMMARY = PROJECT_DIR / "experiments/stage1_probability/reports/phase1b_v4_12pair/summary.json"
P1_REPORT = PROJECT_DIR / "docs/PHASE1_REPORT.md"
P2_SUMMARY = PROJECT_DIR / "experiments/stage2_property/reports/phase2a_v1_12pair/summary.json"
P2_REPORT = PROJECT_DIR / "docs/PHASE2A_REPORT.md"
P4C_REPORT = PROJECT_DIR / "docs/PHASE4C_REPORT.md"
S7_SUMMARY = PROJECT_DIR / "experiments/stage6_inference_causality/reports/stage7_v1_final_v2/stage7_summary.json"
S7_REPORT = PROJECT_DIR / "experiments/stage6_inference_causality/reports/stage7_v1_final_v2/STAGE7_REPORT.md"
S9_SUMMARY = PROJECT_DIR / "experiments/stage9_flow_prior_posterior/reports/stage9a_prior_support_v1/summary.json"
S9_REPORT = PROJECT_DIR / "experiments/stage9_flow_prior_posterior/reports/stage9a_prior_support_v1/STAGE9A_REPORT.md"
S12_SUMMARY = PROJECT_DIR / "experiments/stage12b_fullgeo_probability_bridge/evaluation/stage12b_a/summary.json"
S12_REPORT = PROJECT_DIR / "experiments/stage12b_fullgeo_probability_bridge/reports/STAGE12B_REPORT.md"
S13_REPORT = PROJECT_DIR / "experiments/stage13_binary_label9_bridge/reports/STAGE13A_REPORT.md"
S14_SUMMARY = PROJECT_DIR / "experiments/stage14_gansim_style_geo_guidance/reports/pilot_v1/summary.json"
S14_REPORT = PROJECT_DIR / "experiments/stage14_gansim_style_geo_guidance/reports/pilot_v1/STAGE14_REPORT.md"

INK = "#25313A"
MUTED = "#66737B"
ORACLE = "#B47916"
BLIND = "#2D6E9F"
PASS = "#2F7D5B"
LIMIT = "#A54B42"
NEUTRAL = "#73828A"


def _arrow(ax, start, end, *, color="#AAB2B4", width=0.8, connection="arc3"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=7,
        linewidth=width,
        color=color,
        connectionstyle=connection,
        shrinkA=2,
        shrinkB=2,
        zorder=1,
    )
    ax.add_patch(patch)


def _axis_arrow(ax, y, title, left, right, color):
    ax.text(0.16, y, title, ha="left", va="center", fontsize=6.5, fontweight="bold", color=INK)
    ax.text(3.02, y, left, ha="right", va="center", fontsize=6.1, color=color)
    _arrow(ax, (3.10, y), (9.70, y), color=color, width=0.75)
    ax.text(9.84, y, right, ha="right", va="center", fontsize=6.1, color=color, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4})


def _node(ax, x, y, item, index):
    edge = item["color"]
    patch = FancyBboxPatch(
        (x - 0.57, y - 0.76),
        1.14,
        1.52,
        boxstyle="round,pad=0.02,rounding_size=0.055",
        facecolor=item["face"],
        edgecolor=edge,
        linewidth=0.85 if item["status"] != "reference" else 0.7,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x - 0.48, y + 0.66, str(index), ha="left", va="top", fontsize=6.3, fontweight="bold", color=edge)
    ax.text(x, y + 0.43, item["title"], ha="center", va="center", fontsize=6.45, fontweight="bold", color=INK, linespacing=1.03)
    ax.text(x, y + 0.06, item["role"], ha="center", va="center", fontsize=5.25, color=item["role_color"], fontweight="bold", linespacing=0.92)
    ax.text(x, y - 0.31, item["metric"], ha="center", va="center", fontsize=6.05, color=INK, linespacing=1.10)
    ax.text(x, y - 0.65, item["status_label"], ha="center", va="bottom", fontsize=5.25, color=edge, fontweight="bold")


def _facts() -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    p1 = read_json(P1_SUMMARY)
    p2 = read_json(P2_SUMMARY)
    s7 = read_json(S7_SUMMARY)
    s9 = read_json(S9_SUMMARY)
    s12 = read_json(S12_SUMMARY)
    s14 = read_json(S14_SUMMARY)
    if p1.get("strict_pairing_validated") is not True or int(p1.get("n_pairs", 0)) != 12:
        raise ValueError("unexpected Phase-1 summary")
    if int(p2.get("n_pairs", 0)) != 12 or p2["gates"]["all_pair_gates_pass"] is not True:
        raise ValueError("unexpected Phase-2a summary")
    if s7.get("status") != "completed" or s7.get("truth_used_for_selection") is not False:
        raise ValueError("unexpected Stage-7 summary")
    if s9.get("status") != "complete" or s9["SUPPORT_PASS"] is not False or s9["DISCRIMINATION_PASS"] is not False:
        raise ValueError("unexpected Stage-9A summary")
    if s12.get("status") != "complete" or s12.get("machine_decision") != "STOP_FULLGEO_BRIDGE_NOT_CASE_SPECIFIC":
        raise ValueError("unexpected Stage-12B summary")
    if s14.get("status") != "complete_stop_no_further_experiments":
        raise ValueError("unexpected Stage-14 summary")

    p1_iou = p1["metrics"]["target_iou"]
    p2_iou = p2["metrics"]["target_iou"]
    s7_correct = [next(arm for arm in case["arms"] if arm["optimized_by"] == "correct") for case in s7["native_replicas"]]
    s7_range = [float(row["hidden_target_iou"]) for row in s7_correct]
    s7_rank_first = sum(bool(case["correct_optimized_is_best_against_correct"]) for case in s7["native_replicas"])
    unique_models = sum(int(case["ensemble"]["unique_hard_model_count"]) for case in s9["cases"])
    support_cases = sum(bool(case["SUPPORT_PASS"]) for case in s9["cases"])
    discrimination_cases = sum(bool(case["DISCRIMINATION_PASS"]) for case in s9["cases"])

    facts = {
        "sparse_flow_label9_iou": float(p1_iou["baseline"]["mean"]),
        "oracle_probability_label9_iou": float(p1_iou["guided"]["mean"]),
        "ideal_property_label9_iou": float(p2_iou["guided"]["mean"]),
        "stage7_hidden_iou_range": [min(s7_range), max(s7_range)],
        "stage7_correct_rank_first_count": s7_rank_first,
        "stage7_case_count": len(s7_range),
        "stage9a_unique_hard_models": unique_models,
        "stage9a_support_case_count": support_cases,
        "stage9a_discrimination_case_count": discrimination_cases,
        "stage12b_diagonal_mean_auprc": float(s12["diagonal_mean_auprc"]),
        "stage12b_off_diagonal_mean_auprc": float(s12["off_diagonal_mean_auprc"]),
        "stage12b_diagonal_row_maximum_count": int(s12["diagonal_row_maximum_count"]),
        "stage14_median_hidden_iou_delta": float(s14["overall_paired_median_hidden_label9_iou_delta"]),
        "stage14_positive_case_count": int(s14["positive_case_count"]),
    }
    if unique_models != 3072 or support_cases != 0 or discrimination_cases != 0:
        raise ValueError("Stage-9A aggregate facts changed")

    nodes = [
        {"title": "Sparse-conditioned\nFlow", "role": "REFERENCE", "role_color": NEUTRAL, "metric": f"label-9 IoU\n{facts['sparse_flow_label9_iou']:.3f}", "status": "reference", "status_label": "under-constrained", "color": NEUTRAL, "face": "#F1F3F3", "protocol": "Phase1 alpha-zero baseline"},
        {"title": "Oracle\nprobability", "role": "ORACLE", "role_color": ORACLE, "metric": f"label-9 IoU\n{facts['oracle_probability_label9_iou']:.3f}", "status": "pass", "status_label": "control demonstrated", "color": PASS, "face": "#EEF6F1", "protocol": "Phase1 truth-derived probability"},
        {"title": "Ideal 3-D\nproperties", "role": "UPPER BOUND", "role_color": ORACLE, "metric": f"label-9 IoU\n{facts['ideal_property_label9_iou']:.3f}", "status": "pass", "status_label": "control demonstrated", "color": PASS, "face": "#EEF6F1", "protocol": "Phase2a ideal/noiseless properties"},
        {"title": "Direct\nseismic guidance", "role": "ACQUISITION\nDOMAIN", "role_color": BLIND, "metric": "physical RMSE $\\downarrow$\nhard geology weak", "status": "limit", "status_label": "observability limit", "color": LIMIT, "face": "#F8EEEB", "protocol": "Phase4c single strict pair"},
        {"title": "Structured hard\nseismic", "role": "BOUNDED\nSPACE", "role_color": BLIND, "metric": f"hidden IoU\n{facts['stage7_hidden_iou_range'][0]:.3f}–{facts['stage7_hidden_iou_range'][1]:.3f}\nrank-first {s7_rank_first}/{len(s7_range)}", "status": "pass", "status_label": "structured exception", "color": PASS, "face": "#E8F4ED", "protocol": "Stage7 structured hard inference"},
        {"title": "Unrestricted\nfrozen Flow", "role": "TRUTH-BLIND", "role_color": BLIND, "metric": f"{unique_models} unique\nsupport 0/3\ndiscrimination 0/3", "status": "limit", "status_label": "prior-support limit", "color": LIMIT, "face": "#F8EEEB", "protocol": "Stage9A frozen Flow prior"},
        {"title": "FullGeo\nprobability bridge", "role": "BRIDGE ONLY", "role_color": BLIND, "metric": f"diag AP {facts['stage12b_diagonal_mean_auprc']:.4f}\noff-diag {facts['stage12b_off_diagonal_mean_auprc']:.4f}\nrow max 1/5", "status": "limit", "status_label": "case specificity absent", "color": LIMIT, "face": "#F8EEEB", "protocol": "Stage12B bridge-only gate"},
        {"title": "Probability\n$\\rightarrow$ Flow", "role": "PAIRED\nSYNTHETIC", "role_color": BLIND, "metric": f"median $\\Delta$IoU\n{facts['stage14_median_hidden_iou_delta']:.4f}\npositive 1/5", "status": "limit", "status_label": "current bridge limited", "color": LIMIT, "face": "#F8EEEB", "protocol": "Stage14 paired experiment"},
    ]
    provenance = [
        {"metric": "Sparse Flow and oracle probability label-9 IoU", "source_path": str(P1_SUMMARY.relative_to(PROJECT_DIR)), "json_key": "metrics.target_iou.baseline.mean; metrics.target_iou.guided.mean"},
        {"metric": "Ideal property label-9 IoU", "source_path": str(P2_SUMMARY.relative_to(PROJECT_DIR)), "json_key": "metrics.target_iou.guided.mean"},
        {"metric": "Direct seismic qualitative boundary", "source_path": str(P4C_REPORT.relative_to(PROJECT_DIR)), "reconstruction_rule": "formal decision: hard seismic residual decreases while complete hard-geology gate fails"},
        {"metric": "Structured hidden IoU and correct rank-first", "source_path": str(S7_SUMMARY.relative_to(PROJECT_DIR)), "json_key": "native_replicas[*].arms[optimized_by=correct].hidden_target_iou; correct_optimized_is_best_against_correct"},
        {"metric": "Frozen Flow support/discrimination", "source_path": str(S9_SUMMARY.relative_to(PROJECT_DIR)), "json_key": "cases[*].ensemble.unique_hard_model_count; cases[*].SUPPORT_PASS; cases[*].DISCRIMINATION_PASS"},
        {"metric": "FullGeo bridge specificity", "source_path": str(S12_SUMMARY.relative_to(PROJECT_DIR)), "json_key": "diagonal_mean_auprc; off_diagonal_mean_auprc; diagonal_row_maximum_count"},
        {"metric": "Probability-to-Flow paired outcome", "source_path": str(S14_SUMMARY.relative_to(PROJECT_DIR)), "json_key": "overall_paired_median_hidden_label9_iou_delta; positive_case_count"},
    ]
    return facts, nodes, provenance


def generate() -> dict[str, object]:
    configure_matplotlib()
    ensure_output_dirs()
    facts, nodes, provenance = _facts()

    fig = plt.figure(figsize=(mm_to_inches(DOUBLE_COLUMN_MM), 4.65))
    ax = fig.add_axes([0.018, 0.035, 0.968, 0.94])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.8)
    ax.axis("off")
    ax.text(0.10, 6.66, "Evidence map: from controllability to observability and prior support", fontsize=9.5, fontweight="bold", color=INK, va="top")
    ax.text(9.90, 6.64, "distinct protocols — not paired cross-stage effect sizes", fontsize=6.4, color=MUTED, ha="right", va="top")

    _axis_arrow(ax, 6.10, "Information proximity to target geology", "HIGH", "LOW", ORACLE)
    _axis_arrow(ax, 5.73, "Observation realism", "LOW", "HIGH", BLIND)
    _axis_arrow(ax, 5.36, "Geological controllability", "DEMONSTRATED", "LIMITED / UNKNOWN", LIMIT)

    xs = (0.67, 1.90, 3.13, 4.36, 5.59, 6.82, 8.05, 9.28)
    ys = (3.52, 3.52, 3.52, 3.52, 3.82, 3.52, 3.52, 3.52)
    for left_x, left_y, right_x, right_y in zip(xs[:-1], ys[:-1], xs[1:], ys[1:]):
        _arrow(ax, (left_x + 0.59, left_y), (right_x - 0.59, right_y), connection="arc3,rad=-0.04" if left_y != right_y else "arc3")
    for index, (x, y, item) in enumerate(zip(xs, ys, nodes), start=1):
        _node(ax, x, y, item, index)
    ax.text(5.59, 4.72, "bounded geological hypothesis space", ha="center", va="bottom", fontsize=6.05, color=PASS, fontweight="bold")

    conclusions = (
        (1.76, "CONTROL", "demonstrated with privileged\n3-D evidence", PASS, "#EDF6F1"),
        (5.00, "OBSERVABILITY", "acquisition-domain conversion\nremains limiting", LIMIT, "#F8EEEB"),
        (8.24, "PARAMETERIZATION", "bounded structured inference\nforms the key exception", BLIND, "#EDF3F8"),
    )
    for x, title, text, edge, face in conclusions:
        ax.add_patch(FancyBboxPatch((x - 1.35, 0.46), 2.70, 0.86, boxstyle="round,pad=0.02,rounding_size=0.05", facecolor=face, edgecolor=edge, linewidth=0.75))
        ax.text(x, 1.10, title, ha="center", va="center", fontsize=6.2, color=edge, fontweight="bold")
        ax.text(x, 0.76, text, ha="center", va="center", fontsize=6.55, color=INK, linespacing=1.08)
    ax.text(9.90, 0.15, "Stage 14 is a paired synthetic experiment, not a measured field test.", fontsize=6.15, color=MUTED, ha="right")

    data = {
        "schema": "figure04_evidence_hierarchy_data_v1",
        "cross_protocol_paired_comparison": False,
        "axes": {
            "information_proximity_to_target_geology": "high to low",
            "geological_controllability": "demonstrated to limited/unknown",
            "observation_realism": "low to high",
        },
        "facts": facts,
        "nodes": nodes,
        "metric_provenance": provenance,
        "conclusions": ["Control is demonstrated.", "Acquisition-domain observability remains limiting.", "Inference parameterization matters."],
    }
    data_path = FIGURE_DATA_DIR / f"{FIGURE_ID}.json"
    write_json(data_path, data)
    outputs = save_figure(fig, FIGURES_DIR / FIGURE_ID, title="Evidence hierarchy and inference boundaries")
    source_paths = (
        (P1_SUMMARY, "Phase-1 machine summary"), (P1_REPORT, "Phase-1 report"),
        (P2_SUMMARY, "Phase-2a machine summary"), (P2_REPORT, "Phase-2a report"),
        (P4C_REPORT, "direct acquisition-domain seismic report"),
        (S7_SUMMARY, "Stage-7 structured-inference summary"), (S7_REPORT, "Stage-7 report"),
        (S9_SUMMARY, "Stage-9A unrestricted frozen-Flow summary"), (S9_REPORT, "Stage-9A report"),
        (S12_SUMMARY, "Stage-12B bridge-only summary"), (S12_REPORT, "Stage-12B report"),
        (S13_REPORT, "Stage-13 identifiability boundary report"),
        (S14_SUMMARY, "Stage-14 paired summary"), (S14_REPORT, "Stage-14 report"),
        (STYLE_PATH, "shared paper visual language"), (data_path, "exact evidence-map data"),
    )
    generation = generation_record(SCRIPT_PATH)
    manifest = {
        "schema_version": "paper_figure_manifest_v2",
        "figure_id": FIGURE_ID,
        "source_files": [source_record(path, role) for path, role in source_paths],
        "metric_extraction_keys": provenance,
        "case_ids": ["cond_generation_0", "native_seed20260807", "native_seed20260808", "native_seed20260809", "native_seed20260901", "native_seed20260902", "native_seed20260903", "fullgeo_case01", "fullgeo_case02", "fullgeo_case03", "fullgeo_case04", "fullgeo_case05"],
        "model_hashes": "recorded in source experiment manifests; no model loaded by this figure",
        "camera": None,
        "color_map": generation["style"]["label_colors"],
        "label_meanings": {"label9": "shared pressure-test label; protocol-specific geometry and evidence"},
        "truth_visibility": "aggregate retrospective hard-geology metrics only; no truth selected or displayed",
        "oracle_vs_inference_visible": {
            "oracle": ["Phase1 probability", "Phase2a ideal properties"],
            "truth_blind": ["Phase4c seismic", "Stage7 structured seismic", "Stage9A frozen Flow", "Stage12B bridge", "Stage14 paired Flow"],
        },
        "protocol_mixing_policy": "nodes are distinct protocol evidence, not paired cross-stage effect sizes",
        "scientific_boundaries": {"stage7_structured_inference": True, "stage9_unrestricted_frozen_flow": True, "stage14_measured_field_test": False, "strict_bayesian_posterior_claim": False},
        "metrics": facts,
        "generation": generation,
        "outputs": output_records(outputs),
        "quality_control": {"vector_native": True, "cross_protocol_bar_chart": False, "structured_exception_visible": True, "limitations_visible": True},
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
