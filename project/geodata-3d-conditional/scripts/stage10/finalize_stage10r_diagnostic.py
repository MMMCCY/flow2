#!/usr/bin/env python3
"""Freeze the Stage10R interpretation and repair its publication outputs.

This script only reads frozen Stage10 artifacts and writes inside the already
created diagnostic_addendum directory.  It never calls Flow or inversion.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.stage10.common import EXPERIMENT_DIR
from scripts.stage10.run_stage10r_diagnostic import _save_figure


ADDENDUM = EXPERIMENT_DIR / "diagnostic_addendum"
DECISION_PATH = EXPERIMENT_DIR / "reports/STAGE10_MACHINE_DECISION.json"
PROTECTED_FILES = (
    EXPERIMENT_DIR / "reports/STAGE10_REPORT.md",
    DECISION_PATH,
    EXPERIMENT_DIR / "audit/leakage_audit.json",
    EXPERIMENT_DIR / "audit/property_inversion_provenance.json",
)
PROTECTED_DIRS = (
    EXPERIMENT_DIR / "bridge",
    EXPERIMENT_DIR / "controls",
    EXPERIMENT_DIR / "diagnostics",
)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(ADDENDUM)),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _protected_snapshot() -> dict[str, str]:
    paths = list(PROTECTED_FILES)
    for directory in PROTECTED_DIRS:
        paths.extend(sorted(path for path in directory.rglob("*") if path.is_file()))
    return {str(path.relative_to(EXPERIMENT_DIR)): _sha256(path) for path in paths}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _range(rows: Iterable[dict[str, object]], key: str) -> list[float]:
    values = [float(row[key]) for row in rows]
    return [min(values), max(values)]


def _mean(rows: Iterable[dict[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return float(np.mean(values))


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _transfer_figure_final(
    ap_matrix: np.ndarray,
    case_ids: list[str],
) -> dict[str, dict[str, object]]:
    """Render the final addendum heatmap without changing raw diagnostics."""
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "font.size": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "svg.fonttype": "none",
            "svg.hashsalt": "stage10r-diagnostic-final-v1",
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(4.75, 3.75))
    ax = fig.add_axes([0.21, 0.24, 0.57, 0.64])
    colorbar_axis = fig.add_axes([0.82, 0.28, 0.035, 0.55])
    image = ax.imshow(ap_matrix, cmap="YlOrBr", vmin=0.0, vmax=float(ap_matrix.max()))
    labels = [f"{index + 1:02d}" for index in range(len(case_ids))]
    ax.set_xticks(range(3), labels=[f"T{value}" for value in labels])
    ax.set_yticks(range(3), labels=[f"B{value}" for value in labels])
    ax.set_xlabel("Retrospective truth case")
    ax.set_ylabel("Frozen post-seismic bridge source")
    ax.set_title("Stage 10R all-by-all label-9 AUPRC")
    contrast_threshold = 0.55 * float(ap_matrix.max())
    for row in range(3):
        for column in range(3):
            ax.text(
                column,
                row,
                f"{ap_matrix[row, column]:.3f}",
                ha="center",
                va="center",
                color="white" if ap_matrix[row, column] > contrast_threshold else "#222222",
                fontweight="bold" if row == column else "normal",
            )
            if row == column:
                ax.add_patch(
                    Rectangle(
                        (column - 0.48, row - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor="#1F4E79",
                        linewidth=1.5,
                    )
                )
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Average precision")
    ax.text(
        0.0,
        -0.27,
        "Blue outline: matched bridge/truth case.\nRetrospective only; frozen Stage10 remains FAIL.",
        transform=ax.transAxes,
        fontsize=6.8,
        color="#555555",
    )
    return _save_figure(fig, ADDENDUM, "stage10r_transfer_matrix")


def _report(summary: dict[str, object], raw: dict[str, object], ap: np.ndarray) -> str:
    cases = raw["case_ids"]
    prior_post = raw["prior_vs_post"]
    geometry = raw["truth_geometry_pairwise"]
    similarity = raw["bridge_similarity_pairwise"]
    prior_similarity = [row for row in similarity if row["map_type"] == "prior_only"]
    post_similarity = [row for row in similarity if row["map_type"] == "post_seismic"]
    transfer = raw["transfer_summary"]
    case_labels = [case.replace("native_seed202609", "") for case in cases]
    lines = [
        "# Stage 10R — Geophysical Probability-Bridge Mechanism Diagnostic Addendum",
        "",
        "## Frozen boundary",
        "",
        "Stage10 is not rerun or reinterpreted. Its machine decision remains "
        "`STOP_BRIDGE_NO_GEOPHYSICAL_INFORMATION` (Stage10-A pass 1/3; "
        "Stage10-B/C/D not executed; Stage10 Flow forwards 0). Stage10R is a "
        "retrospective mechanism diagnostic and cannot authorize Stage10-B.",
        "",
        "No Flow sampling, probability guidance, property/seismic inversion, training, "
        "new cases, parameter sweep, smoothing, or sharpening was executed. The fixed "
        "0.5 probability threshold is used only for retrospective Dice/IoU summaries.",
        "",
        "## Diagnostic interpretation",
        "",
        "- Primary: `CASE_GEOMETRY_CONFUNDED`.",
        "- Complementary mechanism finding: `SEISMIC_ADDS_INCREMENTAL_INFORMATION`.",
        "- Not assigned: `PRIOR_TEMPLATE_DOMINATED`, `INCONCLUSIVE`.",
        "",
        "The frozen Stage10 result remains FAIL. These findings diagnose why the wrong-case "
        "control was difficult and whether seismic altered the categorical probability map; "
        "they do not retroactively change the gate.",
        "",
        "## Diagnostic 1 — all-by-all post-seismic bridge transfer",
        "",
        "| bridge \\ truth | T01 | T02 | T03 |",
        "|---|---:|---:|---:|",
    ]
    for row, label in enumerate(case_labels):
        values = " | ".join(_fmt(value) for value in ap[row])
        lines.append(f"| B{label} | {values} |")
    lines.extend(
        [
            "",
            f"Diagonal mean AUPRC = {_fmt(float(transfer['diagonal_mean_auprc']))}; "
            f"off-diagonal mean = {_fmt(float(transfer['off_diagonal_mean_auprc']))}; "
            f"difference = {_fmt(float(transfer['diagonal_minus_off_diagonal_mean_auprc']))}. "
            f"The diagonal is the row maximum in {float(transfer['fraction_diagonal_is_row_maximum']):.0%} "
            f"and the column maximum in {float(transfer['fraction_diagonal_is_column_maximum']):.0%} of cases.",
            "",
            "Full Brier, ROC-AUC, and fixed-threshold Dice/IoU values are in the CSV outputs.",
            "",
            "## Truth geometry across cases",
            "",
            "| pair | label-9 IoU | centroid distance [voxels] | volume ratio | components | matched-body centroid mean/max |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in geometry:
        left = str(row["truth_case_i"]).replace("native_seed202609", "T")
        right = str(row["truth_case_j"]).replace("native_seed202609", "T")
        lines.append(
            f"| {left}–{right} | {_fmt(float(row['label9_iou']))} | "
            f"{_fmt(float(row['centroid_distance_voxels']))} | "
            f"{_fmt(float(row['volume_ratio_min_over_max']))} | "
            f"{row['component_count_i']}/{row['component_count_j']} | "
            f"{_fmt(float(row['matched_body_centroid_distance_mean']))}/"
            f"{_fmt(float(row['matched_body_centroid_distance_max']))} |"
        )
    lines.extend(
        [
            "",
            f"Pairwise truth IoU spans {_fmt(min(float(row['label9_iou']) for row in geometry))}–"
            f"{_fmt(max(float(row['label9_iou']) for row in geometry))}. Every truth has five connected "
            "components, and matched native-body centroids differ by less than 0.66 voxel at worst. "
            "Together with the nearly equal diagonal/off-diagonal transfer AUPRC, this is direct evidence "
            "that the wrong-case gate is confounded by a shared target-location/geometry template.",
            "",
            "## Diagnostic 2 — prior-only versus post-seismic bridge",
            "",
            "| case | AP prior | AP post | ΔAP seismic | Brier prior | Brier post | ΔBrier seismic | Pearson prior/post | Spearman prior/post | MAD |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for case in cases:
        row = prior_post[case]
        label = case.replace("native_seed202609", "")
        lines.append(
            f"| {label} | {_fmt(row['prior_auprc'])} | {_fmt(row['post_auprc'])} | "
            f"{float(row['delta_ap_seismic']):+.4f} | {_fmt(row['prior_brier'])} | "
            f"{_fmt(row['post_brier'])} | {float(row['delta_brier_seismic']):+.4f} | "
            f"{_fmt(row['spatial_pearson_prior_post'])} | {_fmt(row['spatial_spearman_prior_post'])} | "
            f"{_fmt(row['mean_absolute_probability_change'])} |"
        )
    lines.extend(
        [
            "",
            "AP and Brier improve in all three cases. The AP gain is heterogeneous and almost zero in "
            "case 01, while ROC-AUC decreases in all three cases; therefore the evidence is incremental, "
            "not a claim of uniformly better ranking. Probability changes are nontrivial "
            "(MAD 0.0535–0.0571; RMS 0.1207–0.1237), despite high rank similarity.",
            "",
            "## Diagnostic 3 — cross-case bridge-map similarity",
            "",
            f"Prior-only pairwise Pearson spans {_fmt(min(float(row['pearson']) for row in prior_similarity))}–"
            f"{_fmt(max(float(row['pearson']) for row in prior_similarity))}; post-seismic Pearson spans "
            f"{_fmt(min(float(row['pearson']) for row in post_similarity))}–"
            f"{_fmt(max(float(row['pearson']) for row in post_similarity))}. Prior-only cosine spans "
            f"{_fmt(min(float(row['cosine']) for row in prior_similarity))}–"
            f"{_fmt(max(float(row['cosine']) for row in prior_similarity))}; post-seismic cosine spans "
            f"{_fmt(min(float(row['cosine']) for row in post_similarity))}–"
            f"{_fmt(max(float(row['cosine']) for row in post_similarity))}. Post-seismic maps are therefore "
            "less cross-case correlated than the priors, consistent with observation-dependent changes. "
            "Their smaller pairwise MAD reflects lower/sparser probability mass and is not interpreted alone.",
            "",
            "## Answers to the two mechanism questions",
            "",
            "**Q1.** Yes. There is strong evidence that the Stage10 wrong-case control is confounded by "
            "the three StructuralGeo truths sharing almost the same five-body location template. This "
            "diagnoses the control but does not invalidate or replace its frozen FAIL result.",
            "",
            "**Q2.** The prior already contains a strong shared spatial template, but the correct seismic "
            "assimilation adds incremental label-9 information: ΔAP and ΔBrier are positive in 3/3 cases, "
            "and post maps show observation-dependent changes. The evidence is not consistent with a "
            "pure `PRIOR_TEMPLATE_DOMINATED` interpretation.",
            "",
            "## Recommended next research branch (not executed)",
            "",
            "After manual approval, use an independently pre-registered benchmark whose target anchors, "
            "locations, orientations, and body geometries are deliberately diverse, with no outcome-based "
            "case selection. Re-evaluate the probability-bridge concept only on that independent benchmark. "
            "Do not proceed directly to Stage10-B on the current three cases.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    before = _protected_snapshot()
    raw = _read_json(ADDENDUM / "raw_summary.json")
    decision = _read_json(DECISION_PATH)
    if decision.get("machine_decision") != "STOP_BRIDGE_NO_GEOPHYSICAL_INFORMATION":
        raise RuntimeError("frozen Stage10 machine decision changed")
    if int(decision.get("flow_forward_count_stage10", -1)) != 0:
        raise RuntimeError("frozen Stage10 Flow count changed")
    if any(bool(decision.get(key)) for key in ("stage10b_executed", "stage10c_executed", "stage10d_executed")):
        raise RuntimeError("a prohibited Stage10 phase was executed")

    ap_rows = _read_csv(ADDENDUM / "all_by_all_ap_matrix.csv")
    truth_columns = [key for key in ap_rows[0] if key.startswith("truth_")]
    ap = np.asarray([[float(row[key]) for key in truth_columns] for row in ap_rows], dtype=np.float64)
    repaired_figures = _transfer_figure_final(ap, list(raw["case_ids"]))

    geometry = raw["truth_geometry_pairwise"]
    prior_post_rows = list(raw["prior_vs_post"].values())
    similarity = raw["bridge_similarity_pairwise"]
    prior_similarity = [row for row in similarity if row["map_type"] == "prior_only"]
    post_similarity = [row for row in similarity if row["map_type"] == "post_seismic"]
    summary = {
        "schema": "stage10r_mechanism_diagnostic_summary_v1",
        "status": "COMPLETE_STOP",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True).strip(),
        "original_stage10": {
            "machine_decision": decision["machine_decision"],
            "stage10a_pass_count": 1,
            "stage10b_executed": False,
            "stage10c_executed": False,
            "stage10d_executed": False,
            "flow_forward_count": 0,
            "decision_unchanged": True,
        },
        "stage10r_execution": {
            "flow_forward_count": 0,
            "seismic_inversion_count": 0,
            "new_benchmark_cases": 0,
            "diagnostic_probability_threshold": 0.5,
            "threshold_role": "retrospective Dice/IoU visualization only; not a Stage10 gate",
        },
        "interpretation": {
            "primary": "CASE_GEOMETRY_CONFUNDED",
            "complementary_mechanism_finding": "SEISMIC_ADDS_INCREMENTAL_INFORMATION",
            "not_assigned": ["PRIOR_TEMPLATE_DOMINATED", "INCONCLUSIVE"],
            "stage10_b_authorized": False,
            "stage10_reinterpreted": False,
        },
        "transfer": raw["transfer_summary"],
        "truth_geometry": {
            "pairwise_label9_iou_range": _range(geometry, "label9_iou"),
            "pairwise_label9_iou_mean": _mean(geometry, "label9_iou"),
            "centroid_distance_voxels_range": _range(geometry, "centroid_distance_voxels"),
            "component_counts_all_pairs": [[row["component_count_i"], row["component_count_j"]] for row in geometry],
            "matched_body_centroid_distance_max_overall": max(float(row["matched_body_centroid_distance_max"]) for row in geometry),
            "wrong_case_control_geometry_confounded": True,
        },
        "prior_vs_post": {
            "case_metrics": raw["prior_vs_post"],
            "delta_ap_range": _range(prior_post_rows, "delta_ap_seismic"),
            "delta_ap_mean": _mean(prior_post_rows, "delta_ap_seismic"),
            "delta_brier_range": _range(prior_post_rows, "delta_brier_seismic"),
            "delta_brier_mean": _mean(prior_post_rows, "delta_brier_seismic"),
            "positive_delta_ap_cases": sum(float(row["delta_ap_seismic"]) > 0 for row in prior_post_rows),
            "positive_delta_brier_cases": sum(float(row["delta_brier_seismic"]) > 0 for row in prior_post_rows),
            "seismic_adds_incremental_categorical_information": True,
            "roc_auc_direction_note": "post-seismic ROC-AUC is lower than prior-only in all three cases",
        },
        "cross_case_map_similarity": {
            "prior_pearson_range": _range(prior_similarity, "pearson"),
            "post_pearson_range": _range(post_similarity, "pearson"),
            "prior_spearman_range": _range(prior_similarity, "spearman"),
            "post_spearman_range": _range(post_similarity, "spearman"),
            "prior_cosine_range": _range(prior_similarity, "cosine"),
            "post_cosine_range": _range(post_similarity, "cosine"),
            "prior_template_dominated": False,
        },
        "recommended_next_research_branch": (
            "After manual approval, run an independently pre-registered benchmark with deliberately "
            "diverse target locations/geometries and no outcome-based case selection; do not proceed "
            "directly to Stage10-B on the current cases."
        ),
        "figures": {
            "transfer_matrix": repaired_figures,
            "prior_vs_post": {
                suffix: _record(ADDENDUM / f"stage10r_prior_vs_post.{suffix}")
                for suffix in ("pdf", "svg", "png")
            },
        },
    }
    _write_json(ADDENDUM / "summary.json", summary)
    report_path = ADDENDUM / "STAGE10R_DIAGNOSTIC_REPORT.md"
    report_path.write_text(_report(summary, raw, ap), encoding="utf-8")

    manifest = _read_json(ADDENDUM / "manifest.json")
    manifest["status"] = "complete_interpretation_frozen"
    manifest["final_interpretation"] = summary["interpretation"]
    manifest["final_outputs"] = {
        "summary": _record(ADDENDUM / "summary.json"),
        "report": _record(report_path),
        "transfer_matrix": repaired_figures,
    }
    manifest["finalization_generator"] = {
        "path": str(Path(__file__).resolve().relative_to(REPOSITORY_ROOT)),
        "sha256": _sha256(Path(__file__).resolve()),
        "size_bytes": Path(__file__).resolve().stat().st_size,
    }
    _write_json(ADDENDUM / "manifest.json", manifest)

    after = _protected_snapshot()
    if before != after:
        changed = sorted(set(before) | set(after))
        raise RuntimeError(f"protected Stage10 artifact changed during finalization: {changed}")
    print(
        json.dumps(
            {
                "status": "COMPLETE_STOP",
                "original_stage10_decision": decision["machine_decision"],
                "primary_interpretation": "CASE_GEOMETRY_CONFUNDED",
                "seismic_incremental_information": True,
                "protected_stage10_unchanged": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
