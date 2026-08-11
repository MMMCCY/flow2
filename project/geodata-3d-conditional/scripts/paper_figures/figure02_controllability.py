#!/usr/bin/env python3
"""Generate Figure 2 from the authoritative paired Phase-1/Phase-2 artifacts."""

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
    CAMERA,
    DOUBLE_COLUMN_MM,
    FIGURE_DATA_DIR,
    FIGURES_DIR,
    LABEL_COLORS,
    LABEL9_COLOR,
    MANIFESTS_DIR,
    OBSERVATION_COLOR,
    assert_metrics_match,
    configure_matplotlib,
    ensure_output_dirs,
    generation_record,
    load_volume,
    mm_to_inches,
    output_records,
    panel_label,
    read_csv_row,
    read_json,
    render_categorical_panel,
    render_conditions_panel,
    render_target_panel,
    save_figure,
    sha256,
    show_render,
    source_record,
    validate_same_shape,
    write_deterministic_npz,
    write_json,
)


SCRIPT_PATH = Path(__file__).resolve()
FIGURE_ID = "figure02_controllability"
CASE_ID = "cond_generation_0"
SEED = 42
SAMPLE_ID = 0
TARGET_LABEL = 9

TRUTH = PROJECT_DIR / "samples/jupyter-demo/cond_generation_0/true_model.pt"
BOREHOLES = PROJECT_DIR / "samples/jupyter-demo/cond_generation_0/boreholes.pt"
PHASE1_ROOT = (
    PROJECT_DIR
    / "experiments/stage1_probability/runs/cond_generation_0/label9/all/phase1b_v4"
    / "calibrated_reference_windowed/seed42_n4_s32"
)
PHASE2_ROOT = (
    PROJECT_DIR
    / "experiments/stage2_property/runs/cond_generation_0"
    / "ideal_density_susceptibility_label9_contrast_v1/phase2a_v1"
    / "seed42_n4_s32_a025_c025"
)


def _check_pairing(
    phase1_base: dict[str, object],
    phase1_guided: dict[str, object],
    phase2_base: dict[str, object],
    phase2_guided: dict[str, object],
    paths: dict[str, Path],
) -> dict[str, object]:
    configs = (phase1_base, phase1_guided, phase2_base, phase2_guided)
    if any(item.get("run_status") != "completed" for item in configs):
        raise ValueError("all Phase-1/Phase-2 runs must be completed")
    if any(int(item.get("seed", -1)) != SEED for item in configs):
        raise ValueError("mismatched seed in controllability sources")
    if any(int(item.get("target_label", -999)) != TARGET_LABEL for item in configs):
        raise ValueError("mismatched target label in controllability sources")
    for baseline, guided, name in (
        (phase1_base, phase1_guided, "Phase 1"),
        (phase2_base, phase2_guided, "Phase 2"),
    ):
        if float(baseline.get("alpha", np.nan)) != 0.0 or float(guided.get("alpha", 0.0)) <= 0:
            raise ValueError(f"{name}: expected alpha-zero baseline and positive-alpha guidance")
        pairing = guided.get("pairing_validation")
        if not isinstance(pairing, dict) or pairing.get("paired") is not True:
            raise ValueError(f"{name}: saved strict-pair verdict is absent")
        for field in (
            "truth_model_sha256",
            "boreholes_sha256",
            "checkpoint_sha256",
            "initial_noise_sha256",
            "integrator",
            "n_steps",
        ):
            if baseline.get(field) != guided.get(field):
                raise ValueError(f"{name}: strict-pair field differs: {field}")
        if int(baseline.get("max_post_projection_condition_violations", -1)) != 0:
            raise ValueError(f"{name}: baseline condition violation recorded")
        if int(guided.get("max_post_projection_condition_violations", -1)) != 0:
            raise ValueError(f"{name}: guided condition violation recorded")
    p1_hash = sha256(paths["phase1_baseline_sample"])
    p2_hash = sha256(paths["phase2_baseline_sample"])
    if p1_hash != p2_hash:
        raise ValueError("Phase-1 and Phase-2 FLOW_ONLY samples are not byte-identical")
    if phase1_base.get("truth_model_sha256") != phase2_base.get("truth_model_sha256"):
        raise ValueError("Phase-1 and Phase-2 truth hashes differ")
    if phase1_base.get("boreholes_sha256") != phase2_base.get("boreholes_sha256"):
        raise ValueError("Phase-1 and Phase-2 borehole hashes differ")
    return {
        "strict_phase1_pair": True,
        "strict_phase2_pair": True,
        "cross_protocol_flow_only_byte_identical": True,
        "flow_only_sha256": p1_hash,
        "same_truth": True,
        "same_boreholes": True,
        "same_seed": True,
        "sample_selection": "fixed first sample (sample_0); no metric-based selection",
    }


def _metric_line(metrics: dict[str, float]) -> str:
    return "IoU$_9$={:.3f}  P$_9$={:.3f}  R$_9$={:.3f}".format(
        metrics["IoU9"], metrics["Precision9"], metrics["Recall9"]
    )


def generate() -> dict[str, object]:
    configure_matplotlib()
    ensure_output_dirs()
    paths = {
        "truth": TRUTH,
        "boreholes": BOREHOLES,
        "phase1_baseline_sample": PHASE1_ROOT / "baseline/sample_0.pt",
        "phase1_guided_sample": PHASE1_ROOT / "alpha025/sample_0.pt",
        "phase2_baseline_sample": PHASE2_ROOT / "baseline/sample_0.pt",
        "phase2_guided_sample": PHASE2_ROOT / "alpha025/sample_0.pt",
        "phase1_baseline_config": PHASE1_ROOT / "baseline/config.json",
        "phase1_guided_config": PHASE1_ROOT / "alpha025/config.json",
        "phase2_baseline_config": PHASE2_ROOT / "baseline/config.json",
        "phase2_guided_config": PHASE2_ROOT / "alpha025/config.json",
        "phase1_baseline_metrics": PHASE1_ROOT / "baseline/sample_metrics.csv",
        "phase1_guided_metrics": PHASE1_ROOT / "alpha025/sample_metrics.csv",
        "phase2_guided_metrics": PHASE2_ROOT / "alpha025/sample_metrics.csv",
        "phase1_target_mask": PHASE1_ROOT / "alpha025/target_mask.pt",
        "phase1_probability": PHASE1_ROOT / "alpha025/target_probability.pt",
        "phase2_properties": PHASE2_ROOT / "alpha025/target_properties.pt",
    }
    p1_base_cfg = read_json(paths["phase1_baseline_config"])
    p1_guided_cfg = read_json(paths["phase1_guided_config"])
    p2_base_cfg = read_json(paths["phase2_baseline_config"])
    p2_guided_cfg = read_json(paths["phase2_guided_config"])
    pairing_qc = _check_pairing(p1_base_cfg, p1_guided_cfg, p2_base_cfg, p2_guided_cfg, paths)

    truth = load_volume(paths["truth"], dtype=np.int16)
    boreholes = load_volume(paths["boreholes"], dtype=np.int16)
    flow_only = load_volume(paths["phase1_baseline_sample"], dtype=np.int16)
    probability_guided = load_volume(paths["phase1_guided_sample"], dtype=np.int16)
    property_guided = load_volume(paths["phase2_guided_sample"], dtype=np.int16)
    target_mask = load_volume(paths["phase1_target_mask"]).astype(bool)
    validate_same_shape(
        {
            "truth": truth,
            "boreholes": boreholes,
            "flow_only": flow_only,
            "probability_guided": probability_guided,
            "property_guided": property_guided,
            "target_mask": target_mask,
        }
    )
    if not np.array_equal(target_mask, truth == TARGET_LABEL):
        raise ValueError("Phase-1 target mask does not equal truth label 9")
    well_xy = tuple(tuple(int(v) for v in point) for point in p1_base_cfg["conditioning_report"]["full_borehole_xy"])
    observed = boreholes != -1
    for name, array in (
        ("FLOW_ONLY", flow_only),
        ("probability-guided", probability_guided),
        ("property-guided", property_guided),
    ):
        if not np.array_equal(array[observed], boreholes[observed]):
            raise ValueError(f"{name}: saved hard borehole values are not exact")

    metric_rows = {
        "FLOW_ONLY": read_csv_row(paths["phase1_baseline_metrics"], SAMPLE_ID),
        "probability_guided": read_csv_row(paths["phase1_guided_metrics"], SAMPLE_ID),
        "property_guided": read_csv_row(paths["phase2_guided_metrics"], SAMPLE_ID),
    }
    metrics = {
        "FLOW_ONLY": assert_metrics_match(truth, flow_only, metric_rows["FLOW_ONLY"]),
        "probability_guided": assert_metrics_match(truth, probability_guided, metric_rows["probability_guided"]),
        "property_guided": assert_metrics_match(truth, property_guided, metric_rows["property_guided"]),
    }

    # Every geology panel is rendered with the exact camera in style.CAMERA.
    top_images = [
        render_categorical_panel(truth),
        render_conditions_panel(truth, boreholes, well_xy),
        render_categorical_panel(flow_only),
        render_categorical_panel(probability_guided),
        render_categorical_panel(property_guided),
    ]
    bottom_images = [
        render_target_panel(target_mask, well_xy=well_xy),
        render_target_panel(flow_only == TARGET_LABEL, well_xy=well_xy),
        render_target_panel(probability_guided == TARGET_LABEL, well_xy=well_xy),
        render_target_panel(property_guided == TARGET_LABEL, well_xy=well_xy),
    ]

    fig = plt.figure(figsize=(mm_to_inches(DOUBLE_COLUMN_MM), 4.42))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=(1.03, 1.0),
        left=0.018,
        right=0.993,
        top=0.965,
        bottom=0.075,
        hspace=0.16,
    )
    top_grid = outer[0].subgridspec(1, 5, wspace=0.025)
    bottom_grid = outer[1].subgridspec(1, 4, wspace=0.03)
    top_titles = (
        "Truth",
        "Sparse conditions",
        "FLOW_ONLY",
        "Oracle probability guidance$^{\\dagger}$",
        "Oracle property guidance$^{\\dagger}$",
    )
    axes = []
    for index, (image, title) in enumerate(zip(top_images, top_titles)):
        ax = fig.add_subplot(top_grid[0, index])
        show_render(ax, image)
        panel_label(ax, f"({chr(ord('a') + index)})")
        ax.set_title(title, pad=0.0, fontsize=7.5)
        axes.append(ax)
    bottom_titles = (
        "Truth label 9",
        "FLOW_ONLY label 9",
        "Oracle probability guidance\nlabel 9",
        "Oracle property guidance\nlabel 9",
    )
    bottom_keys = (None, "FLOW_ONLY", "probability_guided", "property_guided")
    for index, (image, title, key) in enumerate(zip(bottom_images, bottom_titles, bottom_keys)):
        ax = fig.add_subplot(bottom_grid[0, index])
        show_render(ax, image)
        panel_label(ax, f"({chr(ord('f') + index)})")
        ax.set_title(title, pad=1.0)
        if key is not None:
            ax.text(0.5, -0.015, _metric_line(metrics[key]), transform=ax.transAxes, ha="center", va="top", fontsize=7.2)
        axes.append(ax)
    present_labels = [int(value) for value in sorted(np.unique(truth)) if int(value) != -1]
    legend_handles = [Patch(facecolor=LABEL_COLORS[label], edgecolor="none") for label in present_labels]
    fig.legend(
        legend_handles,
        [str(label) for label in present_labels],
        loc="center",
        bbox_to_anchor=(0.5, 0.548),
        ncol=len(present_labels),
        frameon=False,
        title="Raw lithology label",
        fontsize=6.5,
        title_fontsize=6.7,
        handlelength=0.9,
        handleheight=0.75,
        columnspacing=0.65,
    )
    fig.text(
        0.5,
        0.018,
        "$^{\\dagger}$ truth-derived oracle evidence / controllability upper bound",
        ha="center",
        va="bottom",
        fontsize=7.1,
        color="#555555",
    )

    data_path = FIGURE_DATA_DIR / f"{FIGURE_ID}.npz"
    write_deterministic_npz(
        data_path,
        truth=truth,
        boreholes=boreholes,
        flow_only=flow_only,
        probability_guided=probability_guided,
        property_guided=property_guided,
        target_mask=target_mask.astype(np.uint8),
        well_xy=np.asarray(well_xy, dtype=np.int16),
        metric_values=np.asarray(
            [[metrics[key][name] for name in ("IoU9", "Precision9", "Recall9")] for key in ("FLOW_ONLY", "probability_guided", "property_guided")],
            dtype=np.float64,
        ),
    )
    outputs = save_figure(fig, FIGURES_DIR / FIGURE_ID, title="Controllability under oracle evidence")
    source_roles = {
        "truth": "retrospective truth and oracle construction source",
        "boreholes": "inference-visible sparse conditions",
        "phase1_baseline_sample": "authoritative paired FLOW_ONLY sample",
        "phase1_guided_sample": "authoritative Phase-1 probability-guided sample",
        "phase2_baseline_sample": "authoritative Phase-2 paired FLOW_ONLY identity check",
        "phase2_guided_sample": "authoritative Phase-2 property-guided sample",
        "phase1_baseline_config": "Phase-1 baseline provenance",
        "phase1_guided_config": "Phase-1 guided provenance",
        "phase2_baseline_config": "Phase-2 baseline provenance",
        "phase2_guided_config": "Phase-2 guided provenance",
        "phase1_baseline_metrics": "FLOW_ONLY saved metrics",
        "phase1_guided_metrics": "Phase-1 saved metrics",
        "phase2_guided_metrics": "Phase-2 saved metrics",
        "phase1_target_mask": "truth-derived label-9 target mask",
        "phase1_probability": "truth-derived oracle probability evidence",
        "phase2_properties": "truth-derived oracle property evidence",
    }
    manifest = {
        "schema": "paper_figure_manifest_v1",
        "figure_id": FIGURE_ID,
        "title": "Controllability",
        "source_experiment": ["Phase1 protocol-v4 probability oracle", "Phase2a ideal property upper bound"],
        "source_artifacts": [source_record(paths[name], role) for name, role in source_roles.items()]
        + [source_record(data_path, "exact plotted figure data")],
        "case_id": CASE_ID,
        "candidate_or_sample_ids": [{"seed": SEED, "sample_id": SAMPLE_ID}],
        "sample_selection": "fixed sample_0, not selected by IoU or any plotted metric",
        "metrics_shown": metrics,
        "oracle_panels": [
            "(d) Oracle probability guidance: truth-derived oracle probability evidence",
            "(e) Oracle property guidance: truth-derived full-resolution ideal property evidence",
            "(h) Oracle probability guidance label 9",
            "(i) Oracle property guidance label 9",
        ],
        "truth_blind_panels": [],
        "evidence_caveat": "Neither Phase-1 probability nor Phase-2 property evidence is measured geophysics.",
        "generation": generation_record(SCRIPT_PATH),
        "outputs": output_records(outputs),
        "quality_control": {
            **pairing_qc,
            "no_nan_or_inf": True,
            "shape": list(truth.shape),
            "condition_violations": 0,
            "metrics_recomputed_from_saved_tensors": True,
            "camera_identical_for_all_3d_panels": True,
            "camera": CAMERA,
            "categorical_colors_fixed": True,
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
