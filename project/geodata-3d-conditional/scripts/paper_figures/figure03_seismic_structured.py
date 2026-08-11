#!/usr/bin/env python3
"""Generate Figure 3 from the final Stage-7 truth-blind hard-seismic selector."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
STRUCTURALGEO_SRC = REPOSITORY_ROOT / "StructuralGeo-main/src"
for path in (PROJECT_DIR, STRUCTURALGEO_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from guidance.native_geology_audit import build_structuralgeo_native_case
from guidance.seismic import (
    acoustic_tables_from_config,
    hard_labels_to_acoustic,
    seismic_operator_from_config,
    tensor_sha256,
)
from scripts.paper_figures.style import (
    CAMERA,
    DOUBLE_COLUMN_MM,
    FIGURE_DATA_DIR,
    FIGURES_DIR,
    MANIFESTS_DIR,
    RESIDUAL_CMAP,
    SEISMIC_CMAP,
    configure_matplotlib,
    ensure_output_dirs,
    generation_record,
    load_volume,
    mm_to_inches,
    output_records,
    panel_label,
    read_json,
    render_acquisition_panel,
    render_target_panel,
    repository_path,
    robust_symmetric_limit,
    save_figure,
    sha256,
    show_render,
    source_record,
    validate_array,
    validate_same_shape,
    write_deterministic_npz,
    write_json,
)


SCRIPT_PATH = Path(__file__).resolve()
FIGURE_ID = "figure03_seismic_structured"
CASE_ID = "native_seed20260809"
CASE_SEED = 20260809
SLICE_Y = 42
ROBUST_PERCENTILE = 99.5
TARGET_LABEL = 9

STAGE7_DIR = PROJECT_DIR / "experiments/stage6_inference_causality/reports/stage7_v1_final_v2"
SUMMARY_PATH = STAGE7_DIR / "stage7_summary.json"
CONFIG_INPUT_PATH = STAGE7_DIR / "config_input.json"
SELECTED_LABELS_PATH = STAGE7_DIR / f"states/{CASE_ID}/correct/best_labels.pt"
BASELINE_LABELS_PATH = STAGE7_DIR / f"states/{CASE_ID}/zero/best_labels.pt"
SELECTED_HISTORY_PATH = STAGE7_DIR / f"states/{CASE_ID}/correct/selected_event_history.json"
NATIVE_BUILDER_PATH = PROJECT_DIR / "guidance/native_geology_audit.py"
SEISMIC_SOURCE_PATH = PROJECT_DIR / "guidance/seismic.py"


def _case_summary(summary: dict[str, object]) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    rows = [row for row in summary["native_replicas"] if row["case_id"] == CASE_ID]
    if len(rows) != 1:
        raise ValueError(f"expected one authoritative Stage-7 case {CASE_ID}")
    case = rows[0]
    correct = [row for row in case["arms"] if row["optimized_by"] == "correct"]
    baseline = [row for row in case["arms"] if row["optimized_by"] == "zero"]
    if len(correct) != 1 or len(baseline) != 1:
        raise ValueError("missing Stage-7 correct or zero arm")
    return case, correct[0], baseline[0]


def _hard_response(labels: torch.Tensor, table: torch.Tensor, subsurface: torch.Tensor, operator) -> torch.Tensor:
    acoustic = hard_labels_to_acoustic(labels, table)
    return operator(acoustic[:, 0:1], acoustic[:, 1:2], subsurface)


def _hidden_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth = np.asarray(truth, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    tp = int((truth & prediction).sum())
    fp = int((~truth & prediction).sum())
    fn = int((truth & ~prediction).sum())
    return {
        "hidden_iou": tp / max(tp + fp + fn, 1),
        "hidden_precision": tp / max(tp + fp, 1),
        "hidden_recall": tp / max(tp + fn, 1),
    }


def _plot_seismic(ax, section, *, limit: float, cmap, title: str, panel: str, dt_ms: float):
    nt = section.shape[-1]
    image = ax.imshow(
        section.T,
        origin="upper",
        extent=(-0.5, section.shape[0] - 0.5, nt * dt_ms, 0.0),
        cmap=cmap,
        vmin=-limit,
        vmax=limit,
        aspect="auto",
        interpolation="nearest",
        rasterized=True,
    )
    ax.set_title(title, pad=2.0)
    ax.text(
        0.98,
        0.02,
        "Trace $x$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.0,
        color="#555555",
    )
    ax.set_xticks((0, 16, 32, 48, 63))
    ax.set_yticks((0, 800, 1600, 2400))
    panel_label(ax, panel)
    return image


def generate() -> dict[str, object]:
    configure_matplotlib()
    ensure_output_dirs()
    summary = read_json(SUMMARY_PATH)
    if summary.get("status") != "completed" or summary.get("truth_used_for_selection") is not False:
        raise ValueError("Stage-7 final report is not a completed truth-blind selector")
    if summary.get("git_sha") != "85d5deb4555430117887a8ba173a0222c6b899ae":
        raise ValueError("unexpected authoritative Stage-7 git identity")
    if summary["source_hashes"]["native_case"] != sha256(NATIVE_BUILDER_PATH):
        raise ValueError("current native-case builder differs from the authoritative Stage-7 hash")
    if summary["source_hashes"]["seismic"] != sha256(SEISMIC_SOURCE_PATH):
        raise ValueError("current seismic source differs from the authoritative Stage-7 hash")
    case_summary, selected_summary, baseline_summary = _case_summary(summary)
    if selected_summary.get("selection_criterion") != "minimum hard observed seismic RMSE only":
        raise ValueError("Stage-7 selected candidate used an unexpected criterion")
    if selected_summary.get("truth_used_for_selection") is not False:
        raise ValueError("Stage-7 selected candidate records truth leakage")
    # The case is chosen using the inference-visible hard-attainment column.
    correct_arms = [
        next(arm for arm in case["arms"] if arm["optimized_by"] == "correct")
        for case in summary["native_replicas"]
    ]
    best_by_attainment = max(correct_arms, key=lambda row: float(row["hard_correct_observation_attainment"]))
    if best_by_attainment["case_id"] != CASE_ID:
        raise ValueError("registered strongest hard-attainment Stage-7 case changed")

    config_input = read_json(CONFIG_INPUT_PATH)
    base_config_path = repository_path(config_input["base_config"])
    base_config = read_json(base_config_path)
    acoustic_config_path = repository_path(base_config["acoustic_config"])
    seismic_config_path = repository_path(base_config["seismic_config"])
    acoustic_config = read_json(acoustic_config_path)
    seismic_config = read_json(seismic_config_path)
    if not torch.cuda.is_available():
        raise RuntimeError("exact Stage-7 observation-hash replay requires the original CUDA float32 semantics")
    device = torch.device("cuda")
    tables, _ = acoustic_tables_from_config(acoustic_config, 15)
    table = tables.property_table.to(device)
    operator, seismic_metadata = seismic_operator_from_config(seismic_config, grid_shape=(64, 64, 64))

    native_case, native_metadata = build_structuralgeo_native_case(seed=CASE_SEED)
    if native_metadata["body_voxel_counts"] != case_summary["native_truth_metadata"]["body_voxel_counts"]:
        raise ValueError("deterministically rebuilt truth differs from Stage-7 metadata")
    truth_labels = native_case.truth_labels.cpu()
    hidden_truth_tensor = native_case.body_masks[3:].any(dim=0)
    hidden_truth = hidden_truth_tensor.numpy()
    baseline_expected = truth_labels.clone()
    baseline_expected[0, 0, hidden_truth_tensor] = native_case.background_label
    baseline = load_volume(BASELINE_LABELS_PATH, dtype=np.int16)
    selected = load_volume(SELECTED_LABELS_PATH, dtype=np.int16)
    truth = truth_labels[0, 0].numpy().astype(np.int16)
    validate_same_shape({"truth": truth, "baseline": baseline, "selected": selected, "hidden_truth": hidden_truth})
    if not np.array_equal(baseline, baseline_expected[0, 0].numpy()):
        raise ValueError("saved Stage-7 zero-arm labels do not equal the registered hidden-body baseline")
    if np.any(selected[native_case.condition_mask[0, 0].numpy()] != truth[native_case.condition_mask[0, 0].numpy()]):
        raise ValueError("selected Stage-7 labels violate hard conditions")

    with torch.no_grad():
        subsurface = native_case.subsurface_mask.to(device)
        observed_t = _hard_response(truth_labels.to(device), table, subsurface, operator)
        baseline_t = _hard_response(
            torch.from_numpy(baseline).view(1, 1, 64, 64, 64).long().to(device),
            table,
            subsurface,
            operator,
        )
        selected_t = _hard_response(
            torch.from_numpy(selected).view(1, 1, 64, 64, 64).long().to(device),
            table,
            subsurface,
            operator,
        )
    expected_observation_hash = case_summary["observation_hashes"]["correct"]
    if tensor_sha256(observed_t) != expected_observation_hash:
        raise ValueError("recomputed correct observation hash differs from Stage-7")
    observed = observed_t[0, 0].cpu().numpy()
    baseline_seismic = baseline_t[0, 0].cpu().numpy()
    selected_seismic = selected_t[0, 0].cpu().numpy()
    for name, array in (
        ("observed", observed),
        ("baseline predicted", baseline_seismic),
        ("selected predicted", selected_seismic),
    ):
        validate_array(name, array, ndim=3)
    validate_same_shape({"observed": observed, "baseline": baseline_seismic, "selected": selected_seismic})
    baseline_residual = baseline_seismic - observed
    selected_residual = selected_seismic - observed
    baseline_rmse = float(np.sqrt(np.mean(baseline_residual.astype(np.float64) ** 2)))
    selected_rmse = float(np.sqrt(np.mean(selected_residual.astype(np.float64) ** 2)))
    if abs(baseline_rmse - float(baseline_summary["hard_correct_observation_rmse"])) > 2e-8:
        raise ValueError("recomputed baseline seismic RMSE differs from Stage-7 summary")
    if abs(selected_rmse - float(selected_summary["hard_correct_observation_rmse"])) > 2e-8:
        raise ValueError("recomputed selected seismic RMSE differs from Stage-7 summary")

    domain = np.zeros_like(hidden_truth, dtype=bool)
    domain[4:60, 28:56, 8:52] = True
    baseline_hidden = (baseline == TARGET_LABEL) & domain
    selected_hidden = (selected == TARGET_LABEL) & domain
    baseline_hidden_metrics = _hidden_metrics(hidden_truth, baseline_hidden)
    hidden_metrics = _hidden_metrics(hidden_truth, selected_hidden)
    for computed_name, saved_name in (
        ("hidden_iou", "hidden_target_iou"),
        ("hidden_precision", "hidden_target_precision"),
        ("hidden_recall", "hidden_target_recall"),
    ):
        if abs(hidden_metrics[computed_name] - float(selected_summary[saved_name])) > 5e-8:
            raise ValueError(f"wrong selected candidate: {computed_name} mismatch")

    amplitude_limit = robust_symmetric_limit((observed, baseline_seismic, selected_seismic), ROBUST_PERCENTILE)
    residual_limit = robust_symmetric_limit(
        (baseline_residual, selected_residual),
        ROBUST_PERCENTILE,
        ignore_zeros=True,
    )
    dt_ms = float(seismic_metadata["time_sampling"]["sample_interval_ms"])
    sections = {
        "observed": observed[:, SLICE_Y, :],
        "baseline": baseline_seismic[:, SLICE_Y, :],
        "selected": selected_seismic[:, SLICE_Y, :],
        "baseline_residual": baseline_residual[:, SLICE_Y, :],
        "selected_residual": selected_residual[:, SLICE_Y, :],
    }
    acquisition_image = render_acquisition_panel(hidden_truth, well_xy=native_case.well_xy)
    truth_hidden_image = render_target_panel(hidden_truth)
    baseline_hidden_image = render_target_panel(baseline_hidden)
    selected_hidden_image = render_target_panel(selected_hidden)

    fig = plt.figure(figsize=(mm_to_inches(DOUBLE_COLUMN_MM), 5.02))
    grid = fig.add_gridspec(
        4,
        5,
        height_ratios=(1.0, 0.055, 1.0, 0.055),
        left=0.064,
        right=0.99,
        top=0.945,
        bottom=0.075,
        wspace=0.30,
        hspace=0.30,
    )
    axes = np.empty((2, 5), dtype=object)
    for column in range(5):
        axes[0, column] = fig.add_subplot(grid[0, column])
        axes[1, column] = fig.add_subplot(grid[2, column])
    show_render(axes[0, 0], acquisition_image)
    axes[0, 0].set_title("Acquisition + hidden truth", pad=2.0)
    panel_label(axes[0, 0], "(a)")
    im_amp = _plot_seismic(
        axes[0, 1], sections["observed"], limit=amplitude_limit, cmap=SEISMIC_CMAP,
        title=f"Observed seismic ($y={SLICE_Y}$)", panel="(b)", dt_ms=dt_ms,
    )
    _plot_seismic(
        axes[0, 2], sections["baseline"], limit=amplitude_limit, cmap=SEISMIC_CMAP,
        title=f"Baseline predicted\nRMSE={baseline_rmse:.5f}", panel="(c)", dt_ms=dt_ms,
    )
    _plot_seismic(
        axes[0, 3], sections["selected"], limit=amplitude_limit, cmap=SEISMIC_CMAP,
        title=f"Hard-seismic selected\nRMSE={selected_rmse:.5f}", panel="(d)", dt_ms=dt_ms,
    )
    axes[0, 4].axis("off")
    panel_label(axes[0, 4], "(e)")
    axes[0, 4].set_title("Display / selector scope", pad=2.0)
    axes[0, 4].text(
        0.04,
        0.93,
        "Displayed seismic\n$y=42$ slice\n\n"
        "Selector\nfull $64\\times64\\times320$\nhard seismic volume\n\n"
        "Residuals\nseparate shared\n99.5th-percentile\nrobust scale",
        transform=axes[0, 4].transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        color="#444444",
        linespacing=1.08,
    )
    im_res = _plot_seismic(
        axes[1, 0], sections["baseline_residual"], limit=residual_limit, cmap=RESIDUAL_CMAP,
        title="Baseline residual", panel="(f)", dt_ms=dt_ms,
    )
    _plot_seismic(
        axes[1, 1], sections["selected_residual"], limit=residual_limit, cmap=RESIDUAL_CMAP,
        title="Selected residual", panel="(g)", dt_ms=dt_ms,
    )
    show_render(axes[1, 2], truth_hidden_image)
    axes[1, 2].set_title("True hidden\nlabel 9", pad=1.0, fontsize=7.4)
    panel_label(axes[1, 2], "(h)")
    show_render(axes[1, 3], baseline_hidden_image)
    axes[1, 3].set_title("Baseline hidden\nlabel 9", pad=1.0, fontsize=7.4)
    panel_label(axes[1, 3], "(i)")
    show_render(axes[1, 4], selected_hidden_image)
    axes[1, 4].set_title(
        "Seismic-selected\nhidden label 9\nIoU={:.3f}  R={:.3f}".format(hidden_metrics["hidden_iou"], hidden_metrics["hidden_recall"]),
        pad=1.0,
        fontsize=7.1,
    )
    panel_label(axes[1, 4], "(j)")
    for ax in (axes[0, 2], axes[0, 3], axes[1, 1]):
        ax.set_ylabel("")
        ax.set_yticklabels([])
    axes[0, 1].set_ylabel("TWT [ms]")
    axes[1, 0].set_ylabel("TWT [ms]")
    amp_cax = fig.add_subplot(grid[1, 1:4])
    amp_cb = fig.colorbar(im_amp, cax=amp_cax, orientation="horizontal")
    amp_cb.set_label("Seismic amplitude [a.u.]", labelpad=1.0)
    amp_cb.ax.tick_params(length=2, pad=1)
    res_cax = fig.add_subplot(grid[3, 0:2])
    res_cb = fig.colorbar(im_res, cax=res_cax, orientation="horizontal")
    res_cb.set_label("Prediction $-$ observation [a.u.]", labelpad=1.0)
    res_cb.ax.tick_params(length=2, pad=1)
    note_ax = fig.add_subplot(grid[3, 2:5])
    note_ax.axis("off")
    note_ax.text(
        0.5,
        0.5,
        "selector: full-volume hard observed seismic RMSE only; hidden masks are retrospective",
        ha="center",
        va="center",
        fontsize=6.9,
        color="#555555",
    )

    data_path = FIGURE_DATA_DIR / f"{FIGURE_ID}.npz"
    write_deterministic_npz(
        data_path,
        observed_section=sections["observed"],
        baseline_predicted_section=sections["baseline"],
        selected_predicted_section=sections["selected"],
        baseline_residual_section=sections["baseline_residual"],
        selected_residual_section=sections["selected_residual"],
        hidden_truth=hidden_truth.astype(np.uint8),
        baseline_hidden=baseline_hidden.astype(np.uint8),
        selected_hidden=selected_hidden.astype(np.uint8),
        well_xy=np.asarray(native_case.well_xy, dtype=np.int16),
        amplitude_limit=np.asarray(amplitude_limit, dtype=np.float64),
        residual_limit=np.asarray(residual_limit, dtype=np.float64),
        robust_percentile=np.asarray(ROBUST_PERCENTILE, dtype=np.float64),
        full_volume_rmse=np.asarray([baseline_rmse, selected_rmse], dtype=np.float64),
        hidden_metrics=np.asarray(
            [hidden_metrics["hidden_iou"], hidden_metrics["hidden_precision"], hidden_metrics["hidden_recall"]],
            dtype=np.float64,
        ),
        baseline_hidden_metrics=np.asarray(
            [
                baseline_hidden_metrics["hidden_iou"],
                baseline_hidden_metrics["hidden_precision"],
                baseline_hidden_metrics["hidden_recall"],
            ],
            dtype=np.float64,
        ),
    )
    outputs = save_figure(fig, FIGURES_DIR / FIGURE_ID, title="Stage-7 structured hard-seismic recovery")
    source_paths = (
        (SUMMARY_PATH, "authoritative Stage-7 final summary"),
        (CONFIG_INPUT_PATH, "authoritative Stage-7 frozen configuration"),
        (SELECTED_LABELS_PATH, "truth-blind hard-seismic-selected categorical model"),
        (BASELINE_LABELS_PATH, "Stage-7 hidden-body baseline categorical model"),
        (SELECTED_HISTORY_PATH, "selected structured event provenance"),
        (base_config_path, "Stage-7 base causal configuration"),
        (acoustic_config_path, "frozen acoustic mapping configuration"),
        (seismic_config_path, "frozen convolutional seismic configuration"),
        (NATIVE_BUILDER_PATH, "frozen deterministic StructuralGeo case builder"),
        (SEISMIC_SOURCE_PATH, "frozen hard seismic forward operator"),
        (data_path, "exact plotted figure data"),
    )
    manifest = {
        "schema": "paper_figure_manifest_v1",
        "figure_id": FIGURE_ID,
        "title": "Geophysical hero figure",
        "source_experiment": "Stage7 final structured hard-geophysics inference",
        "source_artifacts": [source_record(path, role) for path, role in source_paths],
        "case_id": CASE_ID,
        "candidate_or_sample_ids": [
            {
                "selector_arm": "correct",
                "selected_objects": selected_summary["selected_objects"],
                "selection_criterion": selected_summary["selection_criterion"],
            }
        ],
        "case_selection": (
            "Highest inference-visible hard-seismic attainment among the three pre-registered "
            "StructuralGeo replicas; hidden IoU was not used."
        ),
        "candidate_selection": "minimum hard observed seismic RMSE only; truth_used_for_selection=false",
        "metrics_shown": {
            "baseline_hard_seismic_rmse": baseline_rmse,
            "selected_hard_seismic_rmse": selected_rmse,
            "baseline_hidden_metrics": baseline_hidden_metrics,
            **hidden_metrics,
        },
        "oracle_panels": ["(a) hidden truth shown for synthetic benchmark context", "(h) retrospective true hidden body"],
        "truth_blind_panels": [
            "(d) predicted seismic of Stage-7 hard-seismic-selected candidate",
            "(g) selected residual",
            "(i) baseline hidden categorical body",
            "(j) selected hidden categorical body",
        ],
        "geophysical_observation": {
            "synthetic": True,
            "measured": False,
            "truth_derived_once_for_benchmark": True,
            "observation_sha256": expected_observation_hash,
        },
        "seismic_display": {
            "slice_y": SLICE_Y,
            "displayed_seismic": "y=42 slice",
            "selector_input": "full 64x64x320 hard seismic volume",
            "interpolation": "nearest (none)",
            "amplitude_shared_across": ["observed", "baseline predicted", "selected predicted"],
            "amplitude_robust_percentile": ROBUST_PERCENTILE,
            "amplitude_limit": [-amplitude_limit, amplitude_limit],
            "residual_shared_across": ["baseline residual", "selected residual"],
            "residual_robust_percentile": ROBUST_PERCENTILE,
            "residual_percentile_population": "nonzero samples pooled across baseline and selected residual volumes",
            "residual_scale_note": "separate shared 99.5th-percentile robust scale",
            "residual_limit": [-residual_limit, residual_limit],
            "sample_interval_ms": dt_ms,
            "units": "unscaled convolutional amplitude [a.u.]",
        },
        "generation": generation_record(SCRIPT_PATH),
        "outputs": output_records(outputs),
        "quality_control": {
            "missing_files": False,
            "no_nan_or_inf": True,
            "forward_replay_device": "cuda (matches authoritative Stage-7 float32 hash semantics)",
            "seismic_shape": list(observed.shape),
            "condition_violations": 0,
            "observation_hash_matches_stage7": True,
            "selected_candidate_metrics_recomputed": True,
            "selected_candidate_matches_frozen_summary": True,
            "shared_amplitude_limit": True,
            "shared_residual_limit": True,
            "display_slice_and_full_volume_selector_explicit": True,
            "baseline_hidden_label9_displayed": True,
            "camera_identical_for_3d_panels": True,
            "camera": CAMERA,
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
