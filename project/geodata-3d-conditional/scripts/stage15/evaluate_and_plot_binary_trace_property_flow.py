#!/usr/bin/env python3
"""Evaluate Stage15-H trace inversion and render Phase1-style Flow results."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime
from scripts.stage10.evaluate_bridge_information import average_precision
from scripts.stage15.common import refuse_nonempty, write_csv, write_json

ROOT = PROJECT_DIR / "experiments/stage15_binary_seismic_consensus"
OUTPUT = ROOT / "reports/binary_trace_property_flow_v4"
SEEDS = (42, 142, 242)


def _row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        return next(csv.DictReader(stream))


def _hard_metrics(prediction: torch.Tensor, truth: torch.Tensor, support: torch.Tensor) -> dict[str, float]:
    positive = prediction.bool() & support
    target = truth.bool() & support
    tp = int((positive & target).sum())
    pp = int(positive.sum())
    tt = int(target.sum())
    return {
        "positive_voxels": pp,
        "precision": tp / pp if pp else 0.0,
        "recall": tp / tt if tt else 0.0,
        "iou": tp / (pp + tt - tp) if pp + tt - tp else 0.0,
    }


def _render_phase1_style(output: Path, truth: torch.Tensor, baseline: torch.Tensor, guided: torch.Tensor, score: torch.Tensor, row: dict[str, object]) -> None:
    import matplotlib.pyplot as plt
    from scripts.paper_figures.style import render_categorical_panel, render_target_panel, show_render

    truth_np = runtime.normalize_single_geology(truth, "truth")[0, 0].numpy()
    baseline_np = runtime.normalize_single_geology(baseline, "baseline")[0, 0].numpy()
    guided_np = runtime.normalize_single_geology(guided, "guided")[0, 0].numpy()
    truth9 = truth_np == 9
    images = (
        render_categorical_panel(truth_np),
        render_categorical_panel(baseline_np),
        render_categorical_panel(guided_np),
        render_target_panel(truth9),
        render_target_panel(baseline_np == 9, ghost_mask=truth9),
        render_target_panel(guided_np == 9, ghost_mask=truth9),
    )
    titles = (
        "(a) Truth categorical model",
        f"(b) Paired Flow-only — seed {row['seed']}",
        "(c) Binary-property-guided Flow",
        "(d) Complete truth label9 geometry",
        f"(e) Flow-only label9 — IoU={row['baseline_target_iou']:.3f}, R={row['baseline_target_recall']:.3f}",
        f"(f) Guided label9 — IoU={row['guided_target_iou']:.3f}, R={row['guided_target_recall']:.3f}",
    )
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5), constrained_layout=True)
    for ax, image, title in zip(axes.flat, images, titles):
        show_render(ax, image)
        ax.set_title(title, loc="left", fontsize=13)
    fig.suptitle("Stage15-H full-trace binary seismic inversion → Phase2-style Flow guidance", fontsize=17)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _render_geophysics(output: Path, observed: torch.Tensor, score: torch.Tensor, boundary: torch.Tensor, truth9: torch.Tensor) -> None:
    import matplotlib.pyplot as plt

    seismic = observed[0, 0].numpy()
    q = score[0, 0].numpy()
    edge = boundary[0, 0].numpy()
    target = truth9[0, 0].numpy()
    footprint = q.max(axis=2)
    truth_footprint = target.any(axis=2)
    x_index = int(np.argmax(target.sum(axis=(1, 2))))
    y_index = int(np.argmax(target.sum(axis=(0, 2))))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    im = axes[0, 0].imshow(seismic[:, y_index, :].T, aspect="auto", cmap="seismic", origin="upper")
    axes[0, 0].set(title=f"Observed binary seismic — y={y_index}", xlabel="X trace", ylabel="time sample")
    fig.colorbar(im, ax=axes[0, 0], shrink=0.8)
    rms = np.sqrt(np.mean(seismic**2, axis=2))
    im = axes[0, 1].imshow(rms.T, origin="lower", cmap="magma")
    axes[0, 1].set(title="Trace RMS amplitude", xlabel="X", ylabel="Y")
    fig.colorbar(im, ax=axes[0, 1], shrink=0.8)
    im = axes[0, 2].imshow(footprint.T, origin="lower", cmap="viridis", vmin=0, vmax=1)
    axes[0, 2].contour(truth_footprint.T, levels=[0.5], colors="white", linewidths=0.8)
    axes[0, 2].set(title="Max inverted binary score\nwhite: truth XY footprint", xlabel="X", ylabel="Y")
    fig.colorbar(im, ax=axes[0, 2], shrink=0.8)
    im = axes[1, 0].imshow(q[x_index].T, origin="lower", cmap="viridis", vmin=0, vmax=1)
    axes[1, 0].contour(target[x_index].T, levels=[0.5], colors="white", linewidths=0.8)
    axes[1, 0].set(title=f"Inverted binary score — x={x_index}", xlabel="Y", ylabel="Z")
    fig.colorbar(im, ax=axes[1, 0], shrink=0.8)
    im = axes[1, 1].imshow(q[:, y_index, :].T, origin="lower", cmap="viridis", vmin=0, vmax=1)
    axes[1, 1].contour(target[:, y_index, :].T, levels=[0.5], colors="white", linewidths=0.8)
    axes[1, 1].set(title=f"Inverted binary score — y={y_index}", xlabel="X", ylabel="Z")
    fig.colorbar(im, ax=axes[1, 1], shrink=0.8)
    im = axes[1, 2].imshow(edge.max(axis=2).T, origin="lower", cmap="inferno", vmin=0, vmax=1)
    axes[1, 2].contour(truth_footprint.T, levels=[0.5], colors="cyan", linewidths=0.8)
    axes[1, 2].set(title="Max vertical boundary strength\ncyan: truth XY footprint", xlabel="X", ylabel="Y")
    fig.colorbar(im, ax=axes[1, 2], shrink=0.8)
    fig.suptitle("Binary seismic and full-trace boundary inversion diagnostics", fontsize=17)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _render_all_seed_summary(
    output: Path,
    truth: torch.Tensor,
    score: torch.Tensor,
    rows: list[dict[str, object]],
) -> None:
    """Show all fixed pairs with the same camera and truth ghost surface."""
    import matplotlib.pyplot as plt
    from scripts.paper_figures.style import render_target_panel, show_render

    truth_np = runtime.normalize_single_geology(truth, "truth")[0, 0].numpy()
    truth9 = truth_np == 9
    fig, axes = plt.subplots(3, 4, figsize=(19.5, 14.5), constrained_layout=True)
    for row_index, row in enumerate(rows):
        seed = int(row["seed"])
        run_root = ROOT / f"trace_boundary/flow_property_seed{seed}_v4"
        baseline = runtime.normalize_single_geology(
            runtime.load_tensor(run_root / "baseline/sample_0.pt", map_location="cpu"),
            f"seed{seed} baseline",
        )[0, 0].numpy()
        guided = runtime.normalize_single_geology(
            runtime.load_tensor(run_root / "guided/sample_0.pt", map_location="cpu"),
            f"seed{seed} guided",
        )[0, 0].numpy()
        panels = (
            render_target_panel(truth9),
            render_target_panel(score[0, 0].numpy() >= 0.5, ghost_mask=truth9),
            render_target_panel(baseline == 9, ghost_mask=truth9),
            render_target_panel(guided == 9, ghost_mask=truth9),
        )
        titles = (
            "Complete truth label9",
            f"seed {seed}: inverted core\nP={row['inversion_core_precision']:.3f}, R={row['inversion_core_recall']:.3f}",
            f"Flow-only\nIoU={row['baseline_target_iou']:.3f}, P={row['baseline_target_precision']:.3f}, R={row['baseline_target_recall']:.3f}",
            f"Property-guided Flow\nIoU={row['guided_target_iou']:.3f}, P={row['guided_target_precision']:.3f}, R={row['guided_target_recall']:.3f}",
        )
        for ax, panel, title in zip(axes[row_index], panels, titles):
            show_render(ax, panel)
            ax.set_title(title, fontsize=12)
    fig.suptitle(
        "Stage15-H all three fixed seeds — identical inversion evidence and paired Flow noise",
        fontsize=17,
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    refuse_nonempty(OUTPUT)
    load = lambda path: runtime.load_tensor(path, map_location="cpu")
    truth = load(PROJECT_DIR / "samples/jupyter-demo/cond_generation_0/true_model.pt").long()
    support = load(ROOT / "observations/cond_generation_0/subsurface_mask.pt").bool()
    observed = load(ROOT / "observations/cond_generation_0/observed_seismic.pt").float()
    score = load(ROOT / "trace_boundary/cond_generation_0_v1/binary_impedance_score.pt").float()
    boundary = load(ROOT / "trace_boundary/cond_generation_0_v1/vertical_boundary_strength.pt").float()
    truth9 = (truth == 9) & support
    true_edge = torch.zeros_like(truth9)
    differences = torch.diff(truth9.float(), dim=-1).abs().bool()
    true_edge[..., :-1] |= differences
    true_edge[..., 1:] |= differences
    core = _hard_metrics(score >= 0.5, truth9, support)
    inversion = {
        "voxel_auprc": average_precision(score[support], truth9[support].float()),
        "truth_mean_score": float(score[truth9].mean()),
        "background_mean_score": float(score[support & ~truth9].mean()),
        "boundary_auprc": average_precision(boundary[support], true_edge[support].float()),
        "xy_footprint_auprc": average_precision(score.max(dim=-1).values[support.any(dim=-1)], truth9.any(dim=-1)[support.any(dim=-1)].float()),
        "fixed_0p5_core": core,
    }
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        root = ROOT / f"trace_boundary/flow_property_seed{seed}_v4"
        baseline = _row(root / "baseline/sample_metrics.csv")
        guided = _row(root / "guided/sample_metrics.csv")
        row: dict[str, object] = {"seed": seed}
        for metric in ("target_iou", "target_precision", "target_recall", "target_centroid_distance", "global_mean_iou", "global_voxel_accuracy", "target_connected_components", "largest_component_fraction"):
            b, g = float(baseline[metric]), float(guided[metric])
            row[f"baseline_{metric}"] = b
            row[f"guided_{metric}"] = g
            row[f"delta_{metric}"] = g - b
        row["condition_violations"] = int(guided["condition_violation_count"])
        row["inversion_core_precision"] = core["precision"]
        row["inversion_core_recall"] = core["recall"]
        rows.append(row)
    selected = max(rows, key=lambda row: float(row["guided_target_iou"]))
    selected_root = ROOT / f"trace_boundary/flow_property_seed{selected['seed']}_v4"
    baseline_model = load(selected_root / "baseline/sample_0.pt").long()
    guided_model = load(selected_root / "guided/sample_0.pt").long()
    OUTPUT.mkdir(parents=True)
    write_csv(OUTPUT / "paired_metrics.csv", rows)
    write_json(OUTPUT / "summary.json", {
        "schema": "stage15_binary_trace_property_flow_evaluation_v1",
        "run_status": "completed",
        "inversion": inversion,
        "flow_pair_count": len(rows),
        "all_iou_improved": all(float(row["delta_target_iou"]) > 0 for row in rows),
        "all_precision_improved": all(float(row["delta_target_precision"]) > 0 for row in rows),
        "all_recall_improved": all(float(row["delta_target_recall"]) > 0 for row in rows),
        "all_global_miou_improved": all(float(row["delta_global_mean_iou"]) > 0 for row in rows),
        "selected_for_figure": selected,
        "selection_policy": "maximum guided label9 IoU among all three reported fixed seeds",
        "truth_used_only_by_evaluator_and_existing_phase2_metric_code": True,
    })
    _render_phase1_style(OUTPUT / "truth_baseline_guided_3d.png", truth, baseline_model, guided_model, score, selected)
    for row in rows:
        seed = int(row["seed"])
        run_root = ROOT / f"trace_boundary/flow_property_seed{seed}_v4"
        seed_baseline = load(run_root / "baseline/sample_0.pt").long()
        seed_guided = load(run_root / "guided/sample_0.pt").long()
        _render_phase1_style(
            OUTPUT / f"truth_baseline_guided_seed{seed}_3d.png",
            truth,
            seed_baseline,
            seed_guided,
            score,
            row,
        )
    _render_all_seed_summary(OUTPUT / "all_three_seed_target_comparison_3d.png", truth, score, rows)
    _render_geophysics(OUTPUT / "binary_seismic_boundary_diagnostics.png", observed, score, boundary, truth9)
    (OUTPUT / "REPORT.md").write_text(
        "# Stage15-H — Full-trace binary boundary inversion and property-guided Flow\n\n"
        f"- Inversion voxel AUPRC: {inversion['voxel_auprc']:.9f}\n"
        f"- Boundary AUPRC: {inversion['boundary_auprc']:.9f}\n"
        f"- XY footprint AUPRC: {inversion['xy_footprint_auprc']:.9f}\n"
        f"- Fixed 0.5 core precision / recall / IoU: {core['precision']:.6f} / {core['recall']:.6f} / {core['iou']:.6f}\n"
        f"- Flow pairs improving IoU/precision/recall/global mIoU: {sum(float(r['delta_target_iou']) > 0 for r in rows)}/3 / {sum(float(r['delta_target_precision']) > 0 for r in rows)}/3 / {sum(float(r['delta_target_recall']) > 0 for r in rows)}/3 / {sum(float(r['delta_global_mean_iou']) > 0 for r in rows)}/3\n"
        f"- Figure seed {selected['seed']} IoU: {selected['baseline_target_iou']:.6f} -> {selected['guided_target_iou']:.6f}\n"
        f"- Figure seed precision: {selected['baseline_target_precision']:.6f} -> {selected['guided_target_precision']:.6f}\n"
        f"- Figure seed recall: {selected['baseline_target_recall']:.6f} -> {selected['guided_target_recall']:.6f}\n"
        f"- Figure seed centroid distance: {selected['baseline_target_centroid_distance']:.6f} -> {selected['guided_target_centroid_distance']:.6f}\n\n"
        "The binary trace inversion uses complete 320-sample traces and never splits the vertical forward model. The normalized binary property endpoint is label9=1 and every other class=0; the continuous inversion score supplies confidence without a threshold.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
