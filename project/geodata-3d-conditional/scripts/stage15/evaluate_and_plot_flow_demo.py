#!/usr/bin/env python3
"""Retrospectively select and render the Stage15 Flow demonstration pair."""

from __future__ import annotations

import argparse
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
from guidance.probability_evaluation import sample_hard_metrics
from scripts.stage15.common import base_manifest, normalize_volume, read_json, refuse_nonempty, write_csv, write_json

ROOT = PROJECT_DIR / "experiments/stage15_binary_seismic_consensus"
DEFAULT_OBSERVATION = ROOT / "observations/cond_generation_0"
DEFAULT_RUN = ROOT / "flow_demo/coarse_occupancy_seed_screen_n8_v2"
DEFAULT_OUTPUT = ROOT / "reports/flow_demo_coarse_occupancy_seed_screen_n8_v3"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--observation-dir", type=Path, default=DEFAULT_OBSERVATION)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def _load(path: Path, name: str) -> torch.Tensor:
    return normalize_volume(runtime.load_tensor(path), name)


def _scatter(ax, mask: np.ndarray, color: str, label: str, alpha: float = 0.55, size: float = 2.0) -> None:
    xyz = np.argwhere(mask)
    if len(xyz):
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=size, c=color, alpha=alpha, label=label, rasterized=True)
    ax.set(xlim=(0, 63), ylim=(0, 63), zlim=(0, 63), xlabel="X", ylabel="Y", zlabel="Z")
    ax.view_init(elev=23, azim=-58)
    ax.invert_zaxis()


def render_figure(
    output: Path,
    evidence: torch.Tensor,
    truth: torch.Tensor,
    baseline: torch.Tensor,
    guided: torch.Tensor,
    row: dict[str, object],
) -> None:
    import matplotlib.pyplot as plt

    truth9 = (truth[0, 0] == 9).numpy()
    base9 = (baseline[0, 0] == 9).numpy()
    guided9 = (guided[0, 0] == 9).numpy()
    evidence_np = evidence[0, 0].numpy()
    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    axes = [fig.add_subplot(2, 3, i + 1, projection="3d") for i in range(6)]
    core = evidence_np >= 0.8 * float(evidence_np.max())
    _scatter(axes[0], core, "#7b3294", "top relative evidence", 0.55, 4)
    axes[0].set_title("Coarse geophysical location evidence\n(display only; loss uses 8³ occupancy)")
    _scatter(axes[1], truth9, "#f2c14e", "truth label9")
    axes[1].set_title("Truth label9")
    _scatter(axes[2], base9, "#2878b5", "Flow-only")
    axes[2].set_title(f"Flow-only\nIoU={row['baseline_target_iou']:.3f}, R={row['baseline_target_recall']:.3f}")
    _scatter(axes[3], guided9, "#2a9d55", "guided")
    axes[3].set_title(f"Evidence-guided Flow\nIoU={row['guided_target_iou']:.3f}, R={row['guided_target_recall']:.3f}")
    _scatter(axes[4], truth9, "#f2c14e", "truth", 0.22, 2)
    _scatter(axes[4], base9, "#2878b5", "baseline", 0.45, 2)
    axes[4].set_title("Truth + Flow-only")
    _scatter(axes[5], truth9, "#f2c14e", "truth", 0.22, 2)
    _scatter(axes[5], guided9, "#2a9d55", "guided", 0.45, 2)
    axes[5].set_title("Truth + evidence-guided")
    for ax in axes:
        ax.legend(loc="upper left", fontsize=7)
    fig.suptitle(
        "Stage15 exploratory binary-seismic → Flow demonstration\n"
        f"post-hoc selected seed {row['source_seed']}; ΔIoU={row['delta_target_iou']:+.3f}, "
        f"Δrecall={row['delta_target_recall']:+.3f}",
        fontsize=14,
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_full_geology_figure(
    output: Path, truth: torch.Tensor, baseline: torch.Tensor, guided: torch.Tensor
) -> None:
    """Use the repository's fixed-camera paper renderer for full categorical models."""
    import matplotlib.pyplot as plt
    from scripts.paper_figures.style import render_categorical_panel

    volumes = [truth[0, 0].numpy(), baseline[0, 0].numpy(), guided[0, 0].numpy()]
    images = [render_categorical_panel(volume) for volume in volumes]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    for ax, image, title in zip(
        axes,
        images,
        ("Truth geology", "Flow-only geology", "Evidence-guided geology"),
    ):
        ax.imshow(image)
        ax.set_title(title, fontsize=14)
        ax.axis("off")
    fig.suptitle("Stage15 selected full 3-D categorical geological models", fontsize=16)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_phase1_style_figure(
    output: Path,
    evidence: torch.Tensor,
    truth: torch.Tensor,
    baseline: torch.Tensor,
    guided: torch.Tensor,
    row: dict[str, object],
) -> None:
    """Render the same full-model/target-layout used by the Phase1 figures."""
    import matplotlib.pyplot as plt
    from scripts.paper_figures.style import (
        render_categorical_panel,
        render_label_frequency_3d,
        render_target_panel,
        show_render,
    )

    truth_np = truth[0, 0].numpy()
    baseline_np = baseline[0, 0].numpy()
    guided_np = guided[0, 0].numpy()
    truth9 = truth_np == 9
    evidence_np = evidence[0, 0].numpy()
    evidence_relative = evidence_np / max(float(evidence_np.max()), 1e-12)
    images = (
        render_categorical_panel(truth_np),
        render_categorical_panel(baseline_np),
        render_categorical_panel(guided_np),
        render_label_frequency_3d(evidence_relative, minimum_frequency=0.8),
        render_target_panel(baseline_np == 9, ghost_mask=truth9),
        render_target_panel(guided_np == 9, ghost_mask=truth9),
    )
    titles = (
        "(a) Truth categorical model",
        "(b) Paired Flow-only model",
        "(c) Coarse-evidence-guided model",
        "(d) Top relative geophysical evidence",
        f"(e) Flow-only label9  IoU={row['baseline_target_iou']:.3f}, R={row['baseline_target_recall']:.3f}",
        f"(f) Guided label9  IoU={row['guided_target_iou']:.3f}, R={row['guided_target_recall']:.3f}",
    )
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5), constrained_layout=True)
    for ax, panel, title in zip(axes.flat, images, titles):
        show_render(ax, panel)
        ax.set_title(title, loc="left", fontsize=13)
    fig.suptitle(
        f"Stage15 binary-geophysical guidance — paired seed {row['source_seed']}\n"
        "same fixed-camera surface presentation as Phase1",
        fontsize=16,
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    refuse_nonempty(args.output_dir)
    run_manifest = read_json(args.run_dir / "run_manifest.json")
    if run_manifest.get("run_status") != "completed" or run_manifest.get("truth_loaded_by_flow_runner") is not False:
        raise ValueError("invalid truth-blind Flow demo")
    observation_manifest = read_json(args.observation_dir / "manifest.json")
    truth_path = Path(str(observation_manifest["phase1_assets"]["truth_model"]["path"]))
    truth = _load(truth_path, "truth").long()
    condition = _load(args.observation_dir / "flow_condition_mask.pt", "condition").bool()
    subsurface = _load(args.observation_dir / "subsurface_mask.pt", "subsurface").bool()
    evidence = _load(args.run_dir / "guidance_evidence.pt", "evidence").float()
    truth_target = (truth == 9) & subsurface
    rows = []
    for sample_id, seed in enumerate(run_manifest["source_seeds"]):
        baseline = _load(args.run_dir / "FLOW_ONLY" / f"sample_{sample_id}.pt", "baseline").long()
        guided = _load(args.run_dir / "GEO_EVIDENCE_GUIDED" / f"sample_{sample_id}.pt", "guided").long()
        base = sample_hard_metrics(baseline, truth, truth_target, subsurface, condition, 9, sample_id)
        guide = sample_hard_metrics(guided, truth, truth_target, subsurface, condition, 9, sample_id, baseline)
        row = {"sample_id": sample_id, "source_seed": int(seed)}
        for name in (
            "target_iou",
            "target_precision",
            "target_recall",
            "target_centroid_distance",
            "target_connected_components",
            "largest_component_fraction",
            "predicted_target_volume",
            "global_voxel_accuracy",
            "global_mean_iou",
        ):
            row[f"baseline_{name}"] = base[name]
            row[f"guided_{name}"] = guide[name]
            row[f"delta_{name}"] = float(guide[name]) - float(base[name])
        row["paired_hard_change_count"] = guide["paired_hard_change_count"]
        row["baseline_condition_violations"] = base["condition_violation_count"]
        row["guided_condition_violations"] = guide["condition_violation_count"]
        rows.append(row)
    selected = max(rows, key=lambda row: (float(row["delta_target_iou"]), float(row["delta_target_recall"])))
    selected_id = int(selected["sample_id"])
    baseline = _load(args.run_dir / "FLOW_ONLY" / f"sample_{selected_id}.pt", "baseline").long()
    guided = _load(args.run_dir / "GEO_EVIDENCE_GUIDED" / f"sample_{selected_id}.pt", "guided").long()
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "all_pair_metrics.csv", rows)
    torch.save(baseline, args.output_dir / "selected_flow_only_model.pt")
    torch.save(guided, args.output_dir / "selected_guided_geological_model.pt")
    render_figure(args.output_dir / "flow_demo_3d.png", evidence, truth, baseline, guided, selected)
    render_full_geology_figure(
        args.output_dir / "flow_demo_full_geology_3d.png", truth, baseline, guided
    )
    render_phase1_style_figure(
        args.output_dir / "flow_demo_phase1_style.png",
        evidence,
        truth,
        baseline,
        guided,
        selected,
    )
    median_delta_iou = float(np.median([float(row["delta_target_iou"]) for row in rows]))
    median_delta_precision = float(np.median([float(row["delta_target_precision"]) for row in rows]))
    median_delta_recall = float(np.median([float(row["delta_target_recall"]) for row in rows]))
    median_delta_global_miou = float(np.median([float(row["delta_global_mean_iou"]) for row in rows]))
    summary = {
        "schema": "stage15_flow_demo_posthoc_evaluation_v1",
        "run_status": "completed",
        "scientific_role": "post_hoc_exploratory_mechanism_demonstration_not_generalization_evidence",
        "pair_count": len(rows),
        "all_pairs_reported": True,
        "selection_policy": "maximum retrospective guided-minus-baseline label9 IoU; recall tie-break",
        "selection_is_truth_informed_and_post_hoc": True,
        "selected_pair": selected,
        "positive_delta_iou_pair_count": sum(float(row["delta_target_iou"]) > 0 for row in rows),
        "positive_delta_recall_pair_count": sum(float(row["delta_target_recall"]) > 0 for row in rows),
        "median_delta_target_iou": median_delta_iou,
        "median_delta_target_precision": median_delta_precision,
        "median_delta_target_recall": median_delta_recall,
        "median_delta_global_mean_iou": median_delta_global_miou,
        "phase1_equivalence": False,
        "phase1_comparison": "coarse evidence improves target localization but does not recover Phase1 oracle geometry and produces excessive small components",
        "truth_loaded_only_by_evaluator": True,
        "threshold_sweep_performed": False,
        "flow_training_performed": False,
    }
    write_json(args.output_dir / "summary.json", summary)
    report = f"""# Stage15 exploratory Flow demonstration

This is a post-hoc, truth-informed visualization selection from all eight strictly paired seeds. It demonstrates a possible mechanism and is not evidence of generalization.

- Positive label9 IoU pairs: {summary['positive_delta_iou_pair_count']}/8
- Positive label9 recall pairs: {summary['positive_delta_recall_pair_count']}/8
- Median delta IoU / precision / recall: {median_delta_iou:+.6f} / {median_delta_precision:+.6f} / {median_delta_recall:+.6f}
- Median global mIoU delta: {median_delta_global_miou:+.6f}
- Selected seed: {selected['source_seed']}
- Label9 IoU: {selected['baseline_target_iou']:.6f} -> {selected['guided_target_iou']:.6f} ({selected['delta_target_iou']:+.6f})
- Label9 precision: {selected['baseline_target_precision']:.6f} -> {selected['guided_target_precision']:.6f}
- Label9 recall: {selected['baseline_target_recall']:.6f} -> {selected['guided_target_recall']:.6f} ({selected['delta_target_recall']:+.6f})
- Centroid distance: {selected['baseline_target_centroid_distance']:.6f} -> {selected['guided_target_centroid_distance']:.6f}
- Label9 connected components: {selected['baseline_target_connected_components']} -> {selected['guided_target_connected_components']}
- Largest-component fraction: {selected['baseline_largest_component_fraction']:.6f} -> {selected['guided_largest_component_fraction']:.6f}
- Hard-condition violations: {selected['baseline_condition_violations']} / {selected['guided_condition_violations']}

The coarse occupancy formulation removes the previous repeated-block target, but it does not equal Phase1: localization improves while the hard decoded label9 becomes excessively fragmented. The Phase1-style figure uses the same fixed-camera surface rendering so that this difference remains visible rather than being hidden by plotting choices.
"""
    (args.output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    manifest = base_manifest("stage15_flow_demo_posthoc_evaluation_run_v1", Path(__file__))
    manifest.update({"run_status": "completed", "truth_loaded_only_by_evaluator": True})
    write_json(args.output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
