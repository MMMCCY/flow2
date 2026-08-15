#!/usr/bin/env python3
"""Retrospective truth evaluation of frozen Stage15-E 4^3 inversions."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Mapping, Sequence

from scipy.stats import pearsonr, spearmanr
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime
from guidance.coarse_binary_seismic import coarse_truth_occupancy
from guidance.seismic import tensor_sha256
from scripts.stage10.evaluate_bridge_information import average_precision
from scripts.stage15.common import (
    base_manifest,
    normalize_volume,
    read_json,
    refuse_nonempty,
    write_csv,
    write_json,
)


EXPERIMENT_ROOT = PROJECT_DIR / "experiments/stage15_binary_seismic_consensus"
DEFAULT_OBSERVATION_DIR = EXPERIMENT_ROOT / "observations/cond_generation_0"
DEFAULT_INVERSION_DIR = EXPERIMENT_ROOT / "coarse_inversion/coarse_4x4x4_n8_v1"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "reports/coarse_4x4x4_truth_evaluation_v1"
MACHINE_DECISIONS = (
    "COARSE_LABEL9_LOCALIZATION_WORKS",
    "COARSE_LABEL9_LOCALIZATION_FAILS",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation-dir", type=Path, default=DEFAULT_OBSERVATION_DIR)
    parser.add_argument("--inversion-dir", type=Path, default=DEFAULT_INVERSION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--decision", choices=MACHINE_DECISIONS, default=None)
    return parser.parse_args()


def weighted_centroid(weights: torch.Tensor, domain: torch.Tensor) -> list[float] | None:
    if weights.ndim != 3 or domain.shape != weights.shape:
        raise ValueError("weighted centroid expects matching 3-D tensors")
    selected = torch.where(domain.bool(), weights.float(), torch.zeros_like(weights.float()))
    total = selected.sum()
    if float(total) <= 0:
        return None
    coordinates = torch.stack(
        torch.meshgrid(
            *(torch.arange(size, dtype=torch.float32) for size in weights.shape),
            indexing="ij",
        ),
        dim=-1,
    )
    return ((coordinates * selected.unsqueeze(-1)).sum(dim=(0, 1, 2)) / total).tolist()


def centroid_distance(left: list[float] | None, right: list[float] | None) -> float:
    if left is None or right is None:
        return float("nan")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def top_k_localization(
    q_pred: torch.Tensor, target_presence: torch.Tensor, domain: torch.Tensor
) -> dict[str, object]:
    flat_domain = domain.reshape(-1).bool()
    domain_indices = torch.nonzero(flat_domain, as_tuple=False).flatten()
    target_flat = target_presence.reshape(-1).bool()
    k = int((target_flat & flat_domain).sum())
    if k <= 0:
        raise ValueError("coarse truth has no target-containing cell")
    scores = q_pred.reshape(-1)[domain_indices]
    order = torch.argsort(scores, descending=True, stable=True)
    selected = domain_indices[order[:k]]
    overlap = int(target_flat[selected].sum())
    return {
        "k": k,
        "overlap_count": overlap,
        "overlap_fraction": overlap / k,
        "selected_flat_indices": selected.tolist(),
        "truth_target_flat_indices": torch.nonzero(
            target_flat & flat_domain, as_tuple=False
        ).flatten().tolist(),
    }


def coarse_metrics(
    q_pred: torch.Tensor,
    q_true: torch.Tensor,
    target_presence: torch.Tensor,
    domain: torch.Tensor,
) -> dict[str, object]:
    if not (q_pred.shape == q_true.shape == target_presence.shape == domain.shape):
        raise ValueError("all coarse metric tensors must match")
    selected_pred = q_pred[domain].double()
    selected_true = q_true[domain].double()
    selected_presence = target_presence[domain].bool()
    pearson = pearsonr(selected_pred.numpy(), selected_true.numpy())
    spearman = spearmanr(selected_pred.numpy(), selected_true.numpy())
    target_mean = float(selected_pred[selected_presence].mean())
    background_mean = float(selected_pred[~selected_presence].mean())
    true_center = weighted_centroid(q_true[0, 0], domain[0, 0])
    predicted_center = weighted_centroid(q_pred[0, 0], domain[0, 0])
    return {
        "pearson_correlation": float(pearson.statistic),
        "pearson_two_sided_p_value": float(pearson.pvalue),
        "spearman_correlation": float(spearman.statistic),
        "spearman_two_sided_p_value": float(spearman.pvalue),
        "coarse_presence_auprc": average_precision(selected_pred, selected_presence),
        "target_containing_cell_count": int(selected_presence.sum()),
        "background_only_cell_count": int((~selected_presence).sum()),
        "mean_q_pred_target_containing_cells": target_mean,
        "mean_q_pred_background_only_cells": background_mean,
        "target_minus_background_mean_q": target_mean - background_mean,
        "predicted_weighted_centroid_xyz": predicted_center,
        "truth_weighted_centroid_xyz": true_center,
        "centroid_distance_coarse_cells": centroid_distance(
            predicted_center, true_center
        ),
        "top_k": top_k_localization(q_pred, target_presence, domain),
    }


def metric_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    fields = (
        "pearson_correlation",
        "spearman_correlation",
        "coarse_presence_auprc",
        "mean_q_pred_target_containing_cells",
        "mean_q_pred_background_only_cells",
        "target_minus_background_mean_q",
        "centroid_distance_coarse_cells",
    )
    result: dict[str, object] = {"run_count": len(rows)}
    for field in fields:
        values = [float(row[field]) for row in rows]
        result[field] = {
            "min": min(values),
            "median": statistics.median(values),
            "max": max(values),
        }
    overlaps = [float(row["top_k_overlap_fraction"]) for row in rows]
    result["top_k_overlap_fraction"] = {
        "min": min(overlaps),
        "median": statistics.median(overlaps),
        "max": max(overlaps),
    }
    return result


def evaluate(
    observation_dir: Path, inversion_dir: Path
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, str]]:
    inversion_manifest_path = inversion_dir / "manifest.json"
    observation_manifest_path = observation_dir / "manifest.json"
    subsurface_path = observation_dir / "subsurface_mask.pt"
    input_paths = [inversion_manifest_path, observation_manifest_path, subsurface_path]
    input_paths.extend(sorted(inversion_dir.glob("run_*/coarse_occupancy.pt")))
    input_paths.append(inversion_dir / "mean_coarse_occupancy.pt")
    hashes = {str(path.resolve()): runtime.file_sha256(path) for path in input_paths}

    inversion_manifest = read_json(inversion_manifest_path)
    if inversion_manifest.get("run_status") != "completed":
        raise ValueError("Stage15-E inversion is not completed")
    if inversion_manifest.get("run_count") != 8:
        raise ValueError("Stage15-E must contain exactly eight inversions")
    if inversion_manifest.get("truth_loaded_by_runner") is not False:
        raise ValueError("Stage15-E runner violated the truth firewall")
    observation_manifest = read_json(observation_manifest_path)
    truth_record = observation_manifest["phase1_assets"]["truth_model"]
    truth_path = Path(str(truth_record["path"]))
    if runtime.file_sha256(truth_path) != truth_record["sha256"]:
        raise ValueError("authoritative cond_generation_0 truth hash changed")
    hashes[str(truth_path.resolve())] = runtime.file_sha256(truth_path)
    truth = normalize_volume(runtime.load_tensor(truth_path), "true_model").long()
    subsurface = normalize_volume(
        runtime.load_tensor(subsurface_path), "subsurface_mask"
    ).bool()
    binary_truth = (truth == 9) & subsurface
    q_true, target_presence, support_count = coarse_truth_occupancy(
        binary_truth, subsurface
    )
    domain = support_count > 0

    q_paths = sorted(inversion_dir.glob("run_*/coarse_occupancy.pt"))
    if len(q_paths) != 8:
        raise ValueError("expected exactly eight coarse occupancy tensors")
    rows: list[dict[str, object]] = []
    q_values: list[torch.Tensor] = []
    for run_index, path in enumerate(q_paths):
        q_pred = runtime.load_tensor(path).float()
        if q_pred.shape != (1, 1, 4, 4, 4):
            raise ValueError(f"invalid q shape: {path}")
        metrics = coarse_metrics(q_pred, q_true, target_presence, domain)
        run_metrics = read_json(path.parent / "metrics.json")
        if tensor_sha256(q_pred) != run_metrics["coarse_occupancy_tensor_sha256"]:
            raise ValueError(f"coarse occupancy hash mismatch: {path}")
        row = {
            "run_index": run_index,
            "seed": int(run_metrics["seed"]),
            "initial_seismic_mse": float(run_metrics["initial_seismic_mse"]),
            "final_seismic_mse": float(run_metrics["final_seismic_mse"]),
            "final_seismic_rmse": float(run_metrics["final_seismic_rmse"]),
            "pearson_correlation": metrics["pearson_correlation"],
            "spearman_correlation": metrics["spearman_correlation"],
            "coarse_presence_auprc": metrics["coarse_presence_auprc"],
            "mean_q_pred_target_containing_cells": metrics[
                "mean_q_pred_target_containing_cells"
            ],
            "mean_q_pred_background_only_cells": metrics[
                "mean_q_pred_background_only_cells"
            ],
            "target_minus_background_mean_q": metrics[
                "target_minus_background_mean_q"
            ],
            "centroid_distance_coarse_cells": metrics[
                "centroid_distance_coarse_cells"
            ],
            "top_k": metrics["top_k"]["k"],
            "top_k_overlap_count": metrics["top_k"]["overlap_count"],
            "top_k_overlap_fraction": metrics["top_k"]["overlap_fraction"],
            "coarse_occupancy_tensor_sha256": tensor_sha256(q_pred),
        }
        rows.append(row)
        q_values.append(q_pred)

    mean_q = runtime.load_tensor(inversion_dir / "mean_coarse_occupancy.pt").float()
    expected_mean = torch.stack(q_values).mean(dim=0)
    if not torch.equal(mean_q, expected_mean):
        raise ValueError("saved mean coarse occupancy differs from eight-run mean")
    ensemble_metrics = coarse_metrics(mean_q, q_true, target_presence, domain)
    summary = {
        "schema": "stage15_e_coarse_binary_seismic_truth_evaluation_v1",
        "run_status": "completed",
        "machine_decision_policy": "manual_alignment_interpretation_without_new_numeric_gate",
        "grid_accounting": {
            "fine_grid_shape": [64, 64, 64],
            "coarse_grid_shape": [4, 4, 4],
            "fine_voxels_per_coarse_cell": [16, 16, 16],
            "evaluated_coarse_cells": int(domain.sum()),
            "target_containing_coarse_cells": int((target_presence & domain).sum()),
            "background_only_coarse_cells": int((~target_presence & domain).sum()),
        },
        "q_true": {
            "min": float(q_true[domain].min()),
            "mean": float(q_true[domain].mean()),
            "max": float(q_true[domain].max()),
            "tensor_sha256": tensor_sha256(q_true),
            "definition": "mean raw-label9 occupancy within subsurface voxels of each 16x16x16 block; not thresholded",
        },
        "per_run_summary": metric_summary(rows),
        "mean_coarse_occupancy_metrics": ensemble_metrics,
        "input_file_sha256": hashes,
        "truth_loaded_only_by_evaluator": True,
        "probability_threshold_applied": False,
        "flow_used": False,
    }
    for path, expected in hashes.items():
        if runtime.file_sha256(Path(path)) != expected:
            raise RuntimeError(f"Stage15-E evaluation input changed: {path}")
    return summary, rows, hashes


def _fmt(value: object) -> str:
    return f"{float(value):.8g}"


def render_report(summary: Mapping[str, object], rows: Sequence[Mapping[str, object]]) -> str:
    ensemble = summary["mean_coarse_occupancy_metrics"]
    lines = [
        "# Stage15-E — 4^3 coarse binary seismic inversion identifiability",
        "",
        f"Machine decision: **{summary['machine_decision']}**",
        "",
        "The truth-blind runner optimized only 64 sigmoid coarse variables against raw mean-squared seismic misfit. It used no Flow, checkpoint, prior, regularizer, threshold, or observation regeneration. A 4^3 grid over 64^3 necessarily uses 16^3 fine voxels per coarse cell.",
        "",
        "## Eight inversions",
        "",
        "| Run | Seed | Final MSE | Pearson | Spearman | Target q | Background q | Top-k overlap | Centroid distance |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run_index']} | {row['seed']} | {_fmt(row['final_seismic_mse'])} | {_fmt(row['pearson_correlation'])} | {_fmt(row['spearman_correlation'])} | {_fmt(row['mean_q_pred_target_containing_cells'])} | {_fmt(row['mean_q_pred_background_only_cells'])} | {row['top_k_overlap_count']}/{row['top_k']} ({_fmt(row['top_k_overlap_fraction'])}) | {_fmt(row['centroid_distance_coarse_cells'])} |"
        )
    lines.extend(
        [
            "",
            "## Eight-run mean coarse occupancy",
            "",
            f"- Pearson / Spearman: {_fmt(ensemble['pearson_correlation'])} / {_fmt(ensemble['spearman_correlation'])}",
            f"- Supplementary coarse-presence AUPRC: {_fmt(ensemble['coarse_presence_auprc'])}",
            f"- Target-containing / background-only mean q: {_fmt(ensemble['mean_q_pred_target_containing_cells'])} / {_fmt(ensemble['mean_q_pred_background_only_cells'])}",
            f"- Top-k overlap: {ensemble['top_k']['overlap_count']}/{ensemble['top_k']['k']} ({_fmt(ensemble['top_k']['overlap_fraction'])})",
            f"- Centroid distance: {_fmt(ensemble['centroid_distance_coarse_cells'])} coarse cells",
            "",
            "## Interpretation",
            "",
            str(summary["decision_rationale"]),
            "",
            "This decision uses the complete frozen localization diagnostics, not seismic-loss reduction alone and not a newly introduced numerical gate.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if not args.preview and args.decision is None:
        raise ValueError("final evaluation requires one of the two --decision values")
    summary, rows, hashes = evaluate(args.observation_dir, args.inversion_dir)
    if args.preview:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    refuse_nonempty(args.output_dir)
    summary["machine_decision"] = args.decision
    summary["machine_decision_allowed_values"] = list(MACHINE_DECISIONS)
    summary["decision_rationale"] = {
        "COARSE_LABEL9_LOCALIZATION_WORKS": (
            "Across independent starts, higher coarse occupancy consistently concentrates "
            "in the true label9-containing coarse cells with coherent correlation, top-k, "
            "separation, and centroid evidence."
        ),
        "COARSE_LABEL9_LOCALIZATION_FAILS": (
            "The frozen inversions do not consistently concentrate high coarse occupancy "
            "at true label9 coarse locations; seismic-loss reduction alone is insufficient."
        ),
    }[args.decision]
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "summary.json", summary)
    write_csv(args.output_dir / "per_run_metrics.csv", rows)
    (args.output_dir / "REPORT.md").write_text(
        render_report(summary, rows), encoding="utf-8"
    )
    manifest = base_manifest(
        "stage15_e_coarse_binary_seismic_truth_evaluation_run_v1", Path(__file__)
    )
    manifest.update(
        {
            "run_status": "completed",
            "machine_decision": args.decision,
            "input_file_sha256_before_and_after": hashes,
            "inputs_unchanged": True,
            "summary": runtime.asset_record(args.output_dir / "summary.json"),
            "per_run_metrics": runtime.asset_record(
                args.output_dir / "per_run_metrics.csv"
            ),
            "report": runtime.asset_record(args.output_dir / "REPORT.md"),
            "truth_loaded_by_evaluator": True,
            "truth_loaded_by_inversion_runner": False,
            "flow_used": False,
        }
    )
    write_json(args.output_dir / "manifest.json", manifest)
    for path, expected in hashes.items():
        if runtime.file_sha256(Path(path)) != expected:
            raise RuntimeError(f"Stage15-E evaluation input changed: {path}")


if __name__ == "__main__":
    main()
