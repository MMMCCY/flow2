#!/usr/bin/env python3
"""Post-hoc gravity-residual ranking of a completed baseline ensemble."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.stage4.audit_gravity_screen import read_numeric_csv
from scripts.stage4.run_gravity_guidance import (
    paired_gravity_config_verdict,
    read_json,
    write_json,
)


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: Sequence[Mapping[str, object]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)


def build_reranking_summary(
    baseline_rows: Sequence[Mapping[str, object]],
    guided_rows: Sequence[Mapping[str, object]],
    *,
    pairing_reason: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Return ranked baseline rows and the immutable selection comparison."""
    if len(baseline_rows) < 2 or len(guided_rows) != len(baseline_rows):
        raise ValueError("reranking requires equal baseline/guided ensembles of size >= 2")
    if any(
        not math.isfinite(float(row["hard_gravity_loss"]))
        for row in (*baseline_rows, *guided_rows)
    ):
        raise ValueError("gravity losses must be finite")
    ordered = sorted(baseline_rows, key=lambda row: float(row["hard_gravity_loss"]))
    ranked = [
        {
            "rank": rank,
            "sample_id": int(row["sample_id"]),
            "hard_gravity_loss": float(row["hard_gravity_loss"]),
            "hard_gravity_rmse_mgal": float(row["hard_gravity_rmse_mgal"]),
            "global_voxel_accuracy": float(row["global_voxel_accuracy"]),
            "truth_present_mean_iou": float(row["truth_present_mean_iou"]),
            "target_iou": float(row["target_iou"]),
            "target_precision": float(row["target_precision"]),
            "target_recall": float(row["target_recall"]),
        }
        for rank, row in enumerate(ordered, start=1)
    ]
    best_baseline = ranked[0]
    best_guided = min(guided_rows, key=lambda row: float(row["hard_gravity_loss"]))
    fields = (
        "hard_gravity_loss",
        "hard_gravity_rmse_mgal",
        "global_voxel_accuracy",
        "truth_present_mean_iou",
        "target_iou",
        "target_precision",
        "target_recall",
    )
    summary = {
        "status": "completed",
        "description": (
            "Post-hoc selection only: no baseline geology was changed and no "
            "gradient used the observation."
        ),
        "strict_pairing": True,
        "pairing_reason": pairing_reason,
        "n_samples": len(baseline_rows),
        "baseline_selected_sample_id": best_baseline["sample_id"],
        "baseline_selected": best_baseline,
        "baseline_ensemble_mean": {
            field: _mean(baseline_rows, field) for field in fields
        },
        "guided_ensemble_mean": {
            field: _mean(guided_rows, field) for field in fields
        },
        "guided_best": {
            "sample_id": int(best_guided["sample_id"]),
            "hard_gravity_loss": float(best_guided["hard_gravity_loss"]),
            "hard_gravity_rmse_mgal": float(best_guided["hard_gravity_rmse_mgal"]),
            "global_voxel_accuracy": float(best_guided["global_voxel_accuracy"]),
            "truth_present_mean_iou": float(best_guided["truth_present_mean_iou"]),
            "target_iou": float(best_guided["target_iou"]),
        },
        "comparisons": {
            "guided_mean_loss_below_reranked_baseline": (
                _mean(guided_rows, "hard_gravity_loss")
                < float(best_baseline["hard_gravity_loss"])
            ),
            "guided_best_loss_below_reranked_baseline": (
                float(best_guided["hard_gravity_loss"])
                < float(best_baseline["hard_gravity_loss"])
            ),
        },
    }
    return ranked, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank an alpha-zero ensemble without modifying any sample."
    )
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--guided-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_config = read_json(args.baseline_dir / "config.json")
    guided_config = read_json(args.guided_dir / "config.json")
    paired, reason = paired_gravity_config_verdict(baseline_config, guided_config)
    if not paired:
        raise ValueError(f"strict pair validation failed: {reason}")
    n_samples = int(baseline_config["n_samples"])
    if n_samples < 2:
        raise ValueError("post-hoc reranking requires at least two baseline samples")
    baseline_rows = read_numeric_csv(args.baseline_dir / "sample_metrics.csv")
    guided_rows = read_numeric_csv(args.guided_dir / "sample_metrics.csv")
    if len(baseline_rows) != n_samples or len(guided_rows) != n_samples:
        raise ValueError("sample metrics are incomplete")
    ranked, summary = build_reranking_summary(
        baseline_rows, guided_rows, pairing_reason=reason
    )
    best_baseline = ranked[0]
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(args.output_dir / "baseline_gravity_ranking.csv", ranked)
    write_json(args.output_dir / "summary.json", summary)
    report = "\n".join(
        [
            "# Phase-4a post-hoc baseline reranking",
            "",
            "This selects an existing alpha-zero sample by hard gravity residual; it does not alter geology.",
            "",
            f"- Selected baseline sample: `{best_baseline['sample_id']}`.",
            f"- Selected baseline gravity loss: `{best_baseline['hard_gravity_loss']:.6f}`.",
            f"- Guided mean beats selected baseline residual: `{summary['comparisons']['guided_mean_loss_below_reranked_baseline']}`.",
            f"- Guided best beats selected baseline residual: `{summary['comparisons']['guided_best_loss_below_reranked_baseline']}`.",
            "",
        ]
    )
    (args.output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
