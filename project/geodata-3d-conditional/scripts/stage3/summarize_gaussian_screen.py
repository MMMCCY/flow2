#!/usr/bin/env python3
"""Summarize the completed Phase-3 seed-42 n=1 Gaussian screen."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import inference_runtime as runtime


MANIFEST_SCHEMA = "phase3_spatial_property_sweep_v1"


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    experiment_root = PROJECT_DIR / "experiments/stage3_spatial_property"
    parser = argparse.ArgumentParser(
        description="Summarize the frozen Phase-3 Gaussian n=1 screen.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=experiment_root / "configs/gaussian_sweep_manifest_v1.json",
    )
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=experiment_root / "reports",
    )
    parser.add_argument(
        "--identity-run-name",
        default="seed42_n1_s32_a025_c025_fix1",
    )
    parser.add_argument(
        "--gaussian-run-name",
        default="seed42_n1_s32_a025_c025",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=experiment_root / "reports/gaussian_screen_seed42_n1",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def screen_decision(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Apply the frozen n=1 screen-to-bracket rule."""
    ordered = sorted(rows, key=lambda row: int(row["order"]))
    if not ordered or str(ordered[0]["role"]) != "implementation_anchor":
        raise ValueError("screen rows must start with the implementation anchor")
    if not bool(ordered[0]["passed"]):
        return {
            "status": "anchor_failed",
            "selected_level": None,
            "bracket_levels": [],
        }
    gaussian_passes = [row for row in ordered[1:] if bool(row["passed"])]
    if not gaussian_passes:
        return {
            "status": "no_nonzero_blur_passed",
            "selected_level": str(ordered[0]["level"]),
            "bracket_levels": [str(ordered[0]["level"]), str(ordered[1]["level"])],
        }
    selected = max(gaussian_passes, key=lambda row: int(row["order"]))
    selected_index = int(selected["order"])
    adjacent_index = min(selected_index + 1, len(ordered) - 1)
    if adjacent_index == selected_index:
        adjacent_index = selected_index - 1
    return {
        "status": "nonzero_blur_candidate_identified",
        "selected_level": str(selected["level"]),
        "bracket_levels": [
            str(selected["level"]),
            str(ordered[adjacent_index]["level"]),
        ],
    }


def _nonincreasing(values: Sequence[float], tolerance: float = 1e-9) -> bool:
    return all(right <= left + tolerance for left, right in zip(values, values[1:]))


def _report(summary: Mapping[str, object]) -> str:
    lines = [
        "# Phase-3 Gaussian spatial-resolution seed-42 n=1 screen",
        "",
        "## Decision",
        "",
        f"**{summary['decision_text']}**",
        "",
        (
            "This is a truth-derived 3-D spatial-degradation screen, not "
            "measured or acquisition-domain geophysics."
        ),
        "",
        "## Frozen levels",
        "",
        "| Level | Gate | Accuracy delta | Fixed mIoU delta | Label-9 IoU / P / R | Major mean recall |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for row in summary["levels"]:
        lines.append(
            f"| {row['level']} | {row['passed']} | "
            f"{float(row['delta_global_voxel_accuracy']):.4f} | "
            f"{float(row['delta_truth_present_mean_iou']):.4f} | "
            f"{float(row['target_iou']):.4f} / "
            f"{float(row['target_precision']):.4f} / "
            f"{float(row['target_recall']):.4f} | "
            f"{float(row['major_component_mean_recall']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen promotion result",
            "",
            f"- Status: `{summary['promotion']['status']}`.",
            f"- Selected level: `{summary['promotion']['selected_level']}`.",
            f"- Seed-42 n=4 bracket: `{summary['promotion']['bracket_levels']}`.",
            f"- Label-9 IoU is non-increasing with blur: `{summary['monotonic']['target_iou']}`.",
            f"- Major-body mean recall is non-increasing with blur: `{summary['monotonic']['major_component_mean_recall']}`.",
            "- Lower observation loss alone did not promote any nonzero blur.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    manifest = _read_json(args.manifest)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"manifest schema must be {MANIFEST_SCHEMA!r}")
    levels = manifest.get("levels")
    if not isinstance(levels, list) or len(levels) < 2:
        raise ValueError("manifest must contain anchor and Gaussian levels")

    rows: list[dict[str, object]] = []
    for level in levels:
        level_id = str(level["id"])
        run_name = (
            args.identity_run_name
            if str(level["role"]) == "implementation_anchor"
            else args.gaussian_run_name
        )
        report_path = args.reports_root / level_id / run_name / "summary.json"
        report = _read_json(report_path)
        config_path = (args.manifest.parent / str(level["config"])).resolve()
        if report.get("level") != level_id:
            raise ValueError(f"report level mismatch: {report_path}")
        if int(report.get("n_samples", -1)) != 1:
            raise ValueError(f"screen report is not n=1: {report_path}")
        if report.get("observation_config_sha256") != runtime.file_sha256(config_path):
            raise ValueError(f"observation config hash mismatch: {level_id}")
        if report.get("strict_pairing") is not True:
            raise ValueError(f"strict pairing failed: {level_id}")
        if report.get("phase2_baseline_regression", {}).get("passed") is not True:
            raise ValueError(f"alpha-zero regression failed: {level_id}")
        guided = report["guided_metrics"]
        delta = report["paired_deltas"]
        gate = report["pair_gate"]
        rows.append(
            {
                "level": level_id,
                "order": int(level["order"]),
                "role": str(level["role"]),
                "passed": bool(gate["passed"]),
                "delta_global_voxel_accuracy": delta["delta_global_voxel_accuracy"],
                "delta_truth_present_mean_iou": delta[
                    "delta_truth_present_mean_iou"
                ],
                "delta_global_mean_iou": delta["delta_global_mean_iou"],
                "delta_hard_observation_loss": delta[
                    "delta_hard_observation_loss"
                ],
                "target_iou": guided["target_iou"],
                "target_precision": guided["target_precision"],
                "target_recall": guided["target_recall"],
                "improved_truth_present_classes": gate[
                    "improved_truth_present_classes"
                ],
                "major_component_min_recall": gate[
                    "major_component_min_recall"
                ],
                "major_component_mean_recall": gate[
                    "major_component_mean_recall"
                ],
                "final_churn_fraction": gate["final_churn_fraction"],
                "report": str(report_path),
            }
        )
    rows.sort(key=lambda row: int(row["order"]))
    promotion = screen_decision(rows)
    decision_text = {
        "anchor_failed": "BLOCKED: identity anchor failed",
        "no_nonzero_blur_passed": (
            "SCREEN CLOSED: no nonzero Gaussian blur passed the complete n=1 gate"
        ),
        "nonzero_blur_candidate_identified": (
            "SCREEN CLOSED: a nonzero Gaussian candidate was identified"
        ),
    }[str(promotion["status"])]
    summary: dict[str, object] = {
        "decision_text": decision_text,
        "scope": "truth-derived Phase-3 Gaussian spatial-resolution n=1 screen",
        "is_measured_geophysics": False,
        "manifest": str(args.manifest),
        "manifest_sha256": runtime.file_sha256(args.manifest),
        "levels": rows,
        "promotion": promotion,
        "monotonic": {
            "target_iou": _nonincreasing(
                [float(row["target_iou"]) for row in rows]
            ),
            "target_precision": _nonincreasing(
                [float(row["target_precision"]) for row in rows]
            ),
            "target_recall": _nonincreasing(
                [float(row["target_recall"]) for row in rows]
            ),
            "major_component_mean_recall": _nonincreasing(
                [float(row["major_component_mean_recall"]) for row in rows]
            ),
        },
        "limitations": [
            "the screen contains one sample per level",
            "identity is the only passing level and is not spatial degradation",
            "the distinct Phase-2a codebook remains truth-derived and label-9 distinctive",
            "continuous observation loss is insufficient for promotion",
        ],
    }
    existing = list(args.output_dir.iterdir()) if args.output_dir.exists() else []
    if existing and not args.overwrite:
        raise FileExistsError(
            f"output directory is non-empty; pass --overwrite: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "summary.json", summary)
    _write_csv(args.output_dir / "level_summary.csv", rows)
    report_text = _report(summary)
    (args.output_dir / "REPORT.md").write_text(report_text, encoding="utf-8")
    print(report_text)


if __name__ == "__main__":
    main()

