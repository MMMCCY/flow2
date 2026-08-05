#!/usr/bin/env python3
"""Audit the predeclared Phase-2b codebook-ambiguity single-sample screen."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import inference_runtime as runtime
from guidance.property_volume import (
    property_codebook_diagnostics,
    property_table_from_config,
)
from scripts.stage2.run_property_guidance import paired_property_config_verdict
from scripts.stage2.summarize_phase2a import (
    _index_rows,
    _last_trace_rows,
    read_json,
    read_numeric_csv,
    sample_gate_audit,
    write_csv,
    write_json,
)


MANIFEST_SCHEMA = "phase2b_codebook_ambiguity_sweep_v1"
EXPERIMENT_STAGE = "phase2b_codebook_ambiguity_v1"
REPORT_FILENAMES = ("REPORT.md", "summary.json", "level_summary.csv")
ANCHOR_GUIDED_DISAGREEMENT_LIMIT = 0.001
ANCHOR_METRIC_TOLERANCES = {
    "global_voxel_accuracy": 0.005,
    "truth_present_mean_iou": 0.005,
    "target_iou": 0.02,
    "target_precision": 0.02,
    "target_recall": 0.02,
}


def parse_args() -> argparse.Namespace:
    experiment_root = PROJECT_DIR / "experiments/stage2_property"
    parser = argparse.ArgumentParser(
        description="Audit completed levels of the Phase-2b seed-42 n=1 screen.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=experiment_root
        / "configs/phase2b_codebook_ambiguity_v1/sweep_manifest.json",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=experiment_root
        / "runs/cond_generation_0/phase2b_codebook_ambiguity_v1",
    )
    parser.add_argument("--level", action="append", dest="levels")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=32)
    parser.add_argument(
        "--run-name",
        default="seed42_n1_s32_a025_c025",
    )
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--guided-name", default="alpha025")
    parser.add_argument(
        "--phase2a-reference-root",
        type=Path,
        default=experiment_root
        / "runs/cond_generation_0"
        / "ideal_density_susceptibility_label9_contrast_v1/phase2a_v1"
        / "seed42_n4_s32_a025_c025",
    )
    parser.add_argument(
        "--require-all-levels",
        action="store_true",
        help="Fail if any predeclared level is absent; use for the final screen decision.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=experiment_root
        / "reports/phase2b_codebook_ambiguity_v1_screen_seed42",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace derived reports only; immutable run directories remain read-only.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, object]:
    manifest = read_json(path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"manifest schema must be {MANIFEST_SCHEMA!r}")
    if manifest.get("experiment_stage") != EXPERIMENT_STAGE:
        raise ValueError(f"manifest stage must be {EXPERIMENT_STAGE!r}")
    levels = manifest.get("levels")
    if not isinstance(levels, list) or not levels:
        raise ValueError("manifest must contain a non-empty levels list")
    ids = [str(level.get("id", "")) for level in levels]
    orders = [int(level.get("order", -1)) for level in levels]
    if any(not level_id for level_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("manifest level IDs must be non-empty and unique")
    if orders != list(range(len(levels))):
        raise ValueError("manifest levels must have contiguous predeclared order")
    if levels[0].get("role") != "implementation_anchor":
        raise ValueError("first manifest level must be the implementation anchor")
    if any(level.get("role") != "ambiguity_sweep" for level in levels[1:]):
        raise ValueError("all non-anchor levels must be ambiguity_sweep levels")
    return manifest


def _resolved_config_path(manifest_path: Path, level: Mapping[str, object]) -> Path:
    return (manifest_path.parent / str(level["config"])).resolve()


def _validate_pair_configs(
    baseline: Mapping[str, object],
    guided: Mapping[str, object],
    config_path: Path,
    seed: int,
    n_samples: int,
    n_steps: int,
) -> None:
    paired, reason = paired_property_config_verdict(baseline, guided)
    if not paired:
        raise ValueError(f"strict Phase-2b pairing failed: {reason}")
    if guided.get("pairing_validation", {}).get("paired") is not True:
        raise ValueError("saved guided pairing verdict is not true")
    expected_config_hash = runtime.file_sha256(config_path)
    for name, config, alpha in (
        ("baseline", baseline, 0.0),
        ("guided", guided, 0.25),
    ):
        expected = {
            "stage": EXPERIMENT_STAGE,
            "seed": seed,
            "n_samples": n_samples,
            "n_steps": n_steps,
            "run_status": "completed",
            "samples_written": n_samples,
            "ema_applied": True,
            "model_weight_source": "ema",
            "max_post_projection_condition_violations": 0,
            "property_config_sha256": expected_config_hash,
        }
        for field, value in expected.items():
            if config.get(field) != value:
                raise ValueError(
                    f"{name} {field}={config.get(field)!r}, expected {value!r}"
                )
        if float(config.get("alpha", float("nan"))) != alpha:
            raise ValueError(f"{name} alpha must be {alpha}")
        if float(config.get("max_guidance_ratio", float("nan"))) != 0.25:
            raise ValueError(f"{name} max_guidance_ratio must be 0.25")
    if baseline.get("initial_noise_sha256") != guided.get("initial_noise_sha256"):
        raise ValueError("paired initial-noise hashes differ")


def _anchor_regression(
    pair_root: Path,
    reference_root: Path,
    current_guided_metrics: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    current_baseline = runtime.load_tensor(pair_root / "baseline/sample_0.pt")
    current_guided = runtime.load_tensor(pair_root / "alpha025/sample_0.pt")
    reference_baseline = runtime.load_tensor(reference_root / "baseline/sample_0.pt")
    reference_guided = runtime.load_tensor(reference_root / "alpha025/sample_0.pt")
    if current_baseline.shape != reference_baseline.shape:
        raise ValueError("anchor/reference baseline shapes differ")
    if current_guided.shape != reference_guided.shape:
        raise ValueError("anchor/reference guided shapes differ")
    baseline_mismatch = int((current_baseline != reference_baseline).sum().item())
    guided_mismatch = int((current_guided != reference_guided).sum().item())
    guided_fraction = guided_mismatch / current_guided.numel()
    reference_metrics = _index_rows(
        read_numeric_csv(reference_root / "alpha025/sample_metrics.csv")
    )[0]
    metric_differences = {
        field: abs(float(current_guided_metrics[0][field]) - float(reference_metrics[field]))
        for field in ANCHOR_METRIC_TOLERANCES
    }
    metrics_pass = all(
        metric_differences[field] <= tolerance
        for field, tolerance in ANCHOR_METRIC_TOLERANCES.items()
    )
    return {
        "passed": (
            baseline_mismatch == 0
            and guided_fraction <= ANCHOR_GUIDED_DISAGREEMENT_LIMIT
            and metrics_pass
        ),
        "baseline_hard_mismatch_voxels": baseline_mismatch,
        "guided_hard_mismatch_voxels": guided_mismatch,
        "guided_hard_disagreement_fraction": guided_fraction,
        "guided_hard_disagreement_limit": ANCHOR_GUIDED_DISAGREEMENT_LIMIT,
        "metric_absolute_differences": metric_differences,
        "metric_tolerances": ANCHOR_METRIC_TOLERANCES,
    }


def promotion_recommendation(
    manifest_levels: Sequence[Mapping[str, object]],
    completed_rows: Sequence[Mapping[str, object]],
    anchor_regression_pass: bool | None,
) -> dict[str, object]:
    """Apply the predeclared screen-to-n4 promotion rule."""
    rows_by_id = {str(row["level_id"]): row for row in completed_rows}
    ordered_ids = [str(level["id"]) for level in manifest_levels]
    missing = [level_id for level_id in ordered_ids if level_id not in rows_by_id]
    if missing:
        return {
            "status": "incomplete_screen",
            "missing_levels": missing,
            "selected_level": None,
            "promote_levels": [],
        }
    if anchor_regression_pass is not True or not bool(
        rows_by_id[ordered_ids[0]]["screen_gate_pass"]
    ):
        return {
            "status": "anchor_failed",
            "missing_levels": [],
            "selected_level": None,
            "promote_levels": [],
        }
    sweep_levels = list(manifest_levels[1:])
    passing = [
        level
        for level in sweep_levels
        if bool(rows_by_id[str(level["id"])]["screen_gate_pass"])
    ]
    if not passing:
        return {
            "status": "no_ambiguity_level_passed",
            "missing_levels": [],
            "selected_level": None,
            "promote_levels": [],
            "observability_bracket": [ordered_ids[0], ordered_ids[1]],
        }
    selected = max(passing, key=lambda level: int(level["order"]))
    selected_index = int(selected["order"])
    if selected_index + 1 < len(manifest_levels):
        neighbor = manifest_levels[selected_index + 1]
    else:
        neighbor = manifest_levels[selected_index - 1]
    return {
        "status": "candidate_identified_not_confirmed",
        "missing_levels": [],
        "selected_level": str(selected["id"]),
        "promote_levels": [str(selected["id"]), str(neighbor["id"])],
    }


def _report_markdown(summary: Mapping[str, object]) -> str:
    lines = [
        "# Phase-2b codebook-ambiguity seed-42 screen",
        "",
        "## Status",
        "",
        f"**{summary['decision']}**",
        "",
        (
            "This is a full-resolution truth-derived property-codebook ablation. "
            "It is not measured geophysics and a one-sample pass is not confirmation."
        ),
        "",
        "## Completed levels",
        "",
        "| Level | Role | Target exact group | Nearest distance | Pair gate | Label-9 IoU / P / R |",
        "|---|---|---|---:|---|---|",
    ]
    for row in summary["levels"]:
        lines.append(
            f"| {row['level_id']} | {row['role']} | "
            f"{row['target_exact_property_group']} | "
            f"{float(row['target_nearest_distance']):.6f} | "
            f"{row['screen_gate_pass']} | "
            f"{float(row['guided_target_iou']):.4f} / "
            f"{float(row['guided_target_precision']):.4f} / "
            f"{float(row['guided_target_recall']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Promotion rule result",
            "",
            f"- Status: `{summary['promotion']['status']}`.",
            f"- Selected level: `{summary['promotion'].get('selected_level')}`.",
            f"- Four-sample bracket: `{summary['promotion'].get('promote_levels', [])}`.",
            "- Continuous property loss is never sufficient for promotion.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    manifest_levels = list(manifest["levels"])
    known = {str(level["id"]): level for level in manifest_levels}
    selected_ids = list(args.levels or known)
    unknown = sorted(set(selected_ids) - set(known))
    if unknown:
        raise ValueError(f"unknown manifest levels: {unknown}")

    existing = list(args.output_dir.iterdir()) if args.output_dir.exists() else []
    if existing and not args.overwrite:
        raise FileExistsError(
            f"output directory is non-empty; pass --overwrite: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    level_rows: list[dict[str, object]] = []
    anchor_regression: dict[str, object] | None = None
    missing_levels: list[str] = []
    for level_id in selected_ids:
        level = known[level_id]
        config_path = _resolved_config_path(args.manifest, level)
        pair_root = args.runs_root / level_id / args.run_name
        baseline_dir = pair_root / args.baseline_name
        guided_dir = pair_root / args.guided_name
        if not baseline_dir.exists() and not guided_dir.exists():
            missing_levels.append(level_id)
            continue
        if not (baseline_dir / "config.json").is_file() or not (
            guided_dir / "config.json"
        ).is_file():
            raise ValueError(f"level {level_id} has an incomplete run directory")
        baseline_config = read_json(baseline_dir / "config.json")
        guided_config = read_json(guided_dir / "config.json")
        _validate_pair_configs(
            baseline_config,
            guided_config,
            config_path,
            args.seed,
            args.n_samples,
            args.n_steps,
        )

        property_config = read_json(config_path)
        table, channel_weights, _ = property_table_from_config(
            property_config,
            num_categories=15,
        )
        diagnostics = property_codebook_diagnostics(
            table,
            channel_weights,
            target_raw_label=9,
        )
        expected_susceptibility = float(level["label9_susceptibility"])
        if not math.isclose(
            float(diagnostics["target_property_values"][1]),
            expected_susceptibility,
            rel_tol=1e-6,
            abs_tol=1e-9,
        ):
            raise ValueError(f"level {level_id} target susceptibility differs from manifest")

        baseline_metrics = _index_rows(
            read_numeric_csv(guided_dir / "paired_baseline_metrics.csv")
        )
        guided_metrics = _index_rows(read_numeric_csv(guided_dir / "sample_metrics.csv"))
        deltas = _index_rows(read_numeric_csv(guided_dir / "paired_deltas.csv"))
        class_deltas = read_numeric_csv(guided_dir / "paired_per_class_deltas.csv")
        components = read_numeric_csv(
            guided_dir / "paired_truth_component_recovery_deltas.csv"
        )
        endpoints = _last_trace_rows(
            read_numeric_csv(guided_dir / "guidance_trace.csv"),
            args.n_samples,
            args.n_steps,
        )
        expected_ids = set(range(args.n_samples))
        if set(baseline_metrics) != expected_ids or set(guided_metrics) != expected_ids:
            raise ValueError(f"level {level_id} sample metric IDs are incomplete")

        pair_audits: list[dict[str, object]] = []
        for sample_id in range(args.n_samples):
            audit = sample_gate_audit(
                guided_metrics[sample_id],
                deltas[sample_id],
                [row for row in class_deltas if int(row["sample_id"]) == sample_id],
                [row for row in components if int(row["sample_id"]) == sample_id],
                float(endpoints[sample_id]["hard_change_fraction"]),
            )
            pair_audits.append(audit)
        first = guided_metrics[0]
        first_delta = deltas[0]
        level_rows.append(
            {
                "level_id": level_id,
                "order": int(level["order"]),
                "role": str(level["role"]),
                "property_config_sha256": runtime.file_sha256(config_path),
                "label9_susceptibility": expected_susceptibility,
                "unique_property_vectors": diagnostics["unique_property_vectors"],
                "target_exact_property_group": diagnostics[
                    "target_exact_property_group"
                ],
                "target_nearest_labels": diagnostics["target_nearest_raw_labels"],
                "target_nearest_distance": diagnostics[
                    "target_nearest_range_normalized_distance"
                ],
                "strict_pairing": True,
                "pair_gate_pass_count": sum(bool(audit["passed"]) for audit in pair_audits),
                "screen_gate_pass": all(bool(audit["passed"]) for audit in pair_audits),
                "guided_global_voxel_accuracy": first["global_voxel_accuracy"],
                "delta_global_voxel_accuracy": first_delta[
                    "delta_global_voxel_accuracy"
                ],
                "guided_truth_present_mean_iou": first["truth_present_mean_iou"],
                "delta_truth_present_mean_iou": first_delta[
                    "delta_truth_present_mean_iou"
                ],
                "guided_hard_property_loss": first["hard_property_loss"],
                "delta_hard_property_loss": first_delta["delta_hard_property_loss"],
                "guided_target_iou": first["target_iou"],
                "guided_target_precision": first["target_precision"],
                "guided_target_recall": first["target_recall"],
                "guided_target_volume": first["predicted_target_volume"],
                "improved_truth_present_classes": pair_audits[0][
                    "improved_truth_present_classes"
                ],
                "major_component_min_recall": pair_audits[0][
                    "major_component_min_recall"
                ],
                "major_component_mean_recall": pair_audits[0][
                    "major_component_mean_recall"
                ],
                "tiny_component_mass_fraction": first[
                    "target_tiny_component_mass_fraction_le_5"
                ],
                "top8_component_mass_fraction": first[
                    "target_top8_component_mass_fraction"
                ],
                "final_churn_fraction": pair_audits[0]["final_churn_fraction"],
            }
        )
        if level["role"] == "implementation_anchor":
            anchor_regression = _anchor_regression(
                pair_root,
                args.phase2a_reference_root,
                guided_metrics,
            )

    if not level_rows:
        raise FileNotFoundError("no completed Phase-2b screen levels were found")
    if (args.require_all_levels or args.levels) and missing_levels:
        raise FileNotFoundError(f"missing requested Phase-2b levels: {missing_levels}")
    level_rows.sort(key=lambda row: int(row["order"]))
    promotion = promotion_recommendation(
        manifest_levels,
        level_rows,
        bool(anchor_regression and anchor_regression["passed"]),
    )
    decision = {
        "incomplete_screen": "INCOMPLETE: run the remaining predeclared levels",
        "anchor_failed": "BLOCKED: Phase-2b implementation anchor regression failed",
        "no_ambiguity_level_passed": "SCREEN RESULT: no ambiguous codebook retained the full gate",
        "candidate_identified_not_confirmed": "SCREEN RESULT: n=4 bracket identified; not yet confirmed",
    }[str(promotion["status"])]
    summary = {
        "decision": decision,
        "scope": "truth-derived full-resolution property-codebook ambiguity screen",
        "is_measured_geophysics": False,
        "manifest": str(args.manifest),
        "manifest_sha256": runtime.file_sha256(args.manifest),
        "seed": args.seed,
        "n_samples": args.n_samples,
        "n_steps": args.n_steps,
        "alpha": 0.25,
        "max_guidance_ratio": 0.25,
        "completed_level_ids": [row["level_id"] for row in level_rows],
        "missing_level_ids": [
            str(level["id"])
            for level in manifest_levels
            if str(level["id"]) not in {row["level_id"] for row in level_rows}
        ],
        "anchor_regression": anchor_regression,
        "levels": level_rows,
        "promotion": promotion,
        "limitations": [
            "single-sample screen is not a Phase-2b confirmation",
            "property target remains truth-derived and full-resolution",
            "controlled codebook values are not calibrated petrophysics",
            "continuous property loss cannot replace hard-label and geometry gates",
        ],
    }
    write_csv(args.output_dir / "level_summary.csv", level_rows)
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "REPORT.md").write_text(
        _report_markdown(summary),
        encoding="utf-8",
    )
    print(args.output_dir / "REPORT.md")
    print(decision)


if __name__ == "__main__":
    main()
