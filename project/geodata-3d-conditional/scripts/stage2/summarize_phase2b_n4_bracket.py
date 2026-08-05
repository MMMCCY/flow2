#!/usr/bin/env python3
"""Audit the Phase-2b seed-42 four-sample observability bracket."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping, Sequence


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import inference_runtime as runtime
from scripts.stage2.summarize_phase2a import (
    _aggregate_metric,
    _index_rows,
    _last_trace_rows,
    read_json,
    read_numeric_csv,
    sample_gate_audit,
    write_csv,
    write_json,
)
from scripts.stage2.summarize_phase2b_screen import (
    _resolved_config_path,
    _validate_pair_configs,
    load_manifest,
)


BRACKET_LEVELS = ("paired_c025", "paired_c010")
N_SAMPLES = 4


def parse_args() -> argparse.Namespace:
    experiment_root = PROJECT_DIR / "experiments/stage2_property"
    parser = argparse.ArgumentParser(
        description="Audit completed Phase-2b seed-42 n=4 bracket levels.",
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
    parser.add_argument("--n-steps", type=int, default=32)
    parser.add_argument("--run-name", default="seed42_n4_s32_a025_c025")
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--guided-name", default="alpha025")
    parser.add_argument(
        "--require-both-levels",
        action="store_true",
        help="Fail unless both frozen bracket levels are complete.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=experiment_root
        / "reports/phase2b_codebook_ambiguity_v1_n4_bracket_seed42",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def classify_n4_level(
    pair_gate_passes: Sequence[bool],
    diversity_pass: bool,
) -> str:
    """Classify one frozen four-sample level without post-hoc thresholds."""
    if len(pair_gate_passes) != N_SAMPLES:
        raise ValueError(f"n=4 classification requires {N_SAMPLES} pair gates")
    pass_count = sum(bool(value) for value in pair_gate_passes)
    if pass_count == N_SAMPLES and diversity_pass:
        return "confirmed_seed42_pass"
    if pass_count == N_SAMPLES:
        return "diversity_gate_failure"
    if pass_count == 0:
        return "confirmed_seed42_failure"
    return "transition_region"


def n4_candidate_selection(
    manifest_levels: Sequence[Mapping[str, object]],
    level_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Select the most degraded 4/4 level and retain the lower result."""
    order = {str(level["id"]): int(level["order"]) for level in manifest_levels}
    confirmed = [
        row
        for row in level_rows
        if row["classification"] == "confirmed_seed42_pass"
    ]
    if not confirmed:
        return {
            "status": "no_seed42_n4_candidate",
            "candidate_level": None,
            "promote_to_multiseed": False,
        }
    candidate = max(confirmed, key=lambda row: order[str(row["level_id"])])
    candidate_order = order[str(candidate["level_id"])]
    lower = [
        row for row in level_rows if order[str(row["level_id"])] > candidate_order
    ]
    lower_row = min(lower, key=lambda row: order[str(row["level_id"])]) if lower else None
    return {
        "status": "seed42_candidate_confirmed",
        "candidate_level": str(candidate["level_id"]),
        "candidate_classification": candidate["classification"],
        "adjacent_lower_level": (
            str(lower_row["level_id"]) if lower_row is not None else None
        ),
        "adjacent_lower_classification": (
            lower_row["classification"] if lower_row is not None else None
        ),
        "promote_to_multiseed": True,
    }


def _mean_metric(
    baseline_rows: Sequence[Mapping[str, object]],
    guided_rows: Sequence[Mapping[str, object]],
    field: str,
) -> tuple[float, float, float]:
    aggregate = _aggregate_metric(baseline_rows, guided_rows, field)
    return (
        float(aggregate["baseline"]["mean"]),
        float(aggregate["guided"]["mean"]),
        float(aggregate["delta"]["mean"]),
    )


def audit_n4_level(
    *,
    level_id: str,
    level: Mapping[str, object],
    config_path: Path,
    pair_root: Path,
    seed: int,
    n_steps: int,
    baseline_name: str = "baseline",
    guided_name: str = "alpha025",
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Apply the frozen Phase-2a pair and diversity gates to one n=4 level."""
    baseline_dir = pair_root / baseline_name
    guided_dir = pair_root / guided_name
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
        seed,
        N_SAMPLES,
        n_steps,
    )
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
        N_SAMPLES,
        n_steps,
    )
    expected_ids = set(range(N_SAMPLES))
    if not (
        set(baseline_metrics) == set(guided_metrics) == set(deltas) == expected_ids
    ):
        raise ValueError(f"level {level_id} sample metric IDs are incomplete")

    pair_rows: list[dict[str, object]] = []
    pair_passes: list[bool] = []
    for sample_id in range(N_SAMPLES):
        audit = sample_gate_audit(
            guided_metrics[sample_id],
            deltas[sample_id],
            [row for row in class_deltas if int(row["sample_id"]) == sample_id],
            [row for row in components if int(row["sample_id"]) == sample_id],
            float(endpoints[sample_id]["hard_change_fraction"]),
        )
        pair_passes.append(bool(audit["passed"]))
        row = {
            "level_id": level_id,
            "sample_id": sample_id,
            "pair_gate_pass": audit["passed"],
            "improved_truth_present_classes": audit[
                "improved_truth_present_classes"
            ],
            "major_component_min_recall": audit["major_component_min_recall"],
            "major_component_mean_recall": audit["major_component_mean_recall"],
            "target_volume_fraction": audit["target_volume_fraction"],
            "final_churn_fraction": audit["final_churn_fraction"],
            **guided_metrics[sample_id],
            **deltas[sample_id],
        }
        for gate_name, passed in audit["checks"].items():
            row[f"gate_{gate_name}"] = passed
        pair_rows.append(row)

    ensemble = read_json(guided_dir / "ensemble_summary.json")["current"]
    diversity_pass = (
        int(ensemble["unique_decoded_samples"]) == N_SAMPLES
        and float(ensemble["mean_pairwise_hard_disagreement_outside_roi"]) > 0
    )
    classification = classify_n4_level(pair_passes, diversity_pass)
    baseline_values = list(baseline_metrics.values())
    guided_values = list(guided_metrics.values())
    _, _, mean_accuracy_delta = _mean_metric(
        baseline_values,
        guided_values,
        "global_voxel_accuracy",
    )
    _, _, mean_miou_delta = _mean_metric(
        baseline_values,
        guided_values,
        "truth_present_mean_iou",
    )
    _, _, mean_property_delta = _mean_metric(
        baseline_values,
        guided_values,
        "hard_property_loss",
    )
    _, mean_iou, _ = _mean_metric(
        baseline_values,
        guided_values,
        "target_iou",
    )
    _, mean_precision, _ = _mean_metric(
        baseline_values,
        guided_values,
        "target_precision",
    )
    _, mean_recall, _ = _mean_metric(
        baseline_values,
        guided_values,
        "target_recall",
    )
    level_row = {
        "level_id": level_id,
        "order": int(level["order"]),
        "property_config_sha256": runtime.file_sha256(config_path),
        "pair_gate_pass_count": sum(pair_passes),
        "diversity_gate_pass": diversity_pass,
        "unique_guided_samples": int(ensemble["unique_decoded_samples"]),
        "outside_roi_disagreement": float(
            ensemble["mean_pairwise_hard_disagreement_outside_roi"]
        ),
        "classification": classification,
        "mean_delta_global_voxel_accuracy": mean_accuracy_delta,
        "mean_delta_truth_present_mean_iou": mean_miou_delta,
        "mean_delta_hard_property_loss": mean_property_delta,
        "mean_guided_target_iou": mean_iou,
        "mean_guided_target_precision": mean_precision,
        "mean_guided_target_recall": mean_recall,
    }
    return pair_rows, level_row


def _report_markdown(summary: Mapping[str, object]) -> str:
    lines = [
        "# Phase-2b seed-42 n=4 observability bracket",
        "",
        "## Decision",
        "",
        f"**{summary['decision']}**",
        "",
        "| Level | Pair gates | Diversity | Classification | Mean label-9 IoU / P / R |",
        "|---|---:|---|---|---|",
    ]
    for row in summary["levels"]:
        lines.append(
            f"| {row['level_id']} | {row['pair_gate_pass_count']}/4 | "
            f"{row['diversity_gate_pass']} | {row['classification']} | "
            f"{float(row['mean_guided_target_iou']):.4f} / "
            f"{float(row['mean_guided_target_precision']):.4f} / "
            f"{float(row['mean_guided_target_recall']):.4f} |"
        )
    selection = summary["selection"]
    lines.extend(
        [
            "",
            "## Frozen selection",
            "",
            f"- Candidate: `{selection.get('candidate_level')}`.",
            f"- Adjacent lower level: `{selection.get('adjacent_lower_level')}` "
            f"(`{selection.get('adjacent_lower_classification')}`).",
            f"- Multi-seed promotion authorized: {selection['promote_to_multiseed']}.",
            "- This remains a truth-derived full-resolution property experiment, not measured geophysics.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    known = {str(level["id"]): level for level in manifest["levels"]}
    selected_ids = tuple(args.levels or BRACKET_LEVELS)
    if any(level_id not in BRACKET_LEVELS for level_id in selected_ids):
        raise ValueError(f"n=4 bracket levels are fixed to {BRACKET_LEVELS}")

    existing = list(args.output_dir.iterdir()) if args.output_dir.exists() else []
    if existing and not args.overwrite:
        raise FileExistsError(
            f"output directory is non-empty; pass --overwrite: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pair_rows: list[dict[str, object]] = []
    level_rows: list[dict[str, object]] = []
    missing: list[str] = []
    for level_id in selected_ids:
        level = known[level_id]
        config_path = _resolved_config_path(args.manifest, level)
        pair_root = args.runs_root / level_id / args.run_name
        baseline_dir = pair_root / args.baseline_name
        guided_dir = pair_root / args.guided_name
        if not baseline_dir.exists() and not guided_dir.exists():
            missing.append(level_id)
            continue
        current_pair_rows, level_row = audit_n4_level(
            level_id=level_id,
            level=level,
            config_path=config_path,
            pair_root=pair_root,
            seed=args.seed,
            n_steps=args.n_steps,
            baseline_name=args.baseline_name,
            guided_name=args.guided_name,
        )
        pair_rows.extend(current_pair_rows)
        level_rows.append(level_row)

    if not level_rows:
        raise FileNotFoundError("no completed Phase-2b n=4 bracket levels found")
    if (args.require_both_levels or args.levels) and missing:
        raise FileNotFoundError(f"missing requested n=4 bracket levels: {missing}")
    level_rows.sort(key=lambda row: int(row["order"]))
    selection = n4_candidate_selection(manifest["levels"], level_rows)
    if missing:
        selection = {
            **selection,
            "status": "incomplete_n4_bracket",
            "promote_to_multiseed": False,
        }
        decision = "INCOMPLETE: run the remaining n=4 bracket level"
    elif selection["promote_to_multiseed"]:
        decision = "SEED42 N=4 RESULT: candidate confirmed for multi-seed testing"
    else:
        decision = "SEED42 N=4 RESULT: no level qualifies for multi-seed testing"
    summary = {
        "decision": decision,
        "scope": "Phase-2b seed-42 four-sample codebook observability bracket",
        "is_measured_geophysics": False,
        "seed": args.seed,
        "n_samples": N_SAMPLES,
        "n_steps": args.n_steps,
        "alpha": 0.25,
        "max_guidance_ratio": 0.25,
        "completed_levels": [row["level_id"] for row in level_rows],
        "missing_levels": missing,
        "levels": level_rows,
        "selection": selection,
    }
    write_csv(args.output_dir / "paired_samples.csv", pair_rows)
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
