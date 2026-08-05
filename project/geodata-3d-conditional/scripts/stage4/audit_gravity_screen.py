#!/usr/bin/env python3
"""Audit a completed strict Phase-4a gravity pair."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import inference_runtime as runtime
from guidance.gravity import tensor_sha256
from scripts.stage2.summarize_phase2a import sample_gate_audit
from scripts.stage4.run_gravity_guidance import (
    PHASE4_STAGE,
    SAVED_PAIR_TENSORS,
    paired_gravity_config_verdict,
    read_json,
    write_json,
)


def _numeric(value: str) -> object:
    if value == "":
        return value
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        integer = int(value)
        if str(integer) == value:
            return integer
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def read_numeric_csv(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as stream:
        return [
            {field: _numeric(value) for field, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def _index_rows(
    rows: Sequence[Mapping[str, object]],
    key: str = "sample_id",
) -> dict[int, Mapping[str, object]]:
    indexed = {int(row[key]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"duplicate {key} values")
    return indexed


def _last_trace_rows(
    rows: Sequence[Mapping[str, object]],
    n_samples: int,
    n_steps: int,
) -> dict[int, Mapping[str, object]]:
    indexed: dict[int, Mapping[str, object]] = {}
    expected_count = n_samples * n_steps
    if len(rows) != expected_count:
        raise ValueError(f"trace has {len(rows)} rows, expected {expected_count}")
    for row in rows:
        if int(row["step"]) == n_steps - 1:
            indexed[int(row["sample_id"])] = row
    if set(indexed) != set(range(n_samples)):
        raise ValueError("trace endpoint sample IDs are incomplete")
    if any(
        not math.isfinite(value)
        for row in rows
        for value in row.values()
        if isinstance(value, float)
    ):
        raise ValueError("trace contains non-finite numeric values")
    return indexed


def _mean(rows: Sequence[Mapping[str, object]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)


def _classification(
    n_samples: int,
    pass_count: int,
    baseline_regression: bool,
) -> tuple[str, str]:
    if not baseline_regression:
        return "baseline_regression_failure", "BLOCKED: alpha-zero regression failed"
    if n_samples == 1:
        if pass_count == 1:
            return "single_sample_pass", "PASS: single-sample gravity screen; not confirmed"
        return "single_sample_failure", "FAIL: single-sample gravity screen"
    if n_samples == 4:
        if pass_count == 4:
            return "confirmed_seed42_pass", "PASS: confirmed seed-42 4/4 gravity gate"
        if pass_count == 0:
            return "confirmed_seed42_failure", "FAIL: confirmed seed-42 0/4 gravity gate"
        return "seed42_transition", f"TRANSITION: seed-42 {pass_count}/4; not a pass"
    if pass_count == n_samples:
        return "all_pairs_pass", f"PASS: all {n_samples}/{n_samples} gravity gates"
    return "incomplete_gate", f"FAIL: only {pass_count}/{n_samples} gravity gates"


def parse_args() -> argparse.Namespace:
    experiment_root = PROJECT_DIR / "experiments/stage4_gravity"
    parser = argparse.ArgumentParser(
        description="Audit one completed Phase-4a gravity strict pair.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=experiment_root / "runs/cond_generation_0/phase4a_gravity_v1",
    )
    parser.add_argument("--run-name", default="seed42_n1_s32_a025_c025")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=32)
    parser.add_argument("--guided-name", default="alpha025")
    parser.add_argument(
        "--observation-dir",
        type=Path,
        default=experiment_root
        / "observations/cond_generation_0/distinct_upper_bound_v1_fix2",
    )
    parser.add_argument(
        "--phase2a-reference-root",
        type=Path,
        default=PROJECT_DIR
        / "experiments/stage2_property/runs/cond_generation_0"
        / "ideal_density_susceptibility_label9_contrast_v1/phase2a_v1"
        / "seed42_n4_s32_a025_c025/baseline",
    )
    parser.add_argument("--reranking-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _validate_config(
    name: str,
    config: Mapping[str, object],
    *,
    manifest_hash: str,
    seed: int,
    n_samples: int,
    n_steps: int,
    alpha: float,
    controller_level_id: str,
    controller_intended_alpha: float,
) -> None:
    expected = {
        "stage": PHASE4_STAGE,
        "seed": seed,
        "n_samples": n_samples,
        "n_steps": n_steps,
        "run_status": "completed",
        "samples_written": n_samples,
        "ema_applied": True,
        "model_weight_source": "ema",
        "max_post_projection_condition_violations": 0,
        "observation_manifest_sha256": manifest_hash,
        "gravity_forward_mode": "full_support_rectangular_prism_fft_v1",
        "gravity_output_unit": "mGal",
        "truth_derived": True,
        "measured_geophysics": False,
        "inverse_crime": True,
        "controller_level_id": controller_level_id,
        "controller_intended_alpha": controller_intended_alpha,
    }
    for field, expected_value in expected.items():
        if config.get(field) != expected_value:
            raise ValueError(
                f"{name} {field}={config.get(field)!r}, expected {expected_value!r}"
            )
    if float(config.get("alpha", float("nan"))) != alpha:
        raise ValueError(f"{name} alpha must be {alpha}")
    if float(config.get("max_guidance_ratio", float("nan"))) != 0.25:
        raise ValueError(f"{name} guidance cap must be 0.25")


def _validate_pair_tensors(
    baseline_dir: Path,
    guided_dir: Path,
    baseline_config: Mapping[str, object],
    guided_config: Mapping[str, object],
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for filename, config_field in SAVED_PAIR_TENSORS.items():
        baseline = runtime.load_tensor(baseline_dir / filename)
        guided = runtime.load_tensor(guided_dir / filename)
        if not torch.equal(baseline, guided):
            raise ValueError(f"paired tensor differs: {filename}")
        digest = tensor_sha256(baseline)
        if digest != baseline_config.get(config_field) or digest != guided_config.get(
            config_field
        ):
            raise ValueError(f"saved pair tensor hash mismatch: {config_field}")
        hashes[config_field] = digest
    return hashes


def _phase2_baseline_regression(
    baseline_dir: Path,
    reference_dir: Path,
    n_samples: int,
) -> dict[str, object]:
    mismatches: list[int] = []
    for sample_id in range(n_samples):
        current = runtime.load_tensor(baseline_dir / f"sample_{sample_id}.pt")
        reference = runtime.load_tensor(reference_dir / f"sample_{sample_id}.pt")
        if current.shape != reference.shape:
            raise ValueError("Phase-4/Phase-2a baseline sample shapes differ")
        mismatches.append(int((current != reference).sum()))
    return {
        "passed": all(value == 0 for value in mismatches),
        "hard_mismatch_voxels_by_sample": mismatches,
        "reference_dir": str(reference_dir),
    }


def _render_report(summary: Mapping[str, object]) -> str:
    guided = summary["aggregate"]["guided_mean"]
    delta = summary["aggregate"]["delta_mean"]
    gate = summary["aggregate"]["gate_mean"]
    reranking = summary["reranking"]
    return "\n".join(
        [
            "# Phase-4a gravity screen",
            "",
            "## Decision",
            "",
            f"**{summary['decision']}**",
            "",
            "This is a truth-derived full-grid synthetic inverse-crime gravity field, not measured geophysics.",
            "",
            "## Strict pair",
            "",
            f"- Pairing and immutable gravity hashes: `{summary['strict_pairing']}`.",
            f"- Phase-2a alpha-zero hard regression: `{summary['phase2_baseline_regression']['passed']}`.",
            f"- Hard conditions exact: `{summary['conditions_exact']}`.",
            f"- Complete geology-plus-gravity gates: `{summary['pair_gate_pass_count']}/{summary['n_samples']}`.",
            f"- Post-hoc baseline reranking: `{reranking['status']}`.",
            "",
            "## Hard result",
            "",
            f"- Global accuracy delta: `{float(delta['delta_global_voxel_accuracy']):.6f}`.",
            f"- Truth-present mIoU delta: `{float(delta['delta_truth_present_mean_iou']):.6f}`.",
            f"- Hard gravity loss delta: `{float(delta['delta_hard_gravity_loss']):.6f}`.",
            f"- Hard gravity RMSE delta: `{float(delta['delta_hard_gravity_rmse_mgal']):.6f}` mGal.",
            "- Label-9 IoU / precision / recall: "
            f"`{float(guided['target_iou']):.4f}` / "
            f"`{float(guided['target_precision']):.4f}` / "
            f"`{float(guided['target_recall']):.4f}`.",
            f"- Improved truth-present classes: `{gate['improved_truth_present_classes']}`.",
            "- Major-component minimum / mean recall: "
            f"`{float(gate['major_component_min_recall']):.4f}` / "
            f"`{float(gate['major_component_mean_recall']):.4f}`.",
            f"- Final hard churn fraction: `{float(gate['final_churn_fraction']):.6f}`.",
            "",
            "A lower gravity residual alone is field fitting, not geological recovery.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    pair_root = args.runs_root / args.run_name
    baseline_dir = pair_root / "baseline"
    guided_dir = pair_root / args.guided_name
    baseline_config = read_json(baseline_dir / "config.json")
    guided_config = read_json(guided_dir / "config.json")
    paired, reason = paired_gravity_config_verdict(baseline_config, guided_config)
    if not paired:
        raise ValueError(f"strict Phase-4 pairing failed: {reason}")
    if guided_config.get("pairing_validation", {}).get("paired") is not True:
        raise ValueError("saved guided pairing verdict is not true")
    manifest_hash = runtime.file_sha256(args.observation_dir / "manifest.json")
    expected_guided_alpha = {
        "alpha025": 0.25,
        "alpha010": 0.10,
    }.get(args.guided_name)
    if expected_guided_alpha is None:
        raise ValueError("guided-name must be alpha025 or alpha010")
    _validate_config(
        "baseline",
        baseline_config,
        manifest_hash=manifest_hash,
        seed=args.seed,
        n_samples=args.n_samples,
        n_steps=args.n_steps,
        alpha=0.0,
        controller_level_id=f"{args.guided_name}_cap025",
        controller_intended_alpha=expected_guided_alpha,
    )
    _validate_config(
        "guided",
        guided_config,
        manifest_hash=manifest_hash,
        seed=args.seed,
        n_samples=args.n_samples,
        n_steps=args.n_steps,
        alpha=expected_guided_alpha,
        controller_level_id=f"{args.guided_name}_cap025",
        controller_intended_alpha=expected_guided_alpha,
    )
    if baseline_config.get("initial_noise_sha256") != guided_config.get(
        "initial_noise_sha256"
    ):
        raise ValueError("paired initial-noise hashes differ")
    pair_hashes = _validate_pair_tensors(
        baseline_dir, guided_dir, baseline_config, guided_config
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
        args.n_samples,
        args.n_steps,
    )
    expected_ids = set(range(args.n_samples))
    if set(baseline_metrics) != expected_ids or set(guided_metrics) != expected_ids:
        raise ValueError("sample metric IDs are incomplete")

    pair_gates: list[dict[str, object]] = []
    for sample_id in range(args.n_samples):
        gate_delta = dict(deltas[sample_id])
        gate_delta["delta_hard_property_loss"] = gate_delta[
            "delta_hard_gravity_loss"
        ]
        audit = sample_gate_audit(
            guided_metrics[sample_id],
            gate_delta,
            [row for row in class_deltas if int(row["sample_id"]) == sample_id],
            [row for row in components if int(row["sample_id"]) == sample_id],
            float(endpoints[sample_id]["hard_change_fraction"]),
        )
        audit["gravity_loss_improved"] = (
            float(gate_delta["delta_hard_gravity_loss"]) < 0
        )
        audit["passed"] = bool(audit["passed"] and audit["gravity_loss_improved"])
        pair_gates.append(audit)

    baseline_regression = _phase2_baseline_regression(
        baseline_dir, args.phase2a_reference_root, args.n_samples
    )
    pass_count = sum(bool(gate["passed"]) for gate in pair_gates)
    classification, decision = _classification(
        args.n_samples, pass_count, bool(baseline_regression["passed"])
    )
    baseline_rows = [baseline_metrics[index] for index in range(args.n_samples)]
    guided_rows = [guided_metrics[index] for index in range(args.n_samples)]
    delta_rows = [deltas[index] for index in range(args.n_samples)]
    metric_fields = (
        "global_voxel_accuracy",
        "truth_present_mean_iou",
        "global_mean_iou",
        "hard_gravity_loss",
        "hard_gravity_rmse_mgal",
        "hard_gravity_mae_mgal",
        "target_iou",
        "target_precision",
        "target_recall",
    )
    delta_fields = tuple(f"delta_{field}" for field in metric_fields)
    reranking: dict[str, object]
    if args.n_samples == 1:
        reranking = {
            "status": "not_applicable_single_sample",
            "reason": "reranking one sample cannot select among alternatives",
        }
    elif args.reranking_summary is None:
        reranking = {
            "status": "required_before_final_promotion",
            "reason": "run the post-hoc baseline reranking comparator",
        }
    else:
        reranking = read_json(args.reranking_summary)
        if reranking.get("strict_pairing") is not True:
            raise ValueError("reranking summary lacks strict pair validation")
        if int(reranking.get("n_samples", -1)) != args.n_samples:
            raise ValueError("reranking summary sample count differs")

    if args.n_samples > 1 and reranking["status"] != "completed":
        classification = "pending_reranking"
        decision = "PENDING: complete post-hoc baseline reranking before promotion"
    elif args.n_samples > 1 and not bool(
        reranking.get("comparisons", {}).get(
            "guided_mean_loss_below_reranked_baseline", False
        )
    ):
        classification = "reranking_not_beaten"
        decision = "FAIL: guided ensemble mean does not beat reranked baseline gravity residual"

    summary: dict[str, object] = {
        "decision": decision,
        "classification": classification,
        "scope": "truth-derived full-grid synthetic inverse-crime gravity",
        "is_measured_geophysics": False,
        "strict_pairing": True,
        "pairing_reason": reason,
        "pair_tensor_hashes": pair_hashes,
        "phase2_baseline_regression": baseline_regression,
        "seed": args.seed,
        "n_samples": args.n_samples,
        "n_steps": args.n_steps,
        "alpha": float(guided_config["alpha"]),
        "conditions_exact": all(
            int(row["condition_violation_count"]) == 0 for row in guided_rows
        ),
        "pair_gate_pass_count": pass_count,
        "pair_gates": pair_gates,
        "baseline_metrics_by_sample": baseline_rows,
        "guided_metrics_by_sample": guided_rows,
        "paired_deltas_by_sample": delta_rows,
        "reranking": reranking,
        "aggregate": {
            "baseline_mean": {
                field: _mean(baseline_rows, field) for field in metric_fields
            },
            "guided_mean": {
                field: _mean(guided_rows, field) for field in metric_fields
            },
            "delta_mean": {
                field: _mean(delta_rows, field) for field in delta_fields
            },
            "gate_mean": {
                "improved_truth_present_classes": _mean(
                    pair_gates, "improved_truth_present_classes"
                ),
                "major_component_min_recall": _mean(
                    pair_gates, "major_component_min_recall"
                ),
                "major_component_mean_recall": _mean(
                    pair_gates, "major_component_mean_recall"
                ),
                "final_churn_fraction": _mean(pair_gates, "final_churn_fraction"),
            },
        },
        "limitations": [
            "synthetic inverse-crime gravity is not measured geophysics",
            "density codebook gives label 9 deliberately strong observability",
            "surface gravity remains depth-nonunique",
            "continuous gravity loss alone is insufficient",
        ],
    }
    output_dir = args.output_dir or (
        PROJECT_DIR / "experiments/stage4_gravity/reports" / args.run_name
    )
    existing = list(output_dir.iterdir()) if output_dir.exists() else []
    if existing and not args.overwrite:
        raise FileExistsError(f"report directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "REPORT.md").write_text(_render_report(summary), encoding="utf-8")
    print(_render_report(summary))


if __name__ == "__main__":
    main()
