#!/usr/bin/env python3
"""Audit one completed Phase-3 seed-42 spatial-property strict pair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import inference_runtime as runtime
from guidance.spatial_property import tensor_sha256
from scripts.stage2.summarize_phase2a import (
    _index_rows,
    _last_trace_rows,
    read_json,
    read_numeric_csv,
    sample_gate_audit,
    write_json,
)
from scripts.stage3.run_spatial_property_guidance import (
    PHASE3_STAGE,
    paired_spatial_property_config_verdict,
)


MANIFEST_SCHEMA = "phase3_spatial_property_sweep_v1"
OBSERVATION_TENSORS = {
    "observation_values.pt": "observation_values_sha256",
    "observation_noiseless.pt": "noiseless_observation_sha256",
    "observation_confidence.pt": "observation_confidence_sha256",
    "observation_noise.pt": "observation_noise_sha256",
}


def parse_args() -> argparse.Namespace:
    experiment_root = PROJECT_DIR / "experiments/stage3_spatial_property"
    parser = argparse.ArgumentParser(
        description="Audit one completed Phase-3 Gaussian screen level.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=experiment_root / "configs/gaussian_sweep_manifest_v1.json",
    )
    parser.add_argument("--level", required=True)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=experiment_root / "runs/cond_generation_0/phase3_spatial_property_v1",
    )
    parser.add_argument("--run-name", default="seed42_n1_s32_a025_c025")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=32)
    parser.add_argument(
        "--phase2a-reference-root",
        type=Path,
        default=PROJECT_DIR
        / "experiments/stage2_property/runs/cond_generation_0"
        / "ideal_density_susceptibility_label9_contrast_v1/phase2a_v1"
        / "seed42_n4_s32_a025_c025/baseline",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_manifest(path: Path) -> tuple[dict[str, object], dict[str, Mapping[str, object]]]:
    manifest = read_json(path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"manifest schema must be {MANIFEST_SCHEMA!r}")
    levels = manifest.get("levels")
    if not isinstance(levels, list) or not levels:
        raise ValueError("manifest must contain levels")
    indexed = {str(level.get("id")): level for level in levels}
    if len(indexed) != len(levels) or "" in indexed:
        raise ValueError("manifest level IDs must be non-empty and unique")
    if [int(level["order"]) for level in levels] != list(range(len(levels))):
        raise ValueError("manifest levels must retain contiguous frozen order")
    return manifest, indexed


def _validate_config(
    name: str,
    config: Mapping[str, object],
    *,
    observation_config_hash: str,
    seed: int,
    n_samples: int,
    n_steps: int,
    alpha: float,
) -> None:
    expected = {
        "stage": PHASE3_STAGE,
        "seed": seed,
        "n_samples": n_samples,
        "n_steps": n_steps,
        "run_status": "completed",
        "samples_written": n_samples,
        "ema_applied": True,
        "model_weight_source": "ema",
        "max_post_projection_condition_violations": 0,
        "observation_config_sha256": observation_config_hash,
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(f"{name} {field}={config.get(field)!r}, expected {value!r}")
    if float(config.get("alpha", float("nan"))) != alpha:
        raise ValueError(f"{name} alpha must be {alpha}")
    if float(config.get("max_guidance_ratio", float("nan"))) != 0.25:
        raise ValueError(f"{name} guidance cap must be 0.25")


def _validate_observation_assets(
    baseline_dir: Path,
    guided_dir: Path,
    baseline_config: Mapping[str, object],
    guided_config: Mapping[str, object],
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for filename, field in OBSERVATION_TENSORS.items():
        baseline = runtime.load_tensor(baseline_dir / filename)
        guided = runtime.load_tensor(guided_dir / filename)
        if not torch.equal(baseline, guided):
            raise ValueError(f"paired observation asset differs: {filename}")
        digest = tensor_sha256(baseline)
        if digest != baseline_config.get(field) or digest != guided_config.get(field):
            raise ValueError(f"saved observation hash mismatch: {field}")
        hashes[field] = digest
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
            raise ValueError("Phase-3/Phase-2a baseline sample shapes differ")
        mismatches.append(int((current != reference).sum().item()))
    return {
        "passed": all(value == 0 for value in mismatches),
        "hard_mismatch_voxels_by_sample": mismatches,
        "reference_dir": str(reference_dir),
    }


def _report(summary: Mapping[str, object]) -> str:
    metrics = summary["aggregate"]["guided_mean"]
    delta = summary["aggregate"]["delta_mean"]
    audit = summary["aggregate"]["gate_mean"]
    return "\n".join(
        [
            f"# Phase-3 spatial screen: {summary['level']}",
            "",
            "## Decision",
            "",
            f"**{summary['decision']}**",
            "",
            "This is a truth-derived degraded 3-D property observation, not measured geophysics.",
            "",
            "## Strict pair",
            "",
            f"- Pairing and observation hashes: `{summary['strict_pairing']}`.",
            "- Phase-2a alpha-zero hard regression: "
            f"`{summary['phase2_baseline_regression']['passed']}`.",
            f"- Hard conditions exact: `{summary['conditions_exact']}`.",
            f"- Complete hard gates: `{summary['pair_gate_pass_count']}/{summary['n_samples']}`.",
            "",
            "## Hard result",
            "",
            f"- Global accuracy delta: `{float(delta['delta_global_voxel_accuracy']):.6f}`.",
            f"- Truth-present mIoU delta: `{float(delta['delta_truth_present_mean_iou']):.6f}`.",
            f"- Hard observation loss delta: `{float(delta['delta_hard_observation_loss']):.6f}`.",
            "- Label-9 IoU / precision / recall: "
            f"`{float(metrics['target_iou']):.4f}` / "
            f"`{float(metrics['target_precision']):.4f}` / "
            f"`{float(metrics['target_recall']):.4f}`.",
            f"- Improved truth-present classes: `{audit['improved_truth_present_classes']}`.",
            "- Major-component minimum / mean recall: "
            f"`{float(audit['major_component_min_recall']):.4f}` / "
            f"`{float(audit['major_component_mean_recall']):.4f}`.",
            f"- Final hard churn fraction: `{float(audit['final_churn_fraction']):.6f}`.",
            "",
            "A lower continuous observation loss alone is not a pass.",
            "",
        ]
    )


def _mean(rows: list[Mapping[str, object]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)


def _classification(
    *,
    n_samples: int,
    pass_count: int,
    baseline_regression_passed: bool,
) -> tuple[str, str]:
    if not baseline_regression_passed:
        return "anchor_regression_failure", "BLOCKED: alpha-zero regression failed"
    if n_samples == 1:
        if pass_count == 1:
            return "single_sample_pass", "PASS: single-sample screen gate; not yet confirmed"
        return "single_sample_failure", "FAIL: do not promote this level"
    if n_samples == 4:
        if pass_count == 4:
            return "confirmed_seed42_pass", "PASS: confirmed seed-42 4/4 gate"
        if pass_count == 0:
            return "confirmed_seed42_failure", "FAIL: confirmed seed-42 0/4 gate"
        return (
            "seed42_transition",
            f"TRANSITION: seed-42 {pass_count}/4 gates; not a pass",
        )
    if pass_count == n_samples:
        return "all_pairs_pass", f"PASS: all {n_samples}/{n_samples} pair gates"
    return "incomplete_pair_gate", f"FAIL: only {pass_count}/{n_samples} pair gates"


def main() -> None:
    args = parse_args()
    manifest, levels = _load_manifest(args.manifest)
    if args.level not in levels:
        raise ValueError(f"unknown frozen level: {args.level}")
    level = levels[args.level]
    config_path = (args.manifest.parent / str(level["config"])).resolve()
    pair_root = args.runs_root / args.level / args.run_name
    baseline_dir = pair_root / "baseline"
    guided_dir = pair_root / "alpha025"
    baseline_config = read_json(baseline_dir / "config.json")
    guided_config = read_json(guided_dir / "config.json")
    paired, reason = paired_spatial_property_config_verdict(
        baseline_config,
        guided_config,
    )
    if not paired:
        raise ValueError(f"strict Phase-3 pairing failed: {reason}")
    if guided_config.get("pairing_validation", {}).get("paired") is not True:
        raise ValueError("saved guided pairing verdict is not true")
    observation_config_hash = runtime.file_sha256(config_path)
    _validate_config(
        "baseline",
        baseline_config,
        observation_config_hash=observation_config_hash,
        seed=args.seed,
        n_samples=args.n_samples,
        n_steps=args.n_steps,
        alpha=0.0,
    )
    _validate_config(
        "guided",
        guided_config,
        observation_config_hash=observation_config_hash,
        seed=args.seed,
        n_samples=args.n_samples,
        n_steps=args.n_steps,
        alpha=0.25,
    )
    if baseline_config.get("initial_noise_sha256") != guided_config.get(
        "initial_noise_sha256"
    ):
        raise ValueError("paired initial-noise hashes differ")
    observation_hashes = _validate_observation_assets(
        baseline_dir,
        guided_dir,
        baseline_config,
        guided_config,
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

    pair_audits: list[dict[str, object]] = []
    for sample_id in range(args.n_samples):
        observation_delta = dict(deltas[sample_id])
        observation_delta["delta_hard_property_loss"] = observation_delta[
            "delta_hard_observation_loss"
        ]
        pair_audits.append(
            sample_gate_audit(
                guided_metrics[sample_id],
                observation_delta,
                [row for row in class_deltas if int(row["sample_id"]) == sample_id],
                [row for row in components if int(row["sample_id"]) == sample_id],
                float(endpoints[sample_id]["hard_change_fraction"]),
            )
        )
    baseline_regression = _phase2_baseline_regression(
        baseline_dir,
        args.phase2a_reference_root,
        args.n_samples,
    )
    pass_count = sum(bool(audit["passed"]) for audit in pair_audits)
    classification, decision = _classification(
        n_samples=args.n_samples,
        pass_count=pass_count,
        baseline_regression_passed=bool(baseline_regression["passed"]),
    )
    baseline_rows = [baseline_metrics[index] for index in range(args.n_samples)]
    guided_rows = [guided_metrics[index] for index in range(args.n_samples)]
    delta_rows = [deltas[index] for index in range(args.n_samples)]
    metric_fields = (
        "global_voxel_accuracy",
        "truth_present_mean_iou",
        "global_mean_iou",
        "hard_observation_loss",
        "target_iou",
        "target_precision",
        "target_recall",
    )
    delta_fields = (
        "delta_global_voxel_accuracy",
        "delta_truth_present_mean_iou",
        "delta_global_mean_iou",
        "delta_hard_observation_loss",
        "delta_target_iou",
        "delta_target_precision",
        "delta_target_recall",
    )
    summary: dict[str, object] = {
        "decision": decision,
        "classification": classification,
        "level": args.level,
        "level_order": int(level["order"]),
        "level_role": str(level["role"]),
        "manifest": str(args.manifest),
        "manifest_sha256": runtime.file_sha256(args.manifest),
        "observation_config": str(config_path),
        "observation_config_sha256": observation_config_hash,
        "scope": "truth-derived spatially degraded 3-D property observation",
        "is_measured_geophysics": False,
        "strict_pairing": True,
        "pairing_reason": reason,
        "observation_hashes": observation_hashes,
        "phase2_baseline_regression": baseline_regression,
        "seed": args.seed,
        "n_samples": args.n_samples,
        "n_steps": args.n_steps,
        "conditions_exact": all(
            int(row["condition_violation_count"]) == 0 for row in guided_rows
        ),
        "pair_gate_pass_count": pass_count,
        "guided_metrics": guided_metrics[0],
        "baseline_metrics": baseline_metrics[0],
        "paired_deltas": deltas[0],
        "pair_gate": pair_audits[0],
        "guided_metrics_by_sample": guided_rows,
        "baseline_metrics_by_sample": baseline_rows,
        "paired_deltas_by_sample": delta_rows,
        "pair_gates": pair_audits,
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
                    pair_audits,
                    "improved_truth_present_classes",
                ),
                "major_component_min_recall": _mean(
                    pair_audits,
                    "major_component_min_recall",
                ),
                "major_component_mean_recall": _mean(
                    pair_audits,
                    "major_component_mean_recall",
                ),
                "final_churn_fraction": _mean(
                    pair_audits,
                    "final_churn_fraction",
                ),
            },
        },
        "limitations": [
            "single-sample screen is not multi-seed confirmation",
            "property codebook remains the distinct truth-derived Phase-2a oracle",
            "spatial observation is not acquisition-domain geophysics",
            "continuous observation loss alone is insufficient",
        ],
    }
    output_dir = args.output_dir or (
        PROJECT_DIR / "experiments/stage3_spatial_property/reports" / args.level / args.run_name
    )
    existing = list(output_dir.iterdir()) if output_dir.exists() else []
    if existing and not args.overwrite:
        raise FileExistsError(f"report directory is non-empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    print(_report(summary), end="")


if __name__ == "__main__":
    main()
