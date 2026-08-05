#!/usr/bin/env python3
"""Audit the single strictly paired Phase-5b flow bridge screen."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Mapping

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import inference_runtime as runtime
from guidance.probability_volume import tensor_sha256
from scripts.stage2.run_property_guidance import paired_property_config_verdict
from scripts.stage2.summarize_phase2a import (
    _index_rows,
    _last_trace_rows,
    read_json,
    read_numeric_csv,
    sample_gate_audit,
    write_json,
)
from scripts.stage4.audit_seismic_identifiability import validate_output_directory
from scripts.stage4.run_seismic_guidance import write_rows


PHASE5B_AUDIT_SCHEMA = "phase5b_inversion_property_bridge_audit_v1"
ASSET_FILES = (
    "property_table.pt",
    "target_properties.pt",
    "property_confidence.pt",
)


def parse_args() -> argparse.Namespace:
    experiment = PROJECT_DIR / "experiments/stage5_acoustic_inversion"
    parser = argparse.ArgumentParser(
        description="Audit the frozen Phase-5b seed-42 n=1 strict pair.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pair-root",
        type=Path,
        default=experiment
        / "runs/cond_generation_0/phase5b_inversion_property_bridge_v1"
        / "seed42_n1_s32_a025_c025",
    )
    parser.add_argument(
        "--phase2a-reference",
        type=Path,
        default=PROJECT_DIR
        / "experiments/stage2_property/runs/cond_generation_0"
        / "ideal_density_susceptibility_label9_contrast_v1/phase2a_v1"
        / "seed42_n4_s32_a025_c025/baseline/sample_0.pt",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _validate_run_config(
    name: str,
    config: Mapping[str, object],
    *,
    alpha: float,
) -> None:
    expected = {
        "stage": "phase5b_inversion_property_bridge_v1",
        "seed": 42,
        "n_samples": 1,
        "n_steps": 32,
        "run_status": "completed",
        "samples_written": 1,
        "ema_applied": True,
        "model_weight_source": "ema",
        "integrator": runtime.PAIRED_INTEGRATOR,
        "initial_noise_policy": runtime.INITIAL_NOISE_POLICY,
        "confidence_mode": "external_posterior_spread_v1",
        "property_sigmas": [0.0],
        "property_scale_weights": [1.0],
        "max_guidance_ratio": 0.25,
        "max_post_projection_condition_violations": 0,
        "external_property_manifest_schema": (
            "phase5b_inversion_property_assets_v1"
        ),
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(f"{name} {field}={config.get(field)!r}, expected {value!r}")
    if float(config.get("alpha", float("nan"))) != alpha:
        raise ValueError(f"{name} alpha must be {alpha}")


def _validate_pair_assets(
    baseline_dir: Path,
    guided_dir: Path,
    baseline: Mapping[str, object],
    guided: Mapping[str, object],
) -> None:
    for filename in ASSET_FILES:
        first = runtime.load_tensor(baseline_dir / filename)
        second = runtime.load_tensor(guided_dir / filename)
        if not torch.equal(first, second):
            raise ValueError(f"strict pair property asset differs: {filename}")
    fields = {
        "property_table.pt": "property_table_sha256",
        "target_properties.pt": "target_properties_sha256",
        "property_confidence.pt": "property_confidence_sha256",
    }
    for filename, field in fields.items():
        digest = tensor_sha256(runtime.load_tensor(baseline_dir / filename))
        if digest != baseline.get(field) or digest != guided.get(field):
            raise ValueError(f"strict pair tensor hash mismatch: {field}")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.pair_root / "audit"
    validate_output_directory(output_dir, overwrite=args.overwrite)
    baseline_dir = args.pair_root / "baseline"
    guided_dir = args.pair_root / "alpha025"
    baseline = read_json(baseline_dir / "config.json")
    guided = read_json(guided_dir / "config.json")
    _validate_run_config("baseline", baseline, alpha=0.0)
    _validate_run_config("guided", guided, alpha=0.25)
    paired, reason = paired_property_config_verdict(baseline, guided)
    if not paired:
        raise ValueError(f"Phase-5b strict pairing failed: {reason}")
    if guided.get("pairing_validation", {}).get("paired") is not True:
        raise ValueError("saved guided pairing validation is not true")
    if baseline.get("initial_noise_sha256") != guided.get("initial_noise_sha256"):
        raise ValueError("Phase-5b initial-noise hashes differ")
    _validate_pair_assets(baseline_dir, guided_dir, baseline, guided)

    baseline_sample = runtime.load_tensor(baseline_dir / "sample_0.pt")
    reference_sample = runtime.load_tensor(args.phase2a_reference)
    if baseline_sample.shape != reference_sample.shape:
        raise ValueError("Phase-5b/Phase-2a baseline sample shapes differ")
    alpha_zero_mismatches = int((baseline_sample != reference_sample).sum().item())
    alpha_zero_regression = alpha_zero_mismatches == 0

    baseline_metrics = _index_rows(
        read_numeric_csv(guided_dir / "paired_baseline_metrics.csv")
    )[0]
    guided_metrics = _index_rows(
        read_numeric_csv(guided_dir / "sample_metrics.csv")
    )[0]
    deltas = _index_rows(read_numeric_csv(guided_dir / "paired_deltas.csv"))[0]
    class_deltas = read_numeric_csv(guided_dir / "paired_per_class_deltas.csv")
    components = read_numeric_csv(
        guided_dir / "paired_truth_component_recovery_deltas.csv"
    )
    endpoint = _last_trace_rows(
        read_numeric_csv(guided_dir / "guidance_trace.csv"), 1, 32
    )[0]
    hard_gate = sample_gate_audit(
        guided_metrics,
        deltas,
        class_deltas,
        components,
        float(endpoint["hard_change_fraction"]),
    )
    observation_improved = float(deltas["delta_hard_observation_loss"]) < 0
    complete_checks = {
        "strict_pairing_and_assets": True,
        "alpha_zero_phase2a_hard_regression": alpha_zero_regression,
        "conditions_exact": (
            int(baseline["max_post_projection_condition_violations"]) == 0
            and int(guided["max_post_projection_condition_violations"]) == 0
        ),
        "hard_inversion_observation_loss_improved": observation_improved,
        **hard_gate["checks"],
    }
    passed = all(complete_checks.values())
    summary = {
        "schema": PHASE5B_AUDIT_SCHEMA,
        "decision": (
            "PASS: single-pair gate; eligible only for a separately frozen seed-42 n=4 confirmation"
            if passed
            else "FAIL: close the no-training inversion-property flow bridge"
        ),
        "passed": passed,
        "n_pairs": 1,
        "checks": complete_checks,
        "strict_pairing_reason": reason,
        "alpha_zero_regression": {
            "passed": alpha_zero_regression,
            "hard_mismatch_voxels": alpha_zero_mismatches,
            "reference": str(args.phase2a_reference),
        },
        "baseline_metrics": baseline_metrics,
        "guided_metrics": guided_metrics,
        "deltas": deltas,
        "gate_diagnostics": {
            key: value for key, value in hard_gate.items() if key != "checks"
        },
        "endpoint": endpoint,
        "continuous_loss_not_sufficient": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", summary)
    write_rows(output_dir / "class_deltas.csv", class_deltas)
    write_rows(output_dir / "truth_component_deltas.csv", components)
    lines = [
        "# Phase 5b inversion-property flow bridge",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        f"- Alpha-zero mismatch voxels: `{alpha_zero_mismatches}`.",
        "- Hard inversion-observation loss delta: "
        f"`{float(deltas['delta_hard_observation_loss']):.8g}`.",
        "- Global accuracy delta: "
        f"`{float(deltas['delta_global_voxel_accuracy']):.8g}`.",
        "- Truth-present mIoU delta: "
        f"`{float(deltas['delta_truth_present_mean_iou']):.8g}`.",
        "- Guided label-9 IoU / precision / recall: "
        f"`{float(guided_metrics['target_iou']):.5f}` / "
        f"`{float(guided_metrics['target_precision']):.5f}` / "
        f"`{float(guided_metrics['target_recall']):.5f}`.",
        "",
        "A lower continuous or hard property loss alone is not a pass.",
        "",
        "## Checks",
        "",
    ]
    lines.extend(
        f"- {'PASS' if value else 'FAIL'} — `{name}`"
        for name, value in complete_checks.items()
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        output_dir / "manifest.json",
        {
            "schema": PHASE5B_AUDIT_SCHEMA,
            "status": "complete",
            "source_assets": {
                "baseline_config": runtime.asset_record(baseline_dir / "config.json"),
                "guided_config": runtime.asset_record(guided_dir / "config.json"),
                "phase2a_reference": runtime.asset_record(args.phase2a_reference),
                "auditor_source": runtime.asset_record(Path(__file__)),
            },
            "outputs": {
                name: runtime.asset_record(output_dir / name)
                for name in (
                    "summary.json",
                    "class_deltas.csv",
                    "truth_component_deltas.csv",
                    "REPORT.md",
                )
            },
        },
    )
    print(summary["decision"])
    print(f"Audit complete: {output_dir}")


if __name__ == "__main__":
    main()

