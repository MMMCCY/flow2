#!/usr/bin/env python3
"""Retrospective Stage18A hard-seismic audit of immutable Stage17C outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime
from guidance.property_evaluation import (
    size_stratified_component_metrics,
    truth_present_mean_iou,
)
from guidance.probability_volume import tensor_sha256
from scripts.stage15.common import normalize_volume, read_json, refuse_nonempty, write_csv, write_json
from scripts.stage17.common import read_csv, validate_asset_record
from scripts.stage18.common import (
    REFERENCE_ARMS,
    binary_physics_from_manifest,
    hard_seismic_metrics,
    load_case_tensors,
    load_stage18_references,
    median,
    reference_index,
    truth_observation_closure,
)


ROOT = PROJECT_DIR / "experiments/stage18_evidence_semantics"
DEFAULT_CONFIG = ROOT / "configs/continuous_property_target_v1.json"
DEFAULT_OUTPUT = ROOT / "hard_seismic_audit/formal_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _float_row(row: dict[str, str]) -> dict[str, object]:
    """Preserve identifiers while parsing Stage17 numeric metric fields."""
    parsed: dict[str, object] = {}
    for key, value in row.items():
        if key in {"case_id", "arm", "evidence_tensor_sha256", "initial_noise_sha256"}:
            parsed[key] = value
            continue
        try:
            parsed[key] = float(value)
        except (TypeError, ValueError):
            parsed[key] = value
    return parsed


def main() -> None:
    args = parse_args()
    refuse_nonempty(args.output_dir)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    config = read_json(args.config)
    stage17, stage17_run_dir, records = load_stage18_references(config)
    metrics_path = validate_asset_record(config["stage17c_sample_metrics"], "Stage17C metrics")
    validate_asset_record(config["stage17c_evaluation_summary"], "Stage17C evaluation")
    reference = reference_index(records)
    existing = {
        (row["case_id"], int(row["source_seed"]), row["arm"]): _float_row(row)
        for row in read_csv(metrics_path)
    }
    if len(existing) != 30:
        raise ValueError("Stage17C sample metrics must contain 30 rows")

    sample_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    closure_rows: list[dict[str, object]] = []
    for case in stage17["cases"]:
        case_id = str(case["case_id"])
        tensors = load_case_tensors(case)
        support = tensors["support"]
        observed = tensors["observed"]
        truth = tensors["truth"]
        properties, operator, _ = binary_physics_from_manifest(
            tensors["observation_manifest"], support.shape[2:]
        )
        closure_rows.append(
            {
                "case_id": case_id,
                **truth_observation_closure(
                    truth, support, observed, properties, operator, device
                ),
            }
        )
        for seed in stage17["source_seeds"]:
            arm_rows: dict[str, dict[str, object]] = {}
            for arm in REFERENCE_ARMS:
                record = reference[(case_id, int(seed), arm)]
                decoded = normalize_volume(
                    runtime.load_tensor(stage17_run_dir / record["decoded_path"]), arm
                ).long()
                if tensor_sha256(decoded) != record["decoded_geology_sha256"]:
                    raise ValueError(f"Stage17C decoded geology changed: {case_id}/{seed}/{arm}")
                topology = size_stratified_component_metrics(decoded == 9)
                row = {
                    **existing[(case_id, int(seed), arm)],
                    "case_id": case_id,
                    "source_seed": int(seed),
                    "arm": arm,
                    "truth_present_mean_iou": truth_present_mean_iou(decoded, truth),
                    **hard_seismic_metrics(
                        decoded, support, observed, properties, operator, device
                    ),
                    "target_top4_component_mass_fraction": topology[
                        "target_top4_component_mass_fraction"
                    ],
                    "target_top8_component_mass_fraction": topology[
                        "target_top8_component_mass_fraction"
                    ],
                }
                sample_rows.append(row)
                arm_rows[arm] = row
            base, guided = arm_rows["FLOW_ONLY"], arm_rows["TRAJECTORY_EVIDENCE"]
            pair_rows.append(
                {
                    "case_id": case_id,
                    "source_seed": int(seed),
                    "delta_hard_seismic_mse": float(guided["hard_seismic_mse"])
                    - float(base["hard_seismic_mse"]),
                    "delta_hard_seismic_rmse": float(guided["hard_seismic_rmse"])
                    - float(base["hard_seismic_rmse"]),
                    "delta_target_iou": float(guided["target_iou"])
                    - float(base["target_iou"]),
                    "delta_target_precision": float(guided["target_precision"])
                    - float(base["target_precision"]),
                    "delta_target_recall": float(guided["target_recall"])
                    - float(base["target_recall"]),
                    "delta_target_absolute_volume_error_fraction": float(
                        guided["target_absolute_volume_error_fraction"]
                    )
                    - float(base["target_absolute_volume_error_fraction"]),
                }
            )

    case_rows: list[dict[str, object]] = []
    for case in stage17["cases"]:
        case_id = str(case["case_id"])
        pairs = [row for row in pair_rows if row["case_id"] == case_id]
        guided = [
            row
            for row in sample_rows
            if row["case_id"] == case_id and row["arm"] == "TRAJECTORY_EVIDENCE"
        ]
        case_rows.append(
            {
                "case_id": case_id,
                "stochastic_replicate_count": 3,
                "median_delta_hard_seismic_rmse": median(
                    [float(row["delta_hard_seismic_rmse"]) for row in pairs]
                ),
                "median_delta_target_iou": median(
                    [float(row["delta_target_iou"]) for row in pairs]
                ),
                "median_guided_target_volume_error_fraction": median(
                    [float(row["target_volume_error_fraction"]) for row in guided]
                ),
                "median_guided_target_absolute_volume_error_fraction": median(
                    [float(row["target_absolute_volume_error_fraction"]) for row in guided]
                ),
            }
        )

    overshoot_cases = sum(
        float(row["median_delta_target_iou"]) > 0
        and float(row["median_delta_hard_seismic_rmse"]) > 0
        for row in case_rows
    )
    compatible_cases = sum(
        float(row["median_delta_hard_seismic_rmse"]) <= 0 for row in case_rows
    )
    overpredicted_cases = sum(
        float(row["median_guided_target_volume_error_fraction"]) > 0 for row in case_rows
    )
    if overshoot_cases >= 3:
        decision = "SURROGATE_OVERSHOOT_SIGNAL"
    elif compatible_cases >= 3 and overpredicted_cases >= 3:
        decision = "HARD_PHYSICS_COMPATIBLE_BUT_VOLUME_UNCALIBRATED"
    else:
        decision = "MIXED_HARD_PHYSICS_RESPONSE"

    summary = {
        "schema": "stage18a_stage17c_hard_seismic_audit_v1",
        "decision": decision,
        "primary_statistical_unit": "independent_geology_case",
        "case_count": 5,
        "replicates_per_case": 3,
        "decoded_geology_count": 30,
        "overshoot_signal_case_count": overshoot_cases,
        "hard_physics_improved_or_tied_case_count": compatible_cases,
        "guided_overprediction_case_count": overpredicted_cases,
        "truth_observation_closure_max_rmse": max(
            float(row["truth_observation_hard_seismic_rmse"]) for row in closure_rows
        ),
        "flow_sampling_performed": False,
        "observation_model_role": "Stage17_binary_observation_model_not_full_lithology",
    }
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "sample_hard_seismic_metrics.csv", sample_rows)
    write_csv(args.output_dir / "paired_hard_seismic_deltas.csv", pair_rows)
    write_csv(args.output_dir / "case_primary_summary.csv", case_rows)
    write_csv(args.output_dir / "truth_observation_closure.csv", closure_rows)
    write_json(args.output_dir / "summary.json", summary)
    report = [
        "# Stage18A Stage17C hard-seismic audit",
        "",
        f"Decision: **{decision}**",
        "",
        f"Hard-seismic overshoot signal: {overshoot_cases}/5 independent cases.",
        f"Hard-seismic improvement/tie: {compatible_cases}/5 cases; positive guided volume error: {overpredicted_cases}/5 cases.",
        "",
        "This is the frozen Stage17 binary observation model, not full-lithology field seismic consistency.",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Stage18A completed: {decision}", flush=True)


if __name__ == "__main__":
    main()
