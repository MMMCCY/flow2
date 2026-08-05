#!/usr/bin/env python3
"""Independently audit geology for a frozen Phase-6P guidance ladder."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys
from typing import Dict, Mapping

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import inference_runtime as runtime
from guidance.probability_evaluation import (
    class_transition_records,
    sample_hard_metrics,
)
from guidance.probability_volume import build_target_mask, dilate_mask
from guidance.property_evaluation import (
    per_class_hard_metrics,
    size_stratified_component_metrics,
    truth_component_recovery_rows,
    truth_present_mean_iou,
)
from scripts.stage4.run_seismic_guidance import read_json, write_json, write_rows
from scripts.stage6.audit_physics_attainment_limit import (
    _numeric_delta,
    _source_path,
    _validate_output_tensor,
)
from scripts.stage6.run_physics_guidance_ladder import PHASE6P_LADDER_RUN_SCHEMA


PHASE6P_LADDER_AUDIT_SCHEMA = "phase6p_trajectory_ladder_truth_audit_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit truth geology only after a Phase-6P ladder is frozen.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--truth-model", type=Path, default=None)
    parser.add_argument("--boreholes", type=Path, default=None)
    parser.add_argument("--target-label", type=int, default=9)
    parser.add_argument("--target-roi-radius", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _major_four(rows: list[Mapping[str, object]], sample_id: int) -> float:
    values = [
        float(row["recall"])
        for row in rows
        if int(row["sample_id"]) == sample_id
        and int(row["truth_component_rank"]) <= 4
    ]
    return float(statistics.mean(values)) if values else float("nan")


def _decision(maximum_attainment: float, resolved: Mapping[str, object]) -> str:
    if maximum_attainment < float(resolved["low_attainment_upper"]):
        return "extreme_trajectory_guidance_low_reachability"
    if maximum_attainment < float(resolved["high_attainment_lower"]):
        return "extreme_trajectory_guidance_partial_reachability"
    return "extreme_trajectory_guidance_high_reachability"


def _report(summary: Mapping[str, object]) -> str:
    lines = [
        "# Phase 6P 极限制导阶梯独立审计",
        "",
        f"- 工程门：`{summary['engineering_pass']}`",
        f"- 最大物理达到率：`{float(summary['maximum_attainment']):.6f}`",
        f"- 物理最优级：`{summary['physically_best_level_id']}`",
        f"- 判定：`{summary['decision']}`",
        "",
        "| 级别 | 强度/上限 | 硬地震 RMSE | 达到率 | Δ全局 mIoU | Δ目标 IoU | Δ目标 recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    geology = summary["level_geology_audits"]
    for physical in summary["level_physics_metrics"]:
        level_id = str(physical["id"])
        item = geology[level_id]
        delta = item["deltas"]
        lines.append(
            f"| {level_id} | {float(physical['alpha']):.2f} | "
            f"{float(physical['hard_seismic_rmse_amplitude']):.8f} | "
            f"{float(physical['attainment']):+.4f} | "
            f"{float(delta['delta_global_mean_iou']):+.4f} | "
            f"{float(delta['delta_target_iou']):+.4f} | "
            f"{float(delta['delta_target_recall']):+.4f} |"
        )
    lines.extend(
        (
            "",
            "所有级别都在运行前冻结并并列报告；真值未参与物理级别选择。",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_config_path = args.run_dir / "config.json"
    run_config = read_json(run_config_path)
    if run_config.get("schema") != PHASE6P_LADDER_RUN_SCHEMA:
        raise ValueError("not a Phase-6P trajectory ladder run")
    if run_config.get("run_status") != "completed":
        raise ValueError("Phase-6P ladder is not completed")
    if run_config.get("truth_metrics_computed_by_runner") is not False:
        raise ValueError("Phase-6P ladder runner is not truth-blind")
    output_records = run_config.get("output_tensor_records")
    resolved = run_config.get("resolved_protocol")
    physical_rows = run_config.get("level_physics_metrics")
    if not isinstance(output_records, Mapping) or not isinstance(
        resolved, Mapping
    ) or not isinstance(physical_rows, list):
        raise ValueError("Phase-6P ladder manifest is incomplete")
    baseline_record = output_records.get("baseline_sample.pt")
    if not isinstance(baseline_record, Mapping):
        raise ValueError("Phase-6P ladder lacks baseline record")
    baseline = _validate_output_tensor(
        args.run_dir, "baseline_sample.pt", baseline_record
    ).long()

    truth_path = _source_path(run_config, "truth_model", args.truth_model)
    boreholes_path = _source_path(run_config, "boreholes", args.boreholes)
    assets = run_config["asset_records"]
    if runtime.file_sha256(truth_path) != assets["truth_model"]["sha256"]:
        raise ValueError("truth differs from the ladder run")
    if runtime.file_sha256(boreholes_path) != assets["boreholes"]["sha256"]:
        raise ValueError("boreholes differ from the ladder run")
    truth = runtime.normalize_single_geology(
        runtime.load_tensor(truth_path, map_location="cpu"), str(truth_path)
    ).long()
    boreholes = runtime.normalize_single_geology(
        runtime.load_tensor(boreholes_path, map_location="cpu"), str(boreholes_path)
    ).long()
    condition_mask = (boreholes != -1) | (truth == -1)
    target_mask, target_metadata = build_target_mask(
        truth, target_label=args.target_label, component_mode="all"
    )
    target_roi = dilate_mask(target_mask, args.target_roi_radius)

    predictions: list[tuple[str, torch.Tensor]] = [("baseline", baseline)]
    for physical in physical_rows:
        level_id = str(physical["id"])
        filename = f"{level_id}_sample.pt"
        record = output_records.get(filename)
        if not isinstance(record, Mapping):
            raise ValueError(f"Phase-6P ladder lacks output: {filename}")
        predictions.append(
            (
                level_id,
                _validate_output_tensor(args.run_dir, filename, record).long(),
            )
        )

    metrics_rows: list[Dict[str, object]] = []
    per_class_rows: list[Dict[str, object]] = []
    component_rows: list[Dict[str, object]] = []
    transition_rows: list[Dict[str, object]] = []
    for sample_id, (role, prediction) in enumerate(predictions):
        row = sample_hard_metrics(
            prediction=prediction,
            truth_model=truth,
            target_mask=target_mask,
            roi_mask=target_roi,
            condition_mask=condition_mask,
            target_label=args.target_label,
            sample_id=sample_id,
            baseline_prediction=baseline if sample_id else None,
        )
        row["truth_present_mean_iou"] = truth_present_mean_iou(
            prediction, truth
        )
        row.update(size_stratified_component_metrics(prediction == args.target_label))
        row["role"] = role
        metrics_rows.append(row)
        class_rows = per_class_hard_metrics(prediction, truth, sample_id)
        for class_row in class_rows:
            class_row["role"] = role
        per_class_rows.extend(class_rows)
        component_rows.extend(
            truth_component_recovery_rows(
                prediction, truth, args.target_label, sample_id
            )
        )
        if sample_id:
            transition_rows.extend(
                {
                    "role": role,
                    **record,
                }
                for record in class_transition_records(
                    baseline, prediction, sample_id
                )
            )

    baseline_metrics = metrics_rows[0]
    baseline_major = _major_four(component_rows, 0)
    geology_audits: Dict[str, object] = {}
    for sample_id, row in enumerate(metrics_rows[1:], start=1):
        role = str(row["role"])
        deltas = _numeric_delta(baseline_metrics, row)
        major = _major_four(component_rows, sample_id)
        material = (
            float(deltas["delta_target_iou"])
            >= float(resolved["target_iou_material_delta"])
            or float(deltas["delta_truth_present_mean_iou"])
            >= float(resolved["truth_present_mean_iou_material_delta"])
        )
        geology_audits[role] = {
            "metrics": row,
            "deltas": deltas,
            "major_four_component_recall": major,
            "delta_major_four_component_recall": major - baseline_major,
            "material_improvement": material,
        }

    maximum_attainment = float(run_config["maximum_attainment"])
    decision = _decision(maximum_attainment, resolved)
    output_dir = args.output_dir or args.run_dir / "truth_audit"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"audit output directory is not empty; refusing to overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, object] = {
        "schema": PHASE6P_LADDER_AUDIT_SCHEMA,
        "status": "completed",
        "run_config": runtime.asset_record(run_config_path),
        "truth_model": runtime.asset_record(truth_path),
        "boreholes": runtime.asset_record(boreholes_path),
        "auditor_source": runtime.asset_record(Path(__file__)),
        "target_label": args.target_label,
        "target_roi_radius": args.target_roi_radius,
        "target_metadata": target_metadata,
        "engineering_pass": bool(run_config.get("engineering_pass")),
        "baseline_geology_metrics": baseline_metrics,
        "baseline_major_four_component_recall": baseline_major,
        "level_physics_metrics": physical_rows,
        "level_geology_audits": geology_audits,
        "physically_best_level_id": run_config["physically_best_level_id"],
        "maximum_attainment": maximum_attainment,
        "decision": (
            decision
            if bool(run_config.get("engineering_pass"))
            else "invalid_engineering_run"
        ),
        "scope_warning": (
            "same-case inverse-crime diagnostic; not held-out evidence or a "
            "mathematical impossibility proof"
        ),
    }
    write_rows(output_dir / "sample_metrics.csv", metrics_rows)
    write_rows(output_dir / "per_class_metrics.csv", per_class_rows)
    write_rows(output_dir / "truth_component_recovery.csv", component_rows)
    write_rows(output_dir / "paired_class_transitions.csv", transition_rows)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    print(
        "Phase-6P ladder truth audit complete: "
        f"decision={summary['decision']}, output={output_dir}"
    )


if __name__ == "__main__":
    main()
