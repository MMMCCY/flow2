#!/usr/bin/env python3
"""Independently audit geology after a frozen Phase-6P physics-only run."""

from __future__ import annotations

import argparse
import math
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
from guidance.seismic import tensor_sha256
from scripts.stage4.run_seismic_guidance import read_json, write_json, write_rows
from scripts.stage6.run_physics_attainment_limit import PHASE6P_RUN_SCHEMA


PHASE6P_AUDIT_SCHEMA = "phase6p_physics_attainment_truth_audit_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run independent truth-geology audit on a completed Phase-6P run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--truth-model", type=Path, default=None)
    parser.add_argument("--boreholes", type=Path, default=None)
    parser.add_argument("--target-label", type=int, default=9)
    parser.add_argument("--target-roi-radius", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _source_path(
    run_config: Mapping[str, object], name: str, override: Path | None
) -> Path:
    if override is not None:
        return override
    assets = run_config.get("asset_records")
    if not isinstance(assets, Mapping) or not isinstance(assets.get(name), Mapping):
        raise ValueError(f"Phase-6P run lacks source asset: {name}")
    return Path(str(assets[name]["path"]))


def _validate_output_tensor(
    run_dir: Path,
    filename: str,
    record: Mapping[str, object],
) -> torch.Tensor:
    path = run_dir / filename
    if runtime.file_sha256(path) != record.get("file_sha256"):
        raise ValueError(f"Phase-6P output file hash mismatch: {filename}")
    value = runtime.load_tensor(path, map_location="cpu")
    if list(value.shape) != record.get("shape") or str(value.dtype) != record.get(
        "dtype"
    ):
        raise ValueError(f"Phase-6P output shape/dtype mismatch: {filename}")
    if tensor_sha256(value) != record.get("raw_tensor_sha256"):
        raise ValueError(f"Phase-6P output tensor hash mismatch: {filename}")
    canonical = record.get("canonical_int64_content_sha256")
    if canonical is not None and tensor_sha256(value.long()) != canonical:
        raise ValueError(f"Phase-6P canonical content hash mismatch: {filename}")
    return value


def _major_component_mean(
    rows: list[Mapping[str, object]], sample_id: int, count: int = 4
) -> float:
    values = [
        float(row["recall"])
        for row in rows
        if int(row["sample_id"]) == sample_id
        and int(row["truth_component_rank"]) <= count
    ]
    return float(statistics.mean(values)) if values else float("nan")


def _numeric_delta(
    baseline: Mapping[str, object], candidate: Mapping[str, object]
) -> Dict[str, float]:
    fields = (
        "global_voxel_accuracy",
        "global_mean_iou",
        "truth_present_mean_iou",
        "target_iou",
        "target_precision",
        "target_recall",
        "selected_roi_iou",
        "selected_roi_precision",
        "selected_roi_recall",
        "condition_violation_count",
    )
    output: Dict[str, float] = {}
    for field in fields:
        left = float(baseline[field])
        right = float(candidate[field])
        output[f"delta_{field}"] = right - left
    return output


def _decision(
    *,
    engineering_pass: bool,
    attainment_band: str,
    geology_material_improvement: bool,
) -> str:
    if not engineering_pass:
        return "invalid_engineering_run"
    if attainment_band == "low":
        return "low_endpoint_reachability_under_frozen_protocol"
    if attainment_band == "partial":
        return "partial_endpoint_reachability_run_phase6p_c_once"
    if geology_material_improvement:
        return "physics_reachable_and_geologically_informative_trajectory_control_gap"
    return "physics_reachable_but_geologically_nonidentifying_or_misaligned"


def _report_markdown(summary: Mapping[str, object]) -> str:
    physical = summary["physical_attainment_diagnostics"]
    deltas = summary["geology_deltas"]
    return "\n".join(
        (
            "# Phase 6P 独立真值审计",
            "",
            f"- 工程门：`{summary['engineering_pass']}`",
            f"- 物理达到率：`{float(physical['attainment']):.6f}` "
            f"（`{physical['attainment_band']}`）",
            f"- 硬地震 RMSE：`{float(physical['baseline_rmse']):.8g}` → "
            f"`{float(physical['candidate_rmse']):.8g}`",
            f"- 全局 mean-IoU 变化：`{float(deltas['delta_global_mean_iou']):+.6f}`",
            f"- 真值出现类别 mean-IoU 变化："
            f"`{float(deltas['delta_truth_present_mean_iou']):+.6f}`",
            f"- 目标体 IoU 变化：`{float(deltas['delta_target_iou']):+.6f}`",
            f"- 目标体 recall 变化：`{float(deltas['delta_target_recall']):+.6f}`",
            f"- 前四大真值连通体平均 recall 变化："
            f"`{float(summary['delta_major_four_component_recall']):+.6f}`",
            f"- 地质实质改善：`{summary['geology_material_improvement']}`",
            f"- 判定：`{summary['decision']}`",
            "",
            "该判定只适用于预先冻结的 200 步终点优化协议，不是数学上的可达性证明。",
            "",
        )
    )


def main() -> None:
    args = parse_args()
    run_config_path = args.run_dir / "config.json"
    run_config = read_json(run_config_path)
    if run_config.get("schema") != PHASE6P_RUN_SCHEMA:
        raise ValueError("not a Phase-6P endpoint-attainment run")
    if run_config.get("run_status") != "completed":
        raise ValueError("Phase-6P run is not completed")
    if run_config.get("truth_metrics_computed_by_optimizer") is not False:
        raise ValueError("optimizer run is not truth-blind")
    output_records = run_config.get("output_tensor_records")
    if not isinstance(output_records, Mapping):
        raise ValueError("Phase-6P run lacks output tensor records")
    for filename in ("baseline_sample.pt", "best_sample.pt"):
        if not isinstance(output_records.get(filename), Mapping):
            raise ValueError(f"Phase-6P run lacks output record: {filename}")
    baseline = _validate_output_tensor(
        args.run_dir,
        "baseline_sample.pt",
        output_records["baseline_sample.pt"],
    ).long()
    candidate = _validate_output_tensor(
        args.run_dir,
        "best_sample.pt",
        output_records["best_sample.pt"],
    ).long()

    truth_path = _source_path(run_config, "truth_model", args.truth_model)
    boreholes_path = _source_path(run_config, "boreholes", args.boreholes)
    assets = run_config["asset_records"]
    if runtime.file_sha256(truth_path) != assets["truth_model"]["sha256"]:
        raise ValueError("truth model differs from the optimizer run")
    if runtime.file_sha256(boreholes_path) != assets["boreholes"]["sha256"]:
        raise ValueError("boreholes differ from the optimizer run")
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

    metrics_rows: list[Dict[str, object]] = []
    component_rows: list[Dict[str, object]] = []
    per_class_rows: list[Dict[str, object]] = []
    for sample_id, (role, prediction) in enumerate(
        (("baseline", baseline), ("physics_best", candidate))
    ):
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
        component_rows.extend(
            truth_component_recovery_rows(
                prediction, truth, args.target_label, sample_id
            )
        )
        class_rows = per_class_hard_metrics(prediction, truth, sample_id)
        for class_row in class_rows:
            class_row["role"] = role
        per_class_rows.extend(class_rows)

    baseline_metrics, candidate_metrics = metrics_rows
    deltas = _numeric_delta(baseline_metrics, candidate_metrics)
    baseline_major = _major_component_mean(component_rows, 0)
    candidate_major = _major_component_mean(component_rows, 1)
    resolved = run_config.get("resolved_protocol")
    if not isinstance(resolved, Mapping):
        raise ValueError("Phase-6P run lacks resolved protocol")
    material = (
        float(deltas["delta_target_iou"])
        >= float(resolved["target_iou_material_delta"])
        or float(deltas["delta_truth_present_mean_iou"])
        >= float(resolved["truth_present_mean_iou_material_delta"])
    )
    physical = run_config.get("physical_attainment_diagnostics")
    if not isinstance(physical, Mapping):
        raise ValueError("Phase-6P run lacks physical attainment diagnostics")
    band = str(physical.get("attainment_band"))
    if band not in {"low", "partial", "high"}:
        raise ValueError("invalid Phase-6P attainment band")
    engineering_pass = bool(run_config.get("engineering_pass"))
    decision = _decision(
        engineering_pass=engineering_pass,
        attainment_band=band,
        geology_material_improvement=material,
    )

    output_dir = args.output_dir or args.run_dir / "truth_audit"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"audit output directory is not empty; refusing to overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, object] = {
        "schema": PHASE6P_AUDIT_SCHEMA,
        "status": "completed",
        "run_config": runtime.asset_record(run_config_path),
        "truth_model": runtime.asset_record(truth_path),
        "boreholes": runtime.asset_record(boreholes_path),
        "auditor_source": runtime.asset_record(Path(__file__)),
        "target_label": args.target_label,
        "target_roi_radius": args.target_roi_radius,
        "target_metadata": target_metadata,
        "engineering_pass": engineering_pass,
        "physical_attainment_diagnostics": dict(physical),
        "baseline_geology_metrics": baseline_metrics,
        "candidate_geology_metrics": candidate_metrics,
        "geology_deltas": deltas,
        "baseline_major_four_component_recall": baseline_major,
        "candidate_major_four_component_recall": candidate_major,
        "delta_major_four_component_recall": candidate_major - baseline_major,
        "geology_material_improvement": material,
        "decision": decision,
        "scope_warning": (
            "diagnostic conclusion under one frozen same-case inverse-crime protocol; "
            "not a mathematical impossibility proof or held-out evidence"
        ),
    }
    write_rows(output_dir / "sample_metrics.csv", metrics_rows)
    write_rows(output_dir / "per_class_metrics.csv", per_class_rows)
    write_rows(output_dir / "truth_component_recovery.csv", component_rows)
    write_rows(
        output_dir / "paired_class_transitions.csv",
        class_transition_records(baseline, candidate, 1),
    )
    write_json(output_dir / "summary.json", summary)
    (output_dir / "REPORT.md").write_text(
        _report_markdown(summary), encoding="utf-8"
    )
    print(
        "Phase-6P truth audit complete: "
        f"decision={decision}, material_geology={material}, output={output_dir}"
    )


if __name__ == "__main__":
    main()
