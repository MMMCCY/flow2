#!/usr/bin/env python3
"""Retrospectively evaluate Stage14 without selecting or ranking samples."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from scipy import ndimage


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for import_root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import inference_runtime as runtime
from guidance.probability_volume import tensor_sha256


DEFAULT_PROTOCOL = (
    PROJECT_DIR
    / "experiments/stage14_gansim_style_geo_guidance/configs/frozen_protocol.json"
)
STAGE12A_ROOT = PROJECT_DIR / "experiments/full_structuralgeo_benchmark"
STAGE12B_ROOT = PROJECT_DIR / "experiments/stage12b_fullgeo_probability_bridge"
HISTORICAL_DECISIONS = {
    "stage10": PROJECT_DIR
    / "experiments/stage10_geophysical_probability_bridge/reports/STAGE10_MACHINE_DECISION.json",
    "stage12b": STAGE12B_ROOT / "reports/STAGE12B_A_MACHINE_DECISION.json",
    "stage13": PROJECT_DIR
    / "experiments/stage13_binary_label9_bridge/reports/STAGE13A_MACHINE_DECISION.json",
}
METRIC_FIELDS = (
    "label9_iou",
    "label9_precision",
    "label9_recall",
    "hidden_label9_iou",
    "hidden_label9_precision",
    "hidden_label9_recall",
    "largest_hidden_component_recall",
    "truth_present_miou",
    "global_accuracy",
)


def _load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _normalize(value: torch.Tensor, name: str) -> torch.Tensor:
    result = runtime.normalize_single_geology(value, name)
    if tuple(result.shape) != (1, 1, 64, 64, 64):
        raise ValueError(f"unexpected {name} shape: {tuple(result.shape)}")
    return result


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _binary_metrics(
    predicted_target: torch.Tensor,
    truth_target: torch.Tensor,
    domain: torch.Tensor,
) -> tuple[float, float, float]:
    predicted = predicted_target & domain
    truth = truth_target & domain
    intersection = int((predicted & truth).sum())
    union = int((predicted | truth).sum())
    return (
        _ratio(intersection, union),
        _ratio(intersection, int(predicted.sum())),
        _ratio(intersection, int(truth.sum())),
    )


def largest_component(mask: torch.Tensor) -> torch.Tensor:
    array = mask.detach().cpu().bool().numpy()
    labels, count = ndimage.label(array, structure=ndimage.generate_binary_structure(3, 1))
    if count == 0:
        return torch.zeros_like(mask, dtype=torch.bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return torch.from_numpy(labels == int(sizes.argmax()))


def truth_present_miou(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    domain: torch.Tensor,
) -> float:
    labels = sorted(int(value) for value in torch.unique(truth[domain]).tolist() if int(value) != -1)
    values: list[float] = []
    for label in labels:
        predicted = (prediction == label) & domain
        actual = (truth == label) & domain
        union = int((predicted | actual).sum())
        values.append(_ratio(int((predicted & actual).sum()), union))
    return float(statistics.fmean(values)) if values else float("nan")


def sample_metrics(
    *,
    prediction: torch.Tensor,
    truth: torch.Tensor,
    condition_values: torch.Tensor,
    condition_mask: torch.Tensor,
    subsurface_mask: torch.Tensor,
    hidden_label9_mask: torch.Tensor,
    largest_hidden_component: torch.Tensor,
) -> dict[str, float | int]:
    predicted = prediction[0, 0].long()
    actual = truth[0, 0].long()
    conditions = condition_values[0, 0].long()
    condition = condition_mask[0, 0].bool()
    subsurface = subsurface_mask[0, 0].bool()
    hidden_domain = subsurface & ~condition
    hidden_target = hidden_label9_mask[0, 0].bool()
    if not torch.equal(hidden_target, (actual == 9) & hidden_domain):
        raise ValueError("registered hidden-label9 mask disagrees with truth/domain")
    predicted_target = predicted == 9
    truth_target = actual == 9
    label_iou, label_precision, label_recall = _binary_metrics(
        predicted_target, truth_target, subsurface
    )
    hidden_iou, hidden_precision, hidden_recall = _binary_metrics(
        predicted_target, hidden_target, hidden_domain
    )
    largest = largest_hidden_component.bool()
    return {
        "label9_iou": label_iou,
        "label9_precision": label_precision,
        "label9_recall": label_recall,
        "hidden_label9_iou": hidden_iou,
        "hidden_label9_precision": hidden_precision,
        "hidden_label9_recall": hidden_recall,
        "largest_hidden_component_recall": _ratio(
            int((predicted_target & largest).sum()), int(largest.sum())
        ),
        "truth_present_miou": truth_present_miou(predicted, actual, subsurface),
        "global_accuracy": _ratio(
            int(((predicted == actual) & subsurface).sum()), int(subsurface.sum())
        ),
        "condition_violation_count": int(((predicted != conditions) & condition).sum()),
    }


def paired_deltas(
    baseline: Mapping[str, object], guided: Mapping[str, object]
) -> dict[str, object]:
    result: dict[str, object] = {
        "case_id": guided["case_id"],
        "sample_id": guided["sample_id"],
        "source_seed": guided["source_seed"],
    }
    for field in METRIC_FIELDS:
        result[f"delta_{field}"] = float(guided[field]) - float(baseline[field])
    result["baseline_condition_violation_count"] = int(
        baseline["condition_violation_count"]
    )
    result["guided_condition_violation_count"] = int(
        guided["condition_violation_count"]
    )
    return result


def _median(rows: Sequence[Mapping[str, object]], field: str) -> float:
    values = [float(row[field]) for row in rows if math.isfinite(float(row[field]))]
    if not values:
        return float("nan")
    return float(statistics.median(values))


def evaluate(run_root: Path, protocol: Mapping[str, object]) -> dict[str, object]:
    run_manifest = _load_json(run_root / "run_manifest.json")
    if run_manifest.get("status") != "completed" or run_manifest.get("truth_loaded_by_flow_runner") is not False:
        raise ValueError("truth-blind Flow run is incomplete or invalid")
    expected_decisions = protocol["historical_machine_decision_sha256"]
    for name, path in HISTORICAL_DECISIONS.items():
        if runtime.file_sha256(path) != expected_decisions[name]:
            raise ValueError(f"historical {name} machine decision changed")

    metric_rows: list[dict[str, object]] = []
    delta_rows: list[dict[str, object]] = []
    case_summaries: dict[str, object] = {}
    for case_id in protocol["case_ids"]:
        case_id = str(case_id)
        case_run = run_root / "cases" / case_id
        case_manifest = _load_json(case_run / "manifest.json")
        if case_manifest.get("status") != "completed":
            raise ValueError(f"incomplete Flow outputs for {case_id}")
        case_root = STAGE12A_ROOT / "cases" / case_id
        truth = _normalize(runtime.load_tensor(case_root / "truth/true_model.pt"), "truth").long()
        hidden = _normalize(
            runtime.load_tensor(case_root / "truth/hidden_label9_mask.pt"),
            "hidden_label9_mask",
        ).bool()
        condition_values = _normalize(
            runtime.load_tensor(case_root / "condition/condition_values.pt"),
            "condition_values",
        ).long()
        condition_mask = _normalize(
            runtime.load_tensor(case_root / "condition/condition_mask.pt"),
            "condition_mask",
        ).bool()
        subsurface_mask = truth != -1
        registered_subsurface = _normalize(
            runtime.load_tensor(STAGE12B_ROOT / "observations" / case_id / "subsurface_mask.pt"),
            "subsurface_mask",
        ).bool()
        if not torch.equal(subsurface_mask, registered_subsurface):
            raise ValueError(f"subsurface mask mismatch for {case_id}")
        largest = largest_component(hidden[0, 0]).unsqueeze(0).unsqueeze(0)
        by_arm_seed: dict[tuple[str, int], dict[str, object]] = {}
        sample_records = case_manifest["samples"]
        for sample_id, source_seed in enumerate(protocol["source_seeds"][case_id]):
            record = sample_records[sample_id]
            if int(record["source_seed"]) != int(source_seed):
                raise ValueError(f"seed manifest mismatch for {case_id}")
            for arm, hash_field in (
                ("BASELINE", "baseline_sample_sha256"),
                ("GEO_PROB_GUIDED", "guided_sample_sha256"),
            ):
                path = case_run / arm / f"source_seed_{source_seed}.pt"
                prediction = _normalize(runtime.load_tensor(path), "prediction").long()
                # The Flow runner saves a canonical raw 3-D categorical tensor;
                # normalize only for evaluation, then hash the original 3-D view.
                if tensor_sha256(prediction[0, 0]) != record[hash_field]:
                    raise ValueError(f"sample hash mismatch: {case_id}/{source_seed}/{arm}")
                row: dict[str, object] = {
                    "case_id": case_id,
                    "sample_id": sample_id,
                    "source_seed": int(source_seed),
                    "arm": arm,
                    **sample_metrics(
                        prediction=prediction,
                        truth=truth,
                        condition_values=condition_values,
                        condition_mask=condition_mask,
                        subsurface_mask=subsurface_mask,
                        hidden_label9_mask=hidden,
                        largest_hidden_component=largest,
                    ),
                }
                metric_rows.append(row)
                by_arm_seed[(arm, int(source_seed))] = row
            delta_rows.append(
                paired_deltas(
                    by_arm_seed[("BASELINE", int(source_seed))],
                    by_arm_seed[("GEO_PROB_GUIDED", int(source_seed))],
                )
            )
        case_deltas = [row for row in delta_rows if row["case_id"] == case_id]
        case_metrics = [row for row in metric_rows if row["case_id"] == case_id]
        arm_medians = {
            arm: {
                field: _median([row for row in case_metrics if row["arm"] == arm], field)
                for field in (*METRIC_FIELDS, "condition_violation_count")
            }
            for arm in ("BASELINE", "GEO_PROB_GUIDED")
        }
        delta_medians = {
            field: _median(case_deltas, f"delta_{field}") for field in METRIC_FIELDS
        }
        case_summaries[case_id] = {
            "arm_medians": arm_medians,
            "paired_delta_medians": delta_medians,
            "positive_median_hidden_label9_iou_delta": delta_medians["hidden_label9_iou"] > 0,
        }

    positive_cases = sum(
        bool(value["positive_median_hidden_label9_iou_delta"])
        for value in case_summaries.values()
    )
    overall_primary_median = _median(delta_rows, "delta_hidden_label9_iou")
    total_condition_violations = sum(
        int(row["condition_violation_count"]) for row in metric_rows
    )
    threshold = float(protocol["decision"]["catastrophic_global_degradation_threshold"])
    catastrophic_cases: list[dict[str, object]] = []
    for case_id, value in case_summaries.items():
        deltas = value["paired_delta_medians"]
        bad = {
            field: float(deltas[field])
            for field in ("truth_present_miou", "global_accuracy")
            if float(deltas[field]) < threshold
        }
        if bad:
            catastrophic_cases.append({"case_id": case_id, "metrics": bad})
    clauses = {
        "at_least_4_of_5_cases_positive_median_hidden_label9_iou_delta": positive_cases >= 4,
        "overall_paired_median_hidden_label9_iou_delta_positive": overall_primary_median > 0,
        "zero_hard_condition_violations": total_condition_violations == 0,
        "no_catastrophic_global_geology_degradation": not catastrophic_cases,
    }
    supported = all(clauses.values())
    decision = (
        protocol["decision"]["supported"]
        if supported
        else protocol["decision"]["not_supported"]
    )
    return {
        "schema": "stage14_gansim_style_evaluation_v1",
        "status": "complete_stop_no_further_experiments",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_role": protocol["experiment_role"],
        "decision": decision,
        "n_cases": len(case_summaries),
        "n_pairs": len(delta_rows),
        "positive_case_count": positive_cases,
        "overall_paired_median_hidden_label9_iou_delta": overall_primary_median,
        "total_condition_violations": total_condition_violations,
        "catastrophic_global_degradation_threshold": threshold,
        "catastrophic_cases": catastrophic_cases,
        "clauses": clauses,
        "case_summaries": case_summaries,
        "metric_rows": metric_rows,
        "delta_rows": delta_rows,
        "historical_machine_decisions_unchanged": True,
        "sample_selection_performed": False,
        "training_performed": False,
        "parameter_sweep_performed": False,
    }


def _fmt(value: object) -> str:
    return f"{float(value):.6f}"


def _report(summary: Mapping[str, object]) -> str:
    lines = [
        "# Stage 14 — GANSim-style geophysical probability guidance pilot",
        "",
        "This is a new end-to-end exploratory experiment testing whether the previous bridge-only stop rule was overly conservative. It does not reopen or alter Stage10, Stage12, or Stage13 decisions.",
        "",
        "## Decision",
        "",
        f"`{summary['decision']}`",
        "",
        f"Primary endpoint: overall paired median hidden-label9 IoU change = **{_fmt(summary['overall_paired_median_hidden_label9_iou_delta'])}**; positive case medians = **{summary['positive_case_count']}/5**.",
        "",
        "| case | baseline hidden IoU | guided hidden IoU | median paired delta | hidden recall delta | largest hidden component recall delta | mIoU delta | accuracy delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case_id, value in summary["case_summaries"].items():
        base = value["arm_medians"]["BASELINE"]
        guided = value["arm_medians"]["GEO_PROB_GUIDED"]
        delta = value["paired_delta_medians"]
        lines.append(
            f"| {case_id} | {_fmt(base['hidden_label9_iou'])} | {_fmt(guided['hidden_label9_iou'])} | {_fmt(delta['hidden_label9_iou'])} | {_fmt(delta['hidden_label9_recall'])} | {_fmt(delta['largest_hidden_component_recall'])} | {_fmt(delta['truth_present_miou'])} | {_fmt(delta['global_accuracy'])} |"
        )
    lines.extend(
        [
            "",
            "## Decision clauses",
            "",
            *[
                f"- {name}: {'PASS' if passed else 'FAIL'}"
                for name, passed in summary["clauses"].items()
            ],
            "",
            f"Hard-condition violations across both arms: **{summary['total_condition_violations']}**.",
            "",
            "The Stage12B post-seismic P(label9) volumes were consumed as continuous volumes through the frozen Phase1 interface, with no bridge redevelopment, pre-Flow AUPRC gate, probability preprocessing, parameter sweep, training, truth-based tuning, or best-sample selection. The experiment stops at this decision.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty report dir: {args.output_dir}")
    protocol = _load_json(args.protocol)
    summary = evaluate(args.run_root, protocol)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(args.output_dir / "per_arm_metrics.csv", summary.pop("metric_rows"))
    _write_rows(args.output_dir / "paired_deltas.csv", summary.pop("delta_rows"))
    _write_json(args.output_dir / "summary.json", summary)
    _write_json(
        args.output_dir / "STAGE14_MACHINE_DECISION.json",
        {
            "schema": "stage14_gansim_style_machine_decision_v1",
            "decision": summary["decision"],
            "status": summary["status"],
            "clauses": summary["clauses"],
            "positive_case_count": summary["positive_case_count"],
            "overall_paired_median_hidden_label9_iou_delta": summary[
                "overall_paired_median_hidden_label9_iou_delta"
            ],
            "historical_machine_decisions_unchanged": True,
        },
    )
    (args.output_dir / "STAGE14_REPORT.md").write_text(
        _report(summary), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
