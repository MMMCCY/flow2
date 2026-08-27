#!/usr/bin/env python3
"""Retrospective truth-aware evaluation for completed Stage16 runs."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Mapping

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime
from guidance.probability_evaluation import (
    paired_metric_deltas,
    sample_hard_metrics,
    summarize_rows,
)
from guidance.probability_volume import build_target_mask, dilate_mask, tensor_sha256
from guidance.property_evaluation import (
    per_class_hard_metrics,
    truth_component_recovery_rows,
)
from scripts.stage15.common import (
    base_manifest,
    normalize_volume,
    read_json,
    refuse_nonempty,
    write_csv,
    write_json,
)


DEFAULT_TRUTH = PROJECT_DIR / "samples/jupyter-demo/cond_generation_0/true_model.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--truth-model", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--body-masks",
        type=Path,
        default=None,
        help="optional pre-existing [K,X,Y,Z] five-body masks for connectivity metrics",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve_run_path(run_dir: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else run_dir / path


def _hidden_metrics(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    condition_mask: torch.Tensor,
    target_label: int,
) -> dict[str, object]:
    predicted_target = prediction == int(target_label)
    hidden_truth = (truth == int(target_label)) & ~condition_mask
    hidden_prediction = predicted_target & ~condition_mask
    intersection = int((hidden_prediction & hidden_truth).sum())
    union = int((hidden_prediction | hidden_truth).sum())
    predicted_count = int(hidden_prediction.sum())
    truth_count = int(hidden_truth.sum())

    def ratio(a: int, b: int) -> float:
        return a / b if b else float("nan")

    return {
        "hidden_target_iou": ratio(intersection, union),
        "hidden_target_precision": ratio(intersection, predicted_count),
        "hidden_target_recall": ratio(intersection, truth_count),
        "hidden_truth_target_volume": truth_count,
        "hidden_predicted_target_volume": predicted_count,
    }


def _optional_body_metrics(
    prediction: torch.Tensor,
    body_masks: torch.Tensor | None,
    target_label: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if body_masks is None:
        return {}, []
    masks = body_masks.bool()
    if masks.ndim == 5 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 4 or tuple(masks.shape[1:]) != tuple(prediction.shape[2:]):
        raise ValueError("body_masks must have shape [K,64,64,64] or [K,1,64,64,64]")
    predicted_target = (prediction == int(target_label))[0, 0].cpu()
    from scripts.stage15.evaluate_five_body_flow import _body_connectivity

    merged_pairs, body_components = _body_connectivity(predicted_target, masks.cpu())
    rows = []
    for body_index, mask in enumerate(masks):
        truth_count = int(mask.sum())
        recovered = int((predicted_target & mask.cpu()).sum())
        rows.append(
            {
                "body_index": body_index,
                "truth_body_voxels": truth_count,
                "recovered_voxels": recovered,
                "truth_body_recall": recovered / truth_count if truth_count else float("nan"),
                "dominant_predicted_component": body_components[body_index],
            }
        )
    return {
        "pairwise_merged_body_count": merged_pairs,
        "body_count": int(masks.shape[0]),
    }, rows


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.run_dir / "evaluation"
    refuse_nonempty(output_dir)
    run_manifest = read_json(args.run_dir / "run_manifest.json")
    if run_manifest.get("run_status") != "completed":
        raise ValueError("Stage16 run is not complete")
    config = read_json(args.run_dir / "config.json")
    target_label = int(config["target_label"])
    truth = normalize_volume(runtime.load_tensor(args.truth_model), "truth_model").long()
    target_mask, _ = build_target_mask(truth, target_label=target_label, component_mode="all")
    roi = dilate_mask(target_mask, int(config.get("evaluation_roi_radius", 6)))

    asset_records = run_manifest["input_assets"]
    if not isinstance(asset_records, Mapping):
        raise TypeError("run manifest is missing input assets")
    if str(config["objective"]) == "seismic":
        condition_mask_path = Path(str(asset_records["condition_mask"]["path"]))
        condition_values_path = Path(str(asset_records["condition_values"]["path"]))
        condition_mask = normalize_volume(
            runtime.load_tensor(condition_mask_path), "condition_mask"
        ).bool()
        condition_values = normalize_volume(
            runtime.load_tensor(condition_values_path), "condition_values"
        ).long()
    else:
        borehole_path = Path(str(asset_records["boreholes"]["path"]))
        boreholes = normalize_volume(runtime.load_tensor(borehole_path), "boreholes").long()
        condition_mask = (boreholes != -1) | (truth == -1)
        condition_values = truth
    if int(((condition_values != truth) & condition_mask).sum()) != 0:
        raise ValueError("condition assets disagree with retrospective truth")

    body_masks = (
        runtime.load_tensor(args.body_masks).bool() if args.body_masks is not None else None
    )
    pair_rows = _read_csv(args.run_dir / "paired_manifest.csv")
    if not pair_rows:
        raise ValueError("paired manifest is empty")
    metric_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    body_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []

    for pair in pair_rows:
        sample_id = int(pair["sample_id"])
        source_seed = int(pair["source_seed"])
        arm_metrics: dict[str, dict[str, object]] = {}
        baseline_prediction: torch.Tensor | None = None
        for arm, field in (
            ("FLOW_ONLY", "baseline_decoded_path"),
            ("DFLOW", "dflow_decoded_path"),
        ):
            prediction = normalize_volume(
                runtime.load_tensor(_resolve_run_path(args.run_dir, pair[field])),
                f"{arm}_prediction",
            ).long()
            metrics = sample_hard_metrics(
                prediction=prediction,
                truth_model=truth,
                target_mask=target_mask,
                roi_mask=roi,
                condition_mask=condition_mask,
                target_label=target_label,
                sample_id=sample_id,
                baseline_prediction=baseline_prediction if arm == "DFLOW" else None,
            )
            body_summary, per_body = _optional_body_metrics(
                prediction, body_masks, target_label
            )
            metrics.update(_hidden_metrics(prediction, truth, condition_mask, target_label))
            metrics.update(body_summary)
            metrics.update(
                {
                    "arm": arm,
                    "source_seed": source_seed,
                    "decoded_model_sha256": tensor_sha256(prediction),
                }
            )
            if int(metrics["condition_violation_count"]) != 0:
                raise RuntimeError(f"hard condition violation in {arm}/seed {source_seed}")
            metric_rows.append(metrics)
            arm_metrics[arm] = metrics
            for row in per_class_hard_metrics(prediction, truth, sample_id):
                class_rows.append({"arm": arm, "source_seed": source_seed, **row})
            for row in truth_component_recovery_rows(
                prediction, truth, target_label, sample_id
            ):
                component_rows.append({"arm": arm, "source_seed": source_seed, **row})
            for row in per_body:
                body_rows.append(
                    {
                        "arm": arm,
                        "sample_id": sample_id,
                        "source_seed": source_seed,
                        **row,
                    }
                )
            if arm == "FLOW_ONLY":
                baseline_prediction = prediction

        delta = paired_metric_deltas(
            arm_metrics["FLOW_ONLY"], arm_metrics["DFLOW"]
        )
        for field_name in (
            "hidden_target_iou",
            "hidden_target_precision",
            "hidden_target_recall",
            "hidden_predicted_target_volume",
        ):
            delta[f"delta_{field_name}"] = float(arm_metrics["DFLOW"][field_name]) - float(
                arm_metrics["FLOW_ONLY"][field_name]
            )
        paired_rows.append({"source_seed": source_seed, **delta})

    output_dir.mkdir(parents=True)
    write_csv(output_dir / "sample_metrics.csv", metric_rows)
    write_csv(output_dir / "per_class_metrics.csv", class_rows)
    write_csv(output_dir / "truth_component_recovery.csv", component_rows)
    write_csv(output_dir / "paired_metric_deltas.csv", paired_rows)
    if body_rows:
        write_csv(output_dir / "five_body_metrics.csv", body_rows)

    finite_paired = {
        key: [float(row[key]) for row in paired_rows if key in row and math.isfinite(float(row[key]))]
        for key in paired_rows[0]
        if key not in {"sample_id", "source_seed"}
    }
    summary = {
        "schema": "stage16_dflow_retrospective_evaluation_v1",
        "objective": config["objective"],
        "pair_count": len(pair_rows),
        "arm_summaries": {
            arm: summarize_rows([row for row in metric_rows if row["arm"] == arm])
            for arm in ("FLOW_ONLY", "DFLOW")
        },
        "paired_delta_means": {
            key: sum(values) / len(values) if values else None
            for key, values in finite_paired.items()
        },
        "truth_role": "retrospective_evaluation_only",
        "truth_loaded_after_inference": True,
        "truth_used_for_sampling_or_optimization": False,
    }
    write_json(output_dir / "summary.json", summary)
    manifest = base_manifest(
        "stage16_dflow_retrospective_evaluation_v1", Path(__file__)
    )
    manifest.update(
        {
            "run_status": "completed",
            "source_run_manifest": runtime.asset_record(args.run_dir / "run_manifest.json"),
            "source_paired_manifest": runtime.asset_record(args.run_dir / "paired_manifest.csv"),
            "truth_model": runtime.asset_record(args.truth_model),
            "truth_tensor_sha256": tensor_sha256(truth),
            "truth_role": "retrospective_evaluation_only",
            "body_masks": runtime.asset_record(args.body_masks) if args.body_masks else None,
            "sample_metrics": runtime.asset_record(output_dir / "sample_metrics.csv"),
            "per_class_metrics": runtime.asset_record(output_dir / "per_class_metrics.csv"),
            "truth_component_recovery": runtime.asset_record(
                output_dir / "truth_component_recovery.csv"
            ),
            "paired_metric_deltas": runtime.asset_record(
                output_dir / "paired_metric_deltas.csv"
            ),
        }
    )
    write_json(output_dir / "evaluation_manifest.json", manifest)


if __name__ == "__main__":
    main()
