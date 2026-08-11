#!/usr/bin/env python3
"""Retrospective Stage9A truth audit after pool and ranking freeze."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Mapping, Sequence

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from guidance import prior_ensemble as ensemble
from guidance.property_evaluation import (
    per_class_hard_metrics,
    size_stratified_component_metrics,
    truth_component_recovery_rows,
    truth_present_mean_iou,
)
from guidance.seismic import tensor_sha256
from guidance.structured_posterior import retrospective_hard_metrics
from scripts.stage9.audit_prior_ranking import (
    RANKING_FILENAMES,
    validate_completed_pool,
)
from scripts.stage9.common import (
    create_staging_directory,
    file_record,
    load_tensor_record,
    publish_staging_directory,
    read_csv,
    read_json,
    utc_now,
    write_csv_x,
    write_json_x,
)
from scripts.stage9.run_prior_ensemble import load_inference_case


def parse_args() -> argparse.Namespace:
    experiment = PROJECT_DIR / "experiments/stage9_flow_prior_posterior"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=experiment / "configs/stage9a_prior_support_v1.json",
    )
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--ranking-dir", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--retrospective-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPOSITORY_ROOT, text=True
    ).strip()


def validate_frozen_pool_and_rankings(
    pool_dir: Path, ranking_dir: Path
) -> dict[str, object]:
    """Complete every inference-visible validation before truth can be opened."""
    pool_manifest, candidate_rows = validate_completed_pool(pool_dir)
    ranking_manifest = read_json(Path(ranking_dir) / "ranking_manifest.json")
    if ranking_manifest.get("schema") != ensemble.STAGE9A_RANKING_SCHEMA:
        raise ValueError("invalid Stage9A ranking schema")
    if ranking_manifest.get("status") != "complete":
        raise RuntimeError("Stage9A ranking is incomplete")
    if ranking_manifest.get("truth_tensor_received") is not False:
        raise RuntimeError("ranking truth firewall failed")
    if ranking_manifest.get("ranking_policy") != ensemble.RANKING_POLICY:
        raise ValueError("ranking policy drifted")
    if ranking_manifest.get("case_id") != pool_manifest.get("case_id"):
        raise ValueError("pool/ranking case mismatch")
    if int(ranking_manifest.get("candidate_count", -1)) != len(candidate_rows):
        raise RuntimeError("pool/ranking candidate count mismatch")
    recorded_pool = ranking_manifest.get("pool_manifest")
    if not isinstance(recorded_pool, Mapping):
        raise ValueError("ranking manifest lacks frozen pool record")
    if ensemble.file_sha256(Path(pool_dir) / "manifest.json") != recorded_pool.get(
        "sha256"
    ):
        raise ValueError("candidate pool changed after ranking")
    expected_ids = {row["candidate_id"] for row in candidate_rows}
    rankings: dict[str, list[dict[str, str]]] = {}
    records = ranking_manifest.get("ranking_files")
    if not isinstance(records, Mapping):
        raise ValueError("ranking manifest lacks ranking file records")
    for name in ensemble.OBSERVATION_NAMES:
        filename = RANKING_FILENAMES[name]
        record = records.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"missing frozen ranking record: {name}")
        path = Path(ranking_dir) / filename
        if ensemble.file_sha256(path) != record.get("sha256"):
            raise ValueError(f"ranking file changed after freeze: {name}")
        rows = read_csv(path)
        if len(rows) != len(candidate_rows):
            raise RuntimeError(f"ranking is incomplete: {name}")
        if {row["candidate_id"] for row in rows} != expected_ids:
            raise ValueError(f"ranking candidate IDs differ: {name}")
        ordered = sorted(
            rows,
            key=lambda row: (
                float(row["hard_seismic_rmse"]),
                row["candidate_id"],
            ),
        )
        if [row["candidate_id"] for row in rows] != [
            row["candidate_id"] for row in ordered
        ]:
            raise ValueError(f"ranking order/tie break is invalid: {name}")
        if [int(row["rank"]) for row in rows] != list(
            range(1, len(rows) + 1)
        ):
            raise ValueError(f"ranking numbers are invalid: {name}")
        rankings[name] = rows
    return {
        "pool_manifest": pool_manifest,
        "pool_manifest_sha256": ensemble.file_sha256(
            Path(pool_dir) / "manifest.json"
        ),
        "ranking_manifest": ranking_manifest,
        "ranking_manifest_sha256": ensemble.file_sha256(
            Path(ranking_dir) / "ranking_manifest.json"
        ),
        "candidate_rows": candidate_rows,
        "rankings": rankings,
    }


def load_retrospective_case(
    retrospective_dir: Path, *, expected_case_id: str
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    manifest = read_json(Path(retrospective_dir) / "manifest.json")
    if manifest.get("schema") != "stage9a_retrospective_case_assets_v1":
        raise ValueError("invalid retrospective case schema")
    if manifest.get("status") != "complete":
        raise RuntimeError("retrospective case assets are incomplete")
    if manifest.get("case_id") != expected_case_id:
        raise ValueError("retrospective case ID mismatch")
    records = manifest.get("tensors")
    if not isinstance(records, Mapping) or set(records) != {
        "truth_labels",
        "native_body_masks",
    }:
        raise ValueError("retrospective tensor set drifted")
    return manifest, {
        name: load_tensor_record(retrospective_dir, record)
        for name, record in records.items()
    }


def validate_then_load_retrospective(
    pool_dir: Path, ranking_dir: Path, retrospective_dir: Path
) -> tuple[
    dict[str, object], tuple[dict[str, object], dict[str, torch.Tensor]]
]:
    """Enforce the firewall ordering in one directly testable API."""
    frozen = validate_frozen_pool_and_rankings(pool_dir, ranking_dir)
    assets = load_retrospective_case(
        retrospective_dir,
        expected_case_id=str(frozen["pool_manifest"]["case_id"]),
    )
    return frozen, assets


def _finite_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _centroid_distance(predicted: torch.Tensor, truth: torch.Tensor) -> float | None:
    predicted_coords = torch.nonzero(predicted, as_tuple=False).float()
    truth_coords = torch.nonzero(truth, as_tuple=False).float()
    if not len(predicted_coords) or not len(truth_coords):
        return None
    return float(
        torch.linalg.vector_norm(
            predicted_coords.mean(dim=0) - truth_coords.mean(dim=0)
        ).item()
    )


def _candidate_metrics(
    prediction: torch.Tensor,
    *,
    candidate_index: int,
    candidate_name: str,
    source_row: Mapping[str, object],
    truth_labels: torch.Tensor,
    condition_values: torch.Tensor,
    condition_mask: torch.Tensor,
    target_label: int,
    thresholds: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    predicted = prediction.long()
    truth = truth_labels.long()
    violations = int(
        ((predicted != condition_values.long()) & condition_mask.bool()).sum().item()
    )
    truth_target = truth == int(target_label)
    predicted_target = predicted == int(target_label)
    intersection = int((truth_target & predicted_target).sum().item())
    union = int((truth_target | predicted_target).sum().item())
    truth_count = int(truth_target.sum().item())
    predicted_count = int(predicted_target.sum().item())
    components = truth_component_recovery_rows(
        predicted, truth, target_label, candidate_index
    )
    major = [
        float(row["recall"])
        for row in components
        if int(row["truth_component_rank"]) in (1, 2, 3, 4)
    ]
    if len(major) != 4:
        raise RuntimeError("truth target lacks four major connected components")
    geometry = retrospective_hard_metrics(
        predicted,
        truth_labels=truth,
        condition_mask=condition_mask,
        target_label=target_label,
    )
    row: dict[str, object] = {
        "candidate_id": candidate_name,
        "candidate_index": candidate_index,
        "source_seed": int(source_row["source_seed"]),
        "source_noise_sha256": source_row["source_noise_sha256"],
        "hard_model_sha256": source_row["hard_model_sha256"],
        "predicted_observation_sha256": source_row[
            "predicted_observation_sha256"
        ],
        "condition_violations": violations,
        "global_accuracy": float((predicted == truth).float().mean().item()),
        "truth_present_mean_iou": truth_present_mean_iou(predicted, truth),
        "label9_iou": _finite_ratio(intersection, union),
        "label9_precision": _finite_ratio(intersection, predicted_count),
        "label9_recall": _finite_ratio(intersection, truth_count),
        "label9_truth_voxels": truth_count,
        "label9_predicted_voxels": predicted_count,
        "label9_absolute_volume_error_fraction": _finite_ratio(
            abs(predicted_count - truth_count), truth_count
        ),
        "label9_centroid_distance_voxels": _centroid_distance(
            predicted_target[0, 0], truth_target[0, 0]
        ),
        "major_component_min_recall": min(major),
        "major_component_mean_recall": sum(major) / len(major),
        "body_recall": geometry["body_recall"],
        "body_precision": geometry["body_precision"],
        "matched_body_count": geometry["matched_body_count"],
        "matched_body_center_error_mean": geometry[
            "matched_body_center_error_mean"
        ],
        "matched_body_relative_size_error_mean": geometry[
            "matched_body_relative_size_error_mean"
        ],
        **size_stratified_component_metrics(predicted_target),
        "retrospective_only": True,
        "used_for_ranking": False,
    }
    checks = ensemble.support_checks(row, thresholds)
    row["support_pass"] = all(checks.values())
    for name, passed in checks.items():
        row[f"support_{name}"] = passed
    class_rows = per_class_hard_metrics(
        predicted, truth, candidate_index, class_ids=None
    )
    for class_row in class_rows:
        class_row.update(
            {
                "candidate_id": candidate_name,
                "retrospective_only": True,
                "used_for_ranking": False,
            }
        )
        for field in ("iou", "precision", "recall"):
            if not math.isfinite(float(class_row[field])):
                class_row[field] = 0.0
    for component in components:
        component.update(
            {
                "candidate_id": candidate_name,
                "retrospective_only": True,
                "used_for_ranking": False,
            }
        )
    return row, class_rows, components


def _best_of_n(
    rows: Sequence[Mapping[str, object]], schedule: Sequence[int]
) -> list[dict[str, object]]:
    result = []
    fields = (
        "global_accuracy",
        "truth_present_mean_iou",
        "label9_iou",
        "label9_precision",
        "label9_recall",
        "major_component_min_recall",
        "major_component_mean_recall",
    )
    for count in schedule:
        if int(count) > len(rows):
            continue
        subset = list(rows[: int(count)])
        row: dict[str, object] = {
            "N": int(count),
            "support_passing_candidate_count": sum(
                bool(value["support_pass"]) for value in subset
            ),
            "support_pass": any(bool(value["support_pass"]) for value in subset),
            "retrospective_oracle_only": True,
            "deployable_selector": False,
        }
        for field in fields:
            best = max(subset, key=lambda value: (float(value[field]), value["candidate_id"]))
            row[f"best_{field}"] = float(best[field])
            row[f"best_{field}_candidate_id"] = best["candidate_id"]
        result.append(row)
    return result


def _enrichment_and_correlations(
    rows: Sequence[Mapping[str, object]],
    rankings: Mapping[str, Sequence[Mapping[str, object]]],
    fractions: Sequence[float],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metric_index = {str(row["candidate_id"]): row for row in rows}
    enrichment: list[dict[str, object]] = []
    correlations: list[dict[str, object]] = []
    for observation in ensemble.OBSERVATION_NAMES:
        ranking = list(rankings[observation])
        ordered_metrics = [metric_index[str(row["candidate_id"])] for row in ranking]
        losses = [float(row["hard_seismic_rmse"]) for row in ranking]
        for field in ensemble.CORRELATION_METRICS:
            rho = ensemble.spearman_rank_correlation(
                losses, [float(row[field]) for row in ordered_metrics]
            )
            correlations.append(
                {
                    "observation": observation,
                    "metric": field,
                    "spearman_rho": rho,
                    "desired_direction": "negative",
                }
            )
        for field in ensemble.TARGET_METRICS:
            full_mean = sum(float(row[field]) for row in ordered_metrics) / len(
                ordered_metrics
            )
            enrichment.append(
                {
                    "observation": observation,
                    "metric": field,
                    "subset": "full",
                    "fraction": 1.0,
                    "count": len(ordered_metrics),
                    "mean": full_mean,
                    "full_mean": full_mean,
                    "enrichment": 0.0,
                }
            )
            for fraction in fractions:
                count = max(1, math.ceil(float(fraction) * len(ordered_metrics)))
                top_mean = sum(
                    float(row[field]) for row in ordered_metrics[:count]
                ) / count
                subset = f"top_{int(round(100 * float(fraction)))}pct"
                enrichment.append(
                    {
                        "observation": observation,
                        "metric": field,
                        "subset": subset,
                        "fraction": float(fraction),
                        "count": count,
                        "mean": top_mean,
                        "full_mean": full_mean,
                        "enrichment": top_mean - full_mean,
                    }
                )
    return enrichment, correlations


def run_retrospective_audit(
    *,
    frozen: Mapping[str, object],
    config: Mapping[str, object],
    pool_dir: Path,
    case_dir: Path,
    retrospective_assets: tuple[dict[str, object], dict[str, torch.Tensor]],
) -> dict[str, object]:
    """Open truth only after the caller supplies a completed frozen validation."""
    pool_manifest = frozen["pool_manifest"]
    case_id = str(pool_manifest["case_id"])
    retrospective_manifest, retrospective_tensors = retrospective_assets
    if retrospective_manifest.get("case_id") != case_id:
        raise ValueError("validated retrospective assets do not match pool case")
    case_manifest, inference_tensors = load_inference_case(case_dir, case_id)
    if retrospective_manifest["inference_manifest_sha256"] != ensemble.file_sha256(
        Path(case_dir) / "manifest.json"
    ):
        raise ValueError("retrospective/inference case link mismatch")
    truth_labels = retrospective_tensors["truth_labels"].long()
    condition_values = inference_tensors["condition_values"].long()
    condition_mask = inference_tensors["condition_mask"].bool()
    candidate_rows = list(frozen["candidate_rows"])
    candidate_index = {row["candidate_id"]: row for row in candidate_rows}
    metrics: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    category_counts = torch.zeros((15, truth_labels.numel()), dtype=torch.int32)
    seen: set[str] = set()
    model_chunks = pool_manifest.get("model_chunks")
    if not isinstance(model_chunks, list) or not model_chunks:
        raise ValueError("candidate pool lacks hard-model chunks")
    for record in model_chunks:
        models = ensemble.load_tensor_gzip(
            Path(pool_dir) / str(record["path"]), expected=record
        )
        start = int(record["candidate_start"])
        stop = int(record["candidate_stop_exclusive"])
        if models.shape[0] != stop - start:
            raise ValueError("hard-model chunk interval mismatch")
        for offset, index in enumerate(range(start, stop)):
            identifier = ensemble.candidate_id(index)
            model = models[offset : offset + 1].long()
            if tensor_sha256(models[offset : offset + 1]) != candidate_index[identifier][
                "hard_model_sha256"
            ]:
                raise ValueError(f"hard-model hash mismatch: {identifier}")
            row, candidate_classes, candidate_components = _candidate_metrics(
                model,
                candidate_index=index,
                candidate_name=identifier,
                source_row=candidate_index[identifier],
                truth_labels=truth_labels,
                condition_values=condition_values,
                condition_mask=condition_mask,
                target_label=int(config["target_label"]),
                thresholds=config["support_thresholds"],
            )
            metrics.append(row)
            class_rows.extend(candidate_classes)
            component_rows.extend(candidate_components)
            categories = (model.reshape(-1) + 1).long().unsqueeze(0)
            category_counts.scatter_add_(
                0, categories, torch.ones_like(categories, dtype=torch.int32)
            )
            seen.add(identifier)
    if seen != set(candidate_index) or len(metrics) != len(candidate_rows):
        raise RuntimeError("retrospective hard-model pool is incomplete")
    metrics.sort(key=lambda row: int(row["candidate_index"]))
    count = len(metrics)
    maximum = category_counts.max(dim=0).values.float()
    mean_modal_disagreement = float((1.0 - maximum / count).mean().item())
    nonunanimous = float((maximum < count).float().mean().item())
    if count > 1:
        pairwise = category_counts.double() * (count - category_counts).double()
        expected_pairwise = float(
            (pairwise.sum(dim=0) / (count * (count - 1))).mean().item()
        )
    else:
        expected_pairwise = 0.0
    best_rows = _best_of_n(metrics, config["best_of_n"])
    enrichment, correlations = _enrichment_and_correlations(
        metrics, frozen["rankings"], config["top_fractions"]
    )
    support_ids = [row["candidate_id"] for row in metrics if row["support_pass"]]
    discrimination = ensemble.discrimination_checks(correlations, enrichment)
    return {
        "metrics": metrics,
        "class_rows": class_rows,
        "component_rows": component_rows,
        "best_of_n": best_rows,
        "enrichment": enrichment,
        "correlations": correlations,
        "case_support_pass": bool(support_ids),
        "support_passing_candidate_ids": support_ids,
        "case_discrimination_pass": bool(discrimination["passed"]),
        "discrimination_checks": discrimination,
        "ensemble": {
            "candidate_count": count,
            "unique_hard_model_count": len(
                {str(row["hard_model_sha256"]) for row in metrics}
            ),
            "mean_voxel_modal_disagreement": mean_modal_disagreement,
            "nonunanimous_voxel_fraction": nonunanimous,
            "expected_pairwise_hard_label_disagreement": expected_pairwise,
        },
        "retrospective_manifest": retrospective_manifest,
        "case_manifest": case_manifest,
    }


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    ensemble.validate_protocol_config(config)
    # This API validates every inference artifact before it can open truth.
    frozen, retrospective_assets = validate_then_load_retrospective(
        args.pool_dir, args.ranking_dir, args.retrospective_dir
    )
    result = run_retrospective_audit(
        frozen=frozen,
        config=config,
        pool_dir=args.pool_dir,
        case_dir=args.case_dir,
        retrospective_assets=retrospective_assets,
    )
    staging = create_staging_directory(args.output_dir)
    outputs = {
        "truth_metrics.csv": result["metrics"],
        "per_class_metrics.csv": result["class_rows"],
        "component_metrics.csv": result["component_rows"],
        "best_of_n.csv": result["best_of_n"],
        "enrichment.csv": result["enrichment"],
        "correlations.csv": result["correlations"],
    }
    records = {}
    for filename, rows in outputs.items():
        write_csv_x(staging / filename, rows)
        records[filename] = file_record(staging / filename, relative_to=staging)
    manifest = {
        "schema": ensemble.STAGE9A_AUDIT_SCHEMA,
        "status": "complete",
        "scientific_evidence": bool(frozen["pool_manifest"]["scientific_evidence"]),
        "mode": frozen["pool_manifest"]["mode"],
        "case_id": frozen["pool_manifest"]["case_id"],
        "candidate_count": frozen["pool_manifest"]["candidate_count"],
        "pool_manifest_sha256_before_truth_load": frozen["pool_manifest_sha256"],
        "ranking_manifest_sha256_before_truth_load": frozen[
            "ranking_manifest_sha256"
        ],
        "truth_loaded_only_after_inference_validation": True,
        "truth_used_for_generation": False,
        "truth_used_for_ranking": False,
        "truth_used_for_metrics_only": True,
        "case_support_pass": result["case_support_pass"],
        "support_passing_candidate_ids": result["support_passing_candidate_ids"],
        "case_discrimination_pass": result["case_discrimination_pass"],
        "discrimination_checks": result["discrimination_checks"],
        "ensemble": result["ensemble"],
        "outputs": records,
        "config": file_record(args.config),
        "pool_manifest": file_record(Path(args.pool_dir) / "manifest.json"),
        "ranking_manifest": file_record(
            Path(args.ranking_dir) / "ranking_manifest.json"
        ),
        "inference_case_manifest": file_record(Path(args.case_dir) / "manifest.json"),
        "retrospective_case_manifest": file_record(
            Path(args.retrospective_dir) / "manifest.json"
        ),
        "completed_at_utc": utc_now(),
        "exact_command": shlex.join([sys.executable, *sys.argv]),
        "git_branch": _git("branch", "--show-current"),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_status": _git("status", "--short"),
        "runner": file_record(Path(__file__)),
        "prior_ensemble": file_record(Path(ensemble.__file__)),
    }
    write_json_x(staging / "audit_manifest.json", manifest)
    publish_staging_directory(staging, args.output_dir)
    print(
        json.dumps(
            {
                "status": "complete",
                "mode": manifest["mode"],
                "case_id": manifest["case_id"],
                "SUPPORT_PASS": manifest["case_support_pass"],
                "DISCRIMINATION_PASS": manifest["case_discrimination_pass"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
