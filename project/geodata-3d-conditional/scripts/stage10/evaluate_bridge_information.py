#!/usr/bin/env python3
"""Freeze Stage10-A controls, then retrospectively measure bridge information."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Mapping

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from guidance.geophysical_probability_bridge import (
    shuffle_xy_probability,
    validate_probabilities,
)
from guidance.prior_ensemble import file_sha256
from scripts.stage10.common import (
    EXPERIMENT_DIR,
    inference_case_dir,
    load_frozen_config,
    load_stage10_inference_case,
    retrospective_case_dir,
    target_probability_channel,
    validate_bridge_collection,
)
from scripts.stage9.audit_prior_truth import load_retrospective_case
from scripts.stage9.common import (
    create_staging_directory,
    file_record,
    load_tensor_record,
    publish_staging_directory,
    read_json,
    save_tensor_x,
    utc_now,
    write_csv_x,
    write_json_x,
)


CONTROL_SCHEMA = "stage10a_frozen_bridge_controls_v1"


def average_precision(scores: torch.Tensor, targets: torch.Tensor) -> float:
    """Tie-aware average precision for binary targets."""
    score = scores.detach().cpu().double().reshape(-1)
    target = targets.detach().cpu().bool().reshape(-1)
    if score.numel() != target.numel() or not score.numel():
        raise ValueError("scores and targets must be non-empty and matching")
    if not torch.isfinite(score).all():
        raise ValueError("scores contain NaN/Inf")
    positive_count = int(target.sum().item())
    if positive_count == 0:
        raise ValueError("average precision requires at least one positive")
    order = torch.argsort(score, descending=True, stable=True)
    sorted_score = score[order]
    sorted_target = target[order].long()
    cumulative_tp = sorted_target.cumsum(0)
    cumulative_fp = (~target[order]).long().cumsum(0)
    group_end = torch.ones_like(sorted_target, dtype=torch.bool)
    group_end[:-1] = sorted_score[:-1] != sorted_score[1:]
    tp = cumulative_tp[group_end].double()
    fp = cumulative_fp[group_end].double()
    previous = torch.cat((tp.new_zeros(1), tp[:-1]))
    return float((((tp - previous) / positive_count) * (tp / (tp + fp))).sum().item())


def roc_auc(scores: torch.Tensor, targets: torch.Tensor) -> float:
    """Tie-aware binary ROC area via average ranks."""
    score = scores.detach().cpu().double().reshape(-1)
    target = targets.detach().cpu().bool().reshape(-1)
    positive = int(target.sum().item())
    negative = int((~target).sum().item())
    if positive == 0 or negative == 0:
        raise ValueError("ROC-AUC requires both classes")
    order = torch.argsort(score, stable=True)
    sorted_score = score[order]
    sorted_target = target[order].double()
    _, inverse, counts = torch.unique_consecutive(
        sorted_score, return_inverse=True, return_counts=True
    )
    ends = counts.cumsum(0).double()
    starts = ends - counts.double()
    average_ranks = 0.5 * ((starts + 1.0) + ends)
    positive_per_group = torch.zeros_like(average_ranks).scatter_add_(
        0, inverse, sorted_target
    )
    rank_sum = (average_ranks * positive_per_group).sum()
    return float(((rank_sum - positive * (positive + 1) / 2) / (positive * negative)).item())


def binary_information_metrics(
    probability: torch.Tensor,
    truth_target: torch.Tensor,
    evaluation_mask: torch.Tensor,
) -> dict[str, float]:
    """Return prevalence, AP, Brier and secondary ROC-AUC on one frozen domain."""
    if probability.shape != truth_target.shape or probability.shape != evaluation_mask.shape:
        raise ValueError("probability, truth and evaluation mask must match")
    active = evaluation_mask.bool()
    score = probability[active].double()
    target = truth_target[active].bool()
    if not score.numel() or bool(((score < 0) | (score > 1)).any()):
        raise ValueError("invalid probability evaluation domain")
    return {
        "voxel_count": int(score.numel()),
        "positive_count": int(target.sum().item()),
        "prevalence": float(target.double().mean().item()),
        "auprc": average_precision(score, target),
        "brier": float((score - target.double()).square().mean().item()),
        "roc_auc": roc_auc(score, target),
        "probability_mean": float(score.mean().item()),
        "probability_minimum": float(score.min().item()),
        "probability_maximum": float(score.max().item()),
    }


def reliability_rows(
    probability: torch.Tensor,
    truth_target: torch.Tensor,
    evaluation_mask: torch.Tensor,
    *,
    bins: int = 10,
) -> list[dict[str, object]]:
    score = probability[evaluation_mask.bool()].double()
    target = truth_target[evaluation_mask.bool()].double()
    result = []
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = (score >= lower) & (score < upper if index + 1 < bins else score <= upper)
        count = int(selected.sum().item())
        result.append(
            {
                "bin_index": index,
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_probability": float(score[selected].mean().item()) if count else None,
                "observed_frequency": float(target[selected].mean().item()) if count else None,
            }
        )
    return result


def _freeze_controls(
    config: Mapping[str, object],
    bridges: Mapping[str, tuple[dict[str, object], dict[str, torch.Tensor]]],
) -> dict[str, dict[str, torch.Tensor]]:
    """Construct controls only after all correct bridge manifests validate."""
    seed_bank = read_json(EXPERIMENT_DIR / "configs/flow_seed_bank.json")
    class_model = read_json(EXPERIMENT_DIR / "configs/petrophysical_class_model.json")
    label9_channel = target_probability_channel(class_model, int(config["target_label"]))
    label9_prior = float(class_model["class_prior"][label9_channel])
    controls_dir = EXPERIMENT_DIR / "controls"
    staging = create_staging_directory(controls_dir)
    case_ids = list(config["case_ids"])
    tensors: dict[str, dict[str, torch.Tensor]] = {}
    records: dict[str, dict[str, object]] = {}
    for index, case_id in enumerate(case_ids):
        correct_all = bridges[case_id][1]["probability_all_classes"]
        correct_label9 = bridges[case_id][1]["probability_label9"]
        shuffled_all, permutation = shuffle_xy_probability(
            correct_all, seed=int(seed_bank["cases"][case_id]["shuffle_seed"])
        )
        shuffled = shuffled_all[:, label9_channel : label9_channel + 1].contiguous()
        wrong_id = case_ids[(index + 1) % len(case_ids)]
        wrong = bridges[wrong_id][1]["probability_label9"].clone().contiguous()
        constant = torch.full_like(correct_label9, label9_prior)
        tensors[case_id] = {
            "constant_prior": constant,
            "shuffled_xy": shuffled,
            "wrong_case": wrong,
        }
        records[case_id] = {
            "constant_prior": save_tensor_x(
                staging / "constant_prior" / case_id / "probability_label9.pt", constant
            ),
            "shuffled_xy": save_tensor_x(
                staging / "shuffled_xy" / case_id / "probability_label9.pt", shuffled
            ),
            "shuffle_permutation": save_tensor_x(
                staging / "shuffled_xy" / case_id / "xy_permutation.pt", permutation
            ),
            "wrong_case": save_tensor_x(
                staging / "wrong_case" / case_id / "probability_label9.pt", wrong
            ),
            "wrong_case_source": wrong_id,
            "shuffle_seed": int(seed_bank["cases"][case_id]["shuffle_seed"]),
            "correct_probability_tensor_sha256": bridges[case_id][0]["generated_tensors"]["probability_label9"]["tensor_sha256"],
        }
        for name in ("constant_prior", "shuffled_xy", "shuffle_permutation", "wrong_case"):
            records[case_id][name]["path"] = str(
                (
                    (staging / ("constant_prior" if name == "constant_prior" else "shuffled_xy" if name in {"shuffled_xy", "shuffle_permutation"} else "wrong_case") / case_id / ("xy_permutation.pt" if name == "shuffle_permutation" else "probability_label9.pt"))
                    .relative_to(staging)
                )
            )
    manifest = {
        "schema": CONTROL_SCHEMA,
        "status": "complete_frozen_before_truth_evaluation",
        "case_ids": case_ids,
        "constant_prior_label9": label9_prior,
        "control_records": records,
        "truth_tensor_received": False,
        "truth_used_for_construction": False,
        "completed_at_utc": utc_now(),
    }
    write_json_x(staging / "manifest.json", manifest)
    publish_staging_directory(staging, controls_dir)
    return tensors


def _load_frozen_controls(
    config: Mapping[str, object],
) -> dict[str, dict[str, torch.Tensor]]:
    root = EXPERIMENT_DIR / "controls"
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema") != CONTROL_SCHEMA or manifest.get("status") != "complete_frozen_before_truth_evaluation":
        raise RuntimeError("Stage10-A controls are not frozen")
    if manifest.get("truth_tensor_received") is not False:
        raise RuntimeError("truth firewall failed during control construction")
    result = {}
    for case_id in config["case_ids"]:
        records = manifest["control_records"][case_id]
        result[case_id] = {
            name: load_tensor_record(root, records[name])
            for name in ("constant_prior", "shuffled_xy", "wrong_case")
        }
    return result


def main() -> None:
    config = load_frozen_config()
    bridges = validate_bridge_collection(config)
    # This must happen before any call that can load retrospective truth.
    _freeze_controls(config, bridges)
    controls = _load_frozen_controls(config)
    diagnostics = EXPERIMENT_DIR / "diagnostics"
    if diagnostics.exists():
        raise FileExistsError(f"refusing to reuse immutable diagnostics: {diagnostics}")
    diagnostics.mkdir(parents=True)
    metric_rows = []
    reliability = []
    control_rows = []
    case_passes = {}
    for case_id in config["case_ids"]:
        inference_manifest, inference_tensors = load_stage10_inference_case(config, case_id)
        retrospective_manifest, retrospective_tensors = load_retrospective_case(
            retrospective_case_dir(config, case_id), expected_case_id=case_id
        )
        if retrospective_manifest["inference_manifest_sha256"] != file_sha256(
            inference_case_dir(config, case_id) / "manifest.json"
        ):
            raise ValueError("retrospective truth is not linked to the frozen inference case")
        truth_target = retrospective_tensors["truth_labels"].long() == int(config["target_label"])
        evaluation_mask = inference_tensors["subsurface_mask"].bool() & ~inference_tensors["condition_mask"].bool()
        probabilities = {
            "correct": bridges[case_id][1]["probability_label9"],
            **controls[case_id],
        }
        case_metrics = {}
        for arm, probability in probabilities.items():
            metrics = binary_information_metrics(probability, truth_target, evaluation_mask)
            case_metrics[arm] = metrics
            metric_rows.append({"case_id": case_id, "arm": arm, **metrics})
            for row in reliability_rows(probability, truth_target, evaluation_mask):
                reliability.append({"case_id": case_id, "arm": arm, **row})
        checks = {
            "correct_auprc_above_constant": case_metrics["correct"]["auprc"] > case_metrics["constant_prior"]["auprc"],
            "correct_auprc_above_shuffled": case_metrics["correct"]["auprc"] > case_metrics["shuffled_xy"]["auprc"],
            "correct_auprc_above_wrong_case": case_metrics["correct"]["auprc"] > case_metrics["wrong_case"]["auprc"],
            "correct_brier_below_constant": case_metrics["correct"]["brier"] < case_metrics["constant_prior"]["brier"],
        }
        case_passes[case_id] = all(checks.values())
        control_rows.append(
            {
                "case_id": case_id,
                **checks,
                "case_pass": case_passes[case_id],
                "wrong_case_source": read_json(EXPERIMENT_DIR / "controls/manifest.json")["control_records"][case_id]["wrong_case_source"],
                "shuffle_seed": read_json(EXPERIMENT_DIR / "controls/manifest.json")["control_records"][case_id]["shuffle_seed"],
            }
        )
    passed_count = sum(case_passes.values())
    passed = passed_count >= int(config["stage10a"]["pass_case_minimum"])
    action = "PROCEED_STAGE10_B_SMALL_PAIRED_FLOW_PILOT" if passed else "STOP_BRIDGE_NO_GEOPHYSICAL_INFORMATION"
    write_csv_x(diagnostics / "bridge_information_metrics.csv", metric_rows)
    write_csv_x(diagnostics / "bridge_controls.csv", control_rows)
    write_json_x(diagnostics / "bridge_reliability.json", reliability)
    write_json_x(
        diagnostics / "stage10a_decision.json",
        {
            "schema": "stage10a_bridge_information_decision_v1",
            "status": "complete",
            "case_passes": case_passes,
            "passing_case_count": passed_count,
            "required_case_count": int(config["stage10a"]["pass_case_minimum"]),
            "stage10a_pass": passed,
            "machine_action": action,
            "truth_loaded_only_after_bridge_and_controls_frozen": True,
            "bridge_collection_manifest": file_record(EXPERIMENT_DIR / "bridge/manifest.json", relative_to=REPOSITORY_ROOT),
            "controls_manifest": file_record(EXPERIMENT_DIR / "controls/manifest.json", relative_to=REPOSITORY_ROOT),
            "completed_at_utc": utc_now(),
        },
    )
    print(json.dumps({"status": "PASS" if passed else "STOP", "machine_action": action, "case_passes": case_passes}))


if __name__ == "__main__":
    main()
