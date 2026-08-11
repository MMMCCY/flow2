#!/usr/bin/env python3
"""Freeze controls, evaluate Stage12B-A, and apply the prospective machine gate."""

from __future__ import annotations

from pathlib import Path
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import inference_runtime as runtime
from guidance.geophysical_probability_bridge import shuffle_xy_probability
from guidance.prior_ensemble import file_sha256
from scripts.stage10.evaluate_bridge_information import binary_information_metrics
from scripts.stage12b.common import (
    CASE_IDS,
    CONFIG_PATH,
    EXPERIMENT_DIR,
    STAGE12A_DIR,
    load_bridge_case,
    load_config,
    load_inference_case,
    verify_stage12a_files,
)
from scripts.stage9.common import (
    create_staging_directory,
    file_record,
    publish_staging_directory,
    save_tensor_x,
    utc_now,
    write_csv_x,
    write_json_x,
)


METRICS = ("auprc", "brier", "roc_auc")


def _record_with_relative_path(
    staging: Path, path: Path, tensor: torch.Tensor
) -> dict[str, object]:
    record = save_tensor_x(path, tensor)
    record["path"] = str(path.relative_to(staging))
    return record


def _freeze_controls(
    config: dict[str, object],
    bridges: dict[str, tuple[dict[str, object], dict[str, torch.Tensor]]],
) -> dict[str, dict[str, torch.Tensor]]:
    final = EXPERIMENT_DIR / "controls"
    staging = create_staging_directory(final)
    controls: dict[str, dict[str, torch.Tensor]] = {}
    tensor_records: dict[str, object] = {}
    for case_id in CASE_IDS:
        post_all = bridges[case_id][1]["probability_all_classes_post"]
        post_label9 = bridges[case_id][1]["probability_label9_post"]
        seed = int(config["stage12b_a"]["shuffle_seeds"][case_id])
        shuffled_all, permutation = shuffle_xy_probability(post_all, seed=seed)
        label9_channel = int(bridges[case_id][0]["target_probability_channel"])
        shuffled_label9 = shuffled_all[:, label9_channel : label9_channel + 1].contiguous()
        constant = torch.full_like(
            post_label9, float(config["stage12b_a"]["constant_label9_probability"])
        )
        controls[case_id] = {
            "constant_prior": constant,
            "shuffled_xy": shuffled_label9,
        }
        case_root = staging / case_id
        tensor_records[case_id] = {
            "constant_prior_label9": _record_with_relative_path(
                staging, case_root / "constant_prior_label9.pt", constant
            ),
            "shuffled_xy_all_classes": _record_with_relative_path(
                staging, case_root / "shuffled_xy_all_classes.pt", shuffled_all
            ),
            "shuffled_xy_label9": _record_with_relative_path(
                staging, case_root / "shuffled_xy_label9.pt", shuffled_label9
            ),
            "xy_permutation": _record_with_relative_path(
                staging, case_root / "xy_permutation.pt", permutation
            ),
            "shuffle_seed": seed,
        }
    write_json_x(
        staging / "manifest.json",
        {
            "schema": "stage12b_a_frozen_controls_v1",
            "status": "complete_frozen_before_truth_evaluation",
            "created_at_utc": utc_now(),
            "truth_tensor_received": False,
            "truth_metric_received": False,
            "case_ids": list(CASE_IDS),
            "constant_label9_probability": config["stage12b_a"][
                "constant_label9_probability"
            ],
            "shuffle_policy": config["stage12b_a"]["shuffle_policy"],
            "wrong_case_policy": "all_20_off_diagonal_post_seismic_bridge_truth_pairs",
            "prior_only_policy": "same_frozen_12_member_property_bank_before_assimilation",
            "tensors": tensor_records,
        },
    )
    publish_staging_directory(staging, final)
    return controls


def _load_truth_evaluation(
    config: dict[str, object], case_id: str
) -> tuple[torch.Tensor, torch.Tensor]:
    source = STAGE12A_DIR / "cases" / case_id / "truth/true_model.pt"
    if file_sha256(source) != config["stage12a"]["case_hashes"][case_id]["true_model"]:
        raise ValueError(f"{case_id}: frozen truth hash changed")
    truth = runtime.load_tensor(source, map_location="cpu").long()
    _, inference = load_inference_case(case_id)
    evaluation = inference["subsurface_mask"].bool() & ~inference["condition_mask"].bool()
    target = truth.eq(int(config["target_label"]))
    if not bool(target[evaluation].any()):
        raise ValueError(f"{case_id}: no hidden label9 positive in evaluation domain")
    return target, evaluation


def _matrix_rows(
    values: dict[tuple[str, str], dict[str, float]], metric: str
) -> list[dict[str, object]]:
    return [
        {
            "bridge_case": bridge,
            **{truth: values[(bridge, truth)][metric] for truth in CASE_IDS},
        }
        for bridge in CASE_IDS
    ]


def main() -> None:
    config = load_config()
    verify_stage12a_files(config)
    bridges = {case_id: load_bridge_case(case_id) for case_id in CASE_IDS}
    # Control construction deliberately completes before the first truth tensor load.
    controls = _freeze_controls(config, bridges)
    truths: dict[str, torch.Tensor] = {}
    evaluation_masks: dict[str, torch.Tensor] = {}
    for case_id in CASE_IDS:
        truths[case_id], evaluation_masks[case_id] = _load_truth_evaluation(config, case_id)

    transfer: dict[tuple[str, str], dict[str, float]] = {}
    transfer_rows: list[dict[str, object]] = []
    for bridge_case in CASE_IDS:
        probability = bridges[bridge_case][1]["probability_label9_post"]
        for truth_case in CASE_IDS:
            metrics = binary_information_metrics(
                probability, truths[truth_case], evaluation_masks[truth_case]
            )
            transfer[(bridge_case, truth_case)] = metrics
            transfer_rows.append(
                {
                    "bridge_case": bridge_case,
                    "truth_case": truth_case,
                    "pair_type": "diagonal" if bridge_case == truth_case else "off_diagonal",
                    **metrics,
                }
            )

    per_case_rows: list[dict[str, object]] = []
    by_case_arm: dict[tuple[str, str], dict[str, float]] = {}
    for case_id in CASE_IDS:
        arms = {
            "post_seismic_correct": bridges[case_id][1]["probability_label9_post"],
            "prior_only": bridges[case_id][1]["probability_label9_prior"],
            "shuffled_xy": controls[case_id]["shuffled_xy"],
            "constant_prior": controls[case_id]["constant_prior"],
        }
        for arm, probability in arms.items():
            metrics = binary_information_metrics(
                probability, truths[case_id], evaluation_masks[case_id]
            )
            by_case_arm[(case_id, arm)] = metrics
            per_case_rows.append({"case_id": case_id, "arm": arm, **metrics})

    diagonal = [transfer[(case_id, case_id)]["auprc"] for case_id in CASE_IDS]
    off_diagonal = [
        transfer[(source, target)]["auprc"]
        for source in CASE_IDS
        for target in CASE_IDS
        if source != target
    ]
    diagonal_mean = sum(diagonal) / len(diagonal)
    off_diagonal_mean = sum(off_diagonal) / len(off_diagonal)
    row_maximum = {
        case_id: transfer[(case_id, case_id)]["auprc"]
        >= max(transfer[(case_id, target)]["auprc"] for target in CASE_IDS)
        for case_id in CASE_IDS
    }
    column_maximum = {
        case_id: transfer[(case_id, case_id)]["auprc"]
        >= max(transfer[(source, case_id)]["auprc"] for source in CASE_IDS)
        for case_id in CASE_IDS
    }
    comparison_rows: list[dict[str, object]] = []
    catastrophic: dict[str, bool] = {}
    for case_id in CASE_IDS:
        post = by_case_arm[(case_id, "post_seismic_correct")]
        prior = by_case_arm[(case_id, "prior_only")]
        shuffled = by_case_arm[(case_id, "shuffled_xy")]
        constant = by_case_arm[(case_id, "constant_prior")]
        brier_increase = post["brier"] - prior["brier"]
        catastrophic[case_id] = (
            brier_increase
            > float(
                config["stage12b_a"]["success_gate"][
                    "catastrophic_brier_absolute_increase"
                ]
            )
            and post["brier"]
            > float(
                config["stage12b_a"]["success_gate"][
                    "catastrophic_brier_relative_factor"
                ]
            )
            * prior["brier"]
        )
        comparison_rows.append(
            {
                "case_id": case_id,
                "correct_auprc": post["auprc"],
                "prior_auprc": prior["auprc"],
                "shuffled_auprc": shuffled["auprc"],
                "constant_auprc": constant["auprc"],
                "post_minus_prior_auprc": post["auprc"] - prior["auprc"],
                "correct_brier": post["brier"],
                "prior_brier": prior["brier"],
                "prior_minus_post_brier": prior["brier"] - post["brier"],
                "correct_roc_auc": post["roc_auc"],
                "prior_roc_auc": prior["roc_auc"],
                "diagonal_is_row_maximum": row_maximum[case_id],
                "diagonal_is_column_maximum": column_maximum[case_id],
                "correct_above_shuffled": post["auprc"] > shuffled["auprc"],
                "correct_above_constant": post["auprc"] > constant["auprc"],
                "post_ap_above_prior_ap": post["auprc"] > prior["auprc"],
                "catastrophic_brier_degradation": catastrophic[case_id],
            }
        )

    gate = config["stage12b_a"]["success_gate"]
    clauses = {
        "diagonal_mean_above_off_diagonal_mean": diagonal_mean > off_diagonal_mean,
        "at_least_4_of_5_diagonal_row_maximum": sum(row_maximum.values())
        >= int(gate["minimum_diagonal_row_maximum_cases"]),
        "at_least_4_of_5_correct_above_shuffled": sum(
            row["correct_above_shuffled"] for row in comparison_rows
        )
        >= int(gate["minimum_correct_above_shuffled_cases"]),
        "all_5_correct_above_constant": sum(
            row["correct_above_constant"] for row in comparison_rows
        )
        >= int(gate["minimum_correct_above_constant_cases"]),
        "at_least_4_of_5_post_ap_above_prior": sum(
            row["post_ap_above_prior_ap"] for row in comparison_rows
        )
        >= int(gate["minimum_post_ap_above_prior_ap_cases"]),
        "no_catastrophic_brier_degradation": not any(catastrophic.values()),
    }
    passed = all(clauses.values())
    machine_decision = (
        "BRIDGE_FULLGEO_PASS" if passed else "STOP_FULLGEO_BRIDGE_NOT_CASE_SPECIFIC"
    )
    final = EXPERIMENT_DIR / "evaluation/stage12b_a"
    staging = create_staging_directory(final)
    write_csv_x(staging / "transfer_metrics.csv", transfer_rows)
    for metric in METRICS:
        write_csv_x(staging / f"transfer_{metric}_matrix.csv", _matrix_rows(transfer, metric))
    write_csv_x(staging / "per_case_arm_metrics.csv", per_case_rows)
    write_csv_x(staging / "per_case_comparisons.csv", comparison_rows)
    summary = {
        "schema": "stage12b_a_bridge_only_summary_v1",
        "status": "complete",
        "created_at_utc": utc_now(),
        "case_ids": list(CASE_IDS),
        "primary_aggregation": "per_case_then_unweighted_macro",
        "pooled_voxels_used_as_primary": False,
        "diagonal_mean_auprc": diagonal_mean,
        "off_diagonal_mean_auprc": off_diagonal_mean,
        "diagonal_minus_off_diagonal_auprc": diagonal_mean - off_diagonal_mean,
        "diagonal_row_maximum_count": sum(row_maximum.values()),
        "diagonal_column_maximum_count": sum(column_maximum.values()),
        "correct_above_shuffled_count": sum(
            row["correct_above_shuffled"] for row in comparison_rows
        ),
        "correct_above_constant_count": sum(
            row["correct_above_constant"] for row in comparison_rows
        ),
        "post_ap_above_prior_count": sum(
            row["post_ap_above_prior_ap"] for row in comparison_rows
        ),
        "catastrophic_brier_case_count": sum(catastrophic.values()),
        "macro_correct_auprc": diagonal_mean,
        "macro_post_minus_prior_auprc": sum(
            row["post_minus_prior_auprc"] for row in comparison_rows
        )
        / len(comparison_rows),
        "macro_prior_minus_post_brier": sum(
            row["prior_minus_post_brier"] for row in comparison_rows
        )
        / len(comparison_rows),
        "gate_clauses": clauses,
        "machine_decision": machine_decision,
        "stage12b_b_authorized": passed,
    }
    write_json_x(staging / "summary.json", summary)
    publish_staging_directory(staging, final)
    write_json_x(
        EXPERIMENT_DIR / "reports/STAGE12B_A_MACHINE_DECISION.json",
        {
            "schema": "stage12b_a_machine_decision_v1",
            "status": "final",
            "created_at_utc": utc_now(),
            "machine_decision": machine_decision,
            "stage12b_b_authorized": passed,
            "gate_clauses": clauses,
            "summary": file_record(final / "summary.json", relative_to=REPOSITORY_ROOT),
            "controls_frozen_before_truth": True,
            "protocol_sha256": file_sha256(CONFIG_PATH),
        },
    )
    print(machine_decision)
    print(f"diagonal mean AUPRC: {diagonal_mean:.9f}")
    print(f"off-diagonal mean AUPRC: {off_diagonal_mean:.9f}")
    for clause, value in clauses.items():
        print(f"{clause}: {value}")


if __name__ == "__main__":
    main()
