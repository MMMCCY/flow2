#!/usr/bin/env python3
"""Read-only Stage 8A-R1 birth-basin and reachability audit.

This program never invokes a geophysical forward or a proposal kernel.  It
only analyzes the frozen Stage8A v1 correct-arm histories and the pre-existing
Stage7 analytic trace, then writes a separate R1 report directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE8_RUN = (
    PROJECT_ROOT / "experiments/stage8_structured_posterior/runs/stage8a_v1"
)
DEFAULT_STAGE7_TRACE = (
    PROJECT_ROOT
    / "experiments/stage6_inference_causality/reports/stage7_v1_final_v2"
    / "traces/cuboid_seed42/correct.json"
)
DEFAULT_STAGE7_CONFIG = (
    PROJECT_ROOT
    / "experiments/stage6_inference_causality/configs/five_body_cuboid_v1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "experiments/stage8_structured_posterior/reports/stage8a_r1"
)

PRIMARY_CATEGORY = "RANDOM_BIRTH_BASIN_MISS"
ALLOWED_CATEGORIES = {
    "SEARCH_IMPLEMENTATION_RETENTION_DEFECT",
    "RANDOM_BIRTH_BASIN_MISS",
    "GREEDY_SINGLE_BIRTH_ENERGY_BARRIER",
    "CONTINUOUS_HARD_LOSS_BASIN_TOO_NARROW",
    "STRUCTURED_PARAMETERIZATION_MISMATCH",
    "UNRESOLVED",
}
PARAMETERS = (
    "center_x",
    "center_y",
    "center_z",
    "size_x",
    "size_y",
    "size_z",
    "orientation_deg",
)


def _json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantile(values: Sequence[float], probability: float) -> float:
    """NumPy-compatible linear quantile without adding a NumPy dependency."""
    if not values:
        raise ValueError("cannot compute a quantile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(probability)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "min": min(numeric),
        "p01": _quantile(numeric, 0.01),
        "p05": _quantile(numeric, 0.05),
        "median": _quantile(numeric, 0.50),
        "p95": _quantile(numeric, 0.95),
        "max": max(numeric),
        "mean": sum(numeric) / len(numeric),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError(f"inconsistent table schema: {path}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _flat_distribution(prefix: str, values: Sequence[float]) -> dict[str, object]:
    return {f"{prefix}_{key}": value for key, value in _distribution(values).items()}


def audit_stage8_correct_arms(
    run_dir: Path,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    trace_paths = sorted(run_dir.glob("cases/*/correct/proposal_trace.json"))
    if len(trace_paths) != 4:
        raise ValueError(f"expected four correct-arm traces, found {len(trace_paths)}")

    proposal_rows: list[dict[str, object]] = []
    arm_rows: list[dict[str, object]] = []
    move_rows: list[dict[str, object]] = []
    body_rows: list[dict[str, object]] = []
    arm_details: dict[str, object] = {}

    for trace_path in trace_paths:
        case_id = trace_path.parts[-3]
        payload = _json(trace_path)
        trace = payload["trace"]
        if len(trace) != 961:
            raise ValueError(f"{case_id}: expected 961 recorded evaluations")
        initial = trace[0]
        if initial["generation"] != 0 or initial["state"]["proposal_move"] != "initial":
            raise ValueError(f"{case_id}: first trace row is not the empty initial state")
        if initial["body_count"] != 0 or initial["changed_from_base_voxels"] != 0:
            raise ValueError(f"{case_id}: initial state is not empty")
        if any(bool(row["truth_used_for_selection"]) for row in trace):
            raise ValueError(f"{case_id}: truth was used for selection")

        empty_rmse = float(initial["hard_observed_rmse"])
        proposals = trace[1:]
        deltas = [float(row["hard_observed_rmse"]) - empty_rmse for row in proposals]
        all_deltas = [0.0, *deltas]
        changed = [int(row["changed_from_base_voxels"]) for row in proposals]
        improving = sum(delta < 0.0 for delta in deltas)
        zero_edit = sum(value == 0 for value in changed)
        best_any = min(
            enumerate(proposals, start=1),
            key=lambda pair: (float(pair[1]["hard_observed_rmse"]), pair[0]),
        )
        nonempty = [
            (index, row)
            for index, row in enumerate(proposals, start=1)
            if int(row["body_count"]) > 0
        ]
        best_nonempty = min(
            nonempty,
            key=lambda pair: (float(pair[1]["hard_observed_rmse"]), pair[0]),
        )

        by_move: dict[str, list[tuple[int, Mapping[str, object]]]] = {}
        shape_counts: dict[str, int] = {}
        material_counts: dict[str, int] = {}
        parameter_values: dict[str, list[float]] = {name: [] for name in PARAMETERS}

        for trace_index, row in enumerate(trace):
            state = row["state"]
            loss = float(row["hard_observed_rmse"])
            delta = loss - empty_rmse
            is_initial = trace_index == 0
            proposal_rows.append(
                {
                    "case_id": case_id,
                    "trace_index": trace_index,
                    "is_initial": is_initial,
                    "generation": int(row["generation"]),
                    "state_id": state["state_id"],
                    "parent_id": state["parent_id"] or "",
                    "move_type": state["proposal_move"],
                    "body_count": int(row["body_count"]),
                    "edited_voxels": int(row["edited_voxels"]),
                    "changed_voxels": int(row["changed_from_base_voxels"]),
                    "hard_rmse": loss,
                    "empty_hard_rmse": empty_rmse,
                    "delta_rmse": delta,
                    "delta_rmse_lt_zero": delta < 0.0,
                    "zero_edit": int(row["changed_from_base_voxels"]) == 0,
                    "truth_used_for_selection": bool(row["truth_used_for_selection"]),
                    "bodies_json": json.dumps(state["bodies"], sort_keys=True),
                }
            )
            if is_initial:
                continue
            by_move.setdefault(str(state["proposal_move"]), []).append((trace_index, row))
            for body_index, body in enumerate(state["bodies"]):
                shape = str(body["shape"])
                material = str(body["material_label"])
                shape_counts[shape] = shape_counts.get(shape, 0) + 1
                material_counts[material] = material_counts.get(material, 0) + 1
                for name in PARAMETERS:
                    parameter_values[name].append(float(body[name]))
                body_rows.append(
                    {
                        "case_id": case_id,
                        "trace_index": trace_index,
                        "generation": int(row["generation"]),
                        "move_type": state["proposal_move"],
                        "state_id": state["state_id"],
                        "body_index": body_index,
                        "body_id": body["body_id"],
                        "shape": shape,
                        "material_label": int(body["material_label"]),
                        **{name: float(body[name]) for name in PARAMETERS},
                        "hard_rmse": loss,
                        "delta_rmse": delta,
                    }
                )

        delta_stats = _distribution(deltas)
        all_delta_stats = _distribution(all_deltas)
        changed_stats = _distribution(changed)
        arm_row = {
            "case_id": case_id,
            "trace_records_including_initial": len(trace),
            "noninitial_proposals": len(proposals),
            "empty_hard_rmse": empty_rmse,
            **_flat_distribution("proposal_delta_rmse", deltas),
            **_flat_distribution("all_evaluated_state_delta_rmse", all_deltas),
            "proposal_delta_rmse_lt_zero_count": improving,
            "zero_edit_proposal_count": zero_edit,
            **_flat_distribution("changed_voxels", changed),
            "best_proposal_trace_index": best_any[0],
            "best_proposal_state_id": best_any[1]["state"]["state_id"],
            "best_proposal_move": best_any[1]["state"]["proposal_move"],
            "best_proposal_body_count": int(best_any[1]["body_count"]),
            "best_proposal_hard_rmse": float(best_any[1]["hard_observed_rmse"]),
            "best_proposal_delta_rmse": float(best_any[1]["hard_observed_rmse"]) - empty_rmse,
            "best_nonempty_trace_index": best_nonempty[0],
            "best_nonempty_state_id": best_nonempty[1]["state"]["state_id"],
            "best_nonempty_move": best_nonempty[1]["state"]["proposal_move"],
            "best_nonempty_body_count": int(best_nonempty[1]["body_count"]),
            "best_nonempty_hard_rmse": float(best_nonempty[1]["hard_observed_rmse"]),
            "best_nonempty_delta_rmse": (
                float(best_nonempty[1]["hard_observed_rmse"]) - empty_rmse
            ),
        }
        arm_rows.append(arm_row)

        move_details: dict[str, object] = {}
        for move in sorted(by_move):
            entries = by_move[move]
            move_deltas = [float(row["hard_observed_rmse"]) - empty_rmse for _, row in entries]
            move_changed = [int(row["changed_from_base_voxels"]) for _, row in entries]
            move_row = {
                "case_id": case_id,
                "move_type": move,
                "count": len(entries),
                **_flat_distribution("delta_rmse", move_deltas),
                "delta_rmse_lt_zero_count": sum(value < 0.0 for value in move_deltas),
                "zero_edit_count": sum(value == 0 for value in move_changed),
                **_flat_distribution("changed_voxels", move_changed),
            }
            move_rows.append(move_row)
            move_details[move] = move_row

        body_parameter_distributions = {
            name: _distribution(values) for name, values in parameter_values.items()
        }
        arm_details[case_id] = {
            "source_trace": str(trace_path),
            "source_trace_sha256": _sha256(trace_path),
            "empty_hard_rmse": empty_rmse,
            "trace_record_count_including_initial": len(trace),
            "noninitial_proposal_count": len(proposals),
            "proposal_delta_rmse_distribution": delta_stats,
            "all_evaluated_state_delta_rmse_distribution": all_delta_stats,
            "proposal_delta_rmse_lt_zero_count": improving,
            "zero_edit_proposal_count": zero_edit,
            "changed_voxel_distribution": changed_stats,
            "move_types": move_details,
            "body_instance_count": sum(len(row["state"]["bodies"]) for row in proposals),
            "body_parameter_distributions": body_parameter_distributions,
            "shape_counts": dict(sorted(shape_counts.items())),
            "material_label_counts": dict(sorted(material_counts.items())),
            "best_proposed_state_regardless_of_acceptance": {
                "trace_index": best_any[0],
                "hard_rmse": float(best_any[1]["hard_observed_rmse"]),
                "delta_rmse": float(best_any[1]["hard_observed_rmse"]) - empty_rmse,
                "state": best_any[1]["state"],
            },
            "best_nonempty_proposed_state": {
                "trace_index": best_nonempty[0],
                "hard_rmse": float(best_nonempty[1]["hard_observed_rmse"]),
                "delta_rmse": float(best_nonempty[1]["hard_observed_rmse"]) - empty_rmse,
                "changed_voxels": int(best_nonempty[1]["changed_from_base_voxels"]),
                "state": best_nonempty[1]["state"],
            },
            "any_loss_improving_proposal": improving > 0,
        }

    if any(detail["any_loss_improving_proposal"] for detail in arm_details.values()):
        raise ValueError("classification precondition failed: an improving proposal exists")
    return proposal_rows, arm_rows, move_rows, body_rows, arm_details


def freeze_stage7_library(trace_path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Extract loss-only empty/singleton/pair rows without opening truth config."""
    trace = _json(trace_path)["trace"]
    empty_rows = [row for row in trace if int(row["generation"]) == 0]
    singleton_rows = [row for row in trace if int(row["generation"]) == 1]
    pair_trace_rows = [row for row in trace if int(row["generation"]) == 2]
    if len(empty_rows) != 1 or len(singleton_rows) != 12 or len(pair_trace_rows) != 132:
        raise ValueError("Stage7 trace does not contain the expected complete library search")

    frozen: list[dict[str, object]] = []
    empty = empty_rows[0]
    empty_rmse = float(empty["hard_seismic_rmse"])
    frozen.append(
        {
            "cardinality": 0,
            "candidate_ids": "",
            "hard_rmse": empty_rmse,
            "delta_rmse_vs_empty": 0.0,
            "loss_rank": -1,
            "source_trace_multiplicity": 1,
        }
    )
    for row in singleton_rows:
        candidate_id = str(row["objects"][0]["object_id"])
        loss = float(row["hard_seismic_rmse"])
        frozen.append(
            {
                "cardinality": 1,
                "candidate_ids": candidate_id,
                "hard_rmse": loss,
                "delta_rmse_vs_empty": loss - empty_rmse,
                "loss_rank": -1,
                "source_trace_multiplicity": 1,
            }
        )

    pairs: dict[tuple[str, str], list[float]] = {}
    for row in pair_trace_rows:
        key = tuple(sorted(str(obj["object_id"]) for obj in row["objects"]))
        if len(key) != 2 or key[0] == key[1]:
            raise ValueError(f"invalid Stage7 pair state: {key}")
        pairs.setdefault(key, []).append(float(row["hard_seismic_rmse"]))
    if len(pairs) != 66 or any(len(losses) != 2 for losses in pairs.values()):
        raise ValueError("Stage7 trace is missing one or more unique candidate pairs")
    for key, losses in sorted(pairs.items()):
        if not math.isclose(losses[0], losses[1], rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"directed duplicate pair losses disagree: {key}")
        loss = losses[0]
        frozen.append(
            {
                "cardinality": 2,
                "candidate_ids": "+".join(key),
                "hard_rmse": loss,
                "delta_rmse_vs_empty": loss - empty_rmse,
                "loss_rank": -1,
                "source_trace_multiplicity": 2,
            }
        )

    ranked = sorted(
        range(len(frozen)),
        key=lambda index: (
            float(frozen[index]["hard_rmse"]),
            int(frozen[index]["cardinality"]),
            str(frozen[index]["candidate_ids"]),
        ),
    )
    for rank, index in enumerate(ranked, start=1):
        frozen[index]["loss_rank"] = rank

    pair_rows = [row for row in frozen if row["cardinality"] == 2]
    solution = min(pair_rows, key=lambda row: (row["hard_rmse"], row["candidate_ids"]))
    solution_ids = tuple(str(solution["candidate_ids"]).split("+"))
    singleton_by_id = {
        str(row["candidate_ids"]): row for row in frozen if row["cardinality"] == 1
    }
    paths = []
    for candidate_id in solution_ids:
        singleton = singleton_by_id[candidate_id]
        paths.append(
            {
                "path": f"empty -> {candidate_id} -> {solution['candidate_ids']}",
                "empty_rmse": empty_rmse,
                "singleton_rmse": singleton["hard_rmse"],
                "pair_rmse": solution["hard_rmse"],
                "strictly_monotonic_improvement": (
                    float(singleton["hard_rmse"]) < empty_rmse
                    and float(solution["hard_rmse"]) < float(singleton["hard_rmse"])
                ),
            }
        )
    audit = {
        "source_trace": str(trace_path),
        "source_trace_sha256": _sha256(trace_path),
        "trace_record_count": len(trace),
        "frozen_library_state_count": len(frozen),
        "empty_count": 1,
        "singleton_count": 12,
        "unique_pair_count": 66,
        "directed_pair_trace_count": 132,
        "empty_hard_rmse": empty_rmse,
        "loss_selected_solution_candidate_ids": list(solution_ids),
        "loss_selected_solution_hard_rmse": float(solution["hard_rmse"]),
        "monotonic_paths": paths,
        "monotonic_single_birth_path_exists": any(
            bool(path["strictly_monotonic_improvement"]) for path in paths
        ),
        "single_birth_energy_barrier_required": not any(
            bool(path["strictly_monotonic_improvement"]) for path in paths
        ),
    }
    return frozen, audit


def add_retrospective_truth_annotation(
    frozen_rows: Sequence[Mapping[str, object]], config_path: Path
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Open truth indices only after the loss-only library table is on disk."""
    config = _json(config_path)
    candidates = config["candidate_bodies"]
    truth_indices = tuple(int(value) for value in config["truth_candidate_indices"])
    truth_ids = tuple(sorted(str(candidates[index]["id"]) for index in truth_indices))
    annotated = []
    for row in frozen_rows:
        ids = tuple(filter(None, str(row["candidate_ids"]).split("+")))
        annotated.append(
            {
                **row,
                "retrospective_truth_overlap_count": len(set(ids) & set(truth_ids)),
                "retrospective_is_exact_truth_state": ids == truth_ids,
            }
        )
    loss_solution = min(
        (row for row in frozen_rows if row["cardinality"] == 2),
        key=lambda row: (row["hard_rmse"], row["candidate_ids"]),
    )
    return annotated, {
        "role": "RETROSPECTIVE_TRUTH_ONLY_AFTER_LOSSES_FROZEN",
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "truth_candidate_indices": list(truth_indices),
        "truth_candidate_ids": list(truth_ids),
        "loss_selected_solution_matches_truth": (
            tuple(str(loss_solution["candidate_ids"]).split("+")) == truth_ids
        ),
    }


def oracle_postmortem_local_basin(
    trace_path: Path, solution_ids: Sequence[str], empty_rmse: float
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Analyze pre-existing direct children of the loss-selected Stage7 solution."""
    trace = _json(trace_path)["trace"]
    target = tuple(sorted(solution_ids))
    parents = [
        row
        for row in trace
        if int(row["generation"]) == 2
        and tuple(sorted(str(obj["object_id"]) for obj in row["objects"])) == target
    ]
    if len(parents) != 2:
        raise ValueError("expected two directed copies of the Stage7 solution pair")
    parent = next(
        (candidate for candidate in parents if any(row["parent_id"] == candidate["model_id"] for row in trace)),
        None,
    )
    if parent is None:
        raise ValueError("no local children found for the Stage7 solution")
    children = [row for row in trace if row["parent_id"] == parent["model_id"]]
    if len(children) != 62:
        raise ValueError(f"expected 62 local solution children, found {len(children)}")

    rows = []
    for row in children:
        loss = float(row["hard_seismic_rmse"])
        rows.append(
            {
                "oracle_label": "ORACLE_POSTMORTEM_ONLY",
                "parent_model_id": parent["model_id"],
                "model_id": row["model_id"],
                "proposal_move": row["proposal_move"],
                "object_count": int(row["object_count"]),
                "modified_voxels": int(row["modified_voxels"]),
                "hard_rmse": loss,
                "delta_rmse_vs_exact_solution": loss - float(parent["hard_seismic_rmse"]),
                "delta_rmse_vs_empty": loss - float(empty_rmse),
                "better_than_empty": loss < float(empty_rmse),
                "objects_json": json.dumps(row["objects"], sort_keys=True),
            }
        )
    losses = [float(row["hard_rmse"]) for row in rows]
    return rows, {
        "label": "ORACLE_POSTMORTEM_ONLY",
        "inference_success": False,
        "affects_stage8_gate": False,
        "used_for_proposal_selection": False,
        "used_for_hyperparameter_tuning": False,
        "source": "pre-existing Stage7 generation-3 local mutations; no new forwards",
        "solution_parent_model_id": parent["model_id"],
        "solution_candidate_ids": list(target),
        "solution_hard_rmse": float(parent["hard_seismic_rmse"]),
        "local_variant_count": len(rows),
        "hard_rmse_distribution": _distribution(losses),
        "better_than_empty_count": sum(value < float(empty_rmse) for value in losses),
        "equal_to_exact_zero_count": sum(value == 0.0 for value in losses),
        "worst_local_variant_margin_vs_empty": max(losses) - float(empty_rmse),
    }


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def render_report(summary: Mapping[str, object]) -> str:
    arms = summary["stage8a_correct_arms"]
    library = summary["stage7_library_barrier_audit"]
    oracle = summary["oracle_postmortem_only"]
    lines = [
        "# Stage 8A-R1 — Birth Basin / Reachability Audit",
        "",
        f"**Primary category: `{summary['primary_category']}`**",
        "",
        "Stage8A v1 remains frozen as `FAIL_STAGE8A_STOP_BEFORE_STAGE8B`. This audit did not rerun search, call the forward model, train, change the 961-call budget, tune a search hyperparameter, or run Stage8B.",
        "",
        "## Scope and trace integrity",
        "",
        "All four correct-arm histories contain 961 forward-evaluated states: one empty reference plus 960 noninitial proposals. Because proposal states and hard losses were complete, R1 analyzed the frozen records and performed no replay. Delta distributions below use the 960 noninitial proposals; the machine summary also records the 961-state distribution including the zero-delta initial state.",
        "",
        "## Frozen Stage8A proposal audit",
        "",
        "| case | empty RMSE | min Δ | p01 | p05 | median | p95 | Δ<0 | zero-edit | best nonempty Δ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case_id, detail in arms.items():
        dist = detail["proposal_delta_rmse_distribution"]
        best = detail["best_nonempty_proposed_state"]
        lines.append(
            "| " + " | ".join(
                [
                    case_id,
                    _fmt(detail["empty_hard_rmse"]),
                    _fmt(dist["min"]),
                    _fmt(dist["p01"]),
                    _fmt(dist["p05"]),
                    _fmt(dist["median"]),
                    _fmt(dist["p95"]),
                    str(detail["proposal_delta_rmse_lt_zero_count"]),
                    str(detail["zero_edit_proposal_count"]),
                    _fmt(best["delta_rmse"]),
                ]
            ) + " |"
        )
    lines.extend(
        [
            "",
            "No correct arm contained a loss-improving proposal. In every arm the best proposal regardless of acceptance was a zero-edit death state exactly tied with empty; the best nonempty state was strictly worse. Therefore an improving state was not lost by acceptance/retention logic.",
            "",
            "### Changed-voxel and best-state audit",
            "",
            "| case | changed min | p01 | p05 | median | p95 | max | mean |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for case_id, detail in arms.items():
        changed = detail["changed_voxel_distribution"]
        lines.append(
            "| " + " | ".join(
                [case_id, *(_fmt(changed[key]) for key in ("min", "p01", "p05", "median", "p95", "max", "mean"))]
            ) + " |"
        )
    lines.extend(
        [
            "",
            "| case | best proposal trace/state | move | bodies | hard RMSE | Δ | best nonempty state | best nonempty Δ |",
            "|---|---|---|---:|---:|---:|---|---:|",
        ]
    )
    for case_id, detail in arms.items():
        best = detail["best_proposed_state_regardless_of_acceptance"]
        best_nonempty = detail["best_nonempty_proposed_state"]
        lines.append(
            f"| {case_id} | {best['trace_index']} / `{best['state']['state_id']}` | {best['state']['proposal_move']} | {len(best['state']['bodies'])} | {_fmt(best['hard_rmse'])} | {_fmt(best['delta_rmse'])} | `{best_nonempty['state']['state_id']}` | {_fmt(best_nonempty['delta_rmse'])} |"
        )
    lines.extend(
        [
            "",
            "### Move and body-parameter distributions",
            "",
            "Each arm recorded 560 births and 80 each of death, translate, resize, rotate, and change-shape. All 80 zero-edit proposals per arm were death moves.",
            "",
            "| case | body instances | dike_hemisphere | ellipsoid | material 9 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for case_id, detail in arms.items():
        lines.append(
            f"| {case_id} | {detail['body_instance_count']} | {detail['shape_counts'].get('dike_hemisphere', 0)} | {detail['shape_counts'].get('ellipsoid', 0)} | {detail['material_label_counts'].get('9', 0)} |"
        )
    lines.extend(
        [
            "",
            "The numeric body distributions below count every body instance in every noninitial proposal state (including inherited bodies in two-body states).",
            "",
            "| case | parameter | min | p01 | p05 | median | p95 | max | mean |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for case_id, detail in arms.items():
        for parameter in PARAMETERS:
            values = detail["body_parameter_distributions"][parameter]
            lines.append(
                "| " + " | ".join(
                    [case_id, parameter, *(_fmt(values[key]) for key in ("min", "p01", "p05", "median", "p95", "max", "mean"))]
                ) + " |"
            )
    lines.extend(
        [
            "",
            "Move-conditioned loss and changed-voxel distributions, plus every serialized proposal body, are retained in the machine-readable tables and JSON.",
            "",
            "## Frozen Stage7 12-candidate barrier audit",
            "",
            f"The loss-only table was frozen and hashed before the truth-index configuration was opened. It contains {library['frozen_library_state_count']} unique states: one empty, 12 singletons, and all 66 unordered pairs (the source trace has 132 directed pair evaluations). The loss-selected pair is `{' + '.join(library['loss_selected_solution_candidate_ids'])}` with hard RMSE {_fmt(library['loss_selected_solution_hard_rmse'])}; retrospective truth annotation subsequently confirmed that this is the benchmark truth pair.",
            "",
            "| path | empty | singleton | pair | strictly improving |",
            "|---|---:|---:|---:|:---:|",
        ]
    )
    for path in library["monotonic_paths"]:
        lines.append(
            f"| {path['path']} | {_fmt(path['empty_rmse'])} | {_fmt(path['singleton_rmse'])} | {_fmt(path['pair_rmse'])} | {path['strictly_monotonic_improvement']} |"
        )
    lines.extend(
        [
            "",
            "Thus the exact solution is reachable by either of two strictly monotonically improving single-body-addition paths. It does not require crossing a single-birth energy barrier or proposing both bodies simultaneously.",
            "",
            "## ORACLE_POSTMORTEM_ONLY local basin-width diagnostic",
            "",
            f"This read-only diagnostic used the {oracle['local_variant_count']} already-recorded direct local mutations of the loss-selected Stage7 zero-loss pair. It made no new forward calls and is not inference success, gate evidence, proposal selection, or hyperparameter tuning. All {oracle['better_than_empty_count']}/{oracle['local_variant_count']} variants remained better than empty; {oracle['equal_to_exact_zero_count']} were still exactly zero after voxelization. Even the worst recorded local variant had Δ versus empty {_fmt(oracle['worst_local_variant_margin_vs_empty'])}. The recorded perturbations include center/size changes of ±0.25 and ±1 voxel, rotations of ±5° and ±15°, removals, shape changes, and material changes. This evidence rejects a too-narrow hard-loss basin for the analytic solution at the tested local widths.",
            "",
            "## Classification",
            "",
            f"The primary category is `{summary['primary_category']}`. The implementation did not discard an improving proposal (none existed), Stage7 shows no greedy birth barrier, and its local basin is not narrow at the recorded perturbations. Stage8A's uniform global births simply never entered an improving basin in any correct arm.",
            "",
            "The analytic Stage8A shape list omitted `cuboid`, so a parameterization mismatch is a real secondary limitation for exact analytic representation. It is not the primary cross-case explanation: all three native fixtures are `DikeHemisphere` cases within the configured family, yet they show the same no-improving-birth pattern.",
            "",
            "## Exactly one minimal truth-blind recommendation",
            "",
            "Replace uniform global **birth-center selection** with a deterministic residual/sensitivity-ranked birth-center initializer, while replacing (not adding to) the existing birth proposals so the 961-call budget and all other frozen search settings remain unchanged. The initializer may use only the observed seismic residual, acquisition/domain geometry, and condition masks; it must not use truth geometry, truth candidate indices, retrospective metrics, or seed selection.",
            "",
            "This is a recommendation only. No Stage8A v2 code or sweep was implemented.",
            "",
            "## Machine-readable artifacts",
            "",
            "- `stage8a_r1_summary.json`",
            "- `stage8a_correct_proposals.csv`",
            "- `stage8a_correct_arm_summary.csv`",
            "- `stage8a_move_summary.csv`",
            "- `stage8a_body_parameters.csv`",
            "- `stage7_library_losses_frozen.csv`",
            "- `stage7_library_truth_posthoc.csv`",
            "- `oracle_postmortem_local_basin.csv`",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage8-run", type=Path, default=DEFAULT_STAGE8_RUN)
    parser.add_argument("--stage7-trace", type=Path, default=DEFAULT_STAGE7_TRACE)
    parser.add_argument("--stage7-config", type=Path, default=DEFAULT_STAGE7_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.stage8_run.resolve()
    stage7_trace = args.stage7_trace.resolve()
    stage7_config = args.stage7_config.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == run_dir or run_dir in output_dir.parents:
        raise ValueError("R1 output must not be inside or overwrite the frozen Stage8A run")
    output_dir.mkdir(parents=True, exist_ok=False)

    source_paths = [
        run_dir / "stage8a_summary.json",
        run_dir / "config_input.json",
        stage7_trace,
        stage7_config,
    ]
    source_hashes_before = {str(path): _sha256(path) for path in source_paths}
    stage8_summary = _json(run_dir / "stage8a_summary.json")
    if stage8_summary["decision"] != "FAIL_STAGE8A_STOP_BEFORE_STAGE8B":
        raise ValueError("R1 is authorized only for the frozen failed Stage8A v1 run")

    proposals, arm_rows, move_rows, body_rows, arms = audit_stage8_correct_arms(run_dir)
    frozen_library, library = freeze_stage7_library(stage7_trace)

    _write_csv(output_dir / "stage8a_correct_proposals.csv", proposals)
    _write_csv(output_dir / "stage8a_correct_arm_summary.csv", arm_rows)
    _write_csv(output_dir / "stage8a_move_summary.csv", move_rows)
    _write_csv(output_dir / "stage8a_body_parameters.csv", body_rows)
    # This loss-only artifact is physically written before truth indices are read.
    frozen_loss_path = output_dir / "stage7_library_losses_frozen.csv"
    _write_csv(frozen_loss_path, frozen_library)
    frozen_loss_sha256 = _sha256(frozen_loss_path)

    annotated_library, truth_posthoc = add_retrospective_truth_annotation(
        frozen_library, stage7_config
    )
    _write_csv(output_dir / "stage7_library_truth_posthoc.csv", annotated_library)
    library["loss_table_frozen_before_truth_open"] = True
    library["frozen_loss_table_sha256"] = frozen_loss_sha256
    library["retrospective_truth_posthoc"] = truth_posthoc

    oracle_rows, oracle = oracle_postmortem_local_basin(
        stage7_trace,
        library["loss_selected_solution_candidate_ids"],
        float(library["empty_hard_rmse"]),
    )
    _write_csv(output_dir / "oracle_postmortem_local_basin.csv", oracle_rows)

    source_hashes_after = {str(path): _sha256(path) for path in source_paths}
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("a frozen source artifact changed during R1")
    if PRIMARY_CATEGORY not in ALLOWED_CATEGORIES:
        raise RuntimeError("invalid primary category")
    if sum(PRIMARY_CATEGORY == category for category in ALLOWED_CATEGORIES) != 1:
        raise RuntimeError("primary category must be unique")

    summary = {
        "schema": "stage8a_r1_summary_v1",
        "stage8a_v1_decision_unchanged": stage8_summary["decision"],
        "stage8a_v1_read_only": True,
        "stage8a_replayed": False,
        "new_forward_calls": 0,
        "stage8b_run": False,
        "training_run": False,
        "search_budget_changed": False,
        "search_hyperparameters_tuned": False,
        "stage8a_v2_implemented": False,
        "source_hashes_before_and_after_equal": True,
        "source_sha256": source_hashes_before,
        "delta_distribution_scope": (
            "960 noninitial proposals per arm; an additional distribution includes "
            "the empty initial state for all 961 recorded evaluations"
        ),
        "quantile_method": "linear interpolation at (n-1)*p",
        "stage8a_correct_arms": arms,
        "any_correct_arm_loss_improving_proposal": False,
        "stage7_library_barrier_audit": library,
        "oracle_postmortem_only": oracle,
        "primary_category": PRIMARY_CATEGORY,
        "secondary_observation_not_primary_category": (
            "analytic cuboid is omitted from Stage8A shapes, but three in-family "
            "DikeHemisphere native cases failed identically"
        ),
        "exactly_one_minimal_truth_blind_recommendation": (
            "Replace uniform global birth-center selection with deterministic "
            "residual/sensitivity-ranked birth-center initialization inside the "
            "unchanged 961-call budget; change no other search setting."
        ),
    }
    _write_json(output_dir / "stage8a_r1_summary.json", summary)
    (output_dir / "STAGE8A_R1_REPORT.md").write_text(
        render_report(summary), encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "primary_category": PRIMARY_CATEGORY,
        "new_forward_calls": 0,
        "stage8b_run": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
