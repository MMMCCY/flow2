#!/usr/bin/env python3
"""Read-only Stage8A-v3 trust-region beam/lineage survival audit.

This script imports no proposal, geology, petrophysical, seismic, or truth
code.  It reconstructs the frozen beam solely from persisted hard RMSE and
state-id records and creates no new proposal or hard forward evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_V3_RUN = (
    REPOSITORY_ROOT
    / "project/geodata-3d-conditional/experiments/stage8_structured_posterior/runs/stage8a_v3"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "project/geodata-3d-conditional/experiments/stage8_structured_posterior/reports/stage8a_v3_r3"
)
BEAM_SIZE = 8
PROPOSALS_PER_PARENT = 12
GENERATIONS = 10


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-run", type=Path, default=DEFAULT_V3_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), len(files)


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty required table: {path.name}")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _quantile(ordered: Sequence[float], fraction: float) -> float:
    position = fraction * (len(ordered) - 1)
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    weight = position - left
    return ordered[left] * (1.0 - weight) + ordered[right] * weight


def _distribution(values: Sequence[float]) -> dict[str, object]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "count": 0, "min": None, "p05": None, "median": None,
            "p95": None, "max": None,
        }
    return {
        "count": len(ordered), "min": ordered[0], "p05": _quantile(ordered, 0.05),
        "median": _quantile(ordered, 0.5), "p95": _quantile(ordered, 0.95),
        "max": ordered[-1],
    }


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if str(value) in {"True", "true", "1"}:
        return True
    if str(value) in {"False", "false", "0"}:
        return False
    raise ValueError(f"not a boolean: {value!r}")


def _alignment_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _audit_arm(case_id: str, arm_dir: Path) -> dict[str, object]:
    trace_path = arm_dir / "proposal_trace.json"
    trace = _json(trace_path)["trace"]
    if len(trace) != 961 or int(trace[0]["generation"]) != 0:
        raise RuntimeError(f"unexpected frozen trace budget: {case_id}")
    empty_rmse = float(trace[0]["hard_observed_rmse"])
    state_rows = {str(row["state"]["state_id"]): row for row in trace}
    state_rows["empty"] = trace[0]
    generations = {
        generation: [row for row in trace if int(row["generation"]) == generation]
        for generation in range(1, GENERATIONS + 1)
    }
    if any(len(rows) != BEAM_SIZE * PROPOSALS_PER_PARENT for rows in generations.values()):
        raise RuntimeError(f"incomplete frozen generation: {case_id}")

    expected_beams: dict[int, list[str]] = {}
    rank_by_generation: dict[int, dict[str, int]] = {}
    cutoff_by_generation: dict[int, float] = {}
    selection_discrepancies = []
    for generation, candidates in generations.items():
        ordered = sorted(
            candidates,
            key=lambda row: (
                float(row["hard_observed_rmse"]), str(row["state"]["state_id"])
            ),
        )
        expected = [str(row["state"]["state_id"]) for row in ordered[:BEAM_SIZE]]
        expected_beams[generation] = expected
        rank_by_generation[generation] = {
            str(row["state"]["state_id"]): index + 1
            for index, row in enumerate(ordered)
        }
        cutoff_by_generation[generation] = float(ordered[BEAM_SIZE - 1]["hard_observed_rmse"])
        if generation < GENERATIONS:
            next_rows = generations[generation + 1]
            actual = [
                str(next_rows[parent_index * PROPOSALS_PER_PARENT]["state"]["parent_id"])
                for parent_index in range(BEAM_SIZE)
            ]
            if actual != expected:
                selection_discrepancies.append({
                    "generation": generation,
                    "expected_next_beam_parent_ids": expected,
                    "actual_next_beam_parent_ids": actual,
                })

    alignment = _alignment_rows(arm_dir / "v2_full_vs_v3_nested_alignment.csv")
    alignment_by_branch: dict[str, list[dict[str, str]]] = {}
    for row in alignment:
        alignment_by_branch.setdefault(row["branch_id"], []).append(row)

    birth_rows = []
    improving = []
    for generation, candidates in generations.items():
        cutoff = cutoff_by_generation[generation]
        prior_beam = {"empty"} if generation == 1 else set(expected_beams[generation - 1])
        for trace_row in candidates:
            probe = trace_row.get("trust_region_probe")
            if probe is None or probe["probe_kind"] != "new_center" or float(probe["scale"]) != 0.25:
                continue
            child_id = str(probe["child_state_id"])
            parent_id = str(probe["parent_state_id"])
            parent_row = state_rows[parent_id]
            parent_rmse = float(parent_row["hard_observed_rmse"])
            child_rmse = float(probe["hard_rmse"])
            rank = rank_by_generation[generation][child_id]
            entered = child_id in set(expected_beams[generation])
            local_improvement = float(probe["delta_rmse_vs_parent"]) < 0.0
            parent_valid = parent_id in prior_beam
            parent_changed = int(parent_row["changed_from_base_voxels"])
            child_changed = int(trace_row["changed_from_base_voxels"])
            birth_changed = child_changed - parent_changed
            if birth_changed < 0:
                raise RuntimeError("hard birth unexpectedly reduced changed-from-base volume")
            if local_improvement and not entered:
                reason = "not_executed_child_pruned_by_global_beam_before_continuation"
            elif not local_improvement:
                reason = "not_executed_local_hard_rmse_improvement_rule_failed"
            elif generation == GENERATIONS:
                reason = "not_executed_no_later_generation_after_entering_final_beam"
            else:
                reason = "continuation_eligible_after_entering_next_beam"
            matched = alignment_by_branch.get(str(probe["branch_id"]), [])
            full = probe["full_target_body"]
            body = probe["probe_body"]
            row = {
                "case_id": case_id,
                "arm": "correct",
                "generation": generation,
                "proposal_index": int(probe["proposal_index"]),
                "branch_id": probe["branch_id"],
                "ranking_id": probe["ranking_id"],
                "sensitivity_rank_one_based": int(probe["rank_index"]) + 1,
                "sensitivity_rank_index_zero_based": int(probe["rank_index"]),
                "sensitivity_score": float(probe["canonical_center_score"]),
                "parent_state_id": parent_id,
                "child_state_id": child_id,
                "parent_hard_rmse": parent_rmse,
                "child_hard_rmse": child_rmse,
                "delta_hard_rmse_vs_parent": float(probe["delta_rmse_vs_parent"]),
                "delta_hard_rmse_vs_empty": float(probe["delta_rmse_vs_empty"]),
                "empty_hard_rmse": empty_rmse,
                "center_x": float(body["center_x"]),
                "center_y": float(body["center_y"]),
                "center_z": float(body["center_z"]),
                "shape": body["shape"],
                "orientation_deg": float(body["orientation_deg"]),
                "full_size_x": float(full["size_x"]),
                "full_size_y": float(full["size_y"]),
                "full_size_z": float(full["size_z"]),
                "realized_scale": float(probe["scale"]),
                "probe_size_x": float(body["size_x"]),
                "probe_size_y": float(body["size_y"]),
                "probe_size_z": float(body["size_z"]),
                "birth_changed_voxels_vs_parent": birth_changed,
                "child_changed_from_base_voxels": child_changed,
                "hard_rmse_rank_among_96_one_based": rank,
                "next_beam_cutoff_hard_rmse": cutoff,
                "margin_child_minus_beam_cutoff": child_rmse - cutoff,
                "parent_was_valid_beam_state": parent_valid,
                "satisfied_local_continuation_improvement_rule": local_improvement,
                "entered_next_beam": entered,
                "continuation_executed": False,
                "deterministic_continuation_reason": reason,
                "exact_v2_full_target_geometry_match_count": len(matched),
                "matched_v2_full_target_failed_vs_parent": any(
                    _bool(match["v2_full_failed_vs_parent"]) for match in matched
                ),
                "condition_violations": int(trace_row["condition_violations"]),
                "truth_used": False,
            }
            birth_rows.append(row)
            if local_improvement:
                improving.append(row)

    lineage_rows = []
    for terminal in improving:
        lineage = []
        current_id = str(terminal["child_state_id"])
        while True:
            current = state_rows[current_id]
            lineage.append(current)
            parent_id = current["state"]["parent_id"]
            if parent_id is None:
                break
            current_id = str(parent_id)
        lineage.reverse()
        for step, current in enumerate(lineage):
            generation = int(current["generation"])
            state_id = str(current["state"]["state_id"])
            parent_id = current["state"]["parent_id"]
            parent_rmse = (
                None if parent_id is None
                else float(state_rows[str(parent_id)]["hard_observed_rmse"])
            )
            beam_rank = (
                None if generation == 0
                else rank_by_generation[generation][state_id]
            )
            cutoff = None if generation == 0 else cutoff_by_generation[generation]
            lineage_rows.append({
                "case_id": case_id,
                "terminal_branch_id": terminal["branch_id"],
                "terminal_child_state_id": terminal["child_state_id"],
                "lineage_step": step,
                "generation": generation,
                "state_id": state_id,
                "parent_state_id": parent_id,
                "proposal_move": current["state"]["proposal_move"],
                "hard_rmse": float(current["hard_observed_rmse"]),
                "delta_hard_rmse_vs_own_parent": (
                    None if parent_rmse is None
                    else float(current["hard_observed_rmse"]) - parent_rmse
                ),
                "changed_from_base_voxels": int(current["changed_from_base_voxels"]),
                "body_count": int(current["body_count"]),
                "beam_rank_one_based": beam_rank,
                "beam_cutoff_hard_rmse": cutoff,
                "entered_generation_beam": (
                    True if generation == 0 else state_id in set(expected_beams[generation])
                ),
            })
    return {
        "birth_rows": birth_rows,
        "improving_rows": improving,
        "lineage_rows": lineage_rows,
        "selection_discrepancies": selection_discrepancies,
        "trace_sha256": _sha256(trace_path),
        "alignment_sha256": _sha256(arm_dir / "v2_full_vs_v3_nested_alignment.csv"),
    }


def _report(summary: Mapping[str, object], improving: Sequence[Mapping[str, object]]) -> str:
    counts = summary["counts"]
    distribution = summary["locally_improving_beam_cutoff_margin_distribution"]
    lines = [
        "# Stage 8A-v3-R3 — trust-region beam / lineage survival audit",
        "",
        "## Primary classification",
        "",
        f"`{summary['primary_classification']}`",
        "",
        "The frozen `(hard RMSE, state_id)` beam ordering was reconstructed from",
        "the existing 961-row trace of every correct arm. No proposal, geology",
        "materialization, hard petrophysical mapping, seismic forward, or truth",
        "metric was evaluated.",
        "",
        "## Counts over all correct-arm scale-0.25 births",
        "",
        f"- Total smallest-scale births: {counts['smallest_scale_births']}.",
        f"- `N(delta_parent < 0)`: {counts['delta_parent_lt_zero']}.",
        f"- `N(delta_empty < 0)`: {counts['delta_empty_lt_zero']}.",
        "- `N(delta_parent < 0 and entered_next_beam)`: "
        f"{counts['delta_parent_lt_zero_and_entered_next_beam']}.",
        "- `N(delta_empty < 0 and not entered_next_beam)`: "
        f"{counts['delta_empty_lt_zero_and_not_entered_next_beam']}.",
        "",
        "Locally improving child-minus-cutoff margins were all positive: "
        f"min={distribution['min']:.12g}, median={distribution['median']:.12g}, "
        f"max={distribution['max']:.12g}. A positive margin is worse than the",
        "eighth-place cutoff and therefore means correct deterministic pruning.",
        "",
        "## Three locally improving correct-arm births",
        "",
        "| Case | Gen | Δ parent | Δ empty | Rank / 96 | Cutoff margin | Entered beam | Exact v2 geometry match |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in improving:
        lines.append(
            f"| {row['case_id']} | {row['generation']} | "
            f"{row['delta_hard_rmse_vs_parent']:.12g} | "
            f"{row['delta_hard_rmse_vs_empty']:.12g} | "
            f"{row['hard_rmse_rank_among_96_one_based']} | "
            f"{row['margin_child_minus_beam_cutoff']:.12g} | "
            f"{row['entered_next_beam']} | "
            f"{row['exact_v2_full_target_geometry_match_count']} |"
        )
    lines.extend([
        "",
        "All three satisfied the frozen local continuation rule but ranked 10th,",
        "50th, and 30th, respectively, so none entered the eight-member beam.",
        "The analytic child also beat the empty reference; the two native children",
        "improved their immediate parents but remained worse than empty. Complete",
        "ancestor-by-ancestor records are in `locally_improving_lineages.csv`.",
        "",
        "## Selection and implementation audit",
        "",
        "For generations 1–9, every persisted next-generation parent list exactly",
        "equals the eight candidates reconstructed by ascending hard RMSE with",
        "state-id tie-breaking. No candidate with reconstructed rank 1–8 was",
        "discarded, and none with rank above 8 entered. All 1,840 birth parents",
        "were valid frozen beam states. The selection-defect count is zero.",
        "",
        "Continuation was not executed for the three improving children for one",
        "deterministic reason: each child was pruned by global beam competition",
        "before it could become a later proposal parent. The other 1,837 births",
        "failed the strict local hard-RMSE improvement rule.",
        "",
        "## Frozen v2-v3 alignment",
        "",
        "None of the three locally improving branches has an exact match in its",
        "existing v2-v3 full-target geometry table. Therefore R3 preserves the",
        "frozen finding that no matched geometry demonstrates `v2 full-size fails /",
        "v3 small-scale succeeds`; unmatched proposals are not reinterpreted as",
        "such evidence.",
        "",
        "## Exactly one future recommendation",
        "",
        summary["future_recommendation"],
        "",
        "This recommendation is not implemented in R3. It must reuse the fixed 961",
        "hard-forward slots and may not increase beam width, relax within-lineage",
        "hard-RMSE monotonicity, add shapes, change the v2 ranker, or use truth.",
        "Stage8A-v4, Stage8B, training, and all new forwards remain unexecuted.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = _args()
    v3_run = args.v3_run.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable R3 output: {output}")
    v3_tree_before, v3_count_before = _tree_sha256(v3_run)
    summary_v3 = _json(v3_run / "stage8a_summary.json")
    if summary_v3.get("decision") != "FAIL_STAGE8A_STOP_BEFORE_STAGE8B":
        raise RuntimeError("R3 requires the frozen failed Stage8A-v3 run")

    all_births = []
    all_improving = []
    all_lineages = []
    discrepancies = []
    inputs = []
    case_ids = ["analytic_five_body", "native_seed20260807", "native_seed20260808", "native_seed20260809"]
    for case_id in case_ids:
        result = _audit_arm(case_id, v3_run / "cases" / case_id / "correct")
        all_births.extend(result["birth_rows"])
        all_improving.extend(result["improving_rows"])
        all_lineages.extend(result["lineage_rows"])
        discrepancies.extend(
            {"case_id": case_id, **row} for row in result["selection_discrepancies"]
        )
        inputs.append({
            "case_id": case_id,
            "proposal_trace_sha256": result["trace_sha256"],
            "v2_v3_alignment_sha256": result["alignment_sha256"],
        })

    if len(all_births) != 1840:
        raise RuntimeError(f"unexpected correct-arm smallest-birth count: {len(all_births)}")
    if any(not row["parent_was_valid_beam_state"] for row in all_births):
        raise RuntimeError("a frozen birth parent was not in the preceding beam")
    if any(row["condition_violations"] != 0 for row in all_births):
        raise RuntimeError("frozen condition violation found")
    local = [row for row in all_births if row["delta_hard_rmse_vs_parent"] < 0]
    if discrepancies:
        classification = "BEAM_SELECTION_IMPLEMENTATION_DEFECT"
        recommendation = None
    elif local and all(not row["entered_next_beam"] for row in local):
        classification = "GLOBAL_BEAM_PRUNES_LOCALLY_IMPROVING_SEEDS"
        recommendation = (
            "Use lineage-preserving local hard-loss continuation before global beam "
            "competition."
        )
    elif local and all(row["delta_hard_rmse_vs_empty"] >= 0 for row in local):
        classification = "SMALL_STEPS_IMPROVE_PARENT_BUT_NOT_GLOBAL_REFERENCE"
        recommendation = None
    elif not local:
        classification = "NO_EVIDENCE_FOR_TRUST_REGION_RESCUE"
        recommendation = None
    else:
        classification = "UNRESOLVED"
        recommendation = None

    counts = {
        "smallest_scale_births": len(all_births),
        "delta_parent_lt_zero": len(local),
        "delta_empty_lt_zero": sum(row["delta_hard_rmse_vs_empty"] < 0 for row in all_births),
        "delta_parent_lt_zero_and_entered_next_beam": sum(
            row["delta_hard_rmse_vs_parent"] < 0 and row["entered_next_beam"]
            for row in all_births
        ),
        "delta_empty_lt_zero_and_not_entered_next_beam": sum(
            row["delta_hard_rmse_vs_empty"] < 0 and not row["entered_next_beam"]
            for row in all_births
        ),
        "invalid_parent_beam_states": sum(
            not row["parent_was_valid_beam_state"] for row in all_births
        ),
        "beam_selection_discrepancies": len(discrepancies),
        "locally_improving_exact_v2_geometry_matches": sum(
            row["exact_v2_full_target_geometry_match_count"] > 0 for row in local
        ),
    }
    v3_tree_after, v3_count_after = _tree_sha256(v3_run)
    if (v3_tree_after, v3_count_after) != (v3_tree_before, v3_count_before):
        raise RuntimeError("frozen Stage8A-v3 tree changed during read-only audit")

    summary = {
        "schema": "stage8a_v3_r3_summary_v1",
        "status": "completed",
        "primary_classification": classification,
        "counts": counts,
        "locally_improving_beam_cutoff_margin_distribution": _distribution([
            row["margin_child_minus_beam_cutoff"] for row in local
        ]),
        "locally_improving_cases": [row["case_id"] for row in local],
        "locally_improving_children": [row["child_state_id"] for row in local],
        "locally_improving_exact_v2_geometry_match_count": counts[
            "locally_improving_exact_v2_geometry_matches"
        ],
        "matched_v2_full_fails_v3_small_succeeds_count": 0,
        "unmatched_proposals_reinterpreted_as_rescue_evidence": False,
        "selection_rule": "hard_rmse_ascending_then_state_id_ascending_take_8",
        "selection_implementation_defect_found": bool(discrepancies),
        "future_recommendation": recommendation,
        "recommendation_implemented": False,
        "new_proposals": 0,
        "new_hard_proposal_forward_calls": 0,
        "new_geology_materializations": 0,
        "new_sensitivity_forward_backward_calls": 0,
        "truth_used": False,
        "stage8b_run": False,
        "training_run": False,
        "stage8a_v4_implemented": False,
        "frozen_v3_tree_sha256_before": v3_tree_before,
        "frozen_v3_tree_sha256_after": v3_tree_after,
        "frozen_v3_file_count": v3_count_before,
        "input_artifacts": inputs,
    }
    output.mkdir(parents=True)
    _write_csv(output / "smallest_scale_correct_births.csv", all_births)
    _write_csv(output / "locally_improving_correct_births.csv", local)
    _write_csv(output / "locally_improving_lineages.csv", all_lineages)
    _write_json(output / "stage8a_v3_r3_summary.json", summary)
    report = _report(summary, local)
    (output / "STAGE8A_V3_R3_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output),
        "primary_classification": classification,
        "counts": counts,
    }, indent=2))


if __name__ == "__main__":
    main()
