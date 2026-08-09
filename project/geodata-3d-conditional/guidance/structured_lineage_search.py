"""Stage8A-v4 lineage-preserving hard-loss continuation search.

This module deliberately has no geological-truth input.  It reuses the frozen
Stage8A proposal kernel and v2 birth-center ranker, but reallocates later slots
inside each generation to finish a locally monotonic birth lineage before the
generation's candidates enter global beam competition.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import time
from typing import Callable, Mapping, Sequence

import torch

from guidance.structured_posterior import (
    BirthCenterRanker,
    HardConditionProjector,
    ObservationLikelihood,
    ProposalKernel,
    SELECTION_CRITERION,
    StructuredBodySpec,
    StructuredState,
    materialize_state,
)
from guidance.structured_trust_region import scaled_body, validate_scale_ladder


LINEAGE_SEARCH_VERSION = "stage8a_v4_lineage_before_global_beam_v1"
ALLOCATION_RULE = (
    "flatten_each_generation_into_frozen_parent_major_slots_0_to_95;"
    "after_an_improving_lineage_probe_reallocate_the_immediately_following_slots_"
    "to_successive_scales_until_first_failure_or_full_scale;"
    "skip_displaced_scheduled_proposals_without_replacement"
)


@dataclass
class _ActiveLineage:
    branch_id: str
    full_body: StructuredBodySpec
    current_scale_index: int
    current_item: tuple[float, StructuredState, torch.Tensor, torch.Tensor]
    ranking_id: str
    rank_index: int
    center_score: float


def _tensor_hash(labels: torch.Tensor) -> str:
    value = labels.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _replace_lineage_body(
    state: StructuredState,
    *,
    full_body: StructuredBodySpec,
    scale: float,
    generation: int,
    evaluation_slot: int,
) -> StructuredState:
    bodies = list(state.bodies)
    matches = [
        index for index, body in enumerate(bodies) if body.body_id == full_body.body_id
    ]
    if len(matches) != 1:
        raise RuntimeError("lineage body is absent or duplicated")
    bodies[matches[0]] = scaled_body(full_body, scale)
    return StructuredState(
        bodies=tuple(bodies),
        state_id=(
            f"g{generation:03d}_s{evaluation_slot:03d}_lineage_growth_"
            f"{state.state_id}"
        ),
        parent_id=state.state_id,
        proposal_move="birth_lineage_growth",
    )


def _branch_id(
    *,
    parent: StructuredState,
    full_child: StructuredState,
    ranking_id: str,
    rank_index: int,
) -> str:
    encoded = json.dumps(
        {
            "parent": parent.state_id,
            "full_child": full_child.record(),
            "ranking_id": ranking_id,
            "rank_index": rank_index,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "v4_branch_" + hashlib.sha256(encoded).hexdigest()[:20]


@torch.no_grad()
def lineage_preserving_structured_search(
    *,
    base_labels: torch.Tensor,
    projector: HardConditionProjector,
    observation: torch.Tensor,
    hard_response: Callable[[torch.Tensor], torch.Tensor],
    proposal_kernel: ProposalKernel,
    beam_size: int,
    generations: int,
    proposals_per_parent: int,
    birth_center_ranker: BirthCenterRanker,
    scale_ladder: Sequence[float],
) -> dict[str, object]:
    """Finish improving hard-loss lineages before each global beam selection."""
    if min(beam_size, generations, proposals_per_parent) <= 0:
        raise ValueError("beam_size, generations and proposals_per_parent must be positive")
    ladder = validate_scale_ladder(scale_ladder)
    if ladder != (0.25, 0.5, 0.75, 1.0):
        raise ValueError("Stage8A-v4 requires its preregistered fixed ladder")
    started = time.perf_counter()
    likelihood = ObservationLikelihood(observation)
    empty = StructuredState((), "empty", None, "initial")
    trace: list[dict[str, object]] = []
    branches: dict[str, dict[str, object]] = {}
    forward_calls = 0
    allocation = {"initial_empty": 0, "new_center": 0, "growth": 0, "nonbirth": 0}
    scheduled_move_counts: dict[str, int] = {}
    displaced_scheduled_move_counts: dict[str, int] = {}
    transition_attempts = {"0.25_to_0.50": 0, "0.50_to_0.75": 0, "0.75_to_1.00": 0}
    transition_successes = {key: 0 for key in transition_attempts}

    def evaluate(
        state: StructuredState,
        generation: int,
        *,
        parent_rmse: float | None = None,
        slot_kind: str,
        evaluation_slot: int | None = None,
        scheduled_move: str | None = None,
        scheduled_parent_state_id: str | None = None,
        birth_guidance: Mapping[str, object] | None = None,
        lineage_probe: Mapping[str, object] | None = None,
    ) -> tuple[float, StructuredState, torch.Tensor, torch.Tensor]:
        nonlocal forward_calls
        labels, audit = materialize_state(
            state, base_labels=base_labels, projector=projector
        )
        response = hard_response(labels)
        forward_calls += 1
        allocation[slot_kind] += 1
        rmse = likelihood.rmse(response)
        row: dict[str, object] = {
            "generation": int(generation),
            "evaluation_slot": evaluation_slot,
            "slot_kind": slot_kind,
            "scheduled_move": scheduled_move,
            "scheduled_parent_state_id": scheduled_parent_state_id,
            "state": state.record(),
            "hard_observed_rmse": rmse,
            "selection_criterion": SELECTION_CRITERION,
            "truth_used_for_selection": False,
            **audit,
        }
        if lineage_probe is not None:
            probe = {
                **lineage_probe,
                "lineage_parent_state_id": state.parent_id,
                "child_state_id": state.state_id,
                "hard_rmse": rmse,
                "delta_rmse_vs_lineage_parent": rmse - float(parent_rmse),
                "delta_rmse_vs_empty": rmse - float(baseline_item[0]),
                "strictly_improves_lineage_parent": rmse < float(parent_rmse),
                "condition_violations": int(audit["condition_violations"]),
            }
            row["lineage_probe"] = probe
            branches[str(probe["branch_id"])]["probes"].append(probe)
        if birth_guidance is not None:
            guidance = {
                **birth_guidance,
                "hard_rmse": rmse,
                "delta_rmse_vs_parent": rmse - float(parent_rmse),
                "delta_rmse_vs_empty": rmse - float(baseline_item[0]),
                "condition_violations": int(audit["condition_violations"]),
            }
            row["birth_guidance"] = guidance
            birth_center_ranker.record_birth_result(
                str(guidance["ranking_id"]), guidance
            )
        trace.append(row)
        return rmse, state, labels.detach().cpu(), response.detach().cpu()

    baseline_item = evaluate(empty, 0, slot_kind="initial_empty")
    beam = [baseline_item for _ in range(beam_size)]
    archive = [baseline_item]
    ranking_cache: dict[tuple[int, str], Mapping[str, object]] = {}
    ranking_use_count: dict[str, int] = {}

    for generation in range(1, generations + 1):
        candidates: list[tuple[float, StructuredState, torch.Tensor, torch.Tensor]] = []
        finalized_branch_ids: list[str] = []
        active: _ActiveLineage | None = None
        slots_per_generation = beam_size * proposals_per_parent
        for slot in range(slots_per_generation):
            scheduled_parent_index = slot // proposals_per_parent
            scheduled_parent_item = beam[scheduled_parent_index]
            scheduled_parent = scheduled_parent_item[1]
            scheduled_move = proposal_kernel.move_for(
                scheduled_parent, proposal_index=slot
            )
            scheduled_move_counts[scheduled_move] = (
                scheduled_move_counts.get(scheduled_move, 0) + 1
            )

            if active is not None:
                displaced_scheduled_move_counts[scheduled_move] = (
                    displaced_scheduled_move_counts.get(scheduled_move, 0) + 1
                )
                previous_scale = ladder[active.current_scale_index]
                next_index = active.current_scale_index + 1
                next_scale = ladder[next_index]
                transition = f"{previous_scale:.2f}_to_{next_scale:.2f}"
                transition_attempts[transition] += 1
                child = _replace_lineage_body(
                    active.current_item[1],
                    full_body=active.full_body,
                    scale=next_scale,
                    generation=generation,
                    evaluation_slot=slot,
                )
                probe_guidance = {
                    "version": LINEAGE_SEARCH_VERSION,
                    "allocation_rule": ALLOCATION_RULE,
                    "branch_id": active.branch_id,
                    "probe_kind": "growth",
                    "transition": transition,
                    "from_scale": previous_scale,
                    "scale": next_scale,
                    "scale_index": next_index,
                    "evaluation_slot": slot,
                    "reallocated_existing_slot": True,
                    "displaced_scheduled_move": scheduled_move,
                    "displaced_scheduled_parent_state_id": scheduled_parent.state_id,
                    "ranking_id": active.ranking_id,
                    "rank_index": active.rank_index,
                    "canonical_center_score": active.center_score,
                    "full_target_body": active.full_body.record(),
                    "probe_body": scaled_body(active.full_body, next_scale).record(),
                    "truth_used": False,
                }
                child_item = evaluate(
                    child,
                    generation,
                    parent_rmse=float(active.current_item[0]),
                    slot_kind="growth",
                    evaluation_slot=slot,
                    scheduled_move=scheduled_move,
                    scheduled_parent_state_id=scheduled_parent.state_id,
                    lineage_probe=probe_guidance,
                )
                improved = child_item[0] < active.current_item[0]
                branch = branches[active.branch_id]
                if improved:
                    transition_successes[transition] += 1
                    active.current_item = child_item
                    active.current_scale_index = next_index
                    branch["maximum_attained_scale"] = next_scale
                    if next_scale == 1.0:
                        branch["termination"] = "full_scale_strictly_improving"
                        branch["final_candidate_state_id"] = child.state_id
                        candidates.append(child_item)
                        finalized_branch_ids.append(active.branch_id)
                        active = None
                else:
                    branch["termination"] = "first_non_improving_growth_step"
                    branch["failed_growth_scale"] = next_scale
                    branch["final_candidate_state_id"] = active.current_item[1].state_id
                    candidates.append(active.current_item)
                    finalized_branch_ids.append(active.branch_id)
                    active = None
                continue

            parent_item = scheduled_parent_item
            parent = scheduled_parent
            if scheduled_move == "birth":
                state_hash = _tensor_hash(parent_item[2])
                cache_key = (generation, state_hash)
                if cache_key not in ranking_cache:
                    ranking_cache[cache_key] = birth_center_ranker.rank(
                        state=parent,
                        current_labels=parent_item[2],
                        current_predicted_seismic=parent_item[3],
                        observed_seismic=observation.detach().cpu(),
                        generation=generation,
                    )
                ranking = ranking_cache[cache_key]
                ranking_id = str(ranking["ranking_id"])
                rank_index = ranking_use_count.get(ranking_id, 0)
                ranked_centers = ranking["ranked_centers"]
                if rank_index >= len(ranked_centers):
                    raise RuntimeError("birth-center ranking shorter than required slots")
                ranked = ranked_centers[rank_index]
                ranking_use_count[ranking_id] = rank_index + 1
                center = tuple(float(value) for value in ranked["center_xyz"])
                full_child = proposal_kernel.propose(
                    parent,
                    generation=generation,
                    proposal_index=slot,
                    birth_center=center,
                )
                full_body = full_child.bodies[-1]
                branch_id = _branch_id(
                    parent=parent,
                    full_child=full_child,
                    ranking_id=ranking_id,
                    rank_index=rank_index,
                )
                seed_body = scaled_body(full_body, ladder[0])
                seed_child = replace(
                    full_child,
                    bodies=(*full_child.bodies[:-1], seed_body),
                    proposal_move="birth_lineage_new_center",
                )
                branches[branch_id] = {
                    "branch_id": branch_id,
                    "generation": generation,
                    "parent_state_id_at_birth": parent.state_id,
                    "parent_hard_rmse_at_birth": float(parent_item[0]),
                    "ranking_id": ranking_id,
                    "rank_index": rank_index,
                    "canonical_center_score": float(ranked["score"]),
                    "full_target_body": full_body.record(),
                    "fixed_scale_ladder": list(ladder),
                    "probes": [],
                    "maximum_attained_scale": 0.0,
                    "maximum_evaluated_scale": 0.25,
                    "failed_growth_scale": None,
                    "termination": None,
                    "final_candidate_state_id": None,
                    "final_global_beam_survival": None,
                    "truth_used": False,
                }
                lineage_guidance = {
                    "version": LINEAGE_SEARCH_VERSION,
                    "allocation_rule": ALLOCATION_RULE,
                    "branch_id": branch_id,
                    "probe_kind": "new_center",
                    "transition": "parent_to_0.25",
                    "from_scale": 0.0,
                    "scale": ladder[0],
                    "scale_index": 0,
                    "evaluation_slot": slot,
                    "reallocated_existing_slot": False,
                    "displaced_scheduled_move": None,
                    "displaced_scheduled_parent_state_id": None,
                    "ranking_id": ranking_id,
                    "rank_index": rank_index,
                    "canonical_center_score": float(ranked["score"]),
                    "full_target_body": full_body.record(),
                    "probe_body": seed_body.record(),
                    "truth_used": False,
                }
                birth_guidance = {
                    "mode": "deterministic_multifield_first_order",
                    "ranking_id": ranking_id,
                    "rank_index": rank_index,
                    "center_xyz": list(center),
                    "predicted_first_order_mse_decrease": float(ranked["score"]),
                    "truth_used": False,
                }
                seed_item = evaluate(
                    seed_child,
                    generation,
                    parent_rmse=float(parent_item[0]),
                    slot_kind="new_center",
                    evaluation_slot=slot,
                    scheduled_move=scheduled_move,
                    scheduled_parent_state_id=parent.state_id,
                    birth_guidance=birth_guidance,
                    lineage_probe=lineage_guidance,
                )
                if seed_item[0] < parent_item[0]:
                    branches[branch_id]["maximum_attained_scale"] = ladder[0]
                    active = _ActiveLineage(
                        branch_id=branch_id,
                        full_body=full_body,
                        current_scale_index=0,
                        current_item=seed_item,
                        ranking_id=ranking_id,
                        rank_index=rank_index,
                        center_score=float(ranked["score"]),
                    )
                else:
                    branches[branch_id]["termination"] = (
                        "smallest_scale_did_not_strictly_improve_parent"
                    )
                    branches[branch_id]["final_candidate_state_id"] = seed_child.state_id
                    candidates.append(seed_item)
                    finalized_branch_ids.append(branch_id)
            else:
                child = proposal_kernel.propose(
                    parent,
                    generation=generation,
                    proposal_index=slot,
                )
                candidates.append(evaluate(
                    child,
                    generation,
                    parent_rmse=float(parent_item[0]),
                    slot_kind="nonbirth",
                    evaluation_slot=slot,
                    scheduled_move=scheduled_move,
                    scheduled_parent_state_id=parent.state_id,
                ))

        if active is not None:
            branch = branches[active.branch_id]
            branch["termination"] = "generation_slots_exhausted_after_improving_step"
            branch["final_candidate_state_id"] = active.current_item[1].state_id
            candidates.append(active.current_item)
            finalized_branch_ids.append(active.branch_id)
            active = None
        if len(candidates) < beam_size:
            raise RuntimeError("lineage slot reallocation left fewer candidates than beam width")
        candidates.sort(key=lambda item: (item[0], item[1].state_id))
        beam = candidates[:beam_size]
        beam_ids = {item[1].state_id for item in beam}
        for branch_id in finalized_branch_ids:
            branch = branches[branch_id]
            branch["final_global_beam_survival"] = (
                branch["final_candidate_state_id"] in beam_ids
            )
        archive.extend(beam)

    best = min(archive, key=lambda item: (item[0], item[1].state_id))
    expected_calls = 1 + generations * beam_size * proposals_per_parent
    if forward_calls != expected_calls or sum(allocation.values()) != expected_calls:
        raise RuntimeError("fixed Stage8A-v4 hard-forward accounting failed")
    if sum(scheduled_move_counts.values()) != expected_calls - 1:
        raise RuntimeError("frozen scheduled-slot accounting failed")
    branch_rows = list(branches.values())
    for branch in branch_rows:
        branch["maximum_evaluated_scale"] = max(
            float(probe["scale"]) for probe in branch["probes"]
        )
        branch["scale_sequence"] = [0.0, *[
            float(probe["scale"]) for probe in branch["probes"]
        ]]
        branch["hard_rmse_sequence"] = [
            float(branch["parent_hard_rmse_at_birth"]),
            *[float(probe["hard_rmse"]) for probe in branch["probes"]],
        ]
    result: dict[str, object] = {
        "best_hard_rmse": best[0],
        "best_state": best[1],
        "best_labels": best[2],
        "best_response": best[3],
        "baseline_hard_rmse": baseline_item[0],
        "baseline_labels": baseline_item[2],
        "baseline_response": baseline_item[3],
        "hard_attainment": (
            1.0 - best[0] / baseline_item[0] if baseline_item[0] > 0 else float("nan")
        ),
        "trace": trace,
        "forward_call_count": forward_calls,
        "fixed_forward_call_budget": expected_calls,
        "runtime_seconds": time.perf_counter() - started,
        "selection_used_truth": False,
        "selection_criterion": SELECTION_CRITERION,
        "proposal_seed": proposal_kernel.seed,
        "birth_center_initializer": dict(birth_center_ranker.summary()),
        "lineage_continuation": {
            "version": LINEAGE_SEARCH_VERSION,
            "allocation_rule": ALLOCATION_RULE,
            "fixed_scale_ladder": list(ladder),
            "slot_allocation": {
                **allocation,
                "total_hard_forward_calls": forward_calls,
                "reallocated_existing_slots": allocation["growth"],
            },
            "scheduled_move_counts": scheduled_move_counts,
            "displaced_scheduled_move_counts": displaced_scheduled_move_counts,
            "locally_improving_scale_0_25_seeds": sum(
                bool(branch["probes"][0]["strictly_improves_lineage_parent"])
                for branch in branch_rows
            ),
            "continuation_attempts": sum(transition_attempts.values()),
            "transition_attempts": transition_attempts,
            "transition_successes": transition_successes,
            "branches_reaching_scale": {
                str(scale): sum(
                    float(branch["maximum_attained_scale"]) >= scale
                    for branch in branch_rows
                )
                for scale in ladder
            },
            "branch_count": len(branch_rows),
            "branch_final_global_beam_survival_count": sum(
                bool(branch["final_global_beam_survival"]) for branch in branch_rows
            ),
            "condition_violation_count": sum(
                int(probe["condition_violations"])
                for branch in branch_rows for probe in branch["probes"]
            ),
            "truth_fields_present": False,
            "branches": branch_rows,
        },
    }
    return result
