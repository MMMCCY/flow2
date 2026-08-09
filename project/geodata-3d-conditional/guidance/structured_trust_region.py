"""Truth-blind nested hard-body continuation for Stage8A-v3.

The controller receives only structured search states and hard-loss results.  It
never receives geological truth, truth body geometry, or retrospective metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Mapping, Sequence

from guidance.structured_posterior import StructuredBodySpec, StructuredState


TRUST_REGION_VERSION = "stage8a_v3_nested_hard_loss_birth_continuation_v1"


def validate_scale_ladder(values: Sequence[float]) -> tuple[float, ...]:
    """Return a strict, finite, increasing ladder ending at full scale."""
    ladder = tuple(float(value) for value in values)
    if len(ladder) < 2:
        raise ValueError("trust-region scale ladder must contain at least two scales")
    if not all(0.0 < value <= 1.0 for value in ladder):
        raise ValueError("trust-region scales must lie in (0, 1]")
    if any(left >= right for left, right in zip(ladder, ladder[1:])):
        raise ValueError("trust-region scale ladder must be strictly increasing")
    if ladder[-1] != 1.0:
        raise ValueError("trust-region scale ladder must terminate at full scale 1")
    return ladder


def scaled_body(full_body: StructuredBodySpec, scale: float) -> StructuredBodySpec:
    """Scale only the three body dimensions about its fixed center."""
    value = float(scale)
    if not 0.0 < value <= 1.0:
        raise ValueError("body scale must lie in (0, 1]")
    return replace(
        full_body,
        size_x=full_body.size_x * value,
        size_y=full_body.size_y * value,
        size_z=full_body.size_z * value,
    )


@dataclass(frozen=True)
class _Continuation:
    branch_id: str
    full_body: StructuredBodySpec
    scale_index: int
    ranking_id: str
    rank_index: int
    center_score: float


class HardLossBirthContinuation:
    """Allocate scheduled birth slots to new centers or one-step growth probes."""

    def __init__(self, *, scale_ladder: Sequence[float]) -> None:
        self.scale_ladder = validate_scale_ladder(scale_ladder)
        self._active: dict[str, _Continuation] = {}
        self._growth_allocated: set[tuple[int, str]] = set()
        self._pending: dict[str, dict[str, object]] = {}
        self._branches: dict[str, dict[str, object]] = {}
        self._probes: list[dict[str, object]] = []

    @staticmethod
    def _replace_body(
        parent: StructuredState, body_id: str, replacement: StructuredBodySpec
    ) -> tuple[StructuredBodySpec, ...]:
        bodies = list(parent.bodies)
        matches = [index for index, body in enumerate(bodies) if body.body_id == body_id]
        if len(matches) != 1:
            raise RuntimeError("active trust-region body is absent or duplicated")
        bodies[matches[0]] = replacement
        return tuple(bodies)

    @staticmethod
    def _state_id(generation: int, proposal_index: int, parent_id: str, kind: str) -> str:
        return f"g{generation:03d}_p{proposal_index:06d}_{kind}_{parent_id}"

    def can_grow(self, *, parent: StructuredState, generation: int) -> bool:
        continuation = self._active.get(parent.state_id)
        return (
            continuation is not None
            and continuation.scale_index + 1 < len(self.scale_ladder)
            and (int(generation), parent.state_id) not in self._growth_allocated
        )

    def plan_growth(
        self,
        *,
        parent: StructuredState,
        generation: int,
        proposal_index: int,
    ) -> tuple[StructuredState, Mapping[str, object]]:
        if not self.can_grow(parent=parent, generation=generation):
            raise RuntimeError("no authorized continuation growth is available")
        continuation = self._active[parent.state_id]
        next_index = continuation.scale_index + 1
        scale = self.scale_ladder[next_index]
        body = scaled_body(continuation.full_body, scale)
        child = StructuredState(
            bodies=self._replace_body(parent, body.body_id, body),
            state_id=self._state_id(generation, proposal_index, parent.state_id, "growth"),
            parent_id=parent.state_id,
            proposal_move="birth_trust_region_growth",
        )
        self._growth_allocated.add((int(generation), parent.state_id))
        guidance = self._register_pending(
            child=child,
            continuation=replace(continuation, scale_index=next_index),
            probe_kind="growth",
            generation=generation,
            proposal_index=proposal_index,
        )
        return child, guidance

    def plan_new_center(
        self,
        *,
        parent: StructuredState,
        full_child: StructuredState,
        generation: int,
        proposal_index: int,
        ranking_id: str,
        rank_index: int,
        center_score: float,
    ) -> tuple[StructuredState, Mapping[str, object]]:
        if full_child.proposal_move != "birth" or len(full_child.bodies) != len(parent.bodies) + 1:
            raise ValueError("new-center continuation requires one frozen birth target")
        full_body = full_child.bodies[-1]
        encoded = json.dumps(
            {
                "parent_id": parent.state_id,
                "state_id": full_child.state_id,
                "ranking_id": str(ranking_id),
                "rank_index": int(rank_index),
                "full_body": full_body.record(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        branch_id = "branch_" + hashlib.sha256(encoded).hexdigest()[:20]
        continuation = _Continuation(
            branch_id=branch_id,
            full_body=full_body,
            scale_index=0,
            ranking_id=str(ranking_id),
            rank_index=int(rank_index),
            center_score=float(center_score),
        )
        first_body = scaled_body(full_body, self.scale_ladder[0])
        child = replace(
            full_child,
            bodies=(*full_child.bodies[:-1], first_body),
            proposal_move="birth_trust_region_new_center",
        )
        self._branches[branch_id] = {
            "branch_id": branch_id,
            "parent_state_id_at_birth": parent.state_id,
            "ranking_id": str(ranking_id),
            "rank_index": int(rank_index),
            "canonical_center_score": float(center_score),
            "full_target_body": full_body.record(),
            "fixed_scale_ladder": list(self.scale_ladder),
            "probes": [],
            "truth_used": False,
        }
        guidance = self._register_pending(
            child=child, continuation=continuation, probe_kind="new_center",
            generation=generation, proposal_index=proposal_index,
        )
        return child, guidance

    def _register_pending(
        self,
        *,
        child: StructuredState,
        continuation: _Continuation,
        probe_kind: str,
        generation: int,
        proposal_index: int,
    ) -> Mapping[str, object]:
        if child.state_id in self._pending:
            raise RuntimeError("duplicate pending trust-region state")
        scale = self.scale_ladder[continuation.scale_index]
        guidance: dict[str, object] = {
            "version": TRUST_REGION_VERSION,
            "branch_id": continuation.branch_id,
            "probe_kind": probe_kind,
            "generation": int(generation),
            "proposal_index": int(proposal_index),
            "scale_index": continuation.scale_index,
            "scale": scale,
            "ranking_id": continuation.ranking_id,
            "rank_index": continuation.rank_index,
            "canonical_center_score": continuation.center_score,
            "full_target_body": continuation.full_body.record(),
            "probe_body": scaled_body(continuation.full_body, scale).record(),
            "truth_used": False,
        }
        self._pending[child.state_id] = {
            "continuation": continuation,
            "guidance": guidance,
        }
        return guidance

    def record_result(
        self,
        *,
        child: StructuredState,
        hard_rmse: float,
        parent_rmse: float,
        empty_rmse: float,
        condition_violations: int,
    ) -> Mapping[str, object]:
        pending = self._pending.pop(child.state_id)
        continuation = pending["continuation"]
        guidance = dict(pending["guidance"])
        delta_parent = float(hard_rmse) - float(parent_rmse)
        delta_empty = float(hard_rmse) - float(empty_rmse)
        improving = delta_parent < 0.0
        has_next = continuation.scale_index + 1 < len(self.scale_ladder)
        if improving and has_next:
            self._active[child.state_id] = continuation
            termination = "continue_on_selected_child"
            next_scale = self.scale_ladder[continuation.scale_index + 1]
        elif improving:
            termination = "full_scale_attained"
            next_scale = None
        else:
            termination = "terminated_non_improving_hard_step"
            next_scale = None
        result = {
            **guidance,
            "child_state_id": child.state_id,
            "parent_state_id": child.parent_id,
            "hard_rmse": float(hard_rmse),
            "delta_rmse_vs_parent": delta_parent,
            "delta_rmse_vs_empty": delta_empty,
            "hard_loss_improving_vs_parent": improving,
            "condition_violations": int(condition_violations),
            "continuation_authorized": bool(improving and has_next),
            "next_scale": next_scale,
            "termination": termination,
        }
        self._probes.append(result)
        branch = self._branches[continuation.branch_id]
        branch["probes"].append(result)
        return result

    def summary(self) -> Mapping[str, object]:
        new_center = [row for row in self._probes if row["probe_kind"] == "new_center"]
        growth = [row for row in self._probes if row["probe_kind"] == "growth"]
        smallest = [row for row in new_center if int(row["scale_index"]) == 0]
        return {
            "version": TRUST_REGION_VERSION,
            "fixed_scale_ladder": list(self.scale_ladder),
            "probe_count": len(self._probes),
            "new_center_probe_count": len(new_center),
            "growth_probe_count": len(growth),
            "smallest_scale_improving_count": sum(
                bool(row["hard_loss_improving_vs_parent"]) for row in smallest
            ),
            "growth_improving_count": sum(
                bool(row["hard_loss_improving_vs_parent"]) for row in growth
            ),
            "full_scale_probe_count": sum(float(row["scale"]) == 1.0 for row in self._probes),
            "full_scale_improving_count": sum(
                float(row["scale"]) == 1.0 and bool(row["hard_loss_improving_vs_parent"])
                for row in self._probes
            ),
            "truth_fields_present": False,
            "probes": list(self._probes),
            "branches": list(self._branches.values()),
        }
