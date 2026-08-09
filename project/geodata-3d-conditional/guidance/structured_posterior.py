"""Truth-blind structured posterior search for Stage 8.

The search types in this module deliberately have no geological-truth input.
Truth may be opened only by :func:`retrospective_hard_metrics`, after a state
has already been selected and serialized by the caller.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import random
import time
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np
from scipy import ndimage
import torch

from guidance.structured_hard_inference import rasterize_object


STAGE8_STRUCTURED_VERSION = "stage8_truth_blind_continuous_structured_v1"
SELECTION_CRITERION = "minimum hard observed seismic RMSE only"


@dataclass(frozen=True)
class StructuredBodySpec:
    """One replayable low-dimensional hard geological event."""

    body_id: str
    center_x: float
    center_y: float
    center_z: float
    size_x: float
    size_y: float
    size_z: float
    orientation_deg: float
    shape: str
    material_label: int

    def record(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_record(cls, value: Mapping[str, object]) -> "StructuredBodySpec":
        required = set(cls.__dataclass_fields__)
        if set(value) != required:
            raise ValueError(
                f"structured body fields differ: missing={sorted(required-set(value))}, "
                f"extra={sorted(set(value)-required)}"
            )
        return cls(
            body_id=str(value["body_id"]),
            center_x=float(value["center_x"]),
            center_y=float(value["center_y"]),
            center_z=float(value["center_z"]),
            size_x=float(value["size_x"]),
            size_y=float(value["size_y"]),
            size_z=float(value["size_z"]),
            orientation_deg=float(value["orientation_deg"]),
            shape=str(value["shape"]),
            material_label=int(value["material_label"]),
        )


@dataclass(frozen=True)
class StructuredState:
    bodies: tuple[StructuredBodySpec, ...]
    state_id: str
    parent_id: str | None
    proposal_move: str

    def record(self) -> dict[str, object]:
        return {
            "bodies": [body.record() for body in self.bodies],
            "state_id": self.state_id,
            "parent_id": self.parent_id,
            "proposal_move": self.proposal_move,
        }

    @classmethod
    def from_record(cls, value: Mapping[str, object]) -> "StructuredState":
        return cls(
            bodies=tuple(StructuredBodySpec.from_record(item) for item in value["bodies"]),
            state_id=str(value["state_id"]),
            parent_id=None if value.get("parent_id") is None else str(value["parent_id"]),
            proposal_move=str(value["proposal_move"]),
        )


@dataclass(frozen=True)
class StructuredBounds:
    center_x: tuple[float, float]
    center_y: tuple[float, float]
    center_z: tuple[float, float]
    size_x: tuple[float, float]
    size_y: tuple[float, float]
    size_z: tuple[float, float]
    orientation_deg: tuple[float, float]
    shapes: tuple[str, ...]
    material_labels: tuple[int, ...]
    maximum_body_count: int

    def __post_init__(self) -> None:
        for name in (
            "center_x", "center_y", "center_z", "size_x", "size_y", "size_z",
            "orientation_deg",
        ):
            low, high = getattr(self, name)
            if not all(math.isfinite(float(v)) for v in (low, high)) or low >= high:
                raise ValueError(f"{name} must have finite increasing bounds")
        if not self.shapes or any(v not in {"cuboid", "ellipsoid", "dike_hemisphere"} for v in self.shapes):
            raise ValueError("shapes must be a non-empty supported set")
        if not self.material_labels:
            raise ValueError("material_labels must not be empty")
        if self.maximum_body_count <= 0:
            raise ValueError("maximum_body_count must be positive")

    def record(self) -> dict[str, object]:
        value = asdict(self)
        return {key: list(item) if isinstance(item, tuple) else item for key, item in value.items()}


class HardConditionProjector:
    """Apply edits only in a declared domain, then restore exact conditions."""

    def __init__(
        self,
        *,
        condition_values: torch.Tensor,
        condition_mask: torch.Tensor,
        edit_mask: torch.Tensor,
    ) -> None:
        if condition_values.ndim != 5 or condition_values.shape[1] != 1:
            raise ValueError("condition_values must have shape [1,1,X,Y,Z]")
        if condition_mask.shape != condition_values.shape or edit_mask.shape != condition_values.shape:
            raise ValueError("condition and edit masks must match condition_values")
        self.condition_values = condition_values.long().clone()
        self.condition_mask = condition_mask.bool().clone()
        self.edit_mask = edit_mask.bool().clone()
        if bool((self.condition_mask & self.edit_mask).any()):
            raise ValueError("edit_mask must exclude every conditioned voxel")

    def project(self, labels: torch.Tensor) -> torch.Tensor:
        if labels.shape != self.condition_values.shape:
            raise ValueError("labels must match condition_values")
        projected = labels.long().clone()
        mask = self.condition_mask.to(projected.device)
        values = self.condition_values.to(projected.device)
        projected[mask] = values[mask]
        return projected

    def violation_count(self, labels: torch.Tensor) -> int:
        mask = self.condition_mask.to(labels.device)
        values = self.condition_values.to(labels.device)
        return int((labels[mask] != values[mask]).sum())


def _as_stage7_object(body: StructuredBodySpec):
    # Reuse the already validated hard rasterizer without exposing Stage-7's
    # finite proposal-library search.
    from guidance.structured_hard_inference import StructuredObject

    return StructuredObject(
        object_id=body.body_id,
        presence=True,
        center_x=body.center_x,
        center_y=body.center_y,
        center_z=body.center_z,
        size_x=body.size_x,
        size_y=body.size_y,
        size_z=body.size_z,
        orientation_deg=body.orientation_deg,
        shape=body.shape,
        material_label=body.material_label,
        source_family=STAGE8_STRUCTURED_VERSION,
    )


def materialize_state(
    state: StructuredState,
    *,
    base_labels: torch.Tensor,
    projector: HardConditionProjector,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Rasterize ordered hard events, restrict edits, and project conditions."""
    if base_labels.shape != projector.condition_values.shape:
        raise ValueError("base_labels and projector must have matching shapes")
    labels = projector.project(base_labels)
    shape = labels.shape[2:]
    edited = torch.zeros_like(projector.edit_mask, device=labels.device)
    for body in state.bodies:
        if body.material_label not in {-1, *range(14)}:
            raise ValueError(f"invalid raw material label: {body.material_label}")
        mask3 = rasterize_object(_as_stage7_object(body), shape, device=labels.device)
        mask = mask3.view(1, 1, *shape) & projector.edit_mask.to(labels.device)
        labels[mask] = int(body.material_label)
        edited |= mask
    labels = projector.project(labels)
    violations = projector.violation_count(labels)
    if violations:
        raise RuntimeError("hard-condition projection failed")
    return labels, {
        "condition_violations": violations,
        "body_count": len(state.bodies),
        "edited_voxels": int(edited.sum()),
        "changed_from_base_voxels": int((labels != projector.project(base_labels)).sum()),
    }


class ObservationLikelihood:
    """Hard-field likelihood containing inference-visible quantities only."""

    def __init__(self, observation: torch.Tensor) -> None:
        if not observation.is_floating_point() or not torch.isfinite(observation).all():
            raise ValueError("observation must be a finite floating tensor")
        self.observation = observation.detach().clone()

    def rmse(self, predicted: torch.Tensor) -> float:
        if predicted.shape != self.observation.shape:
            raise ValueError("predicted response must match observation")
        return float(
            (predicted - self.observation.to(predicted)).square().mean().sqrt().detach().cpu()
        )


class ProposalKernel:
    """Deterministic bounded birth/death and continuous local moves."""

    MOVE_ORDER = ("birth", "death", "translate", "resize", "rotate", "change_shape")

    def __init__(self, bounds: StructuredBounds, *, seed: int) -> None:
        self.bounds = bounds
        self.seed = int(seed)

    @staticmethod
    def _uniform(rng: random.Random, interval: tuple[float, float]) -> float:
        return rng.uniform(float(interval[0]), float(interval[1]))

    @staticmethod
    def _clamp(value: float, interval: tuple[float, float]) -> float:
        return min(float(interval[1]), max(float(interval[0]), float(value)))

    def _birth(
        self,
        rng: random.Random,
        serial: int,
        *,
        birth_center: tuple[float, float, float] | None = None,
    ) -> StructuredBodySpec:
        # Always consume the three original random draws.  Stage8A-v2 may
        # replace only their values; every later size/orientation/shape draw
        # therefore remains bit-for-bit on the frozen v1 random stream.
        sampled_center = (
            self._uniform(rng, self.bounds.center_x),
            self._uniform(rng, self.bounds.center_y),
            self._uniform(rng, self.bounds.center_z),
        )
        center = sampled_center if birth_center is None else tuple(float(v) for v in birth_center)
        for value, interval, name in zip(
            center,
            (self.bounds.center_x, self.bounds.center_y, self.bounds.center_z),
            ("center_x", "center_y", "center_z"),
        ):
            if value < interval[0] or value > interval[1]:
                raise ValueError(f"ranked {name} lies outside the frozen bounds")
        return StructuredBodySpec(
            body_id=f"body_{serial:08d}",
            center_x=center[0],
            center_y=center[1],
            center_z=center[2],
            size_x=self._uniform(rng, self.bounds.size_x),
            size_y=self._uniform(rng, self.bounds.size_y),
            size_z=self._uniform(rng, self.bounds.size_z),
            orientation_deg=self._uniform(rng, self.bounds.orientation_deg),
            shape=rng.choice(self.bounds.shapes),
            material_label=int(rng.choice(self.bounds.material_labels)),
        )

    def propose(
        self,
        parent: StructuredState,
        *,
        generation: int,
        proposal_index: int,
        birth_center: tuple[float, float, float] | None = None,
    ) -> StructuredState:
        serial = int(generation) * 1_000_000 + int(proposal_index)
        rng = random.Random((self.seed << 32) ^ serial ^ _stable_state_seed(parent))
        bodies = list(parent.bodies)
        move = self.move_for(parent, proposal_index=proposal_index)
        if birth_center is not None and move != "birth":
            raise ValueError("birth_center may be supplied only for a birth move")
        if move == "birth":
            bodies.append(self._birth(rng, serial, birth_center=birth_center))
        elif move == "death":
            bodies.pop(rng.randrange(len(bodies)))
        else:
            index = rng.randrange(len(bodies))
            body = bodies[index]
            if move == "translate":
                axis = rng.choice(("x", "y", "z"))
                field = f"center_{axis}"
                interval = getattr(self.bounds, field)
                scale = 0.08 * (interval[1] - interval[0])
                body = replace(body, **{field: self._clamp(getattr(body, field) + rng.gauss(0, scale), interval)})
            elif move == "resize":
                axis = rng.choice(("x", "y", "z"))
                field = f"size_{axis}"
                interval = getattr(self.bounds, field)
                scale = 0.08 * (interval[1] - interval[0])
                body = replace(body, **{field: self._clamp(getattr(body, field) + rng.gauss(0, scale), interval)})
            elif move == "rotate":
                interval = self.bounds.orientation_deg
                body = replace(body, orientation_deg=self._clamp(body.orientation_deg + rng.gauss(0, 12.0), interval))
            elif move == "change_shape":
                choices = [value for value in self.bounds.shapes if value != body.shape]
                body = replace(body, shape=rng.choice(choices) if choices else body.shape)
            else:  # pragma: no cover - MOVE_ORDER is closed above
                raise RuntimeError(move)
            bodies[index] = body
        return StructuredState(
            bodies=tuple(bodies),
            state_id=f"g{generation:03d}_p{proposal_index:06d}_{parent.state_id}",
            parent_id=parent.state_id,
            proposal_move=move,
        )

    def move_for(self, parent: StructuredState, *, proposal_index: int) -> str:
        """Return the frozen v1 move schedule without constructing a proposal."""
        moves = list(self.MOVE_ORDER)
        if not parent.bodies:
            moves = ["birth"]
        elif len(parent.bodies) >= self.bounds.maximum_body_count:
            moves.remove("birth")
        return moves[int(proposal_index) % len(moves)]


def _stable_state_seed(state: StructuredState) -> int:
    encoded = json.dumps(state.record(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


class BirthCenterRanker(Protocol):
    """Truth-free proposal-guidance interface used only by Stage8A-v2."""

    def rank(
        self,
        *,
        state: StructuredState,
        current_labels: torch.Tensor,
        current_predicted_seismic: torch.Tensor,
        observed_seismic: torch.Tensor,
        generation: int,
    ) -> Mapping[str, object]: ...

    def record_birth_result(self, ranking_id: str, record: Mapping[str, object]) -> None: ...

    def summary(self) -> Mapping[str, object]: ...


class BirthTrustRegionController(Protocol):
    """Truth-free hard-loss continuation interface used only by Stage8A-v3."""

    def can_grow(self, *, parent: StructuredState, generation: int) -> bool: ...

    def plan_growth(
        self, *, parent: StructuredState, generation: int, proposal_index: int
    ) -> tuple[StructuredState, Mapping[str, object]]: ...

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
    ) -> tuple[StructuredState, Mapping[str, object]]: ...

    def record_result(
        self,
        *,
        child: StructuredState,
        hard_rmse: float,
        parent_rmse: float,
        empty_rmse: float,
        condition_violations: int,
    ) -> Mapping[str, object]: ...

    def summary(self) -> Mapping[str, object]: ...


@torch.no_grad()
def structured_search(
    *,
    base_labels: torch.Tensor,
    projector: HardConditionProjector,
    observation: torch.Tensor,
    hard_response: Callable[[torch.Tensor], torch.Tensor],
    proposal_kernel: ProposalKernel,
    beam_size: int,
    generations: int,
    proposals_per_parent: int,
    birth_center_ranker: BirthCenterRanker | None = None,
    birth_trust_region_controller: BirthTrustRegionController | None = None,
) -> dict[str, object]:
    """Select a structured state exclusively by hard observed-field RMSE.

    The signature intentionally cannot receive hidden geological truth.
    Every arm performs exactly ``1 + generations*beam_size*proposals_per_parent``
    hard-forward calls (the empty state is repeated to fill the initial beam).
    """
    if min(beam_size, generations, proposals_per_parent) <= 0:
        raise ValueError("beam_size, generations and proposals_per_parent must be positive")
    if birth_trust_region_controller is not None and birth_center_ranker is None:
        raise ValueError("trust-region births require the frozen birth-center ranker")
    started = time.perf_counter()
    likelihood = ObservationLikelihood(observation)
    empty = StructuredState((), "empty", None, "initial")
    trace: list[dict[str, object]] = []
    forward_calls = 0

    def evaluate(
        state: StructuredState,
        generation: int,
        *,
        parent_rmse: float | None = None,
        birth_guidance: Mapping[str, object] | None = None,
        trust_region_guidance: Mapping[str, object] | None = None,
        scheduled_move: str | None = None,
    ) -> tuple[float, StructuredState, torch.Tensor, torch.Tensor]:
        nonlocal forward_calls
        labels, audit = materialize_state(state, base_labels=base_labels, projector=projector)
        response = hard_response(labels)
        forward_calls += 1
        rmse = likelihood.rmse(response)
        trace_row = {
            "generation": int(generation),
            "state": state.record(),
            "hard_observed_rmse": rmse,
            "selection_criterion": SELECTION_CRITERION,
            "truth_used_for_selection": False,
            **audit,
        }
        if scheduled_move is not None:
            trace_row["scheduled_move"] = scheduled_move
        if trust_region_guidance is not None:
            trace_row["trust_region_probe"] = dict(
                birth_trust_region_controller.record_result(
                    child=state,
                    hard_rmse=rmse,
                    parent_rmse=float(parent_rmse),
                    empty_rmse=float(baseline_item[0]),
                    condition_violations=int(audit["condition_violations"]),
                )
            )
        if birth_guidance is not None:
            trace_row["birth_guidance"] = {
                **birth_guidance,
                "hard_rmse": rmse,
                "delta_rmse_vs_parent": rmse - float(parent_rmse),
                "delta_rmse_vs_empty": rmse - float(baseline_item[0]),
                "condition_violations": int(audit["condition_violations"]),
            }
            birth_center_ranker.record_birth_result(
                str(birth_guidance["ranking_id"]), trace_row["birth_guidance"]
            )
        trace.append(trace_row)
        return rmse, state, labels.detach().cpu(), response.detach().cpu()

    baseline_item = evaluate(empty, 0)
    # A fixed-size beam makes the evaluation budget identical across controls.
    beam = [baseline_item for _ in range(beam_size)]
    archive = [baseline_item]
    ranking_cache: dict[tuple[int, str], Mapping[str, object]] = {}
    ranking_use_count: dict[str, int] = {}
    for generation in range(1, generations + 1):
        candidates = []
        for parent_index, parent_item in enumerate(beam):
            parent = parent_item[1]
            for local_index in range(proposals_per_parent):
                proposal_index = parent_index * proposals_per_parent + local_index
                move = proposal_kernel.move_for(parent, proposal_index=proposal_index)
                birth_center = None
                birth_guidance = None
                trust_region_guidance = None
                if (
                    move == "birth"
                    and birth_trust_region_controller is not None
                    and birth_trust_region_controller.can_grow(
                        parent=parent, generation=generation
                    )
                ):
                    child, trust_region_guidance = (
                        birth_trust_region_controller.plan_growth(
                            parent=parent,
                            generation=generation,
                            proposal_index=proposal_index,
                        )
                    )
                elif move == "birth" and birth_center_ranker is not None:
                    labels_for_hash = parent_item[2].detach().cpu().contiguous()
                    state_digest = hashlib.sha256()
                    state_digest.update(str(labels_for_hash.dtype).encode("utf-8"))
                    state_digest.update(str(tuple(labels_for_hash.shape)).encode("utf-8"))
                    state_digest.update(labels_for_hash.view(torch.uint8).numpy().tobytes())
                    state_hash = state_digest.hexdigest()
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
                        raise RuntimeError("birth-center ranking is shorter than the required proposals")
                    ranked = ranked_centers[rank_index]
                    ranking_use_count[ranking_id] = rank_index + 1
                    center_values = ranked["center_xyz"]
                    birth_center = tuple(float(value) for value in center_values)
                    birth_guidance = {
                        "mode": "deterministic_multifield_first_order",
                        "ranking_id": ranking_id,
                        "rank_index": int(rank_index),
                        "center_xyz": list(birth_center),
                        "predicted_first_order_mse_decrease": float(ranked["score"]),
                        "truth_used": False,
                    }
                    child = proposal_kernel.propose(
                        parent,
                        generation=generation,
                        proposal_index=proposal_index,
                        birth_center=birth_center,
                    )
                    if birth_trust_region_controller is not None:
                        child, trust_region_guidance = (
                            birth_trust_region_controller.plan_new_center(
                                parent=parent,
                                full_child=child,
                                generation=generation,
                                proposal_index=proposal_index,
                                ranking_id=ranking_id,
                                rank_index=rank_index,
                                center_score=float(ranked["score"]),
                            )
                        )
                else:
                    child = proposal_kernel.propose(
                        parent,
                        generation=generation,
                        proposal_index=proposal_index,
                        birth_center=birth_center,
                    )
                candidates.append(evaluate(
                    child,
                    generation,
                    parent_rmse=float(parent_item[0]),
                    birth_guidance=birth_guidance,
                    trust_region_guidance=trust_region_guidance,
                    scheduled_move=move,
                ))
        candidates.sort(key=lambda item: (item[0], item[1].state_id))
        beam = candidates[:beam_size]
        archive.extend(beam)
    best = min(archive, key=lambda item: (item[0], item[1].state_id))
    expected_calls = 1 + generations * beam_size * proposals_per_parent
    if forward_calls != expected_calls:
        raise RuntimeError("fixed forward-call budget accounting failed")
    result = {
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
    }
    if birth_center_ranker is not None:
        result["birth_center_initializer"] = dict(birth_center_ranker.summary())
    if birth_trust_region_controller is not None:
        result["birth_trust_region"] = dict(birth_trust_region_controller.summary())
    return result


def inference_visible_audit(
    labels: torch.Tensor,
    *,
    base_labels: torch.Tensor,
    projector: HardConditionProjector,
    predicted_response: torch.Tensor,
    observation: torch.Tensor,
) -> dict[str, object]:
    """Metrics permitted before selection is frozen."""
    return {
        "hard_condition_violations": projector.violation_count(labels),
        "hard_observed_seismic_rmse": ObservationLikelihood(observation).rmse(predicted_response),
        "edit_volume_from_base": int((labels != base_labels).sum()),
        "truth_fields_present": False,
    }


def controlled_observations(
    correct: torch.Tensor,
    *,
    wrong_case: torch.Tensor,
    shuffle_seed: int,
) -> dict[str, torch.Tensor]:
    """Return the four frozen Stage-8 observation arms.

    The shuffled control applies one fixed permutation to lateral trace
    locations while preserving every trace waveform and value distribution.
    """
    if correct.ndim != 5 or correct.shape[1] != 1:
        raise ValueError("correct observation must have shape [1,1,X,Y,T]")
    if wrong_case.shape != correct.shape:
        raise ValueError("wrong_case observation must match correct")
    generator = torch.Generator(device="cpu").manual_seed(int(shuffle_seed))
    lateral = correct.shape[2] * correct.shape[3]
    permutation = torch.randperm(lateral, generator=generator)
    shuffled = correct.detach().cpu().reshape(1, 1, lateral, correct.shape[-1])
    shuffled = shuffled[:, :, permutation].reshape_as(correct.detach().cpu())
    return {
        "correct": correct.detach().clone(),
        "zero": torch.zeros_like(correct),
        "shuffled_xy": shuffled.to(correct),
        "wrong_case_observation": wrong_case.detach().clone(),
    }


def retrospective_hard_metrics(
    selected_labels: torch.Tensor,
    *,
    truth_labels: torch.Tensor,
    condition_mask: torch.Tensor,
    target_label: int,
    base_labels: torch.Tensor | None = None,
    truth_target_mask: torch.Tensor | None = None,
    evaluation_mask: torch.Tensor | None = None,
) -> dict[str, object]:
    """Truth-only metrics; callers must invoke this after selection."""
    if selected_labels.shape != truth_labels.shape or condition_mask.shape != truth_labels.shape:
        raise ValueError("retrospective tensors must have matching shapes")
    predicted = selected_labels.long()
    truth = truth_labels.long()
    evaluation = ~condition_mask.bool()
    if evaluation_mask is not None:
        if evaluation_mask.shape != truth.shape:
            raise ValueError("evaluation_mask must match truth_labels")
        evaluation &= evaluation_mask.bool()
    predicted_target = (predicted == int(target_label)) & evaluation
    if truth_target_mask is None:
        truth_target = (truth == int(target_label)) & evaluation
    else:
        if truth_target_mask.shape != truth.shape:
            raise ValueError("truth_target_mask must match truth_labels")
        truth_target = truth_target_mask.bool() & evaluation
    intersection = int((predicted_target & truth_target).sum())
    union = int((predicted_target | truth_target).sum())
    predicted_count = int(predicted_target.sum())
    truth_count = int(truth_target.sum())
    result: dict[str, object] = {
        "global_hard_label_accuracy": float((predicted == truth).float().mean()),
        "concealed_target_iou": intersection / union if union else 1.0,
        "concealed_target_precision": intersection / predicted_count if predicted_count else 0.0,
        "concealed_target_recall": intersection / truth_count if truth_count else 1.0,
        "concealed_target_predicted_volume": predicted_count,
        "concealed_target_truth_volume": truth_count,
        "retrospective_only": True,
        "used_for_selection": False,
    }
    class_rows = {}
    for label in sorted(set(torch.unique(truth).tolist()) | set(torch.unique(predicted).tolist())):
        truth_class = truth == int(label)
        predicted_class = predicted == int(label)
        class_intersection = int((truth_class & predicted_class).sum())
        class_union = int((truth_class | predicted_class).sum())
        class_rows[str(int(label))] = {
            "iou": class_intersection / class_union if class_union else 1.0,
            "truth_volume": int(truth_class.sum()),
            "predicted_volume": int(predicted_class.sum()),
        }
    result["per_label"] = class_rows
    result.update(_component_match_metrics(predicted_target, truth_target))
    if base_labels is not None:
        changed = predicted != base_labels.long()
        result.update({
            "edit_volume_from_base": int(changed.sum()),
            "wrong_lithology_substitution_volume": int(
                (changed & (predicted != int(target_label)) & evaluation).sum()
            ),
        })
    return result


def _component_match_metrics(
    predicted_target: torch.Tensor,
    truth_target: torch.Tensor,
    *,
    minimum_component_voxels: int = 20,
) -> dict[str, object]:
    """Greedy retrospective body matching by intersection-over-union."""
    structure = ndimage.generate_binary_structure(3, 1)
    predicted_np = predicted_target.detach().cpu().numpy()[0, 0]
    truth_np = truth_target.detach().cpu().numpy()[0, 0]
    predicted_ids, predicted_count = ndimage.label(predicted_np, structure=structure)
    truth_ids, truth_count = ndimage.label(truth_np, structure=structure)
    predicted_components = [
        predicted_ids == index
        for index in range(1, predicted_count + 1)
        if int((predicted_ids == index).sum()) >= minimum_component_voxels
    ]
    truth_components = [
        truth_ids == index
        for index in range(1, truth_count + 1)
        if int((truth_ids == index).sum()) >= minimum_component_voxels
    ]
    candidates = []
    for truth_index, truth_component in enumerate(truth_components):
        for predicted_index, predicted_component in enumerate(predicted_components):
            intersection = int((truth_component & predicted_component).sum())
            union = int((truth_component | predicted_component).sum())
            iou = intersection / union if union else 0.0
            candidates.append((iou, truth_index, predicted_index))
    matched_truth: set[int] = set()
    matched_predicted: set[int] = set()
    matches = []
    for iou, truth_index, predicted_index in sorted(candidates, reverse=True):
        if iou <= 0 or truth_index in matched_truth or predicted_index in matched_predicted:
            continue
        matched_truth.add(truth_index)
        matched_predicted.add(predicted_index)
        truth_coords = np.argwhere(truth_components[truth_index])
        predicted_coords = np.argwhere(predicted_components[predicted_index])
        matches.append({
            "truth_body_index": truth_index,
            "predicted_body_index": predicted_index,
            "iou": iou,
            "center_error_voxels": float(
                np.linalg.norm(truth_coords.mean(axis=0) - predicted_coords.mean(axis=0))
            ),
            "absolute_size_error_voxels": abs(len(predicted_coords) - len(truth_coords)),
            "relative_size_error": abs(len(predicted_coords) - len(truth_coords)) / len(truth_coords),
        })
    return {
        "body_match_minimum_component_voxels": minimum_component_voxels,
        "truth_body_count": len(truth_components),
        "predicted_body_count": len(predicted_components),
        "matched_body_count": len(matches),
        "body_recall": len(matches) / len(truth_components) if truth_components else 1.0,
        "body_precision": len(matches) / len(predicted_components) if predicted_components else 0.0,
        "matched_body_center_error_mean": (
            sum(row["center_error_voxels"] for row in matches) / len(matches)
            if matches else None
        ),
        "matched_body_relative_size_error_mean": (
            sum(row["relative_size_error"] for row in matches) / len(matches)
            if matches else None
        ),
        "body_matches": matches,
    }
