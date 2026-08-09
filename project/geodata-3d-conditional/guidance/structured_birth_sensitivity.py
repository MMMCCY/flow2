"""Deterministic truth-blind Stage8A-v2 birth-center sensitivity ranking."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from guidance.seismic import hard_labels_to_acoustic, tensor_sha256
from guidance.structured_posterior import (
    HardConditionProjector,
    StructuredBounds,
    StructuredState,
)


BIRTH_CENTER_SCORE_VERSION = "stage8a_v2_multifield_first_order_mse_decrease_v1"
PROPERTY_FIELDS = ("impedance", "slowness")
TIE_BREAKING = "score_descending_then_center_x_then_center_y_then_center_z_ascending"


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_ellipsoid_kernel(
    size_xyz: Sequence[float], *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if len(size_xyz) != 3 or any(not math.isfinite(float(v)) or float(v) <= 0 for v in size_xyz):
        raise ValueError("canonical_size_xyz must contain three finite positive values")
    radii = tuple(float(value) / 2.0 for value in size_xyz)
    extents = tuple(int(math.ceil(radius)) for radius in radii)
    coordinates = [
        torch.arange(-extent, extent + 1, device=device, dtype=dtype)
        for extent in extents
    ]
    x, y, z = torch.meshgrid(*coordinates, indexing="ij")
    mask = (
        (x / radii[0]).square()
        + (y / radii[1]).square()
        + (z / radii[2]).square()
    ) <= 1.0
    if not bool(mask.any()):
        raise ValueError("canonical insertion kernel is empty")
    return mask.to(dtype=dtype).view(1, 1, *mask.shape)


class DeterministicSensitivityBirthCenterRanker:
    """Rank centers by the negative first-order two-property MSE derivative.

    This API intentionally receives no geological truth.  The differentiable
    forward/backward is proposal guidance only; hard proposal evaluation and
    selection remain outside this class.
    """

    def __init__(
        self,
        *,
        projector: HardConditionProjector,
        property_table: torch.Tensor,
        subsurface_mask: torch.Tensor,
        seismic_forward: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        bounds: StructuredBounds,
        target_label: int,
        canonical_size_xyz: Sequence[float],
        audit_dir: Path,
        ranked_center_record_count: int,
    ) -> None:
        if property_table.ndim != 2 or property_table.shape[0] != 2:
            raise ValueError("property_table must contain impedance and slowness")
        category = int(target_label) + 1
        if category < 0 or category >= property_table.shape[1]:
            raise ValueError("target_label is absent from the fixed property table")
        if int(ranked_center_record_count) <= 0:
            raise ValueError("ranked_center_record_count must be positive")
        self.projector = projector
        self.property_table = property_table.detach().clone()
        self.subsurface_mask = subsurface_mask.detach().clone()
        self.seismic_forward = seismic_forward
        self.bounds = bounds
        self.target_label = int(target_label)
        self.canonical_size_xyz = tuple(float(value) for value in canonical_size_xyz)
        self.audit_dir = Path(audit_dir)
        self.ranked_center_record_count = int(ranked_center_record_count)
        self._records: list[dict[str, object]] = []
        self._differentiable_forward_calls = 0
        self._backward_calls = 0
        self._runtime_seconds = 0.0
        self._valid_center_mask_cpu = self._build_valid_center_mask()

    def _build_valid_center_mask(self) -> torch.Tensor:
        edit = self.projector.edit_mask.detach().cpu().bool()[0, 0]
        valid = torch.zeros_like(edit)
        intervals = (self.bounds.center_x, self.bounds.center_y, self.bounds.center_z)
        slices = []
        for size, interval in zip(edit.shape, intervals):
            start = max(0, int(math.ceil(float(interval[0]))))
            stop = min(int(size) - 1, int(math.floor(float(interval[1]))))
            if stop < start:
                raise ValueError("frozen center bounds contain no integer candidate")
            slices.append(slice(start, stop + 1))
        valid[tuple(slices)] = True
        valid &= edit
        if not bool(valid.any()):
            raise ValueError("no valid truth-blind birth centers")
        return valid

    def rank(
        self,
        *,
        state: StructuredState,
        current_labels: torch.Tensor,
        current_predicted_seismic: torch.Tensor,
        observed_seismic: torch.Tensor,
        generation: int,
    ) -> Mapping[str, object]:
        started = time.perf_counter()
        device = self.property_table.device
        labels = current_labels.to(device=device, dtype=torch.long)
        predicted_hard = current_predicted_seismic.to(
            device=device, dtype=self.property_table.dtype
        )
        observed = observed_seismic.to(device=device, dtype=self.property_table.dtype)
        if predicted_hard.shape != observed.shape:
            raise ValueError("current prediction and observation must match")

        with torch.enable_grad():
            properties = hard_labels_to_acoustic(labels, self.property_table).detach()
            properties.requires_grad_(True)
            predicted_graph = self.seismic_forward(
                properties[:, :1], properties[:, 1:2], self.subsurface_mask.to(device)
            )
            self._differentiable_forward_calls += 1
            consistency = float((predicted_graph.detach() - predicted_hard).abs().max().cpu())
            if consistency > 1e-7:
                raise RuntimeError(
                    f"differentiable ranking forward disagrees with hard prediction: {consistency}"
                )
            mismatch_mse = (predicted_graph - observed).square().mean()
            gradient = torch.autograd.grad(mismatch_mse, properties, only_inputs=True)[0]
            self._backward_calls += 1

        target_properties = self.property_table[:, self.target_label + 1].view(1, 2, 1, 1, 1)
        property_direction = target_properties - properties.detach()
        editable = self.projector.edit_mask.to(device=device, dtype=gradient.dtype)
        per_property_decrease = -(gradient.detach() * property_direction) * editable
        directional_decrease = per_property_decrease.sum(dim=1, keepdim=True)
        kernel = _canonical_ellipsoid_kernel(
            self.canonical_size_xyz, device=device, dtype=directional_decrease.dtype
        )
        padding = tuple(int(size // 2) for size in kernel.shape[2:])
        sensitivity_map = F.conv3d(directional_decrease, kernel, padding=padding)[0, 0]
        if not bool(torch.isfinite(sensitivity_map).all()):
            raise FloatingPointError("birth-center sensitivity map is non-finite")

        valid = self._valid_center_mask_cpu
        coordinates = torch.nonzero(valid, as_tuple=False).numpy()
        scores = sensitivity_map.detach().cpu()[valid].numpy().astype(np.float64, copy=False)
        order = np.lexsort(
            (coordinates[:, 2], coordinates[:, 1], coordinates[:, 0], -scores)
        )
        keep = min(self.ranked_center_record_count, len(order))
        ranked_centers = [
            {
                "rank": rank,
                "center_xyz": [int(value) for value in coordinates[index]],
                "score": float(scores[index]),
            }
            for rank, index in enumerate(order[:keep])
        ]

        state_record = state.record()
        state_record_hash = _json_sha256(state_record)
        ranking_id = f"g{int(generation):03d}_{state_record_hash[:16]}_{len(self._records):04d}"
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        map_path = self.audit_dir / f"{ranking_id}.pt"
        torch.save(
            {
                "score_version": BIRTH_CENTER_SCORE_VERSION,
                "sensitivity_map": sensitivity_map.detach().cpu(),
                "valid_center_mask": valid,
                "canonical_shape": "ellipsoid",
                "canonical_size_xyz": self.canonical_size_xyz,
                "tie_breaking": TIE_BREAKING,
            },
            map_path,
        )
        residual = predicted_hard.detach() - observed
        runtime_seconds = time.perf_counter() - started
        self._runtime_seconds += runtime_seconds
        record: dict[str, object] = {
            "ranking_id": ranking_id,
            "generation": int(generation),
            "state_id": state.state_id,
            "current_state_record_sha256": state_record_hash,
            "current_state_labels_sha256": tensor_sha256(labels),
            "current_predicted_seismic_sha256": tensor_sha256(predicted_hard),
            "observed_seismic_sha256": tensor_sha256(observed),
            "residual_sha256": tensor_sha256(residual),
            "sensitivity_map_sha256": tensor_sha256(sensitivity_map),
            "sensitivity_map_artifact": str(map_path),
            "sensitivity_map_artifact_sha256": _file_sha256(map_path),
            "valid_center_mask_sha256": tensor_sha256(valid),
            "valid_center_count": int(valid.sum()),
            "property_fields": list(PROPERTY_FIELDS),
            "property_gradient_sha256": {
                name: tensor_sha256(gradient[:, index : index + 1])
                for index, name in enumerate(PROPERTY_FIELDS)
            },
            "property_directional_decrease_sha256": {
                name: tensor_sha256(per_property_decrease[:, index : index + 1])
                for index, name in enumerate(PROPERTY_FIELDS)
            },
            "score_definition": (
                "score(c)=-sum_{p in {impedance,slowness}} sum_x "
                "dMSE/dP_p(x) * mask_c(x) * (P_p(label9)-P_p(current,x))"
            ),
            "mismatch_mse": float(mismatch_mse.detach().cpu()),
            "hard_prediction_recompute_max_abs_difference": consistency,
            "canonical_shape": "ellipsoid",
            "canonical_size_xyz": list(self.canonical_size_xyz),
            "canonical_orientation_deg": 0.0,
            "tie_breaking": TIE_BREAKING,
            "ranked_center_record_count": len(ranked_centers),
            "ranked_centers": ranked_centers,
            "full_ranking_reconstructible_from_map_mask_and_tie_breaking": True,
            "selected_births": [],
            "differentiable_forward_calls": 1,
            "backward_calls": 1,
            "runtime_seconds": runtime_seconds,
            "truth_fields_present": False,
            "truth_used": False,
        }
        self._records.append(record)
        return {"ranking_id": ranking_id, "ranked_centers": ranked_centers}

    def record_birth_result(self, ranking_id: str, record: Mapping[str, object]) -> None:
        matches = [item for item in self._records if item["ranking_id"] == ranking_id]
        if len(matches) != 1:
            raise RuntimeError(f"unknown or duplicate ranking id: {ranking_id}")
        matches[0]["selected_births"].append(dict(record))

    def summary(self) -> Mapping[str, object]:
        selected = [birth for record in self._records for birth in record["selected_births"]]
        return {
            "mode": "deterministic_multifield_first_order",
            "score_version": BIRTH_CENTER_SCORE_VERSION,
            "proposal_guidance_only": True,
            "hard_selection_criterion_unchanged": True,
            "truth_fields_present": False,
            "truth_used": False,
            "canonical_shape": "ellipsoid",
            "canonical_size_xyz": list(self.canonical_size_xyz),
            "property_fields": list(PROPERTY_FIELDS),
            "tie_breaking": TIE_BREAKING,
            "ranking_computation_count": len(self._records),
            "differentiable_forward_calls": self._differentiable_forward_calls,
            "backward_calls": self._backward_calls,
            "sensitivity_runtime_seconds": self._runtime_seconds,
            "birth_proposal_count": len(selected),
            "loss_improving_birth_count_vs_parent": sum(
                float(item["delta_rmse_vs_parent"]) < 0.0 for item in selected
            ),
            "loss_improving_birth_count_vs_empty": sum(
                float(item["delta_rmse_vs_empty"]) < 0.0 for item in selected
            ),
            "condition_violation_count_across_births": sum(
                int(item["condition_violations"]) for item in selected
            ),
            "rankings": self._records,
        }
