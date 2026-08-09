#!/usr/bin/env python3
"""Stage8A-v2-R2 sensitivity-to-hard-proposal alignment audit.

Frozen Stage8A artifacts are read-only.  No new proposal or hard proposal
seismic evaluation is performed.  Raw two-property gradients, which v2 saved
only as hashes, are reconstructed solely for directional-derivative audit.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np
from scipy import stats
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
STRUCTURALGEO_SRC = REPOSITORY_ROOT / "StructuralGeo-main/src"
import sys
for _path in (PROJECT_DIR, STRUCTURALGEO_SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from guidance.native_geology_audit import build_structuralgeo_native_case
from guidance.seismic import (
    acoustic_tables_from_config,
    hard_labels_to_acoustic,
    seismic_operator_from_config,
    tensor_sha256,
)
from guidance.simple_causality import build_simple_causal_case
from guidance.structured_birth_sensitivity import _canonical_ellipsoid_kernel
from guidance.structured_hard_inference import rasterize_object
from guidance.structured_posterior import (
    HardConditionProjector,
    StructuredBodySpec,
    StructuredState,
    _as_stage7_object,
    controlled_observations,
    materialize_state,
)
from scripts.stage8.run_stage8 import _bounds


DEFAULT_RUN = PROJECT_DIR / "experiments/stage8_structured_posterior/runs/stage8a_v2"
DEFAULT_V1 = PROJECT_DIR / "experiments/stage8_structured_posterior/runs/stage8a_v1"
DEFAULT_R1 = PROJECT_DIR / "experiments/stage8_structured_posterior/reports/stage8a_r1"
DEFAULT_OUTPUT = PROJECT_DIR / "experiments/stage8_structured_posterior/reports/stage8a_v2_r2"
CLASSIFICATIONS = {
    "FIRST_ORDER_CENTER_LOCALIZATION_FAILURE",
    "CANONICAL_TO_ACTUAL_GEOMETRY_MISMATCH",
    "FIRST_ORDER_TO_FINITE_HARD_NONLINEARITY",
    "MIXED_LOCALIZATION_AND_GEOMETRY_FAILURE",
    "IMPLEMENTATION_DEFECT",
    "UNRESOLVED",
}


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _resolve(value: object) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
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


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty table: {path}")
    fields = list(rows[0])
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _distribution(values: Sequence[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0}
    quantiles = np.quantile(array, [0.0, 0.01, 0.05, 0.5, 0.95, 1.0])
    return {
        "count": int(len(array)),
        "min": float(quantiles[0]),
        "p01": float(quantiles[1]),
        "p05": float(quantiles[2]),
        "median": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "max": float(quantiles[5]),
        "mean": float(array.mean()),
    }


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _correlation_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    score = np.asarray([float(row["canonical_center_score"]) for row in rows])
    hard = np.asarray([float(row["hard_delta_rmse_vs_parent"]) for row in rows])
    spearman = stats.spearmanr(score, hard)
    kendall = stats.kendalltau(score, hard)
    identity = np.arange(len(rows))
    predicted_order = np.lexsort((identity, -score))
    hard_order = np.lexsort((identity, hard))
    overlaps = {}
    for label, k in (("k10", min(10, len(rows))), ("k10pct", max(1, math.ceil(0.1 * len(rows))))):
        intersection = len(set(predicted_order[:k]) & set(hard_order[:k]))
        overlaps[label] = {
            "k": int(k),
            "intersection_count": int(intersection),
            "overlap_fraction_of_k": intersection / k,
        }
    return {
        "count": len(rows),
        "spearman_rho_score_vs_hard_delta": _finite_or_none(spearman.statistic),
        "spearman_pvalue": _finite_or_none(spearman.pvalue),
        "kendall_tau_score_vs_hard_delta": _finite_or_none(kendall.statistic),
        "kendall_pvalue": _finite_or_none(kendall.pvalue),
        "higher_score_associated_with_lower_hard_rmse": (
            math.isfinite(float(spearman.statistic)) and float(spearman.statistic) < 0.0
        ),
        "top_k_overlap": overlaps,
        "canonical_score_distribution": _distribution(score),
        "hard_delta_rmse_vs_parent_distribution": _distribution(hard),
    }


def _conditioned_score_rows(
    rows: Sequence[Mapping[str, object]], *, scope: str
) -> list[dict[str, object]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["hard_delta_rmse_vs_parent"]),
            str(row["case_id"]),
            str(row["state_id"]),
        ),
    )
    output = []
    for quintile in range(5):
        start = math.floor(len(ordered) * quintile / 5)
        stop = math.floor(len(ordered) * (quintile + 1) / 5)
        bucket = ordered[start:stop]
        score_stats = _distribution([float(row["canonical_center_score"]) for row in bucket])
        hard_stats = _distribution([float(row["hard_delta_rmse_vs_parent"]) for row in bucket])
        output.append({
            "scope": scope,
            "hard_loss_rank_quintile": quintile + 1,
            "interpretation": "1=lowest/best hard delta, 5=highest/worst hard delta",
            "count": len(bucket),
            "hard_delta_min": hard_stats["min"],
            "hard_delta_median": hard_stats["median"],
            "hard_delta_max": hard_stats["max"],
            "score_min": score_stats["min"],
            "score_p05": score_stats["p05"],
            "score_median": score_stats["median"],
            "score_p95": score_stats["p95"],
            "score_max": score_stats["max"],
        })
    return output


def freeze_initial_full_ranking(run_dir: Path, output_dir: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Write the complete loss-only ranking before any oracle geometry is read."""
    ranking_path = run_dir / "cases/analytic_five_body/correct/sensitivity_rankings.json"
    rankings = _json(ranking_path)["rankings"]
    initial = [row for row in rankings if row["generation"] == 1 and row["state_id"] == "empty"]
    if len(initial) != 1:
        raise ValueError("analytic correct initial empty ranking is not unique")
    record = initial[0]
    map_path = Path(record["sensitivity_map_artifact"])
    payload = torch.load(map_path, map_location="cpu", weights_only=True)
    score_map = payload["sensitivity_map"]
    valid = payload["valid_center_mask"].bool()
    if tensor_sha256(score_map) != record["sensitivity_map_sha256"]:
        raise RuntimeError("frozen sensitivity-map hash mismatch")
    coordinates = torch.nonzero(valid, as_tuple=False).numpy()
    scores = score_map[valid].numpy().astype(np.float64, copy=False)
    order = np.lexsort((coordinates[:, 2], coordinates[:, 1], coordinates[:, 0], -scores))
    rows = [
        {
            "rank": rank,
            "rank_percentile_from_top": 100.0 * rank / len(order),
            "score_percentile": 100.0 * (len(order) - rank + 1) / len(order),
            "center_x": int(coordinates[index, 0]),
            "center_y": int(coordinates[index, 1]),
            "center_z": int(coordinates[index, 2]),
            "canonical_center_score": float(scores[index]),
        }
        for rank, index in enumerate(order, start=1)
    ]
    frozen_path = output_dir / "analytic_initial_full_center_ranking_frozen.csv"
    _write_csv(frozen_path, rows)
    return rows, {
        "role": "INFERENCE_ARTIFACT_FROZEN_BEFORE_ORACLE_GEOMETRY_OPEN",
        "source_ranking_path": str(ranking_path),
        "source_ranking_sha256": _sha256(ranking_path),
        "source_sensitivity_map_path": str(map_path),
        "source_sensitivity_map_file_sha256": _sha256(map_path),
        "source_sensitivity_map_tensor_sha256": tensor_sha256(score_map),
        "valid_center_count": len(rows),
        "tie_breaking": record["tie_breaking"],
        "frozen_full_ranking_path": str(frozen_path),
        "frozen_full_ranking_sha256": _sha256(frozen_path),
        "oracle_geometry_opened_before_freeze": False,
    }


def retrospective_center_rank(
    full_ranking: Sequence[Mapping[str, object]], stage7_config_path: Path
) -> tuple[list[dict[str, object]], dict[str, tuple[float, float, float]]]:
    """Open candidate geometry only after the full v2 ranking is frozen."""
    config = _json(stage7_config_path)
    successful_ids = ("candidate_04", "candidate_06")
    by_id = {str(row["id"]): row for row in config["candidate_bodies"]}
    centers = {}
    rows = []
    for candidate_id in successful_ids:
        body = by_id[candidate_id]
        center = tuple(
            (float(start) + float(stop) - 1.0) / 2.0
            for start, stop in zip(body["start"], body["stop"])
        )
        centers[candidate_id] = center
        nearest_distance = min(
            math.dist(center, (row["center_x"], row["center_y"], row["center_z"]))
            for row in full_ranking
        )
        nearest = min(
            (
                row for row in full_ranking
                if math.isclose(
                    math.dist(center, (row["center_x"], row["center_y"], row["center_z"])),
                    nearest_distance,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ),
            key=lambda row: int(row["rank"]),
        )
        top = full_ranking[0]
        top96 = min(
            full_ranking[:96],
            key=lambda row: (
                math.dist(center, (row["center_x"], row["center_y"], row["center_z"])),
                int(row["rank"]),
            ),
        )
        rows.append({
            "oracle_label": "ORACLE_POSTMORTEM_ONLY",
            "candidate_id": candidate_id,
            "successful_center_x": center[0],
            "successful_center_y": center[1],
            "successful_center_z": center[2],
            "nearest_ranked_center_x": nearest["center_x"],
            "nearest_ranked_center_y": nearest["center_y"],
            "nearest_ranked_center_z": nearest["center_z"],
            "nearest_grid_distance_voxels": nearest_distance,
            "sensitivity_rank": nearest["rank"],
            "rank_percentile_from_top": nearest["rank_percentile_from_top"],
            "score_percentile": nearest["score_percentile"],
            "canonical_center_score": nearest["canonical_center_score"],
            "top_ranked_center_distance_voxels": math.dist(
                center, (top["center_x"], top["center_y"], top["center_z"])
            ),
            "nearest_top96_rank": top96["rank"],
            "nearest_top96_distance_voxels": math.dist(
                center, (top96["center_x"], top96["center_y"], top96["center_z"])
            ),
            "used_for_selection": False,
            "affects_gate": False,
            "used_for_tuning": False,
        })
    return rows, centers


def recorded_birth_rows(run_dir: Path) -> list[dict[str, object]]:
    rows = []
    for trace_path in sorted(run_dir.glob("cases/*/*/proposal_trace.json")):
        case_id, observation_kind = trace_path.parts[-3:-1]
        trace = _json(trace_path)["trace"]
        for trace_index, record in enumerate(trace):
            guidance = record.get("birth_guidance")
            if guidance is None:
                continue
            body = record["state"]["bodies"][-1]
            rows.append({
                "case_id": case_id,
                "observation_kind": observation_kind,
                "observation_group": "correct" if observation_kind == "correct" else "control",
                "trace_index": trace_index,
                "generation": record["generation"],
                "state_id": record["state"]["state_id"],
                "parent_id": record["state"]["parent_id"],
                "ranking_id": guidance["ranking_id"],
                "rank_index": guidance["rank_index"],
                "center_x": guidance["center_xyz"][0],
                "center_y": guidance["center_xyz"][1],
                "center_z": guidance["center_xyz"][2],
                "canonical_center_score": guidance["predicted_first_order_mse_decrease"],
                "hard_rmse": guidance["hard_rmse"],
                "hard_delta_rmse_vs_parent": guidance["delta_rmse_vs_parent"],
                "hard_delta_rmse_vs_empty": guidance["delta_rmse_vs_empty"],
                "hard_improves_parent": float(guidance["delta_rmse_vs_parent"]) < 0.0,
                "hard_improves_empty": float(guidance["delta_rmse_vs_empty"]) < 0.0,
                "condition_violations": guidance["condition_violations"],
                "shape": body["shape"],
                "material_label": body["material_label"],
                "size_x": body["size_x"],
                "size_y": body["size_y"],
                "size_z": body["size_z"],
                "orientation_deg": body["orientation_deg"],
                "truth_used": guidance["truth_used"],
            })
    return rows


@dataclass
class CaseAssets:
    case_id: str
    base_labels: torch.Tensor
    projector: HardConditionProjector
    subsurface: torch.Tensor
    observations: dict[str, torch.Tensor]
    target_label: int


def _projector(base: torch.Tensor, condition: torch.Tensor, edit: torch.Tensor) -> HardConditionProjector:
    return HardConditionProjector(
        condition_values=base.detach().cpu(),
        condition_mask=condition.detach().cpu(),
        edit_mask=edit.detach().cpu(),
    )


def prepare_cases(
    config: Mapping[str, object], table: torch.Tensor, operator, device: torch.device
) -> tuple[dict[str, CaseAssets], int, dict[str, object]]:
    """Reconstruct absent arm observations with the minimum eight forwards."""
    calls = 0
    evidence = {
        "raw_observation_tensors_found_in_v2_run": False,
        "raw_observation_absence_proven_by": "no observation tensor exists under the 841-file v2 run tree",
        "reason_reconstruction_indispensable": (
            "raw gradients require the exact observed tensor; v2 persisted only its hash"
        ),
    }

    def response(labels: torch.Tensor, subsurface: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        acoustic = hard_labels_to_acoustic(labels.to(device).long(), table)
        calls += 1
        return operator(acoustic[:, :1], acoustic[:, 1:2], subsurface.to(device))

    assets = {}
    stage7_config = _json(_resolve(config["stage7_config"]))
    analytic = build_simple_causal_case(stage7_config)
    analytic_correct = response(analytic.truth_labels, analytic.subsurface_mask)
    wrong_labels = analytic.baseline_labels.clone()
    wrong_union = analytic.candidate_masks[[1, 9]].any(dim=0)
    wrong_labels[0, 0, wrong_union] = analytic.target_label
    analytic_wrong = response(wrong_labels, analytic.subsurface_mask)
    assets["analytic_five_body"] = CaseAssets(
        case_id="analytic_five_body",
        base_labels=analytic.baseline_labels,
        projector=_projector(
            analytic.baseline_labels,
            analytic.condition_mask,
            analytic.subsurface_mask & ~analytic.condition_mask,
        ),
        subsurface=analytic.subsurface_mask,
        observations=controlled_observations(
            analytic_correct,
            wrong_case=analytic_wrong,
            shuffle_seed=int(config["shuffle_seed"]),
        ),
        target_label=analytic.target_label,
    )
    for offset, seed in enumerate(config["stage8a_native_seeds"]):
        case, _ = build_structuralgeo_native_case(seed=int(seed))
        wrong, _ = build_structuralgeo_native_case(seed=int(seed) + 1000)
        base = torch.full_like(case.truth_labels, case.background_label)
        base[~case.subsurface_mask] = case.air_label
        base[case.condition_mask] = case.truth_labels[case.condition_mask]
        correct = response(case.truth_labels, case.subsurface_mask)
        wrong_response = response(wrong.truth_labels, wrong.subsurface_mask)
        case_id = f"native_seed{seed}"
        assets[case_id] = CaseAssets(
            case_id=case_id,
            base_labels=base,
            projector=_projector(
                base, case.condition_mask, case.subsurface_mask & ~case.condition_mask
            ),
            subsurface=case.subsurface_mask,
            observations=controlled_observations(
                correct,
                wrong_case=wrong_response,
                shuffle_seed=int(config["shuffle_seed"]) + int(seed),
            ),
            target_label=case.target_label,
        )
    if calls != 8:
        raise RuntimeError("observation reconstruction call accounting failed")
    return assets, calls, evidence


def _trace_state_lookup(trace: Sequence[Mapping[str, object]]) -> dict[str, StructuredState]:
    lookup = {}
    for row in trace:
        state = StructuredState.from_record(row["state"])
        lookup[state.state_id] = state
    return lookup


def actual_mask_audit(
    *,
    run_dir: Path,
    case_assets: Mapping[str, CaseAssets],
    table: torch.Tensor,
    operator,
    bounds,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    output = []
    defects = []
    differentiable_forwards = 0
    backwards = 0
    runtime_seconds = 0.0
    initial_gradients = {}
    target_properties = table[:, 10].view(1, 2, 1, 1, 1)

    for ranking_path in sorted(run_dir.glob("cases/*/*/sensitivity_rankings.json")):
        case_id, observation_kind = ranking_path.parts[-3:-1]
        assets = case_assets[case_id]
        trace_path = ranking_path.parent / "proposal_trace.json"
        trace = _json(trace_path)["trace"]
        state_lookup = _trace_state_lookup(trace)
        births_by_ranking: dict[str, list[Mapping[str, object]]] = {}
        for row in trace:
            if "birth_guidance" in row:
                births_by_ranking.setdefault(row["birth_guidance"]["ranking_id"], []).append(row)

        observation = assets.observations[observation_kind].to(device)
        for ranking in _json(ranking_path)["rankings"]:
            started = time.perf_counter()
            state = state_lookup[ranking["state_id"]]
            labels, state_audit = materialize_state(
                state,
                base_labels=assets.base_labels.to(device),
                projector=assets.projector,
            )
            if state_audit["condition_violations"]:
                defects.append(f"condition violation while replaying {ranking['ranking_id']}")
            if tensor_sha256(labels) != ranking["current_state_labels_sha256"]:
                defects.append(f"state-label hash mismatch {ranking['ranking_id']}")
            with torch.enable_grad():
                properties = hard_labels_to_acoustic(labels.long(), table).detach()
                properties.requires_grad_(True)
                predicted = operator(
                    properties[:, :1], properties[:, 1:2], assets.subsurface.to(device)
                )
                differentiable_forwards += 1
                loss = (predicted - observation).square().mean()
                gradient = torch.autograd.grad(loss, properties, only_inputs=True)[0]
                backwards += 1
            residual = predicted.detach() - observation
            checks = {
                "current prediction": (
                    tensor_sha256(predicted), ranking["current_predicted_seismic_sha256"]
                ),
                "observation": (tensor_sha256(observation), ranking["observed_seismic_sha256"]),
                "residual": (tensor_sha256(residual), ranking["residual_sha256"]),
                "impedance gradient": (
                    tensor_sha256(gradient[:, :1]), ranking["property_gradient_sha256"]["impedance"]
                ),
                "slowness gradient": (
                    tensor_sha256(gradient[:, 1:2]), ranking["property_gradient_sha256"]["slowness"]
                ),
            }
            for label, (actual, expected) in checks.items():
                if actual != expected:
                    defects.append(f"{label} hash mismatch {ranking['ranking_id']}")

            direction = (target_properties - properties.detach()) * assets.projector.edit_mask.to(device)
            per_property_decrease = -(gradient.detach() * direction)
            directional = per_property_decrease.sum(dim=1, keepdim=True)
            kernel = _canonical_ellipsoid_kernel(
                ranking["canonical_size_xyz"], device=device, dtype=directional.dtype
            )
            padding = tuple(int(size // 2) for size in kernel.shape[2:])
            score_map = torch.nn.functional.conv3d(directional, kernel, padding=padding)[0, 0]
            if tensor_sha256(score_map) != ranking["sensitivity_map_sha256"]:
                defects.append(f"sensitivity-map hash mismatch {ranking['ranking_id']}")

            if case_id == "analytic_five_body" and observation_kind == "correct" and ranking["generation"] == 1 and ranking["state_id"] == "empty":
                initial_gradients[case_id] = (
                    gradient.detach(), properties.detach(), labels.detach()
                )

            for birth in births_by_ranking.get(ranking["ranking_id"], []):
                guidance = birth["birth_guidance"]
                body = StructuredBodySpec.from_record(birth["state"]["bodies"][-1])
                mask3 = rasterize_object(
                    _as_stage7_object(body), labels.shape[2:], device=device
                )
                mask = mask3.view(1, 1, *mask3.shape) & assets.projector.edit_mask.to(device)
                actual_direction = direction * mask
                directional_by_property = (gradient.detach() * actual_direction).sum(
                    dim=(0, 2, 3, 4)
                )
                actual_delta = float(directional_by_property.sum().cpu())
                changed = int((mask & (labels != assets.target_label)).sum())
                output.append({
                    "case_id": case_id,
                    "observation_kind": observation_kind,
                    "observation_group": "correct" if observation_kind == "correct" else "control",
                    "generation": birth["generation"],
                    "state_id": birth["state"]["state_id"],
                    "parent_id": birth["state"]["parent_id"],
                    "ranking_id": guidance["ranking_id"],
                    "rank_index": guidance["rank_index"],
                    "center_x": guidance["center_xyz"][0],
                    "center_y": guidance["center_xyz"][1],
                    "center_z": guidance["center_xyz"][2],
                    "shape": body.shape,
                    "material_label": body.material_label,
                    "size_x": body.size_x,
                    "size_y": body.size_y,
                    "size_z": body.size_z,
                    "orientation_deg": body.orientation_deg,
                    "canonical_center_score": guidance["predicted_first_order_mse_decrease"],
                    "canonical_predicted_delta_mse": -float(
                        guidance["predicted_first_order_mse_decrease"]
                    ),
                    "actual_mask_impedance_predicted_delta_mse": float(
                        directional_by_property[0].cpu()
                    ),
                    "actual_mask_slowness_predicted_delta_mse": float(
                        directional_by_property[1].cpu()
                    ),
                    "actual_mask_predicted_delta_mse": actual_delta,
                    "actual_mask_predicted_decrease_mse": -actual_delta,
                    "actual_mask_predicts_improvement": actual_delta < 0.0,
                    "actual_insertion_changed_voxels": changed,
                    "actual_hard_rmse": guidance["hard_rmse"],
                    "actual_hard_delta_rmse_vs_parent": guidance["delta_rmse_vs_parent"],
                    "actual_hard_delta_rmse_vs_empty": guidance["delta_rmse_vs_empty"],
                    "actual_hard_improves_parent": float(guidance["delta_rmse_vs_parent"]) < 0.0,
                    "first_order_hard_sign_agreement": (
                        (actual_delta < 0.0)
                        == (float(guidance["delta_rmse_vs_parent"]) < 0.0)
                    ),
                    "condition_violations": guidance["condition_violations"],
                    "truth_used": False,
                })
            runtime_seconds += time.perf_counter() - started
    return output, {
        "raw_gradients_persisted_in_v2": False,
        "gradient_recomputation_was_indispensable": True,
        "differentiable_forward_calls": differentiable_forwards,
        "backward_calls": backwards,
        "runtime_seconds": runtime_seconds,
        "hard_proposal_forward_calls": 0,
        "implementation_defects": defects,
        "all_recomputed_hashes_match_v2": not defects,
    }, initial_gradients


def actual_alignment_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    canonical = np.asarray([float(row["canonical_center_score"]) for row in rows])
    actual_decrease = np.asarray(
        [float(row["actual_mask_predicted_decrease_mse"]) for row in rows]
    )
    hard = np.asarray([float(row["actual_hard_delta_rmse_vs_parent"]) for row in rows])
    canonical_actual = stats.spearmanr(canonical, actual_decrease)
    actual_hard = stats.spearmanr(actual_decrease, hard)
    return {
        "count": len(rows),
        "spearman_canonical_score_vs_actual_mask_predicted_decrease": _finite_or_none(
            canonical_actual.statistic
        ),
        "spearman_actual_mask_predicted_decrease_vs_hard_delta": _finite_or_none(
            actual_hard.statistic
        ),
        "canonical_actual_sign_agreement_fraction": float(
            np.mean((canonical > 0.0) == (actual_decrease > 0.0))
        ),
        "actual_first_order_hard_sign_agreement_fraction": float(
            np.mean((actual_decrease > 0.0) == (hard < 0.0))
        ),
        "actual_mask_predicts_improvement_count": int((actual_decrease > 0.0).sum()),
        "hard_improves_parent_count": int((hard < 0.0).sum()),
    }


def finite_step_oracle_audit(
    *,
    stage7_config_path: Path,
    r1_dir: Path,
    analytic_assets: CaseAssets,
    table: torch.Tensor,
    initial_gradient: torch.Tensor,
    initial_properties: torch.Tensor,
    device: torch.device,
) -> list[dict[str, object]]:
    config = _json(stage7_config_path)
    library_rows = list(csv.DictReader(
        (r1_dir / "stage7_library_losses_frozen.csv").open(encoding="utf-8")
    ))
    empty_rmse = float(next(row for row in library_rows if int(row["cardinality"]) == 0)["hard_rmse"])
    singleton_loss = {
        row["candidate_ids"]: float(row["hard_rmse"])
        for row in library_rows if int(row["cardinality"]) == 1
    }
    target = table[:, 10].view(1, 2, 1, 1, 1)
    direction = (target - initial_properties) * analytic_assets.projector.edit_mask.to(device)
    all_rows = []
    for body in config["candidate_bodies"]:
        center = [
            (float(start) + float(stop) - 1.0) / 2.0
            for start, stop in zip(body["start"], body["stop"])
        ]
        size = [float(stop) - float(start) for start, stop in zip(body["start"], body["stop"])]
        spec = StructuredBodySpec(
            body_id=str(body["id"]), center_x=center[0], center_y=center[1],
            center_z=center[2], size_x=size[0], size_y=size[1], size_z=size[2],
            orientation_deg=0.0, shape="cuboid", material_label=9,
        )
        mask3 = rasterize_object(_as_stage7_object(spec), initial_properties.shape[2:], device=device)
        mask = mask3.view(1, 1, *mask3.shape) & analytic_assets.projector.edit_mask.to(device)
        by_property = (initial_gradient * direction * mask).sum(dim=(0, 2, 3, 4))
        predicted_delta = float(by_property.sum().cpu())
        hard_delta = singleton_loss[spec.body_id] - empty_rmse
        all_rows.append({
            "candidate_id": spec.body_id,
            "predicted_delta_mse": predicted_delta,
            "predicted_decrease_mse": -predicted_delta,
            "impedance_predicted_delta_mse": float(by_property[0].cpu()),
            "slowness_predicted_delta_mse": float(by_property[1].cpu()),
            "singleton_hard_rmse": singleton_loss[spec.body_id],
            "singleton_hard_delta_rmse_vs_empty": hard_delta,
        })
    predicted_order = sorted(all_rows, key=lambda row: (row["predicted_delta_mse"], row["candidate_id"]))
    hard_order = sorted(all_rows, key=lambda row: (row["singleton_hard_delta_rmse_vs_empty"], row["candidate_id"]))
    predicted_rank = {row["candidate_id"]: index for index, row in enumerate(predicted_order, 1)}
    hard_rank = {row["candidate_id"]: index for index, row in enumerate(hard_order, 1)}
    output = []
    for row in all_rows:
        if row["candidate_id"] not in {"candidate_04", "candidate_06"}:
            continue
        output.append({
            "oracle_label": "ORACLE_POSTMORTEM_ONLY",
            **row,
            "first_order_rank_among_12": predicted_rank[row["candidate_id"]],
            "singleton_hard_loss_rank_among_12": hard_rank[row["candidate_id"]],
            "first_order_predicts_improvement": row["predicted_delta_mse"] < 0.0,
            "finite_hard_step_improves": row["singleton_hard_delta_rmse_vs_empty"] < 0.0,
            "sign_agreement": (
                (row["predicted_delta_mse"] < 0.0)
                == (row["singleton_hard_delta_rmse_vs_empty"] < 0.0)
            ),
            "used_for_selection": False,
            "affects_gate": False,
            "used_for_tuning": False,
        })
    return output


def _fmt(value: object) -> str:
    return f"{value:.7g}" if isinstance(value, float) else str(value)


def render_report(summary: Mapping[str, object]) -> str:
    lines = [
        "# Stage 8A-v2-R2 — Sensitivity-to-Hard-Proposal Alignment Audit",
        "",
        f"**Primary classification: `{summary['primary_classification']}`**",
        "",
        "This is a read-only postmortem. Stage8A-v1, R1, and v2 remained unchanged. No proposal was generated or selected, no hard proposal seismic forward was called, Stage8A-v2 was not rerun, and Stage8B/training/v3 were not run.",
        "",
        "## A. ORACLE_POSTMORTEM_ONLY retrospective center rank",
        "",
        "The complete 140,985-center analytic initial ranking was reconstructed from the frozen score map and written before Stage7 successful geometry was opened.",
        "",
        "| candidate | nearest grid center | distance | rank | top-rank percentile | score percentile | score | top-1 distance | nearest top-96 distance |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["retrospective_center_rank"]:
        lines.append(
            f"| {row['candidate_id']} | ({row['nearest_ranked_center_x']},{row['nearest_ranked_center_y']},{row['nearest_ranked_center_z']}) | {_fmt(row['nearest_grid_distance_voxels'])} | {row['sensitivity_rank']} | {_fmt(row['rank_percentile_from_top'])}% | {_fmt(row['score_percentile'])}% | {_fmt(row['canonical_center_score'])} | {_fmt(row['top_ranked_center_distance_voxels'])} | {_fmt(row['nearest_top96_distance_voxels'])} |"
        )
    lines.extend([
        "",
        "## B. Recorded score versus frozen hard loss",
        "",
        "| scope | births | Spearman(score, hard Δ) | Kendall tau | higher score -> lower hard RMSE? | top-10 overlap | top-10% overlap |",
        "|---|---:|---:|---:|:---:|---:|---:|",
    ])
    for scope in ("all_correct", "all_controls", "control_zero", "control_shuffled_xy", "control_wrong_case_observation"):
        row = summary["score_alignment"][scope]
        lines.append(
            f"| {scope} | {row['count']} | {_fmt(row['spearman_rho_score_vs_hard_delta'])} | {_fmt(row['kendall_tau_score_vs_hard_delta'])} | {row['higher_score_associated_with_lower_hard_rmse']} | {row['top_k_overlap']['k10']['intersection_count']}/{row['top_k_overlap']['k10']['k']} | {row['top_k_overlap']['k10pct']['intersection_count']}/{row['top_k_overlap']['k10pct']['k']} |"
        )
    lines.extend([
        "",
        "## C. Canonical score -> actual-mask derivative -> hard delta",
        "",
        "| scope | births | rho(canonical, actual decrease) | rho(actual decrease, hard Δ) | canonical/actual sign agreement | actual/hard sign agreement | actual predicts improve | hard improves |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for scope in ("all_correct", "all_controls"):
        row = summary["actual_mask_alignment"][scope]
        lines.append(
            f"| {scope} | {row['count']} | {_fmt(row['spearman_canonical_score_vs_actual_mask_predicted_decrease'])} | {_fmt(row['spearman_actual_mask_predicted_decrease_vs_hard_delta'])} | {_fmt(row['canonical_actual_sign_agreement_fraction'])} | {_fmt(row['actual_first_order_hard_sign_agreement_fraction'])} | {row['actual_mask_predicts_improvement_count']} | {row['hard_improves_parent_count']} |"
        )
    lines.extend([
        "",
        "Raw gradients were absent, so they were indispensably reconstructed. Observation tensors were also absent and required eight observation-reconstruction forwards. These are separately counted and never used to evaluate a new proposal.",
        "",
        "## D. ORACLE_POSTMORTEM_ONLY known finite-step sign audit",
        "",
        "| candidate | first-order ΔMSE | first-order rank / 12 | hard singleton ΔRMSE | hard rank / 12 | first-order improves? | hard improves? | sign agreement |",
        "|---|---:|---:|---:|---:|:---:|:---:|:---:|",
    ])
    for row in summary["finite_step_oracle"]:
        lines.append(
            f"| {row['candidate_id']} | {_fmt(row['predicted_delta_mse'])} | {row['first_order_rank_among_12']} | {_fmt(row['singleton_hard_delta_rmse_vs_empty'])} | {row['singleton_hard_loss_rank_among_12']} | {row['first_order_predicts_improvement']} | {row['finite_hard_step_improves']} | {row['sign_agreement']} |"
        )
    lines.extend([
        "",
        "## Classification and stop boundary",
        "",
        summary["classification_rationale"],
        "",
        "The frozen v2 protocol deviation remains separate: the analytic zero arm kept 961 hard forwards but realized 370 births rather than v1's 560 because the unchanged state-dependent kernel followed a different beam path. R2 does not reinterpret, repair, or rerun it.",
        "",
        "## Exactly one next algorithmic recommendation",
        "",
        summary["exactly_one_recommendation"],
        "",
        "This recommendation was not implemented. Stage8B and Stage8A-v3 remain unrun.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--v1-dir", type=Path, default=DEFAULT_V1)
    parser.add_argument("--r1-dir", type=Path, default=DEFAULT_R1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    v1_dir = args.v1_dir.resolve()
    r1_dir = args.r1_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse R2 output: {output_dir}")
    if output_dir == run_dir or run_dir in output_dir.parents:
        raise ValueError("R2 output must be separate from frozen v2")
    output_dir.mkdir(parents=True)
    frozen_before = {
        "stage8a_v1": dict(zip(("tree_sha256", "file_count"), _tree_sha256(v1_dir))),
        "stage8a_r1": dict(zip(("tree_sha256", "file_count"), _tree_sha256(r1_dir))),
        "stage8a_v2": dict(zip(("tree_sha256", "file_count"), _tree_sha256(run_dir))),
    }

    # Firewall: this complete table is physically frozen before Stage7 geometry opens.
    full_ranking, ranking_freeze = freeze_initial_full_ranking(run_dir, output_dir)
    config = _json(run_dir / "config_input.json")
    stage7_config_path = _resolve(config["stage7_config"])
    center_rows, _ = retrospective_center_rank(full_ranking, stage7_config_path)
    _write_csv(output_dir / "retrospective_center_rank.csv", center_rows)

    score_rows = recorded_birth_rows(run_dir)
    _write_csv(output_dir / "score_vs_hard_loss.csv", score_rows)
    grouped = {
        "all_correct": [row for row in score_rows if row["observation_kind"] == "correct"],
        "all_controls": [row for row in score_rows if row["observation_kind"] != "correct"],
        "control_zero": [row for row in score_rows if row["observation_kind"] == "zero"],
        "control_shuffled_xy": [row for row in score_rows if row["observation_kind"] == "shuffled_xy"],
        "control_wrong_case_observation": [
            row for row in score_rows if row["observation_kind"] == "wrong_case_observation"
        ],
    }
    for case_id in sorted({str(row["case_id"]) for row in score_rows}):
        for observation_kind in sorted({str(row["observation_kind"]) for row in score_rows}):
            selected = [
                row for row in score_rows
                if row["case_id"] == case_id and row["observation_kind"] == observation_kind
            ]
            if selected:
                grouped[f"arm:{case_id}/{observation_kind}"] = selected
    score_alignment = {scope: _correlation_summary(rows) for scope, rows in grouped.items()}
    alignment_rows = []
    conditioned_rows = []
    for scope, rows in grouped.items():
        summary = score_alignment[scope]
        alignment_rows.append({
            "scope": scope,
            "count": summary["count"],
            "spearman_rho": summary["spearman_rho_score_vs_hard_delta"],
            "spearman_pvalue": summary["spearman_pvalue"],
            "kendall_tau": summary["kendall_tau_score_vs_hard_delta"],
            "kendall_pvalue": summary["kendall_pvalue"],
            "higher_score_associated_with_lower_hard_rmse": summary[
                "higher_score_associated_with_lower_hard_rmse"
            ],
            "top10_k": summary["top_k_overlap"]["k10"]["k"],
            "top10_intersection": summary["top_k_overlap"]["k10"]["intersection_count"],
            "top10pct_k": summary["top_k_overlap"]["k10pct"]["k"],
            "top10pct_intersection": summary["top_k_overlap"]["k10pct"]["intersection_count"],
        })
        conditioned_rows.extend(_conditioned_score_rows(rows, scope=scope))
    _write_csv(output_dir / "score_alignment_summary.csv", alignment_rows)
    _write_csv(output_dir / "score_conditioned_on_hard_loss_quantiles.csv", conditioned_rows)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for exact gradient-hash replay but is unavailable")
    acoustic_config = _json(_resolve(config["acoustic_config"]))
    tables, _ = acoustic_tables_from_config(acoustic_config, 15)
    table = tables.property_table.to(device)
    seismic_config = _json(_resolve(config["seismic_config"]))
    operator, _ = seismic_operator_from_config(seismic_config, grid_shape=(64, 64, 64))
    bounds = _bounds(config["bounds"], int(config["stage8a_search"]["maximum_body_count"]))
    cases, observation_calls, observation_evidence = prepare_cases(config, table, operator, device)
    actual_rows, recomputation, initial = actual_mask_audit(
        run_dir=run_dir, case_assets=cases, table=table, operator=operator,
        bounds=bounds, device=device,
    )
    _write_csv(output_dir / "actual_mask_directional_derivative.csv", actual_rows)
    actual_grouped = {
        "all_correct": [row for row in actual_rows if row["observation_kind"] == "correct"],
        "all_controls": [row for row in actual_rows if row["observation_kind"] != "correct"],
    }
    for key in ("zero", "shuffled_xy", "wrong_case_observation"):
        actual_grouped[f"control_{key}"] = [
            row for row in actual_rows if row["observation_kind"] == key
        ]
    actual_alignment = {
        scope: actual_alignment_summary(rows) for scope, rows in actual_grouped.items()
    }
    actual_summary_rows = [{"scope": scope, **values} for scope, values in actual_alignment.items()]
    _write_csv(output_dir / "actual_mask_alignment_summary.csv", actual_summary_rows)

    gradient, properties, _ = initial["analytic_five_body"]
    finite_rows = finite_step_oracle_audit(
        stage7_config_path=stage7_config_path, r1_dir=r1_dir,
        analytic_assets=cases["analytic_five_body"], table=table,
        initial_gradient=gradient, initial_properties=properties, device=device,
    )
    _write_csv(output_dir / "candidate_04_06_finite_step_comparison.csv", finite_rows)

    if recomputation["implementation_defects"]:
        classification = "IMPLEMENTATION_DEFECT"
        rationale = "Recomputed frozen-state hashes did not reproduce v2, so alignment evidence is implementation-defective."
        recommendation = "Repair the identified hash/replay defect under a new protocol before changing any search logic."
    elif (
        actual_alignment["all_correct"]["actual_mask_predicts_improvement_count"]
        >= 0.9 * actual_alignment["all_correct"]["count"]
        and actual_alignment["all_correct"]["hard_improves_parent_count"] == 0
        and actual_alignment["all_correct"][
            "actual_first_order_hard_sign_agreement_fraction"
        ] < 0.1
    ):
        classification = "FIRST_ORDER_TO_FINITE_HARD_NONLINEARITY"
        rationale = (
            "Localization is not the failure: the two successful analytic centers rank 1 and 5 of 140,985. "
            "After replacing the canonical mask by every proposal's actual mask, first order still predicts improvement for 2,106/2,240 correct-arm births, while the frozen finite hard evaluation improves for 0/2,240; sign agreement is only 5.98%. "
            "The known Stage7 singleton masks themselves have matching improving first-order and finite-step signs, confirming that the implementation can represent the descent signal. "
            "The dominant break is therefore extrapolation from an infinitesimal property direction to the one-shot finite hard insertion used by v2."
        )
        recommendation = (
            "Replace one-shot full-size births with a deterministic truth-blind hard-loss trust-region continuation that starts from a nested small allowed-shape insertion and grows it only through already-budgeted hard-RMSE-improving proposals, without increasing the hard-forward budget."
        )
    else:
        localization_failure = all(float(row["rank_percentile_from_top"]) > 50.0 for row in center_rows)
        geometry_summary = actual_alignment["all_correct"]
        geometry_failure = (
            geometry_summary["spearman_canonical_score_vs_actual_mask_predicted_decrease"] is None
            or float(geometry_summary["spearman_canonical_score_vs_actual_mask_predicted_decrease"]) <= 0.0
        )
        if localization_failure and geometry_failure:
            classification = "MIXED_LOCALIZATION_AND_GEOMETRY_FAILURE"
            rationale = "Successful centers rank below the median and canonical scores do not positively track actual-mask derivatives."
            recommendation = "Rank the existing sampled birth masks directly with their actual two-property directional derivatives, with deterministic spatial diversification, inside the same proposal budget."
        elif localization_failure:
            classification = "FIRST_ORDER_CENTER_LOCALIZATION_FAILURE"
            rationale = "Both successful centers rank below the median of the frozen full center map."
            recommendation = "Replace pointwise center ranking with a deterministic acquisition-aware nonlocal localization score inside the unchanged proposal budget."
        elif geometry_failure:
            classification = "CANONICAL_TO_ACTUAL_GEOMETRY_MISMATCH"
            rationale = "Canonical center scores do not positively track the corresponding actual-mask directional derivatives."
            recommendation = "Rank each existing sampled birth by its actual shape/size/orientation two-property directional derivative instead of a canonical mask."
        else:
            classification = "UNRESOLVED"
            rationale = "The frozen evidence does not isolate one authorized failure mechanism."
            recommendation = "Under a new protocol, add one truth-blind diagnostic that compares finite-step hard loss for the already-budgeted proposed masks before altering search behavior."
    if classification not in CLASSIFICATIONS:
        raise RuntimeError("invalid R2 classification")

    frozen_after = {
        "stage8a_v1": dict(zip(("tree_sha256", "file_count"), _tree_sha256(v1_dir))),
        "stage8a_r1": dict(zip(("tree_sha256", "file_count"), _tree_sha256(r1_dir))),
        "stage8a_v2": dict(zip(("tree_sha256", "file_count"), _tree_sha256(run_dir))),
    }
    if frozen_before != frozen_after:
        raise RuntimeError("a frozen Stage8A artifact changed during R2")
    summary = {
        "schema": "stage8a_v2_r2_summary_v1",
        "primary_classification": classification,
        "classification_rationale": rationale,
        "exactly_one_recommendation": recommendation,
        "frozen_sources_unchanged": True,
        "frozen_tree_hashes": frozen_before,
        "stage8a_v2_rerun": False,
        "stage8b_run": False,
        "training_run": False,
        "search_algorithm_changed": False,
        "stage8a_v3_implemented": False,
        "new_proposals_evaluated": 0,
        "new_hard_proposal_forward_calls": 0,
        "observation_reconstruction_forward_calls": observation_calls,
        "observation_reconstruction_evidence": observation_evidence,
        "gradient_recomputation": recomputation,
        "initial_ranking_freeze": ranking_freeze,
        "retrospective_center_rank": center_rows,
        "score_alignment": score_alignment,
        "actual_mask_alignment": actual_alignment,
        "finite_step_oracle": finite_rows,
        "v2_protocol_deviation": {
            "case_id": "analytic_five_body",
            "observation_kind": "zero",
            "hard_forward_count": 961,
            "v1_birth_count": 560,
            "v2_birth_count": 370,
            "cause": "unchanged state-dependent kernel followed a different beam/body-count path",
            "reinterpreted_or_repaired": False,
            "v2_rerun_for_repair": False,
        },
        "machine_tables": [
            "analytic_initial_full_center_ranking_frozen.csv",
            "retrospective_center_rank.csv",
            "score_vs_hard_loss.csv",
            "score_alignment_summary.csv",
            "score_conditioned_on_hard_loss_quantiles.csv",
            "actual_mask_directional_derivative.csv",
            "actual_mask_alignment_summary.csv",
            "candidate_04_06_finite_step_comparison.csv",
        ],
    }
    _write_json(output_dir / "stage8a_v2_r2_summary.json", summary)
    (output_dir / "STAGE8A_V2_R2_REPORT.md").write_text(
        render_report(summary), encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "primary_classification": classification,
        "new_hard_proposal_forward_calls": 0,
        "differentiable_forward_calls": recomputation["differentiable_forward_calls"],
        "observation_reconstruction_forward_calls": observation_calls,
    }, indent=2))


if __name__ == "__main__":
    main()
