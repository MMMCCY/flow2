#!/usr/bin/env python3
"""Run the frozen Stage-8A or Stage-8B structured-posterior protocol."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import shlex
import socket
import subprocess
import sys
from typing import Mapping

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
STRUCTURALGEO_SRC = REPOSITORY_ROOT / "StructuralGeo-main/src"
for path in (PROJECT_DIR, STRUCTURALGEO_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import inference_runtime as runtime
from guidance.native_geology_audit import build_structuralgeo_native_case
from guidance.seismic import hard_labels_to_acoustic, seismic_operator_from_config
from guidance.simple_causality import build_simple_causal_case
from guidance.structured_birth_sensitivity import (
    BIRTH_CENTER_SCORE_VERSION,
    DeterministicSensitivityBirthCenterRanker,
    TIE_BREAKING,
)
from guidance.structured_hard_inference import (
    StructuredObject as Stage7Object,
    beam_evolutionary_search as stage7_library_search,
)
from guidance.structured_posterior import (
    HardConditionProjector,
    ProposalKernel,
    StructuredBounds,
    controlled_observations,
    inference_visible_audit,
    retrospective_hard_metrics,
    structured_search,
)
from guidance.structured_trust_region import (
    HardLossBirthContinuation,
    TRUST_REGION_VERSION,
    validate_scale_ladder,
)
from guidance.structured_lineage_search import (
    ALLOCATION_RULE as V4_ALLOCATION_RULE,
    LINEAGE_SEARCH_VERSION,
    lineage_preserving_structured_search,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("8a", "8b"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage8a-summary", type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _resolve(value: object) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _write_csv(path: Path, rows: list[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _distribution(values: list[float]) -> dict[str, object]:
    """Small deterministic machine-readable distribution summary."""
    if not values:
        return {"count": 0, "min": None, "p05": None, "median": None, "p95": None, "max": None}
    ordered = sorted(float(value) for value in values)

    def quantile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        left = int(position)
        right = min(left + 1, len(ordered) - 1)
        weight = position - left
        return ordered[left] * (1.0 - weight) + ordered[right] * weight

    return {
        "count": len(ordered), "min": ordered[0], "p05": quantile(0.05),
        "median": quantile(0.5), "p95": quantile(0.95), "max": ordered[-1],
    }


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPOSITORY_ROOT, text=True).strip()


def _validate_preflight(config: Mapping[str, object]) -> dict[str, object]:
    """Refuse asset drift from the frozen preflight manifest."""
    manifest_path = _resolve(config["preflight_manifest"])
    manifest = _json(manifest_path)
    checked = []
    for asset in manifest["assets"]:
        path_text = str(asset["path"])
        expected = asset["sha256"]
        if isinstance(expected, str):
            path = _resolve(path_text)
            actual = runtime.file_sha256(path)
            if actual != expected:
                raise RuntimeError(f"preflight asset hash drift: {path}")
            checked.append({"path": path_text, "sha256": actual})
        elif "{0,1,2,3}" in path_text:
            paths = [_resolve(path_text.replace("{0,1,2,3}", str(index))) for index in range(4)]
            actual = [runtime.file_sha256(path) for path in paths]
            if actual != list(expected):
                raise RuntimeError("preflight cached flow ensemble hash drift")
            checked.extend(
                {"path": str(path.relative_to(REPOSITORY_ROOT)), "sha256": digest}
                for path, digest in zip(paths, actual)
            )
        else:
            raise ValueError(f"unsupported preflight grouped asset: {path_text}")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": runtime.file_sha256(manifest_path),
        "checked_assets": checked,
    }


def _directory_tree_sha256(root: Path) -> tuple[str, int]:
    """Hash every relative path and file digest without changing the tree."""
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(runtime.file_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), len(files)


def _validate_v2_preflight(config: Mapping[str, object]) -> dict[str, object]:
    """Enforce the single authorized v1->v2 algorithmic difference."""
    v1_path = _resolve(config["frozen_stage8a_v1_config"])
    v1 = _json(v1_path)
    added = {
        "birth_center_initializer",
        "frozen_stage8a_v1_config",
        "stage8a_v2_preflight_manifest",
    }
    if set(config) != set(v1) | added:
        raise RuntimeError(
            f"Stage8A-v2 config keys drifted: {sorted(set(config) ^ (set(v1) | added))}"
        )
    for key, value in v1.items():
        if key == "schema":
            continue
        if config[key] != value:
            raise RuntimeError(f"Stage8A-v2 changed frozen v1 config field: {key}")
    initializer = config["birth_center_initializer"]
    expected_initializer_keys = {
        "mode",
        "score_version",
        "candidate_center_grid",
        "canonical_shape",
        "canonical_size_xyz",
        "property_fields",
        "tie_breaking",
        "ranked_center_record_count",
    }
    if set(initializer) != expected_initializer_keys:
        raise RuntimeError("unexpected Stage8A-v2 initializer configuration fields")
    if initializer["mode"] != "deterministic_multifield_first_order":
        raise RuntimeError("Stage8A-v2 must use deterministic first-order ranking")
    if initializer["score_version"] != BIRTH_CENTER_SCORE_VERSION:
        raise RuntimeError("Stage8A-v2 score definition drifted")
    if initializer["candidate_center_grid"] != "integer_voxel_centers_within_v1_bounds_and_edit_mask":
        raise RuntimeError("Stage8A-v2 candidate-center domain drifted")
    if initializer["canonical_shape"] != "ellipsoid":
        raise RuntimeError("Stage8A-v2 canonical score shape must remain ellipsoid")
    midpoint_size = [
        (float(config["bounds"][name][0]) + float(config["bounds"][name][1])) / 2.0
        for name in ("size_x", "size_y", "size_z")
    ]
    if [float(value) for value in initializer["canonical_size_xyz"]] != midpoint_size:
        raise RuntimeError("canonical score size must be the untuned midpoint of v1 bounds")
    if initializer["property_fields"] != ["impedance", "slowness"]:
        raise RuntimeError("both seismic property fields must enter the birth score")
    if initializer["tie_breaking"] != TIE_BREAKING:
        raise RuntimeError("Stage8A-v2 deterministic tie-breaking drifted")
    maximum_births = int(config["stage8a_search"]["beam_size"]) * int(
        config["stage8a_search"]["proposals_per_parent"]
    )
    if int(initializer["ranked_center_record_count"]) != maximum_births:
        raise RuntimeError("ranked-center record count must cover a full frozen generation")
    if "cuboid" in config["bounds"]["shapes"]:
        raise RuntimeError("Stage8A-v2 is forbidden from adding cuboid")

    manifest_path = _resolve(config["stage8a_v2_preflight_manifest"])
    manifest = _json(manifest_path)
    if manifest.get("schema") != "stage8a_v2_preflight_manifest_v1":
        raise RuntimeError("unexpected Stage8A-v2 preflight manifest")
    source_checks = []
    for item in manifest["source_files"]:
        path = _resolve(item["path"])
        actual = runtime.file_sha256(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"Stage8A-v2 frozen source drift: {path}")
        source_checks.append({"path": item["path"], "sha256": actual})
    artifact_checks = []
    for item in manifest["frozen_artifact_trees"]:
        root = _resolve(item["path"])
        actual, count = _directory_tree_sha256(root)
        if actual != item["tree_sha256"] or count != int(item["file_count"]):
            raise RuntimeError(f"frozen Stage8A artifact tree drift: {root}")
        artifact_checks.append({
            "path": item["path"], "tree_sha256": actual, "file_count": count
        })
    gate_path = _resolve(config["gate"])
    gate_sha256 = runtime.file_sha256(gate_path)
    if gate_sha256 != manifest["unchanged_stage8_gate_sha256"]:
        raise RuntimeError("original Stage8A gate changed")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": runtime.file_sha256(manifest_path),
        "frozen_v1_config_path": str(v1_path),
        "frozen_v1_config_sha256": runtime.file_sha256(v1_path),
        "unchanged_stage8_gate_sha256": gate_sha256,
        "source_files": source_checks,
        "frozen_artifact_trees": artifact_checks,
        "only_authorized_change_validated": True,
    }


def _validate_v3_preflight(config: Mapping[str, object]) -> dict[str, object]:
    """Enforce the sole v2->v3 change and all frozen artifact hashes."""
    v2_path = _resolve(config["frozen_stage8a_v2_config"])
    v2 = _json(v2_path)
    added = {
        "birth_trust_region",
        "frozen_stage8a_v2_config",
        "stage8a_v3_preflight_manifest",
    }
    if set(config) != set(v2) | added:
        raise RuntimeError(
            f"Stage8A-v3 config keys drifted: {sorted(set(config) ^ (set(v2) | added))}"
        )
    for key, value in v2.items():
        if key == "schema":
            continue
        if config[key] != value:
            raise RuntimeError(f"Stage8A-v3 changed frozen v2 config field: {key}")
    trust = config["birth_trust_region"]
    if set(trust) != {"mode", "version", "scale_ladder", "continuation_rule"}:
        raise RuntimeError("unexpected Stage8A-v3 trust-region configuration fields")
    if trust["mode"] != "nested_hard_loss_birth_continuation":
        raise RuntimeError("Stage8A-v3 trust-region mode drifted")
    if trust["version"] != TRUST_REGION_VERSION:
        raise RuntimeError("Stage8A-v3 trust-region implementation version drifted")
    if validate_scale_ladder(trust["scale_ladder"]) != (0.25, 0.5, 0.75, 1.0):
        raise RuntimeError("Stage8A-v3 frozen scale ladder drifted")
    if trust["continuation_rule"] != "continue_only_if_hard_rmse_strictly_improves_parent":
        raise RuntimeError("Stage8A-v3 hard-loss continuation rule drifted")
    if config["birth_center_initializer"] != v2["birth_center_initializer"]:
        raise RuntimeError("Stage8A-v3 changed the frozen v2 center ranker configuration")
    if "cuboid" in config["bounds"]["shapes"]:
        raise RuntimeError("Stage8A-v3 is forbidden from adding cuboid")

    manifest_path = _resolve(config["stage8a_v3_preflight_manifest"])
    manifest = _json(manifest_path)
    if manifest.get("schema") != "stage8a_v3_preflight_manifest_v1":
        raise RuntimeError("unexpected Stage8A-v3 preflight manifest")
    source_checks = []
    for item in manifest["source_files"]:
        path = _resolve(item["path"])
        actual = runtime.file_sha256(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"Stage8A-v3 frozen source drift: {path}")
        source_checks.append({"path": item["path"], "sha256": actual})
    artifact_checks = []
    for item in manifest["frozen_artifact_trees"]:
        root = _resolve(item["path"])
        actual, count = _directory_tree_sha256(root)
        if actual != item["tree_sha256"] or count != int(item["file_count"]):
            raise RuntimeError(f"frozen Stage8A artifact tree drift: {root}")
        artifact_checks.append({
            "path": item["path"], "tree_sha256": actual, "file_count": count
        })
    gate_path = _resolve(config["gate"])
    gate_sha256 = runtime.file_sha256(gate_path)
    if gate_sha256 != manifest["unchanged_stage8_gate_sha256"]:
        raise RuntimeError("original Stage8A gate changed")
    ranker_path = PROJECT_DIR / "guidance/structured_birth_sensitivity.py"
    ranker_sha256 = runtime.file_sha256(ranker_path)
    if ranker_sha256 != manifest["unchanged_v2_birth_ranker_sha256"]:
        raise RuntimeError("frozen Stage8A-v2 sensitivity ranker changed")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": runtime.file_sha256(manifest_path),
        "frozen_v2_config_path": str(v2_path),
        "frozen_v2_config_sha256": runtime.file_sha256(v2_path),
        "unchanged_stage8_gate_sha256": gate_sha256,
        "unchanged_v2_birth_ranker_sha256": ranker_sha256,
        "source_files": source_checks,
        "frozen_artifact_trees": artifact_checks,
        "only_authorized_change_validated": True,
    }


def _validate_v4_preflight(config: Mapping[str, object]) -> dict[str, object]:
    """Enforce the sole final v3->v4 lineage-allocation change."""
    v3_path = _resolve(config["frozen_stage8a_v3_config"])
    v3 = _json(v3_path)
    added = {
        "lineage_preserving_continuation",
        "frozen_stage8a_v3_config",
        "stage8a_v4_preflight_manifest",
    }
    if set(config) != set(v3) | added:
        raise RuntimeError(
            f"Stage8A-v4 config keys drifted: {sorted(set(config) ^ (set(v3) | added))}"
        )
    for key, value in v3.items():
        if key == "schema":
            continue
        if config[key] != value:
            raise RuntimeError(f"Stage8A-v4 changed frozen v3 config field: {key}")
    lineage = config["lineage_preserving_continuation"]
    expected_keys = {
        "mode", "version", "allocation_rule", "scale_ladder",
        "within_lineage_rule", "global_competition_rule",
    }
    if set(lineage) != expected_keys:
        raise RuntimeError("unexpected Stage8A-v4 lineage configuration fields")
    if lineage["mode"] != "local_monotonic_lineage_before_global_beam":
        raise RuntimeError("Stage8A-v4 lineage mode drifted")
    if lineage["version"] != LINEAGE_SEARCH_VERSION:
        raise RuntimeError("Stage8A-v4 lineage implementation version drifted")
    if lineage["allocation_rule"] != V4_ALLOCATION_RULE:
        raise RuntimeError("Stage8A-v4 deterministic slot allocation drifted")
    if validate_scale_ladder(lineage["scale_ladder"]) != (0.25, 0.5, 0.75, 1.0):
        raise RuntimeError("Stage8A-v4 frozen scale ladder drifted")
    if lineage["scale_ladder"] != config["birth_trust_region"]["scale_ladder"]:
        raise RuntimeError("Stage8A-v4 changed the frozen v3 ladder")
    if lineage["within_lineage_rule"] != "strict_hard_rmse_improvement_each_step":
        raise RuntimeError("Stage8A-v4 within-lineage rule drifted")
    if lineage["global_competition_rule"] != "best_reached_lineage_state_only":
        raise RuntimeError("Stage8A-v4 global competition rule drifted")
    if config["birth_center_initializer"] != v3["birth_center_initializer"]:
        raise RuntimeError("Stage8A-v4 changed the frozen v2 sensitivity ranker")
    if "cuboid" in config["bounds"]["shapes"]:
        raise RuntimeError("Stage8A-v4 is forbidden from adding cuboid")

    manifest_path = _resolve(config["stage8a_v4_preflight_manifest"])
    manifest = _json(manifest_path)
    if manifest.get("schema") != "stage8a_v4_preflight_manifest_v1":
        raise RuntimeError("unexpected Stage8A-v4 preflight manifest")
    source_checks = []
    for item in manifest["source_files"]:
        path = _resolve(item["path"])
        actual = runtime.file_sha256(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"Stage8A-v4 frozen source drift: {path}")
        source_checks.append({"path": item["path"], "sha256": actual})
    artifact_checks = []
    for item in manifest["frozen_artifact_trees"]:
        root = _resolve(item["path"])
        actual, count = _directory_tree_sha256(root)
        if actual != item["tree_sha256"] or count != int(item["file_count"]):
            raise RuntimeError(f"frozen Stage8A artifact tree drift: {root}")
        artifact_checks.append({
            "path": item["path"], "tree_sha256": actual, "file_count": count
        })
    gate_sha256 = runtime.file_sha256(_resolve(config["gate"]))
    if gate_sha256 != manifest["unchanged_stage8_gate_sha256"]:
        raise RuntimeError("original Stage8A gate changed")
    ranker_sha256 = runtime.file_sha256(
        PROJECT_DIR / "guidance/structured_birth_sensitivity.py"
    )
    if ranker_sha256 != manifest["unchanged_v2_birth_ranker_sha256"]:
        raise RuntimeError("frozen Stage8A-v2 sensitivity ranker changed")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": runtime.file_sha256(manifest_path),
        "frozen_v3_config_path": str(v3_path),
        "frozen_v3_config_sha256": runtime.file_sha256(v3_path),
        "unchanged_stage8_gate_sha256": gate_sha256,
        "unchanged_v2_birth_ranker_sha256": ranker_sha256,
        "source_files": source_checks,
        "frozen_artifact_trees": artifact_checks,
        "only_authorized_change_validated": True,
        "standalone_stage8a_terminal_iteration": True,
    }


def _bounds(config: Mapping[str, object], maximum: int) -> StructuredBounds:
    return StructuredBounds(
        center_x=tuple(config["center_x"]), center_y=tuple(config["center_y"]),
        center_z=tuple(config["center_z"]), size_x=tuple(config["size_x"]),
        size_y=tuple(config["size_y"]), size_z=tuple(config["size_z"]),
        orientation_deg=tuple(config["orientation_deg"]),
        shapes=tuple(config["shapes"]), material_labels=tuple(config["material_labels"]),
        maximum_body_count=int(maximum),
    )


def _hard_response(labels, table, subsurface, operator):
    acoustic = hard_labels_to_acoustic(labels.long(), table)
    return operator(acoustic[:, :1], acoustic[:, 1:2], subsurface)


def _search_arm(
    *,
    arm_dir: Path,
    base_labels: torch.Tensor,
    projector: HardConditionProjector,
    observation: torch.Tensor,
    response_fn,
    bounds: StructuredBounds,
    search_config: Mapping[str, object],
    proposal_seed: int,
    table: torch.Tensor | None = None,
    subsurface: torch.Tensor | None = None,
    operator=None,
    target_label: int | None = None,
    birth_initializer_config: Mapping[str, object] | None = None,
    birth_trust_region_config: Mapping[str, object] | None = None,
    lineage_continuation_config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    arm_dir.mkdir(parents=True, exist_ok=False)
    birth_center_ranker = None
    if birth_initializer_config is not None:
        if table is None or subsurface is None or operator is None or target_label is None:
            raise ValueError("sensitivity birth initialization requires the fixed physics inputs")
        if birth_initializer_config.get("mode") != "deterministic_multifield_first_order":
            raise ValueError("unsupported Stage8A-v2 birth initializer")
        birth_center_ranker = DeterministicSensitivityBirthCenterRanker(
            projector=projector,
            property_table=table,
            subsurface_mask=subsurface,
            seismic_forward=operator,
            bounds=bounds,
            target_label=int(target_label),
            canonical_size_xyz=birth_initializer_config["canonical_size_xyz"],
            audit_dir=arm_dir / "sensitivity_maps",
            ranked_center_record_count=int(
                birth_initializer_config["ranked_center_record_count"]
            ),
        )
    birth_trust_region_controller = None
    if birth_trust_region_config is not None:
        if birth_center_ranker is None:
            raise ValueError("trust-region continuation requires the frozen v2 ranker")
        birth_trust_region_controller = HardLossBirthContinuation(
            scale_ladder=birth_trust_region_config["scale_ladder"]
        )
    common_search = {
        "base_labels": base_labels,
        "projector": projector,
        "observation": observation,
        "hard_response": response_fn,
        "proposal_kernel": ProposalKernel(bounds, seed=proposal_seed),
        "beam_size": int(search_config["beam_size"]),
        "generations": int(search_config["generations"]),
        "proposals_per_parent": int(search_config["proposals_per_parent"]),
        "birth_center_ranker": birth_center_ranker,
    }
    if lineage_continuation_config is not None:
        if birth_center_ranker is None or birth_trust_region_controller is not None:
            raise ValueError("v4 lineage continuation requires only the frozen v2 ranker")
        result = lineage_preserving_structured_search(
            **common_search,
            scale_ladder=lineage_continuation_config["scale_ladder"],
        )
    else:
        result = structured_search(
            **common_search,
            birth_trust_region_controller=birth_trust_region_controller,
        )
    torch.save(result["best_labels"], arm_dir / "selected_labels.pt")
    torch.save(result["best_response"], arm_dir / "selected_response.pt")
    selection = {
        "best_hard_rmse": result["best_hard_rmse"],
        "baseline_hard_rmse": result["baseline_hard_rmse"],
        "hard_attainment": result["hard_attainment"],
        "best_state": result["best_state"].record(),
        "forward_call_count": result["forward_call_count"],
        "fixed_forward_call_budget": result["fixed_forward_call_budget"],
        "runtime_seconds": result["runtime_seconds"],
        "selection_used_truth": result["selection_used_truth"],
        "selection_criterion": result["selection_criterion"],
        "proposal_seed": result["proposal_seed"],
        "selected_labels_sha256": runtime.file_sha256(arm_dir / "selected_labels.pt"),
        "selected_response_sha256": runtime.file_sha256(arm_dir / "selected_response.pt"),
        "selection_frozen_before_retrospective_evaluation": True,
        "realized_proposal_move_counts": {
            move: sum(
                row["state"]["proposal_move"] == move for row in result["trace"][1:]
            )
            for move in sorted({
                row["state"]["proposal_move"] for row in result["trace"][1:]
            })
        },
        "frozen_scheduled_move_counts": {
            move: sum(row.get("scheduled_move") == move for row in result["trace"][1:])
            for move in sorted({
                row.get("scheduled_move") for row in result["trace"][1:]
                if row.get("scheduled_move") is not None
            })
        },
    }
    initializer = result.get("birth_center_initializer")
    if initializer is not None:
        initializer = dict(initializer)
        rankings = initializer.pop("rankings")
        _write_json(arm_dir / "sensitivity_rankings.json", {"rankings": rankings})
        birth_rows = []
        for row in result["trace"]:
            if "birth_guidance" not in row:
                continue
            guidance = row["birth_guidance"]
            birth_rows.append({
                "generation": row["generation"],
                "state_id": row["state"]["state_id"],
                "parent_id": row["state"]["parent_id"],
                "ranking_id": guidance["ranking_id"],
                "rank_index": guidance["rank_index"],
                "center_x": guidance["center_xyz"][0],
                "center_y": guidance["center_xyz"][1],
                "center_z": guidance["center_xyz"][2],
                "predicted_first_order_mse_decrease": guidance[
                    "predicted_first_order_mse_decrease"
                ],
                "hard_rmse": guidance["hard_rmse"],
                "delta_rmse_vs_parent": guidance["delta_rmse_vs_parent"],
                "delta_rmse_vs_empty": guidance["delta_rmse_vs_empty"],
                "condition_violations": guidance["condition_violations"],
                "truth_used": guidance["truth_used"],
            })
        _write_csv(arm_dir / "birth_proposals.csv", birth_rows)
        selection["birth_center_initializer"] = initializer
        selection["sensitivity_rankings_sha256"] = runtime.file_sha256(
            arm_dir / "sensitivity_rankings.json"
        )
        selection["birth_proposals_sha256"] = runtime.file_sha256(
            arm_dir / "birth_proposals.csv"
        )
    trust_region = result.get("birth_trust_region")
    if trust_region is not None:
        trust_region = dict(trust_region)
        probes = trust_region.pop("probes")
        branches = trust_region.pop("branches")
        probe_rows = []
        for row in probes:
            probe_body = row["probe_body"]
            full_body = row["full_target_body"]
            probe_rows.append({
                "branch_id": row["branch_id"],
                "probe_kind": row["probe_kind"],
                "generation": row["generation"],
                "proposal_index": row["proposal_index"],
                "child_state_id": row["child_state_id"],
                "parent_state_id": row["parent_state_id"],
                "ranking_id": row["ranking_id"],
                "rank_index": row["rank_index"],
                "canonical_center_score": row["canonical_center_score"],
                "center_x": full_body["center_x"],
                "center_y": full_body["center_y"],
                "center_z": full_body["center_z"],
                "shape": full_body["shape"],
                "material_label": full_body["material_label"],
                "orientation_deg": full_body["orientation_deg"],
                "full_size_x": full_body["size_x"],
                "full_size_y": full_body["size_y"],
                "full_size_z": full_body["size_z"],
                "scale_index": row["scale_index"],
                "scale": row["scale"],
                "probe_size_x": probe_body["size_x"],
                "probe_size_y": probe_body["size_y"],
                "probe_size_z": probe_body["size_z"],
                "hard_rmse": row["hard_rmse"],
                "delta_rmse_vs_parent": row["delta_rmse_vs_parent"],
                "delta_rmse_vs_empty": row["delta_rmse_vs_empty"],
                "hard_loss_improving_vs_parent": row["hard_loss_improving_vs_parent"],
                "condition_violations": row["condition_violations"],
                "continuation_authorized": row["continuation_authorized"],
                "next_scale": row["next_scale"],
                "termination": row["termination"],
                "truth_used": row["truth_used"],
            })
        _write_csv(arm_dir / "trust_region_probes.csv", probe_rows)
        _write_json(arm_dir / "trust_region_branches.json", {"branches": branches})
        nonbirth = result["forward_call_count"] - 1 - len(probes)
        trust_region["slot_allocation"] = {
            "initial_empty": 1,
            "new_center": trust_region["new_center_probe_count"],
            "growth": trust_region["growth_probe_count"],
            "nonbirth": nonbirth,
            "total_hard_forward_calls": result["forward_call_count"],
        }
        trust_region["trust_region_probes_sha256"] = runtime.file_sha256(
            arm_dir / "trust_region_probes.csv"
        )
        trust_region["trust_region_branches_sha256"] = runtime.file_sha256(
            arm_dir / "trust_region_branches.json"
        )
        growth_rows = [row for row in probes if row["probe_kind"] == "growth"]
        trust_region["growth_delta_rmse_vs_parent_distribution"] = _distribution(
            [row["delta_rmse_vs_parent"] for row in growth_rows]
        )
        trust_region["improving_growth_delta_rmse_vs_parent_distribution"] = _distribution(
            [
                row["delta_rmse_vs_parent"] for row in growth_rows
                if row["hard_loss_improving_vs_parent"]
            ]
        )
        trust_region["probe_diagnostics_by_scale"] = {
            str(scale): {
                "probe_count": sum(float(row["scale"]) == float(scale) for row in probes),
                "improving_count": sum(
                    float(row["scale"]) == float(scale)
                    and bool(row["hard_loss_improving_vs_parent"])
                    for row in probes
                ),
                "delta_rmse_vs_parent_distribution": _distribution([
                    row["delta_rmse_vs_parent"] for row in probes
                    if float(row["scale"]) == float(scale)
                ]),
            }
            for scale in trust_region["fixed_scale_ladder"]
        }
        trust_region["probe_diagnostics_by_shape"] = {
            shape: {
                "new_center_probe_count": sum(
                    row["probe_kind"] == "new_center"
                    and row["full_target_body"]["shape"] == shape
                    for row in probes
                ),
                "growth_probe_count": sum(
                    row["probe_kind"] == "growth"
                    and row["full_target_body"]["shape"] == shape
                    for row in probes
                ),
                "improving_probe_count": sum(
                    row["full_target_body"]["shape"] == shape
                    and bool(row["hard_loss_improving_vs_parent"])
                    for row in probes
                ),
                "delta_rmse_vs_parent_distribution": _distribution([
                    row["delta_rmse_vs_parent"] for row in probes
                    if row["full_target_body"]["shape"] == shape
                ]),
            }
            for shape in sorted({row["full_target_body"]["shape"] for row in probes})
        }
        smallest_improved_full_reached = 0
        smaller_improved_full_failed = 0
        branch_best_scales = []
        for branch in branches:
            branch_probes = branch["probes"]
            if not branch_probes:
                continue
            best_probe = min(
                branch_probes,
                key=lambda row: (float(row["hard_rmse"]), float(row["scale"])),
            )
            branch_best_scales.append(float(best_probe["scale"]))
            smallest_improved = bool(branch_probes[0]["hard_loss_improving_vs_parent"])
            full = next(
                (row for row in branch_probes if float(row["scale"]) == 1.0), None
            )
            if smallest_improved and full is not None:
                smallest_improved_full_reached += 1
            if (
                any(
                    float(row["scale"]) < 1.0
                    and bool(row["hard_loss_improving_vs_parent"])
                    for row in branch_probes
                )
                and full is not None
                and not bool(full["hard_loss_improving_vs_parent"])
            ):
                smaller_improved_full_failed += 1
        trust_region["mechanism_diagnostics"] = {
            "smallest_scale_improved_then_full_scale_reached_count": smallest_improved_full_reached,
            "smaller_scale_improved_but_full_scale_failed_count": smaller_improved_full_failed,
            "best_nested_scale_distribution": _distribution(branch_best_scales),
        }
        selection["birth_trust_region"] = trust_region
    lineage = result.get("lineage_continuation")
    if lineage is not None:
        lineage = dict(lineage)
        branches = lineage.pop("branches")
        probe_rows = []
        for trace_row in result["trace"]:
            probe = trace_row.get("lineage_probe")
            if probe is None:
                continue
            full = probe["full_target_body"]
            body = probe["probe_body"]
            probe_rows.append({
                "branch_id": probe["branch_id"],
                "generation": trace_row["generation"],
                "evaluation_slot": probe["evaluation_slot"],
                "probe_kind": probe["probe_kind"],
                "transition": probe["transition"],
                "from_scale": probe["from_scale"],
                "scale": probe["scale"],
                "ranking_id": probe["ranking_id"],
                "rank_index": probe["rank_index"],
                "canonical_center_score": probe["canonical_center_score"],
                "lineage_parent_state_id": trace_row["state"]["parent_id"],
                "child_state_id": trace_row["state"]["state_id"],
                "center_x": full["center_x"],
                "center_y": full["center_y"],
                "center_z": full["center_z"],
                "shape": full["shape"],
                "material_label": full["material_label"],
                "orientation_deg": full["orientation_deg"],
                "full_size_x": full["size_x"],
                "full_size_y": full["size_y"],
                "full_size_z": full["size_z"],
                "probe_size_x": body["size_x"],
                "probe_size_y": body["size_y"],
                "probe_size_z": body["size_z"],
                "hard_rmse": probe["hard_rmse"],
                "delta_rmse_vs_lineage_parent": probe[
                    "delta_rmse_vs_lineage_parent"
                ],
                "delta_rmse_vs_empty": probe["delta_rmse_vs_empty"],
                "strictly_improves_lineage_parent": probe[
                    "strictly_improves_lineage_parent"
                ],
                "reallocated_existing_slot": probe["reallocated_existing_slot"],
                "displaced_scheduled_move": probe["displaced_scheduled_move"],
                "displaced_scheduled_parent_state_id": probe[
                    "displaced_scheduled_parent_state_id"
                ],
                "condition_violations": probe["condition_violations"],
                "truth_used": probe["truth_used"],
            })
        _write_csv(arm_dir / "lineage_probes.csv", probe_rows)
        _write_json(arm_dir / "lineage_branches.json", {"branches": branches})
        lineage["lineage_probes_sha256"] = runtime.file_sha256(
            arm_dir / "lineage_probes.csv"
        )
        lineage["lineage_branches_sha256"] = runtime.file_sha256(
            arm_dir / "lineage_branches.json"
        )
        lineage["growth_delta_rmse_distribution"] = _distribution([
            probe["delta_rmse_vs_lineage_parent"] for probe in probe_rows
            if probe["probe_kind"] == "growth"
        ])
        lineage["successful_growth_delta_rmse_distribution"] = _distribution([
            probe["delta_rmse_vs_lineage_parent"] for probe in probe_rows
            if probe["probe_kind"] == "growth"
            and probe["strictly_improves_lineage_parent"]
        ])
        lineage["maximum_attained_scale_distribution"] = _distribution([
            branch["maximum_attained_scale"] for branch in branches
        ])
        selection["lineage_continuation"] = lineage
    _write_json(arm_dir / "selection.json", selection)
    _write_json(arm_dir / "proposal_trace.json", {"trace": result["trace"]})
    return {**selection, "labels": result["best_labels"], "response": result["best_response"]}


def _condition_projector(values, condition, edit) -> HardConditionProjector:
    return HardConditionProjector(
        condition_values=values.detach().cpu(),
        condition_mask=condition.detach().cpu(),
        edit_mask=edit.detach().cpu(),
    )


def _case_run(
    *,
    case_id: str,
    case_dir: Path,
    base_labels: torch.Tensor,
    condition_mask: torch.Tensor,
    edit_mask: torch.Tensor,
    truth_labels: torch.Tensor,
    retrospective_evaluation_mask: torch.Tensor,
    retrospective_truth_target_mask: torch.Tensor,
    observations: Mapping[str, torch.Tensor],
    table: torch.Tensor,
    subsurface: torch.Tensor,
    operator,
    bounds: StructuredBounds,
    search_config: Mapping[str, object],
    proposal_seed: int,
    target_label: int,
    birth_initializer_config: Mapping[str, object] | None = None,
    birth_trust_region_config: Mapping[str, object] | None = None,
    lineage_continuation_config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    projector = _condition_projector(base_labels, condition_mask, edit_mask)
    response_fn = lambda labels: _hard_response(
        labels.to(table.device), table, subsurface.to(table.device), operator
    )
    arms = []
    saved = {}
    # Selection files are frozen before the retrospective loop below opens truth.
    for name, observation in observations.items():
        selected = _search_arm(
            arm_dir=case_dir / name,
            base_labels=base_labels.to(table.device), projector=projector,
            observation=observation.to(table.device), response_fn=response_fn,
            bounds=bounds, search_config=search_config, proposal_seed=proposal_seed,
            table=table, subsurface=subsurface.to(table.device), operator=operator,
            target_label=target_label,
            birth_initializer_config=birth_initializer_config,
            birth_trust_region_config=birth_trust_region_config,
            lineage_continuation_config=lineage_continuation_config,
        )
        saved[name] = selected
    correct = observations["correct"].to(table.device)
    base_response = response_fn(base_labels.to(table.device))
    base_correct_rmse = float((base_response - correct).square().mean().sqrt().cpu())
    for name, selected in saved.items():
        labels = selected.pop("labels")
        response = selected.pop("response").to(table.device)
        inference = inference_visible_audit(
            labels, base_labels=base_labels.cpu(), projector=projector,
            predicted_response=response.cpu(), observation=observations[name].cpu(),
        )
        retrospective = retrospective_hard_metrics(
            labels, truth_labels=truth_labels.cpu(),
            condition_mask=condition_mask.cpu(), target_label=target_label,
            base_labels=base_labels.cpu(),
            truth_target_mask=retrospective_truth_target_mask.cpu(),
            evaluation_mask=retrospective_evaluation_mask.cpu(),
        )
        correct_rmse = float((response - correct).square().mean().sqrt().cpu())
        arms.append({
            "case_id": case_id, "optimized_by": name,
            **selected, **inference, **retrospective,
            "hard_correct_observation_rmse": correct_rmse,
            "hard_correct_observation_attainment": (
                1.0 - correct_rmse / base_correct_rmse if base_correct_rmse > 0 else None
            ),
        })
        _write_json(case_dir / name / "retrospective_metrics.json", retrospective)
    ranked = sorted(arms, key=lambda row: (row["hard_correct_observation_rmse"], row["optimized_by"]))
    correct_first = (
        ranked[0]["optimized_by"] == "correct"
        and ranked[0]["hard_correct_observation_rmse"] < ranked[1]["hard_correct_observation_rmse"]
    )
    return {
        "case_id": case_id,
        "arms": arms,
        "correct_evaluation_ranking": [row["optimized_by"] for row in ranked],
        "correct_optimized_is_strictly_best_against_correct": correct_first,
        "base_correct_observation_rmse": base_correct_rmse,
    }


def _body_geometry_key(body: Mapping[str, object]) -> str:
    """Match a frozen v2 full target to a v3 full target without body serials."""
    fields = (
        "center_x", "center_y", "center_z", "size_x", "size_y", "size_z",
        "orientation_deg", "shape", "material_label",
    )
    return json.dumps(
        {field: body[field] for field in fields},
        sort_keys=True,
        separators=(",", ":"),
    )


def _attach_frozen_v2_full_size_alignment(
    *,
    case_results: list[dict[str, object]],
    v3_output_dir: Path,
    frozen_v2_run_dir: Path,
) -> None:
    """Post-selection mechanism comparison; never enters selection or gates."""
    for case in case_results:
        case_id = str(case["case_id"])
        for arm in case["arms"]:
            arm_name = str(arm["optimized_by"])
            v2_trace_path = frozen_v2_run_dir / "cases" / case_id / arm_name / "proposal_trace.json"
            v3_branch_path = v3_output_dir / "cases" / case_id / arm_name / "trust_region_branches.json"
            if not v2_trace_path.is_file() or not v3_branch_path.is_file():
                raise FileNotFoundError("frozen v2/v3 mechanism artifact is missing")
            v2_by_geometry: dict[str, list[Mapping[str, object]]] = {}
            for trace_row in _json(v2_trace_path)["trace"]:
                guidance = trace_row.get("birth_guidance")
                if guidance is None:
                    continue
                body = trace_row["state"]["bodies"][-1]
                v2_by_geometry.setdefault(_body_geometry_key(body), []).append({
                    "state_id": trace_row["state"]["state_id"],
                    "hard_rmse": guidance["hard_rmse"],
                    "delta_rmse_vs_parent": guidance["delta_rmse_vs_parent"],
                    "delta_rmse_vs_empty": guidance["delta_rmse_vs_empty"],
                })
            rows = []
            for branch in _json(v3_branch_path)["branches"]:
                matches = v2_by_geometry.get(_body_geometry_key(branch["full_target_body"]), [])
                probes = branch["probes"]
                best = min(probes, key=lambda row: (row["hard_rmse"], row["scale"]))
                smallest = probes[0]
                full = next((row for row in probes if float(row["scale"]) == 1.0), None)
                for match in matches:
                    body = branch["full_target_body"]
                    rows.append({
                        "branch_id": branch["branch_id"],
                        "ranking_id": branch["ranking_id"],
                        "rank_index": branch["rank_index"],
                        "center_x": body["center_x"],
                        "center_y": body["center_y"],
                        "center_z": body["center_z"],
                        "shape": body["shape"],
                        "full_size_x": body["size_x"],
                        "full_size_y": body["size_y"],
                        "full_size_z": body["size_z"],
                        "v2_full_state_id": match["state_id"],
                        "v2_full_delta_rmse_vs_parent": match["delta_rmse_vs_parent"],
                        "v2_full_delta_rmse_vs_empty": match["delta_rmse_vs_empty"],
                        "v2_full_failed_vs_parent": float(match["delta_rmse_vs_parent"]) >= 0.0,
                        "v3_smallest_scale": smallest["scale"],
                        "v3_smallest_delta_rmse_vs_parent": smallest["delta_rmse_vs_parent"],
                        "v3_smallest_improved_vs_parent": smallest["hard_loss_improving_vs_parent"],
                        "v3_best_evaluated_scale": best["scale"],
                        "v3_best_evaluated_delta_rmse_vs_parent": best["delta_rmse_vs_parent"],
                        "v3_full_scale_evaluated": full is not None,
                        "v3_full_scale_delta_rmse_vs_parent": (
                            None if full is None else full["delta_rmse_vs_parent"]
                        ),
                        "oracle_or_truth_used": False,
                        "post_selection_diagnostic_only": True,
                    })
            alignment_path = (
                v3_output_dir / "cases" / case_id / arm_name
                / "v2_full_vs_v3_nested_alignment.csv"
            )
            _write_csv(alignment_path, rows)
            diagnostic = {
                "matched_full_geometry_count": len(rows),
                "v2_full_failed_v3_smallest_improved_count": sum(
                    bool(row["v2_full_failed_vs_parent"])
                    and bool(row["v3_smallest_improved_vs_parent"])
                    for row in rows
                ),
                "post_selection_diagnostic_only": True,
                "used_for_gate": False,
                "truth_used": False,
                "table_sha256": runtime.file_sha256(alignment_path) if rows else None,
            }
            arm["birth_trust_region"]["frozen_v2_full_size_alignment"] = diagnostic


def _analytic_library(case) -> list[Stage7Object]:
    values = []
    for body in case.candidate_bodies:
        center = [(left + right - 1) / 2 for left, right in zip(body.start, body.stop)]
        size = [right - left for left, right in zip(body.start, body.stop)]
        values.append(Stage7Object(
            object_id=body.id, presence=True,
            center_x=center[0], center_y=center[1], center_z=center[2],
            size_x=size[0], size_y=size[1], size_z=size[2], orientation_deg=0,
            shape="cuboid", material_label=case.target_label,
            source_family="immutable Stage-7 analytic regression library",
        ))
    return values


def run_stage8a(config, output_dir: Path, device: torch.device) -> dict[str, object]:
    base_config = _json(_resolve(config["stage7_config"]))
    acoustic = _json(_resolve(config["acoustic_config"]))
    seismic = _json(_resolve(config["seismic_config"]))
    from guidance.seismic import acoustic_tables_from_config
    tables, acoustic_meta = acoustic_tables_from_config(acoustic, 15)
    table = tables.property_table.to(device)
    operator, seismic_meta = seismic_operator_from_config(seismic, grid_shape=(64, 64, 64))
    search_config = config["stage8a_search"]
    bounds = _bounds(config["bounds"], int(search_config["maximum_body_count"]))
    proposal_seed = int(config["proposal_seed"])
    shuffle_seed = int(config["shuffle_seed"])
    birth_initializer_config = config.get("birth_center_initializer")
    lineage_continuation_config = config.get("lineage_preserving_continuation")
    birth_trust_region_config = (
        None if lineage_continuation_config is not None
        else config.get("birth_trust_region")
    )

    analytic = build_simple_causal_case(base_config)
    response_fn = lambda labels: _hard_response(
        labels.to(device), table, analytic.subsurface_mask.to(device), operator
    )
    analytic_correct = response_fn(analytic.truth_labels)
    wrong_labels = analytic.baseline_labels.clone()
    wrong_union = analytic.candidate_masks[[1, 9]].any(dim=0)
    wrong_labels[0, 0, wrong_union] = analytic.target_label
    observations = controlled_observations(
        analytic_correct, wrong_case=response_fn(wrong_labels), shuffle_seed=shuffle_seed
    )
    analytic_result = _case_run(
        case_id="analytic_five_body", case_dir=output_dir / "cases/analytic_five_body",
        base_labels=analytic.baseline_labels, condition_mask=analytic.condition_mask,
        edit_mask=analytic.subsurface_mask & ~analytic.condition_mask,
        truth_labels=analytic.truth_labels,
        retrospective_evaluation_mask=analytic.candidate_masks.any(dim=0).view(1, 1, 64, 64, 64),
        retrospective_truth_target_mask=analytic.candidate_masks[
            list(analytic.truth_candidate_indices)
        ].any(dim=0).view(1, 1, 64, 64, 64),
        observations=observations, table=table, subsurface=analytic.subsurface_mask,
        operator=operator, bounds=bounds, search_config=search_config,
        proposal_seed=proposal_seed, target_label=analytic.target_label,
        birth_initializer_config=birth_initializer_config,
        birth_trust_region_config=birth_trust_region_config,
        lineage_continuation_config=lineage_continuation_config,
    )
    # Immutable exact-library regression remains separate from the continuous gate.
    regression = stage7_library_search(
        baseline_labels=analytic.baseline_labels.to(device),
        condition_mask=analytic.condition_mask.to(device), air_start_z=56,
        observation=analytic_correct, hard_response=response_fn,
        proposal_library=_analytic_library(analytic), allowed_material_labels=(0, 9),
        kmax=2, beam_size=12, local_generations=1,
    )
    analytic_result["stage7_exact_library_regression"] = {
        "best_hard_rmse": regression["best_hard_rmse"],
        "forward_call_count": regression["forward_call_count"],
        "selection_used_truth": regression["selection_used_truth"],
        "regression_only_not_continuous_gate": True,
    }

    native_results = []
    for offset, seed in enumerate(config["stage8a_native_seeds"]):
        case, metadata = build_structuralgeo_native_case(seed=int(seed))
        wrong_case, wrong_metadata = build_structuralgeo_native_case(seed=int(seed) + 1000)
        hidden = case.body_masks[3:].any(dim=0).view(1, 1, 64, 64, 64)
        # The authoritative continuous/native start uses observed conditions
        # only.  It does not erase hidden bodies from a truth-derived model or
        # retain unobserved geometry of the drilled bodies.
        base = torch.full_like(case.truth_labels, case.background_label)
        base[~case.subsurface_mask] = case.air_label
        base[case.condition_mask] = case.truth_labels[case.condition_mask]
        correct = _hard_response(case.truth_labels.to(device), table, case.subsurface_mask.to(device), operator)
        wrong = _hard_response(wrong_case.truth_labels.to(device), table, wrong_case.subsurface_mask.to(device), operator)
        result = _case_run(
            case_id=f"native_seed{seed}", case_dir=output_dir / f"cases/native_seed{seed}",
            base_labels=base, condition_mask=case.condition_mask,
            edit_mask=case.subsurface_mask & ~case.condition_mask,
            truth_labels=case.truth_labels,
            retrospective_evaluation_mask=case.subsurface_mask & ~case.condition_mask,
            retrospective_truth_target_mask=hidden,
            observations=controlled_observations(
                correct, wrong_case=wrong, shuffle_seed=shuffle_seed + int(seed)
            ),
            table=table, subsurface=case.subsurface_mask, operator=operator,
            bounds=bounds, search_config=search_config,
            proposal_seed=proposal_seed + offset + 1, target_label=case.target_label,
            birth_initializer_config=birth_initializer_config,
            birth_trust_region_config=birth_trust_region_config,
            lineage_continuation_config=lineage_continuation_config,
        )
        result["fixture_metadata"] = metadata
        result["wrong_case_metadata"] = wrong_metadata
        result["true_concealed_body_count_retrospective_only"] = int(case.body_masks[3:].shape[0])
        native_results.append(result)

    gate_config = _json(_resolve(config["gate"]))["stage8a"]
    cases = [analytic_result, *native_results]
    v2_config_path = _resolve(config["frozen_stage8a_v2_config"]) if birth_trust_region_config else None
    if v2_config_path is not None:
        frozen_v2_run_dir = v2_config_path.parent.parent / "runs/stage8a_v2"
        _attach_frozen_v2_full_size_alignment(
            case_results=cases,
            v3_output_dir=output_dir,
            frozen_v2_run_dir=frozen_v2_run_dir,
        )
    correct_rows = [next(row for row in case["arms"] if row["optimized_by"] == "correct") for case in cases]
    analytic_correct_row = correct_rows[0]
    native_correct = correct_rows[1:]
    gates = {
        "all_required_cases_present": {case["case_id"] for case in cases} == set(gate_config["required_cases"]),
        "correct_specificity": sum(case["correct_optimized_is_strictly_best_against_correct"] for case in cases) / len(cases) >= gate_config["correct_arm_strictly_first_against_correct_required_fraction"],
        "analytic_hidden_iou": analytic_correct_row["concealed_target_iou"] >= gate_config["analytic_hidden_iou_minimum"],
        "analytic_hidden_recall": analytic_correct_row["concealed_target_recall"] >= gate_config["analytic_hidden_recall_minimum"],
        "native_mean_iou": sum(row["concealed_target_iou"] for row in native_correct) / len(native_correct) >= gate_config["native_mean_concealed_iou_minimum"],
        "native_mean_recall": sum(row["concealed_target_recall"] for row in native_correct) / len(native_correct) >= gate_config["native_mean_concealed_recall_minimum"],
        "geometry_improves_from_empty": all(row["concealed_target_iou"] > 0 for row in native_correct),
        "conditions_exact": all(row["hard_condition_violations"] <= gate_config["maximum_condition_violations"] for case in cases for row in case["arms"]),
        "truth_blind_selection": all(not row["selection_used_truth"] for case in cases for row in case["arms"]),
        "identical_budgets": all(len({row["forward_call_count"] for row in case["arms"]}) == 1 for case in cases),
        "unknown_count": all(abs(len(row["best_state"]["bodies"]) - case["true_concealed_body_count_retrospective_only"]) <= gate_config["unknown_count_absolute_error_maximum"] for row, case in zip(native_correct, native_results)),
    }
    correct_birth_diagnostics = {
        row["case_id"]: {
            "birth_proposal_count": row.get("birth_center_initializer", {}).get(
                "birth_proposal_count"
            ),
            "delta_rmse_lt_zero_vs_parent_count": row.get(
                "birth_center_initializer", {}
            ).get("loss_improving_birth_count_vs_parent"),
            "delta_rmse_lt_zero_vs_empty_count": row.get(
                "birth_center_initializer", {}
            ).get("loss_improving_birth_count_vs_empty"),
            "smallest_scale_delta_rmse_lt_zero_vs_parent_count": row.get(
                "birth_trust_region", {}
            ).get("smallest_scale_improving_count"),
            "growth_probe_count": row.get("birth_trust_region", {}).get(
                "growth_probe_count"
            ),
            "growth_delta_rmse_lt_zero_vs_parent_count": row.get(
                "birth_trust_region", {}
            ).get("growth_improving_count"),
            "growth_delta_rmse_vs_parent_distribution": row.get(
                "birth_trust_region", {}
            ).get("growth_delta_rmse_vs_parent_distribution"),
            "v2_full_failed_v3_smallest_improved_count": row.get(
                "birth_trust_region", {}
            ).get("frozen_v2_full_size_alignment", {}).get(
                "v2_full_failed_v3_smallest_improved_count"
            ),
            "v4_locally_improving_scale_0_25_seeds": row.get(
                "lineage_continuation", {}
            ).get("locally_improving_scale_0_25_seeds"),
            "v4_continuation_attempts": row.get(
                "lineage_continuation", {}
            ).get("continuation_attempts"),
            "v4_transition_attempts": row.get(
                "lineage_continuation", {}
            ).get("transition_attempts"),
            "v4_transition_successes": row.get(
                "lineage_continuation", {}
            ).get("transition_successes"),
            "v4_branch_final_global_beam_survival_count": row.get(
                "lineage_continuation", {}
            ).get("branch_final_global_beam_survival_count"),
            "v4_slot_allocation": row.get("lineage_continuation", {}).get(
                "slot_allocation"
            ),
        }
        for row in correct_rows
    }
    summary = {
        "schema": (
            "stage8a_summary_v4"
            if lineage_continuation_config is not None
            else (
                "stage8a_summary_v3"
                if birth_trust_region_config is not None
                else ("stage8a_summary_v2" if birth_initializer_config is not None else "stage8a_summary_v1")
            )
        ), "status": "completed",
        "decision": "PASS_STAGE8A" if all(gates.values()) else "FAIL_STAGE8A_STOP_BEFORE_STAGE8B",
        "gates": gates, "analytic": analytic_result, "native_cases": native_results,
        "acoustic_metadata": acoustic_meta, "seismic_metadata": seismic_meta,
        "correct_arm_birth_diagnostics": correct_birth_diagnostics,
    }
    return summary


def _condition_assets(config, device):
    observation_dir = _resolve(config["observation_dir"])
    boreholes = runtime.normalize_single_geology(runtime.load_tensor(_resolve(config["boreholes"])), "boreholes").long()
    subsurface = runtime.load_tensor(observation_dir / "subsurface_mask.pt").bool()
    values = torch.zeros_like(boreholes)
    values[~subsurface] = -1
    values[boreholes != -1] = boreholes[boreholes != -1]
    condition = (boreholes != -1) | ~subsurface
    return boreholes, subsurface.to(device), values, condition


def _deep_edit_mask(subsurface: torch.Tensor, condition: torch.Tensor, minimum_depth: int) -> torch.Tensor:
    counts = subsurface.cpu().sum(dim=-1, keepdim=True)
    z = torch.arange(subsurface.shape[-1]).view(1, 1, 1, 1, -1)
    depth = counts - 1 - z
    return subsurface.cpu() & ~condition.cpu() & (depth >= int(minimum_depth))


def run_stage8b(config, output_dir: Path, device: torch.device, stage8a_path: Path) -> dict[str, object]:
    stage8a = _json(stage8a_path.resolve())
    if stage8a.get("decision") != "PASS_STAGE8A":
        raise RuntimeError("Stage 8A gate did not pass; Stage 8B is forbidden")
    observation_dir = _resolve(config["observation_dir"])
    seismic_config = _json(_resolve(config["seismic_config"]))
    operator, seismic_meta = seismic_operator_from_config(seismic_config, grid_shape=(64, 64, 64))
    observed = runtime.load_tensor(observation_dir / "observed_seismic.pt").to(device)
    table = runtime.load_tensor(observation_dir / "acoustic_property_table.pt").to(
        device=device, dtype=observed.dtype
    )
    _, subsurface, condition_values, condition = _condition_assets(config, device)
    edit = _deep_edit_mask(
        subsurface, condition, int(config["stage8b_screen"]["minimum_cells_below_local_surface"])
    )
    projector = _condition_projector(condition_values, condition, edit)
    response_fn = lambda labels: _hard_response(labels.to(device), table, subsurface, operator)
    screen = config["stage8b_screen"]
    flow_dir = _resolve(screen["flow_baseline_dir"])
    flow_samples = {
        int(sample_id): runtime.normalize_single_geology(
            runtime.load_tensor(flow_dir / f"sample_{sample_id}.pt"), f"flow_sample_{sample_id}"
        ).long()
        for sample_id in screen["sample_ids"]
    }
    for sample_id, labels in flow_samples.items():
        if projector.violation_count(labels):
            raise RuntimeError(f"cached flow sample {sample_id} violates hard conditions")
    wrong_labels = runtime.normalize_single_geology(
        runtime.load_tensor(_resolve(screen["wrong_case_flow_sample"])), "wrong_case_flow_sample"
    ).long()
    wrong_observation = response_fn(wrong_labels)
    observations = controlled_observations(
        observed, wrong_case=wrong_observation, shuffle_seed=int(config["shuffle_seed"])
    )
    bounds = _bounds(config["bounds"], int(config["stage8b_search"]["maximum_body_count"]))
    search_config = config["stage8b_search"]
    truth_path = _resolve(config["truth"])

    observed_borehole = condition_values[(condition_values != -1) & condition]
    labels, counts = torch.unique(observed_borehole, return_counts=True)
    background = int(labels[counts.argmax()])
    structured_base = torch.full_like(condition_values, background)
    structured_base[~subsurface.cpu()] = -1
    structured_base = projector.project(structured_base)

    rows = []
    member_records = []
    # Selection phase: no truth tensor has been loaded.
    for sample_id, flow_base in flow_samples.items():
        flow_response = response_fn(flow_base)
        flow_rmse = float((flow_response - observed.to(flow_response)).square().mean().sqrt().cpu())
        structured = _search_arm(
            arm_dir=output_dir / f"members/member_{sample_id}/STRUCTURED_ONLY/correct",
            base_labels=structured_base.to(device), projector=projector,
            observation=observed, response_fn=response_fn, bounds=bounds,
            search_config=search_config, proposal_seed=int(config["proposal_seed"]) + sample_id,
        )
        combined = {}
        for name, observation in observations.items():
            combined[name] = _search_arm(
                arm_dir=output_dir / f"members/member_{sample_id}/FLOW_PLUS_STRUCTURED/{name}",
                base_labels=flow_base.to(device), projector=projector,
                observation=observation, response_fn=response_fn, bounds=bounds,
                search_config=search_config, proposal_seed=int(config["proposal_seed"]) + sample_id,
            )
        ranking = sorted(
            ((float((value["response"].to(device) - observed.to(device)).square().mean().sqrt().cpu()), name) for name, value in combined.items())
        )
        record = {
            "member_id": sample_id,
            "flow_base": flow_base,
            "flow_response": flow_response.cpu(),
            "flow_correct_rmse": flow_rmse,
            "structured": structured,
            "combined": combined,
            "combined_correct_ranking": [name for _, name in ranking],
            "combined_correct_is_strictly_first": ranking[0][1] == "correct" and ranking[0][0] < ranking[1][0],
        }
        member_records.append(record)
        for name, value in combined.items():
            correct_rmse = float((value["response"].to(device) - observed.to(device)).square().mean().sqrt().cpu())
            rows.append({
                "member_id": sample_id, "method": "FLOW_PLUS_STRUCTURED", "optimized_by": name,
                "hard_correct_observation_rmse": correct_rmse,
                "hard_optimized_observation_rmse": value["best_hard_rmse"],
                "forward_call_count": value["forward_call_count"],
                "condition_violations": projector.violation_count(value["labels"]),
                "truth_used_for_selection": value["selection_used_truth"],
            })
        rows.append({
            "member_id": sample_id, "method": "FLOW_ONLY", "optimized_by": "none",
            "hard_correct_observation_rmse": flow_rmse,
            "hard_optimized_observation_rmse": flow_rmse,
            "forward_call_count": 0,
            "condition_violations": projector.violation_count(flow_base),
            "truth_used_for_selection": False,
        })
    combined_rows = [row for row in rows if row["method"] == "FLOW_PLUS_STRUCTURED"]
    correct_rows = [row for row in combined_rows if row["optimized_by"] == "correct"]
    control_names = ("zero", "shuffled_xy", "wrong_case_observation")
    means = {
        name: sum(row["hard_correct_observation_rmse"] for row in combined_rows if row["optimized_by"] == name) / len(flow_samples)
        for name in ("correct", *control_names)
    }
    inference_gates = {
        "conditions_exact": all(row["condition_violations"] == 0 for row in rows),
        "correct_specificity_member_fraction": sum(record["combined_correct_is_strictly_first"] for record in member_records) / len(member_records) >= 0.75,
        "correct_mean_below_controls": all(means["correct"] < means[name] for name in control_names),
        "hard_seismic_improves_all_members": all(
            row["hard_correct_observation_rmse"] < record["flow_correct_rmse"]
            for row, record in zip(correct_rows, member_records)
        ),
        "truth_blind": all(not row["truth_used_for_selection"] for row in rows),
        "identical_combined_budgets": all(
            len({row["forward_call_count"] for row in combined_rows if row["member_id"] == sample_id}) == 1
            for sample_id in flow_samples
        ),
    }
    expansion_authorized = all(inference_gates.values())
    _write_json(output_dir / "inference_visible_screening_verdict.json", {
        "gates": inference_gates,
        "expansion_authorized_before_truth_evaluation": expansion_authorized,
        "truth_metrics_inspected": False,
    })

    # Retrospective phase starts only after the inference-visible verdict exists.
    truth = runtime.normalize_single_geology(runtime.load_tensor(truth_path), "truth").long()
    for record in member_records:
        flow_metrics = retrospective_hard_metrics(
            record["flow_base"], truth_labels=truth, condition_mask=condition,
            target_label=int(config["target_label"]), base_labels=record["flow_base"],
        )
        structured_metrics = retrospective_hard_metrics(
            record["structured"]["labels"], truth_labels=truth, condition_mask=condition,
            target_label=int(config["target_label"]), base_labels=structured_base,
        )
        combined_metrics = {}
        for name, value in record["combined"].items():
            combined_metrics[name] = retrospective_hard_metrics(
                value["labels"], truth_labels=truth, condition_mask=condition,
                target_label=int(config["target_label"]), base_labels=record["flow_base"],
            )
            _write_json(
                output_dir / f"members/member_{record['member_id']}/FLOW_PLUS_STRUCTURED/{name}/retrospective_metrics.json",
                combined_metrics[name],
            )
        record["retrospective"] = {
            "FLOW_ONLY": flow_metrics,
            "STRUCTURED_ONLY": structured_metrics,
            "FLOW_PLUS_STRUCTURED": combined_metrics,
        }
        record["combined_iou_delta"] = combined_metrics["correct"]["concealed_target_iou"] - flow_metrics["concealed_target_iou"]
    gate_config = _json(_resolve(config["gate"]))["stage8b_screening"]
    iou_deltas = [record["combined_iou_delta"] for record in member_records]
    wrong_fractions = [
        record["retrospective"]["FLOW_PLUS_STRUCTURED"]["correct"]["wrong_lithology_substitution_volume"] / truth.numel()
        for record in member_records
    ]
    final_gates = {
        **inference_gates,
        "concealed_iou_improvement_fraction": sum(value > 0 for value in iou_deltas) / len(iou_deltas) >= gate_config["combined_concealed_iou_improvement_member_fraction_minimum"],
        "mean_concealed_iou_delta": sum(iou_deltas) / len(iou_deltas) >= gate_config["combined_mean_concealed_iou_delta_minimum"],
        "wrong_lithology_substitution": max(wrong_fractions) <= gate_config["wrong_lithology_substitution_fraction_maximum"],
    }
    serializable_members = []
    for record in member_records:
        serializable_members.append({
            "member_id": record["member_id"],
            "flow_correct_rmse": record["flow_correct_rmse"],
            "combined_correct_ranking": record["combined_correct_ranking"],
            "combined_correct_is_strictly_first": record["combined_correct_is_strictly_first"],
            "retrospective": record["retrospective"],
            "combined_iou_delta": record["combined_iou_delta"],
        })
    return {
        "schema": "stage8b_summary_v1", "status": "completed",
        "decision": "PASS_STAGE8B_SCREEN" if all(final_gates.values()) else "FAIL_STAGE8B_STOP",
        "inference_visible_expansion_authorized": expansion_authorized,
        "inference_visible_gates": inference_gates, "final_gates": final_gates,
        "correct_field_mean_rmse_by_optimized_arm": means,
        "members": serializable_members, "seismic_metadata": seismic_meta,
        "split_provenance": "unknown; same-distribution integration/mechanism benchmark",
    }, rows


def _report(stage: str, summary: Mapping[str, object]) -> str:
    lines = [f"# Stage {stage.upper()} report", "", f"Decision: **{summary['decision']}**", ""]
    gates = summary["gates"] if stage == "8a" else summary["final_gates"]
    lines.extend(["## Frozen gates", ""])
    for name, value in gates.items():
        lines.append(f"- `{name}`: `{value}`")
    if stage == "8a" and summary.get("correct_arm_birth_diagnostics"):
        lines.extend(["", "## Correct-arm birth diagnostics", ""])
        for case_id, values in summary["correct_arm_birth_diagnostics"].items():
            lines.append(
                f"- `{case_id}`: births={values['birth_proposal_count']}, "
                f"delta_rmse<0 vs parent={values['delta_rmse_lt_zero_vs_parent_count']}, "
                f"delta_rmse<0 vs empty={values['delta_rmse_lt_zero_vs_empty_count']}"
            )
            if values.get("smallest_scale_delta_rmse_lt_zero_vs_parent_count") is not None:
                lines.append(
                    f"  - v3 smallest-scale improving births="
                    f"{values['smallest_scale_delta_rmse_lt_zero_vs_parent_count']}; "
                    f"growth probes={values['growth_probe_count']}; "
                    f"improving growth probes="
                    f"{values['growth_delta_rmse_lt_zero_vs_parent_count']}; "
                    f"matched v2 full-size failures rescued at v3 smallest scale="
                    f"{values['v2_full_failed_v3_smallest_improved_count']}"
                )
            if values.get("v4_locally_improving_scale_0_25_seeds") is not None:
                lines.append(
                    f"  - v4 improving scale-0.25 seeds="
                    f"{values['v4_locally_improving_scale_0_25_seeds']}; "
                    f"continuation attempts={values['v4_continuation_attempts']}; "
                    f"transition successes={values['v4_transition_successes']}; "
                    f"final lineage candidates surviving global beam="
                    f"{values['v4_branch_final_global_beam_survival_count']}"
                )
    lines.extend([
        "", "Selection used hard observed seismic RMSE only. Truth metrics were computed only after selection files were frozen.",
        "", "No training, fine-tuning, LoRA, Stage-9 SMC/RJMCMC, gravity, or magnetics were run.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = _args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse immutable output: {output_dir}")
    config_path = args.config.resolve()
    config = _json(config_path)
    schema = config.get("schema")
    if schema not in {
        "stage8_structured_posterior_config_v1",
        "stage8_structured_posterior_config_v2",
        "stage8_structured_posterior_config_v3",
        "stage8_structured_posterior_config_v4",
    }:
        raise ValueError("unexpected Stage-8 config schema")
    if schema in {
        "stage8_structured_posterior_config_v2",
        "stage8_structured_posterior_config_v3",
        "stage8_structured_posterior_config_v4",
    } and args.stage != "8a":
        raise RuntimeError("Stage8A-v2/v3/v4 configuration is forbidden from running Stage8B")
    if config.get("formal_training_authorized") or config.get("stage9_authorized"):
        raise ValueError("Stage 8 forbids training and Stage 9")
    if "truth" in inspect.signature(structured_search).parameters:
        raise RuntimeError("truth-leakage firewall failed: search accepts truth")
    for target in (
        DeterministicSensitivityBirthCenterRanker.__init__,
        DeterministicSensitivityBirthCenterRanker.rank,
    ):
        if "truth" in inspect.signature(target).parameters:
            raise RuntimeError("truth-leakage firewall failed: birth ranker accepts truth")
    for target in (
        HardLossBirthContinuation.__init__,
        HardLossBirthContinuation.plan_new_center,
        HardLossBirthContinuation.plan_growth,
        HardLossBirthContinuation.record_result,
    ):
        if "truth" in inspect.signature(target).parameters:
            raise RuntimeError("truth-leakage firewall failed: continuation accepts truth")
    if "truth" in inspect.signature(lineage_preserving_structured_search).parameters:
        raise RuntimeError("truth-leakage firewall failed: v4 lineage search accepts truth")
    preflight_validation = _validate_preflight(config)
    v2_preflight_validation = (
        _validate_v2_preflight(config)
        if schema == "stage8_structured_posterior_config_v2"
        else None
    )
    v3_preflight_validation = (
        _validate_v3_preflight(config)
        if schema == "stage8_structured_posterior_config_v3"
        else None
    )
    v4_preflight_validation = (
        _validate_v4_preflight(config)
        if schema == "stage8_structured_posterior_config_v4"
        else None
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output_dir.mkdir(parents=True)
    _write_json(output_dir / "config_input.json", config)
    if args.stage == "8a":
        summary = run_stage8a(config, output_dir, device)
        csv_rows = [row for case in [summary["analytic"], *summary["native_cases"]] for row in case["arms"]]
        _write_csv(output_dir / "stage8a_paired.csv", csv_rows)
        summary_name = "stage8a_summary.json"
        report_name = "STAGE8A_REPORT.md"
    else:
        if args.stage8a_summary is None:
            raise ValueError("--stage8a-summary is required for Stage 8B")
        summary, csv_rows = run_stage8b(config, output_dir, device, args.stage8a_summary)
        _write_csv(output_dir / "stage8b_paired.csv", csv_rows)
        summary_name = "stage8b_summary.json"
        report_name = "STAGE8B_REPORT.md"
    status = _git("status", "--short")
    summary = {
        **summary,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "exact_command": shlex.join([sys.executable, *sys.argv]),
        "git_branch": _git("branch", "--show-current"),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_status": status,
        "config_sha256": runtime.file_sha256(config_path),
        "checkpoint_sha256": runtime.file_sha256(_resolve(config["checkpoint"])),
        "truth_sha256": runtime.file_sha256(_resolve(config["truth"])),
        "truth_role": "observation-generation/retrospective only; never search selection",
        "condition_sha256": runtime.file_sha256(_resolve(config["boreholes"])),
        "observation_sha256": runtime.file_sha256(_resolve(config["observation_dir"]) / "observed_seismic.pt"),
        "preflight_validation": preflight_validation,
        "stage8a_v2_preflight_validation": v2_preflight_validation,
        "stage8a_v3_preflight_validation": v3_preflight_validation,
        "stage8a_v4_preflight_validation": v4_preflight_validation,
        "runtime": {
            "hostname": socket.gethostname(), "torch": torch.__version__,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    _write_json(output_dir / summary_name, summary)
    with (output_dir / report_name).open("x", encoding="utf-8") as stream:
        stream.write(_report(args.stage, summary))
    print(json.dumps({"output_dir": str(output_dir), "decision": summary["decision"]}, indent=2))


if __name__ == "__main__":
    main()
