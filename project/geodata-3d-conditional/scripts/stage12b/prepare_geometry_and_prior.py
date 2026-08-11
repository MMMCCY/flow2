#!/usr/bin/env python3
"""Freeze the Stage12B prior bank and write the descriptive truth-geometry audit."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy import ndimage
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import inference_runtime as runtime
from guidance.prior_ensemble import file_sha256
from scripts.stage10.build_probability_bridge import _load_fixed_prior_members
from scripts.stage12b.common import (
    CASE_IDS,
    EXPERIMENT_DIR,
    STAGE12A_DIR,
    load_config,
    resolve_path,
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


def _load_truth_mask(case_id: str) -> torch.Tensor:
    root = STAGE12A_DIR / "cases" / case_id
    truth = runtime.load_tensor(root / "truth/true_model.pt", map_location="cpu")
    mask = truth.long().eq(9)
    frozen_mask = runtime.load_tensor(root / "truth/label9_mask.pt", map_location="cpu").bool()
    if not torch.equal(mask, frozen_mask):
        raise ValueError(f"{case_id}: label9 mask differs from true_model == 9")
    return mask


def _centroid(mask: torch.Tensor) -> np.ndarray:
    points = torch.nonzero(mask[0, 0], as_tuple=False).double()
    if not points.numel():
        raise ValueError("registered case contains no label9 voxels")
    return points.mean(0).numpy()


def _component_count(mask: torch.Tensor, connectivity: int) -> int:
    rank = 1 if connectivity == 6 else 3
    structure = ndimage.generate_binary_structure(3, rank)
    _, count = ndimage.label(mask[0, 0].numpy(), structure=structure)
    return int(count)


def _matrix_rows(values: dict[tuple[str, str], float]) -> list[dict[str, object]]:
    return [
        {"source_case": source, **{target: values[(source, target)] for target in CASE_IDS}}
        for source in CASE_IDS
    ]


def _build_geometry_audit(masks: dict[str, torch.Tensor]) -> None:
    final = EXPERIMENT_DIR / "geometry"
    staging = create_staging_directory(final)
    counts = {case_id: int(mask.sum().item()) for case_id, mask in masks.items()}
    centroids = {case_id: _centroid(mask) for case_id, mask in masks.items()}
    components_6 = {case_id: _component_count(mask, 6) for case_id, mask in masks.items()}
    components_26 = {case_id: _component_count(mask, 26) for case_id, mask in masks.items()}
    long_rows: list[dict[str, object]] = []
    iou: dict[tuple[str, str], float] = {}
    distance: dict[tuple[str, str], float] = {}
    ratio: dict[tuple[str, str], float] = {}
    for source in CASE_IDS:
        for target in CASE_IDS:
            intersection = int((masks[source] & masks[target]).sum().item())
            union = int((masks[source] | masks[target]).sum().item())
            key = (source, target)
            iou[key] = intersection / union
            distance[key] = float(np.linalg.norm(centroids[source] - centroids[target]))
            ratio[key] = min(counts[source], counts[target]) / max(counts[source], counts[target])
            long_rows.append(
                {
                    "source_case": source,
                    "target_case": target,
                    "pair_type": "diagonal" if source == target else "off_diagonal",
                    "intersection_voxels": intersection,
                    "union_voxels": union,
                    "iou": iou[key],
                    "centroid_distance_voxels": distance[key],
                    "source_volume_voxels": counts[source],
                    "target_volume_voxels": counts[target],
                    "symmetric_volume_ratio_min_over_max": ratio[key],
                    "source_components_6": components_6[source],
                    "target_components_6": components_6[target],
                    "source_components_26": components_26[source],
                    "target_components_26": components_26[target],
                }
            )
    write_csv_x(staging / "truth_similarity.csv", long_rows)
    write_csv_x(staging / "pairwise_iou_matrix.csv", _matrix_rows(iou))
    write_csv_x(staging / "centroid_distance_voxels_matrix.csv", _matrix_rows(distance))
    write_csv_x(staging / "volume_ratio_matrix.csv", _matrix_rows(ratio))
    off = [row for row in long_rows if row["pair_type"] == "off_diagonal"]
    write_json_x(
        staging / "summary.json",
        {
            "schema": "stage12b_pre_geophysics_truth_geometry_audit_v1",
            "status": "complete_descriptive_only",
            "created_at_utc": utc_now(),
            "case_ids": list(CASE_IDS),
            "cohort_changed": False,
            "role": "retrospective_descriptive_only_never_case_selection",
            "target_label": 9,
            "case_statistics": {
                case_id: {
                    "volume_voxels": counts[case_id],
                    "centroid_index_xyz": centroids[case_id].tolist(),
                    "connected_components_6": components_6[case_id],
                    "connected_components_26": components_26[case_id],
                }
                for case_id in CASE_IDS
            },
            "off_diagonal_summary": {
                "pair_count_directed": len(off),
                "mean_iou": float(np.mean([row["iou"] for row in off])),
                "maximum_iou": float(max(row["iou"] for row in off)),
                "mean_centroid_distance_voxels": float(
                    np.mean([row["centroid_distance_voxels"] for row in off])
                ),
                "minimum_centroid_distance_voxels": float(
                    min(row["centroid_distance_voxels"] for row in off)
                ),
                "mean_symmetric_volume_ratio": float(
                    np.mean([row["symmetric_volume_ratio_min_over_max"] for row in off])
                ),
            },
        },
    )
    publish_staging_directory(staging, final)


def _freeze_prior_bank(config: dict[str, object]) -> None:
    final = EXPERIMENT_DIR / "prior_bank"
    staging = create_staging_directory(final)
    record = config["property_prior_bank"]
    pool = resolve_path(record["source_pool"])
    if file_sha256(pool / "manifest.json") != record["source_pool_manifest_sha256"]:
        raise ValueError("frozen Stage9A source-pool manifest hash mismatch")
    if file_sha256(pool / "candidate_pool.csv") != record["candidate_pool_csv_sha256"]:
        raise ValueError("frozen Stage9A candidate-pool CSV hash mismatch")
    labels, members, source_manifest = _load_fixed_prior_members(
        pool, [int(index) for index in record["candidate_indices"]]
    )
    tensor_record = save_tensor_x(staging / "hard_prior_members.pt", labels.to(torch.int8))
    write_json_x(
        staging / "manifest.json",
        {
            "schema": "stage12b_common_property_prior_bank_v1",
            "status": "frozen_before_stage12b_seismic_generation",
            "created_at_utc": utc_now(),
            "truth_tensor_received": False,
            "seismic_observation_received": False,
            "case_metric_received": False,
            "selection_uses_ranking": False,
            "new_flow_sampling": False,
            "policy": record["policy"],
            "qualification": record["qualification"],
            "candidate_indices": record["candidate_indices"],
            "members": members,
            "hard_prior_members": tensor_record,
            "source_pool": file_record(pool / "manifest.json", relative_to=REPOSITORY_ROOT),
            "source_candidate_pool": file_record(
                pool / "candidate_pool.csv", relative_to=REPOSITORY_ROOT
            ),
            "source_pool_truth_tensor_received": source_manifest["truth_tensor_received"],
        },
    )
    publish_staging_directory(staging, final)


def main() -> None:
    config = load_config()
    verify_stage12a_files(config)
    masks = {case_id: _load_truth_mask(case_id) for case_id in CASE_IDS}
    _build_geometry_audit(masks)
    _freeze_prior_bank(config)
    write_json_x(
        EXPERIMENT_DIR / "audit/pre_geophysics_freeze.json",
        {
            "schema": "stage12b_pre_geophysics_freeze_audit_v1",
            "status": "complete",
            "created_at_utc": utc_now(),
            "fixed_case_ids": list(CASE_IDS),
            "case_replacement": False,
            "geometry_used_for_selection": False,
            "prior_bank_frozen_before_stage12b_seismic": True,
            "prior_bank_received_truth": False,
            "prior_bank_received_seismic": False,
        },
    )
    print("Stage12B pre-geophysics geometry audit and common prior freeze complete")


if __name__ == "__main__":
    main()
