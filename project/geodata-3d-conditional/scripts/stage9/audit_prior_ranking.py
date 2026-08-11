#!/usr/bin/env python3
"""Freeze Stage9A rankings from cached predictions without geological truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Mapping

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from guidance import prior_ensemble as ensemble
from guidance.seismic import tensor_sha256
from scripts.stage9.common import (
    create_staging_directory,
    file_record,
    publish_staging_directory,
    read_csv,
    read_json,
    utc_now,
    write_csv_x,
    write_json_x,
)
from scripts.stage9.run_prior_ensemble import load_inference_case


RANKING_FILENAMES = {
    "correct": "ranking_correct.csv",
    "zero": "ranking_zero.csv",
    "shuffled_xy": "ranking_shuffled.csv",
    "wrong_case": "ranking_wrong_case.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPOSITORY_ROOT, text=True
    ).strip()


def validate_completed_pool(
    pool_dir: Path,
) -> tuple[dict[str, object], list[dict[str, str]]]:
    manifest = read_json(Path(pool_dir) / "manifest.json")
    if manifest.get("schema") != ensemble.STAGE9A_POOL_SCHEMA:
        raise ValueError("invalid Stage9A candidate-pool schema")
    if manifest.get("status") != "complete":
        raise RuntimeError("candidate pool is incomplete")
    if manifest.get("truth_tensor_received") is not False:
        raise RuntimeError("candidate pool truth firewall failed")
    if manifest.get("cache_policy") != ensemble.PREDICTION_CACHE_POLICY:
        raise ValueError("candidate prediction cache policy drifted")
    csv_record = manifest.get("candidate_pool_csv")
    if not isinstance(csv_record, Mapping):
        raise ValueError("candidate pool lacks CSV record")
    csv_path = Path(pool_dir) / str(csv_record["path"])
    if ensemble.file_sha256(csv_path) != csv_record.get("sha256"):
        raise ValueError("candidate-pool CSV hash mismatch")
    rows = read_csv(csv_path)
    if len(rows) != int(manifest["candidate_count"]):
        raise RuntimeError("candidate-pool row count is incomplete")
    expected_ids = [ensemble.candidate_id(index) for index in range(len(rows))]
    if [row["candidate_id"] for row in rows] != expected_ids:
        raise ValueError("candidate IDs or order drifted")
    if any(int(row["condition_violations"]) != 0 for row in rows):
        raise RuntimeError("candidate pool contains hard-condition violation")
    return manifest, rows


def compute_rankings(
    pool_dir: Path, inference_case_dir: Path
) -> tuple[dict[str, object], dict[str, list[dict[str, object]]]]:
    """Compute all rankings; the signature intentionally cannot receive truth."""
    pool_manifest, candidate_rows = validate_completed_pool(pool_dir)
    case_manifest, tensors = load_inference_case(
        inference_case_dir, str(pool_manifest["case_id"])
    )
    observations = {
        name: tensors[f"observation_{name}"].float()
        for name in ensemble.OBSERVATION_NAMES
    }
    scores: dict[str, dict[str, float]] = {
        row["candidate_id"]: {} for row in candidate_rows
    }
    row_index = {row["candidate_id"]: row for row in candidate_rows}
    seen: set[str] = set()
    chunk_records = pool_manifest.get("prediction_chunks")
    if not isinstance(chunk_records, list) or not chunk_records:
        raise ValueError("candidate pool lacks prediction chunks")
    for record in chunk_records:
        predicted = ensemble.load_tensor_gzip(
            Path(pool_dir) / str(record["path"]), expected=record
        )
        if predicted.dtype != torch.float32:
            raise ValueError("Stage9A ranking requires lossless float32 predictions")
        start = int(record["candidate_start"])
        stop = int(record["candidate_stop_exclusive"])
        if predicted.shape[0] != stop - start:
            raise ValueError("prediction chunk candidate interval mismatch")
        batch_rmse = {
            name: ensemble.plain_rmse(predicted, observation)
            for name, observation in observations.items()
        }
        for offset, index in enumerate(range(start, stop)):
            identifier = ensemble.candidate_id(index)
            if identifier in seen:
                raise ValueError(f"candidate prediction repeated: {identifier}")
            expected_hash = row_index[identifier]["predicted_observation_sha256"]
            if tensor_sha256(predicted[offset : offset + 1]) != expected_hash:
                raise ValueError(f"candidate prediction hash mismatch: {identifier}")
            scores[identifier] = {
                name: float(values[offset].item())
                for name, values in batch_rmse.items()
            }
            seen.add(identifier)
    if seen != set(scores):
        raise RuntimeError("candidate prediction cache is incomplete")
    rankings = ensemble.rank_scores(scores)
    observation_hashes = {
        name: tensor_sha256(value) for name, value in observations.items()
    }
    for name, rows in rankings.items():
        for row in rows:
            candidate = row_index[str(row["candidate_id"])]
            row.update(
                {
                    "observation": name,
                    "observation_sha256": observation_hashes[name],
                    "predicted_observation_sha256": candidate[
                        "predicted_observation_sha256"
                    ],
                    "hard_model_sha256": candidate["hard_model_sha256"],
                    "condition_violations": int(candidate["condition_violations"]),
                }
            )
    metadata = {
        "pool_manifest": pool_manifest,
        "case_manifest": case_manifest,
        "observation_hashes": observation_hashes,
    }
    return metadata, rankings


def main() -> None:
    args = parse_args()
    metadata, rankings = compute_rankings(args.pool_dir, args.case_dir)
    staging = create_staging_directory(args.output_dir)
    ranking_records = {}
    for name, rows in rankings.items():
        filename = RANKING_FILENAMES[name]
        write_csv_x(staging / filename, rows)
        ranking_records[name] = file_record(staging / filename, relative_to=staging)
    pool_manifest = metadata["pool_manifest"]
    manifest = {
        "schema": ensemble.STAGE9A_RANKING_SCHEMA,
        "status": "complete",
        "scientific_evidence": bool(pool_manifest["scientific_evidence"]),
        "mode": pool_manifest["mode"],
        "case_id": pool_manifest["case_id"],
        "candidate_count": pool_manifest["candidate_count"],
        "ranking_policy": ensemble.RANKING_POLICY,
        "tie_break": "candidate_id_ascending",
        "observations": list(ensemble.OBSERVATION_NAMES),
        "observation_hashes": metadata["observation_hashes"],
        "pool_manifest": file_record(Path(args.pool_dir) / "manifest.json"),
        "pool_candidate_csv": file_record(Path(args.pool_dir) / "candidate_pool.csv"),
        "inference_case_manifest": file_record(Path(args.case_dir) / "manifest.json"),
        "ranking_files": ranking_records,
        "completed_at_utc": utc_now(),
        "exact_command": shlex.join([sys.executable, *sys.argv]),
        "git_branch": _git("branch", "--show-current"),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_status": _git("status", "--short"),
        "runner": file_record(Path(__file__)),
        "prior_ensemble": file_record(Path(ensemble.__file__)),
        "truth_tensor_received": False,
        "truth_metrics_computed": False,
    }
    write_json_x(staging / "ranking_manifest.json", manifest)
    publish_staging_directory(staging, args.output_dir)
    print(
        json.dumps(
            {
                "status": "complete",
                "case_id": pool_manifest["case_id"],
                "candidate_count": pool_manifest["candidate_count"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
