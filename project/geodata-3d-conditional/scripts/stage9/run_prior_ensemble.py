#!/usr/bin/env python3
"""Generate one truth-inaccessible frozen Stage9A Flow candidate pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
from typing import Mapping

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import inference_runtime as runtime
from guidance import generator_posterior as generator_module
from guidance import prior_ensemble as ensemble
from guidance import seismic as seismic_module
from guidance.seismic import seismic_operator_from_config, tensor_sha256
from scripts.stage9.common import (
    create_staging_directory,
    file_record,
    load_tensor_record,
    publish_staging_directory,
    read_json,
    utc_now,
    write_csv_x,
    write_json_x,
)


def parse_args() -> argparse.Namespace:
    experiment = PROJECT_DIR / "experiments/stage9_flow_prior_posterior"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=experiment / "configs/stage9a_prior_support_v1.json",
    )
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _resolve(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPOSITORY_ROOT, text=True
    ).strip()


def _ordered_digest(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _case_config(
    config: Mapping[str, object], case_id: str
) -> dict[str, object]:
    matches = [
        dict(value)
        for value in config["primary_cases"]
        if str(value["case_id"]) == str(case_id)
    ]
    if len(matches) != 1:
        raise ValueError(f"case ID is not uniquely frozen in config: {case_id}")
    return matches[0]


def load_inference_case(case_dir: Path, case_id: str) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    """Load only inference-visible assets; this API cannot receive truth."""
    manifest = read_json(Path(case_dir) / "manifest.json")
    expected = {
        "schema": "stage9a_inference_case_assets_v1",
        "status": "complete",
        "case_id": str(case_id),
        "geological_labels_available_to_inference": False,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"inference case {field} must be {value!r}")
    records = manifest.get("tensors")
    if not isinstance(records, Mapping):
        raise ValueError("inference case lacks tensor records")
    required = {
        "condition_values",
        "condition_mask",
        "subsurface_mask",
        "acoustic_property_table",
        "observation_correct",
        "observation_zero",
        "observation_shuffled_xy",
        "observation_wrong_case",
    }
    if set(records) != required:
        raise ValueError("inference tensor set drifted")
    tensors = {
        name: load_tensor_record(case_dir, record)
        for name, record in records.items()
    }
    return manifest, tensors


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    resolved = ensemble.validate_protocol_config(config)
    case_config = _case_config(config, args.case_id)
    inference_manifest, tensors = load_inference_case(args.case_dir, args.case_id)
    if int(inference_manifest["case_index"]) != int(case_config["case_index"]):
        raise ValueError("case index differs from frozen config")
    checkpoint = args.checkpoint or _resolve(config["checkpoint"])
    if runtime.file_sha256(checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("checkpoint hash differs from frozen Stage9A config")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Stage9A candidate generation requires SSH-visible CUDA")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse immutable pool: {args.output_dir}")

    from model_train_sh_inference_cond import Geo3DStochInterp

    model, load_report = runtime.load_model_with_weight_policy(
        Geo3DStochInterp, checkpoint, device, "ema"
    )
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if (
        load_report["ema_applied"] is not True
        or load_report["ema_missing_trainable"]
        or load_report["ema_shape_mismatches"]
        or load_report["ema_excluded_frozen_parameters"] != ["embedding.weight"]
    ):
        raise RuntimeError("checkpoint/EMA policy mismatch")

    condition_values = tensors["condition_values"].long()
    condition_mask = tensors["condition_mask"].bool()
    subsurface_mask = tensors["subsurface_mask"].bool().to(device)
    property_table = tensors["acoustic_property_table"].float().to(device)
    if condition_values.shape != (1, 1, 64, 64, 64):
        raise ValueError("Stage9A condition shape must be [1,1,64,64,64]")
    embedded_conditions = model.embed(condition_values.to(device))
    conditioning = torch.where(
        condition_mask.to(device).expand_as(embedded_conditions),
        embedded_conditions,
        torch.zeros_like(embedded_conditions),
    )
    seismic_config = read_json(_resolve(config["seismic_config"]))
    forward_operator, seismic_metadata = seismic_operator_from_config(
        seismic_config, grid_shape=(64, 64, 64)
    )
    count = (
        int(resolved["smoke_count"])
        if args.mode == "smoke"
        else int(resolved["formal_count"])
    )
    batch_size = int(resolved["batch_size"])
    staging = create_staging_directory(args.output_dir)
    cache_dir = staging / "cache"
    cache_dir.mkdir()
    candidate_rows: list[dict[str, object]] = []
    model_chunks: list[dict[str, object]] = []
    prediction_chunks: list[dict[str, object]] = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    hard_forward_count = 0
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        sources = []
        source_hashes = []
        source_seeds = []
        for index in range(start, stop):
            seed = ensemble.source_seed(
                config,
                case_index=int(case_config["case_index"]),
                candidate_index=index,
                mode=args.mode,
            )
            source = ensemble.gaussian_source(
                (1, model.embedding_dim, *model.data_shape), seed=seed
            )
            sources.append(source)
            source_seeds.append(seed)
            source_hashes.append(tensor_sha256(source))
        initial_state = torch.cat(sources, dim=0).to(device)
        batch_started = time.perf_counter()
        labels = ensemble.generate_prior_batch(
            model,
            initial_state,
            conditioning,
            embedded_conditions,
            condition_mask,
            condition_values,
            n_steps=int(resolved["n_steps"]),
        )
        violations = (
            (labels != condition_values.to(device))
            & condition_mask.to(device)
        ).flatten(1).sum(dim=1)
        if bool((violations != 0).any()):
            raise RuntimeError("hard condition violation in candidate batch")
        predicted = ensemble.hard_seismic_response(
            labels,
            property_table=property_table,
            subsurface_mask=subsurface_mask,
            forward_operator=forward_operator,
        )
        hard_forward_count += stop - start
        torch.cuda.synchronize(device)
        batch_seconds = time.perf_counter() - batch_started
        labels_cpu = labels.detach().cpu().to(torch.int8).contiguous()
        predicted_cpu = predicted.detach().cpu().float().contiguous()
        model_name = f"models_{start:06d}_{stop - 1:06d}.pt.gz"
        prediction_name = f"predictions_{start:06d}_{stop - 1:06d}.pt.gz"
        model_record = ensemble.save_tensor_gzip(cache_dir / model_name, labels_cpu)
        prediction_record = ensemble.save_tensor_gzip(
            cache_dir / prediction_name, predicted_cpu
        )
        for record, kind, name in (
            (model_record, "hard_models", model_name),
            (prediction_record, "predicted_observations", prediction_name),
        ):
            record.update(
                {
                    "path": f"cache/{name}",
                    "kind": kind,
                    "candidate_start": start,
                    "candidate_stop_exclusive": stop,
                }
            )
        model_chunks.append(model_record)
        prediction_chunks.append(prediction_record)
        for offset, index in enumerate(range(start, stop)):
            candidate_rows.append(
                {
                    "candidate_id": ensemble.candidate_id(index),
                    "candidate_index": index,
                    "source_seed": source_seeds[offset],
                    "source_noise_sha256": source_hashes[offset],
                    "hard_model_sha256": tensor_sha256(labels_cpu[offset : offset + 1]),
                    "predicted_observation_sha256": tensor_sha256(
                        predicted_cpu[offset : offset + 1]
                    ),
                    "condition_violations": int(violations[offset].item()),
                    "model_chunk": f"cache/{model_name}",
                    "prediction_chunk": f"cache/{prediction_name}",
                    "chunk_offset": offset,
                    "batch_seconds": batch_seconds,
                }
            )
        del initial_state, labels, predicted, labels_cpu, predicted_cpu
        if stop == count or stop % 64 == 0:
            print(
                json.dumps(
                    {
                        "status": "running",
                        "mode": args.mode,
                        "case_id": args.case_id,
                        "candidates_completed": stop,
                        "candidate_count": count,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )

    if len(candidate_rows) != count or hard_forward_count != count:
        raise RuntimeError("candidate pool is incomplete")
    write_csv_x(staging / "candidate_pool.csv", candidate_rows)
    elapsed = time.perf_counter() - started
    manifest = {
        "schema": ensemble.STAGE9A_POOL_SCHEMA,
        "status": "complete",
        "scientific_evidence": args.mode == "formal",
        "mode": args.mode,
        "case_id": args.case_id,
        "case_index": int(case_config["case_index"]),
        "candidate_count": count,
        "candidate_ids": [row["candidate_id"] for row in candidate_rows],
        "candidate_batch_size": batch_size,
        "cache_policy": ensemble.PREDICTION_CACHE_POLICY,
        "cache_is_lossless": True,
        "prediction_dtype": "torch.float32",
        "n_euler_steps": int(resolved["n_steps"]),
        "integrator": runtime.PAIRED_INTEGRATOR,
        "condition_projection": generator_module.CONDITION_PROJECTION_POLICY,
        "hard_decode": config["hard_decode"],
        "source_policy": "independent_per_candidate_cpu_standard_gaussian_v1",
        "source_seed_first": int(candidate_rows[0]["source_seed"]),
        "source_seed_last": int(candidate_rows[-1]["source_seed"]),
        "source_hash_sequence_sha256": _ordered_digest(
            [str(row["source_noise_sha256"]) for row in candidate_rows]
        ),
        "hard_model_hash_sequence_sha256": _ordered_digest(
            [str(row["hard_model_sha256"]) for row in candidate_rows]
        ),
        "predicted_observation_hash_sequence_sha256": _ordered_digest(
            [str(row["predicted_observation_sha256"]) for row in candidate_rows]
        ),
        "hard_condition_violation_maximum": max(
            int(row["condition_violations"]) for row in candidate_rows
        ),
        "hard_seismic_forward_count": hard_forward_count,
        "flow_velocity_forward_count": count * int(resolved["n_steps"]),
        "runtime_seconds": elapsed,
        "seconds_per_candidate": elapsed / count,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "completed_at_utc": utc_now(),
        "exact_command": shlex.join([sys.executable, *sys.argv]),
        "git_branch": _git("branch", "--show-current"),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_status": _git("status", "--short"),
        "runtime": {
            "hostname": socket.gethostname(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
        },
        "checkpoint": load_report["checkpoint"],
        "checkpoint_sha256": runtime.file_sha256(checkpoint),
        "model_weight_source": "ema",
        "model_load_report": load_report,
        "config": file_record(args.config),
        "spec": file_record(PROJECT_DIR / "docs/STAGE9A_FLOW_PRIOR_SUPPORT_SPEC.md"),
        "inference_case_manifest": file_record(args.case_dir / "manifest.json"),
        "candidate_pool_csv": file_record(staging / "candidate_pool.csv", relative_to=staging),
        "source_files": {
            "runner": file_record(Path(__file__)),
            "prior_ensemble": file_record(Path(ensemble.__file__)),
            "projected_sampler": file_record(Path(generator_module.__file__)),
            "seismic": file_record(Path(seismic_module.__file__)),
            "runtime": file_record(Path(runtime.__file__)),
        },
        "condition_tensor_hashes": {
            name: inference_manifest["tensors"][name]["tensor_sha256"]
            for name in ("condition_values", "condition_mask", "subsurface_mask")
        },
        "seismic_metadata": seismic_metadata,
        "model_chunks": model_chunks,
        "prediction_chunks": prediction_chunks,
        "truth_tensor_received": False,
        "truth_metrics_computed": False,
        "ranking_computed": False,
    }
    write_json_x(staging / "manifest.json", manifest)
    publish_staging_directory(staging, args.output_dir)
    print(
        json.dumps(
            {
                "status": "complete",
                "mode": args.mode,
                "case_id": args.case_id,
                "candidate_count": count,
                "runtime_seconds": elapsed,
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
