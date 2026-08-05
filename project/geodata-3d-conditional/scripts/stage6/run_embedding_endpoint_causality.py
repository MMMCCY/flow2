#!/usr/bin/env python3
"""Run Phase-6Q Q3 with the real frozen checkpoint embedding and no flow U-Net."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import socket
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from guidance.gravity import density_table_from_config, gravity_operator_from_config
from guidance.seismic import acoustic_tables_from_config, seismic_operator_from_config
from guidance.simple_causality import (
    EMBEDDING_METHODS,
    OBSERVATION_CONTROLS,
    OBSERVATION_MODES,
    AnalyticObservationSuite,
    build_simple_causal_case,
    build_voxel_search_mask,
    optimize_embedding_endpoint,
    validate_embedding_endpoint_config,
    validate_simple_causality_config,
    validate_voxel_reconstruction_config,
)
from scripts.stage6.run_simple_causality import (
    _file_sha256,
    _parse_csv_choices,
    _read_json,
    _resolve_repo_path,
    _tensor_sha256,
    _write_json,
    _write_rows,
)
from scripts.stage6.run_voxel_causality import _assert_finite_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase-6Q frozen-checkpoint embedding endpoint diagnostics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--modes", default=None)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--seismic-controls", default=None)
    return parser.parse_args()


def _checkpoint_embedding(checkpoint_path: Path, key: str) -> torch.Tensor:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict) or key not in state:
        raise KeyError(f"checkpoint lacks frozen embedding key: {key}")
    value = state[key]
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError("checkpoint embedding must be a rank-2 tensor")
    if not torch.isfinite(value).all():
        raise ValueError("checkpoint embedding contains non-finite values")
    return value.detach().float().contiguous()


def _report(summary: dict[str, object]) -> str:
    lines = [
        "# Phase 6Q Q3 frozen embedding endpoint report",
        "",
        f"Completed: {summary['completed_at_utc']}",
        "",
        "Only the checkpoint embedding matrix was loaded. The flow U-Net was not instantiated, and no training was performed.",
        "",
        "| Mode | Control | Method | Hard attainment | Hidden IoU/P/R | Body recalls | Hard voxels | Air voxels | Best step |",
        "|---|---|---|---:|---|---|---:|---:|---:|",
    ]
    for result in summary["optimization"]:
        best = result["best_metrics"]
        lines.append(
            "| {mode} | {control} | {method} | {attain:.6f} | {iou:.4f}/{precision:.4f}/{recall:.4f} | {b0:.4f}/{b1:.4f} | {voxels} | {air} | {step} |".format(
                mode=result["mode"],
                control=result["control"],
                method=result["method"],
                attain=float(best["hard_attainment"]),
                iou=float(best["hidden_iou"]),
                precision=float(best["hidden_precision"]),
                recall=float(best["hidden_recall"]),
                b0=float(best["hidden_body_0_recall"]),
                b1=float(best["hidden_body_1_recall"]),
                voxels=int(best["predicted_hidden_voxels"]),
                air=int(best["hard_air_voxels_in_search"]),
                step=int(result["best_step"]),
            )
        )
    lines.extend(
        [
            "",
            "Best-state selection used hard physics only. All geology metrics are post-run audits.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = _read_json(config_path)
    voxel_path = _resolve_repo_path(config.get("voxel_config"), "voxel_config")
    voxel_hash = _file_sha256(voxel_path)
    if voxel_hash != str(config.get("voxel_config_sha256", "")):
        raise ValueError("voxel config hash mismatch")
    voxel_config = _read_json(voxel_path)
    base_path = _resolve_repo_path(config.get("base_config"), "base_config")
    base_hash = _file_sha256(base_path)
    if base_hash != str(config.get("base_config_sha256", "")):
        raise ValueError("base config hash mismatch")
    if base_path != _resolve_repo_path(voxel_config.get("base_config"), "voxel base_config"):
        raise ValueError("embedding and voxel configs must reference the same base case")
    base_config = _read_json(base_path)
    base_resolved = validate_simple_causality_config(base_config)
    voxel_resolved = validate_voxel_reconstruction_config(
        voxel_config, grid_shape=base_resolved["grid_shape"]
    )
    resolved = validate_embedding_endpoint_config(
        config, grid_shape=base_resolved["grid_shape"]
    )
    if resolved["search_region"] != voxel_resolved["search_region"]:
        raise ValueError("Q3 must reuse the exact frozen Q2 search region")
    modes = _parse_csv_choices(
        args.modes,
        default=resolved["observation_modes"],
        allowed=OBSERVATION_MODES,
        name="modes",
    )
    methods = _parse_csv_choices(
        args.methods,
        default=resolved["methods"],
        allowed=EMBEDDING_METHODS,
        name="methods",
    )
    controls = _parse_csv_choices(
        args.seismic_controls,
        default=resolved["seismic_controls"],
        allowed=OBSERVATION_CONTROLS,
        name="seismic-controls",
    )
    checkpoint_path = _resolve_repo_path(config.get("checkpoint"), "checkpoint")
    checkpoint_hash = _file_sha256(checkpoint_path)
    if checkpoint_hash != str(config.get("checkpoint_sha256", "")):
        raise ValueError("checkpoint hash mismatch")
    embedding_key = str(config.get("checkpoint_embedding_key", ""))
    embedding_weight = _checkpoint_embedding(checkpoint_path, embedding_key)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Phase 6Q refuses to reuse any output path: {output_dir}")
    output_dir.mkdir(parents=True)
    started_at = datetime.now(timezone.utc).isoformat()

    case = build_simple_causal_case(base_config)
    search_mask, search_report = build_voxel_search_mask(case, voxel_config)
    acoustic_path = _resolve_repo_path(base_config.get("acoustic_config"), "acoustic_config")
    seismic_path = _resolve_repo_path(base_config.get("seismic_config"), "seismic_config")
    density_path = _resolve_repo_path(base_config.get("density_config"), "density_config")
    gravity_path = _resolve_repo_path(base_config.get("gravity_config"), "gravity_config")
    acoustic_tables, acoustic_metadata = acoustic_tables_from_config(_read_json(acoustic_path), 15)
    density_table, density_metadata = density_table_from_config(_read_json(density_path), 15)
    seismic_operator, seismic_metadata = seismic_operator_from_config(
        _read_json(seismic_path), grid_shape=base_resolved["grid_shape"]
    )
    gravity_operator, gravity_metadata = gravity_operator_from_config(
        _read_json(gravity_path), grid_shape=base_resolved["grid_shape"]
    )
    suite = AnalyticObservationSuite(
        case,
        acoustic_property_table=acoustic_tables.property_table.to(device),
        density_table=density_table.to(device),
        seismic_operator=seismic_operator,
        gravity_operator=gravity_operator,
        blur_sigma_voxels=float(base_resolved["blur_sigma_voxels"]),
    )
    public = {key: value for key, value in resolved.items() if key != "temperatures"}
    public.update(
        {
            "selected_modes": modes,
            "selected_methods": methods,
            "selected_seismic_controls": controls,
            "device": str(device),
            "config_sha256": _file_sha256(config_path),
            "voxel_config_sha256": voxel_hash,
            "base_config_sha256": base_hash,
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_embedding_key": embedding_key,
            "checkpoint_embedding_shape": list(embedding_weight.shape),
            "checkpoint_embedding_tensor_sha256": _tensor_sha256(embedding_weight),
            "causality_source_sha256": _file_sha256(PROJECT_DIR / "guidance/simple_causality.py"),
            "runner_source_sha256": _file_sha256(Path(__file__).resolve()),
            "started_at_utc": started_at,
        }
    )
    _write_json(output_dir / "config_input.json", config)
    _write_json(output_dir / "config_resolved.json", public)
    search_path = output_dir / "tensors/search_mask.pt"
    search_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(search_mask, search_path)
    torch.save(embedding_weight, output_dir / "tensors/checkpoint_embedding.pt")
    _write_json(
        output_dir / "input_validation.json",
        {
            "case": case.validation,
            "search": search_report,
            "search_mask_file_sha256": _file_sha256(search_path),
            "search_mask_tensor_sha256": _tensor_sha256(search_mask),
            "checkpoint_embedding_tensor_sha256": _tensor_sha256(embedding_weight),
            "checkpoint_embedding_row_norm_min": float(torch.linalg.vector_norm(embedding_weight, dim=1).min()),
            "checkpoint_embedding_row_norm_max": float(torch.linalg.vector_norm(embedding_weight, dim=1).max()),
        },
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    results: list[dict[str, object]] = []
    for mode in modes:
        mode_controls = controls if mode == "seismic" else ["correct"]
        for control in mode_controls:
            for method in methods:
                result = optimize_embedding_endpoint(
                    suite,
                    mode,
                    search_mask=search_mask,
                    embedding_weight=embedding_weight,
                    control=control,
                    method=method,
                    temperatures=resolved["temperatures"],
                    learning_rate=float(resolved["learning_rate"]),
                    weight_decay=float(resolved["weight_decay"]),
                    gradient_clip_norm=float(resolved["gradient_clip_norm"]),
                    hard_check_interval=int(resolved["hard_check_interval"]),
                    max_state_norm_to_embedding_norm=float(resolved["max_state_norm_to_embedding_norm"]),
                    shuffle_seed=int(resolved["shuffle_seed"]),
                )
                _assert_finite_trace(result["trace"])
                base = output_dir / "optimization" / mode / control / method
                _write_rows(base / "trace.csv", result["trace"])
                torch.save(result["best_vectors"], base / "best_vectors.pt")
                torch.save(result["best_hard_occupancy"], base / "best_hard_occupancy.pt")
                torch.save(result["best_labels"].to(torch.int8), base / "best_labels.pt")
                results.append(
                    {
                        "mode": mode,
                        "control": control,
                        "method": method,
                        "baseline_rmse": result["baseline_rmse"],
                        "best_step": result["best_step"],
                        "best_metrics": result["best_metrics"],
                        "final_metrics": result["final_metrics"],
                        "trace_path": str((base / "trace.csv").relative_to(output_dir)),
                        "best_occupancy_path": str((base / "best_hard_occupancy.pt").relative_to(output_dir)),
                        "best_labels_path": str((base / "best_labels.pt").relative_to(output_dir)),
                    }
                )

    completed_at = datetime.now(timezone.utc).isoformat()
    summary = {
        "stage": "phase6q_q3_checkpoint_embedding_endpoint",
        "status": "completed",
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "search_report": search_report,
        "optimization": results,
        "acoustic_metadata": acoustic_metadata,
        "density_metadata": density_metadata,
        "seismic_metadata": seismic_metadata,
        "gravity_metadata": gravity_metadata,
        "runtime": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        },
        "formal_training_performed": False,
        "flow_checkpoint_loaded": True,
        "flow_unet_loaded": False,
        "checkpoint_tensors_used": [embedding_key],
    }
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": str(output_dir),
                "optimization_runs": len(results),
                "search_voxels": search_report["search_voxels"],
                "gpu_peak_allocated_bytes": summary["runtime"]["gpu_peak_allocated_bytes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
