#!/usr/bin/env python3
"""Run Phase-6Q Q2b monotone hard-coordinate physics proposals."""

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
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from guidance.gravity import density_table_from_config, gravity_operator_from_config
from guidance.seismic import acoustic_tables_from_config, seismic_operator_from_config
from guidance.simple_causality import (
    OBSERVATION_CONTROLS,
    OBSERVATION_MODES,
    AnalyticObservationSuite,
    build_simple_causal_case,
    build_voxel_search_mask,
    optimize_hard_coordinates,
    validate_hard_coordinate_config,
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
        description="Run Phase-6Q monotone hard-coordinate diagnostics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--modes", default=None)
    parser.add_argument("--seismic-controls", default=None)
    return parser.parse_args()


def _report(summary: dict[str, object]) -> str:
    lines = [
        "# Phase 6Q Q2b monotone hard-coordinate report",
        "",
        f"Completed: {summary['completed_at_utc']}",
        "",
        "Every accepted update was selected by hard physics RMSE; no soft endpoint, shape/count prior, regularizer, flow checkpoint or training was used.",
        "",
        "| Mode | Control | Hard attainment | Hidden IoU/P/R | Body recalls | Hard voxels | Accepted/total |",
        "|---|---|---:|---|---|---:|---:|",
    ]
    for result in summary["optimization"]:
        final = result["final_metrics"]
        lines.append(
            "| {mode} | {control} | {attain:.6f} | {iou:.4f}/{precision:.4f}/{recall:.4f} | {b0:.4f}/{b1:.4f} | {voxels} | {accepted}/{total} |".format(
                mode=result["mode"],
                control=result["control"],
                attain=float(final["hard_attainment"]),
                iou=float(final["hidden_iou"]),
                precision=float(final["hidden_precision"]),
                recall=float(final["hidden_recall"]),
                b0=float(final["hidden_body_0_recall"]),
                b1=float(final["hidden_body_1_recall"]),
                voxels=int(final["predicted_hidden_voxels"]),
                accepted=int(result["accepted_iterations"]),
                total=int(result["iterations_completed"]),
            )
        )
    lines.extend(["", "Geometry metrics are post-run audits and were not used for proposal selection.", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = _read_json(config_path)
    resolved = validate_hard_coordinate_config(config)
    voxel_path = _resolve_repo_path(config.get("voxel_config"), "voxel_config")
    actual_voxel_hash = _file_sha256(voxel_path)
    if actual_voxel_hash != str(config.get("voxel_config_sha256", "")):
        raise ValueError("voxel config hash mismatch")
    voxel_config = _read_json(voxel_path)
    base_path = _resolve_repo_path(voxel_config.get("base_config"), "base_config")
    if _file_sha256(base_path) != str(voxel_config.get("base_config_sha256", "")):
        raise ValueError("base config hash mismatch")
    base_config = _read_json(base_path)
    base_resolved = validate_simple_causality_config(base_config)
    validate_voxel_reconstruction_config(
        voxel_config, grid_shape=base_resolved["grid_shape"]
    )
    modes = _parse_csv_choices(
        args.modes,
        default=resolved["observation_modes"],
        allowed=OBSERVATION_MODES,
        name="modes",
    )
    controls = _parse_csv_choices(
        args.seismic_controls,
        default=resolved["seismic_controls"],
        allowed=OBSERVATION_CONTROLS,
        name="seismic-controls",
    )
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
    public = dict(resolved)
    public.update(
        {
            "selected_modes": modes,
            "selected_seismic_controls": controls,
            "device": str(device),
            "config_sha256": _file_sha256(config_path),
            "voxel_config_sha256": actual_voxel_hash,
            "base_config_sha256": _file_sha256(base_path),
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
    _write_json(
        output_dir / "input_validation.json",
        {
            "case": case.validation,
            "search": search_report,
            "search_mask_file_sha256": _file_sha256(search_path),
            "search_mask_tensor_sha256": _tensor_sha256(search_mask),
        },
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    results: list[dict[str, object]] = []
    for mode in modes:
        mode_controls = controls if mode == "seismic" else ["correct"]
        for control in mode_controls:
            result = optimize_hard_coordinates(
                suite,
                mode,
                search_mask=search_mask,
                control=control,
                max_iterations=int(resolved["max_iterations"]),
                proposal_flip_counts=resolved["proposal_flip_counts"],
                improvement_tolerance=float(resolved["improvement_tolerance"]),
                shuffle_seed=int(voxel_config["shuffle_seed"]),
            )
            _assert_finite_trace(result["trace"])
            base = output_dir / "optimization" / mode / control / "hard_coordinate"
            _write_rows(base / "trace.csv", result["trace"])
            torch.save(result["final_hard_occupancy"], base / "final_hard_occupancy.pt")
            results.append(
                {
                    "mode": mode,
                    "control": control,
                    "baseline_rmse": result["baseline_rmse"],
                    "iterations_completed": result["iterations_completed"],
                    "accepted_iterations": result["accepted_iterations"],
                    "final_metrics": result["final_metrics"],
                    "trace_path": str((base / "trace.csv").relative_to(output_dir)),
                    "final_occupancy_path": str((base / "final_hard_occupancy.pt").relative_to(output_dir)),
                }
            )

    completed_at = datetime.now(timezone.utc).isoformat()
    summary = {
        "stage": "phase6q_q2b_hard_coordinate",
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
        "flow_checkpoint_loaded": False,
    }
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": str(output_dir),
                "runs": len(results),
                "gpu_peak_allocated_bytes": summary["runtime"]["gpu_peak_allocated_bytes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
