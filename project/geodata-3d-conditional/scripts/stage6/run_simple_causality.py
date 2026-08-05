#!/usr/bin/env python3
"""Run Phase-6Q Q0/Q1 analytic five-body causal diagnostics."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
from typing import Mapping, Sequence

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
    OPTIMIZATION_METHODS,
    PHASE6Q_IMPLEMENTATION_VERSION,
    AnalyticObservationSuite,
    build_simple_causal_case,
    enumerate_hard_pairs,
    optimize_candidate_logits,
    validate_simple_causality_config,
)


def _read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _resolve_repo_path(value: object, name: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name}: {path}")
    return path


def _parse_csv_choices(
    value: str | None,
    *,
    default: Sequence[str],
    allowed: Sequence[str],
    name: str,
) -> list[str]:
    parsed = list(default) if value is None else [item.strip() for item in value.split(",") if item.strip()]
    if not parsed or len(parsed) != len(set(parsed)) or any(item not in allowed for item in parsed):
        raise ValueError(f"{name} must be unique comma-separated members of {tuple(allowed)}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase-6Q hard enumeration and candidate-relaxation diagnostics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--modes", default=None, help="Comma-separated observation subset")
    parser.add_argument("--methods", default=None, help="Comma-separated optimizer subset")
    parser.add_argument(
        "--seismic-controls",
        default=None,
        help="Comma-separated seismic controls; non-seismic modes always use correct",
    )
    parser.add_argument("--enumeration-only", action="store_true")
    return parser.parse_args()


def _public_config(config: Mapping[str, object], resolved: Mapping[str, object]) -> dict[str, object]:
    optimization = dict(resolved["optimization"])
    optimization.pop("temperatures", None)
    return {
        "schema": resolved["schema"],
        "id": resolved["id"],
        "description": resolved["description"],
        "grid_shape": resolved["grid_shape"],
        "air_start_z": resolved["air_start_z"],
        "air_label": resolved["air_label"],
        "background_label": resolved["background_label"],
        "target_label": resolved["target_label"],
        "fixed_bodies": list(config["fixed_bodies"]),
        "candidate_bodies": list(config["candidate_bodies"]),
        "truth_candidate_indices": resolved["truth_candidate_indices"],
        "observation_modes": resolved["observation_modes"],
        "blur_sigma_voxels": resolved["blur_sigma_voxels"],
        "optimization": optimization,
        "seismic_controls": resolved["seismic_controls"],
        "shuffle_seed": resolved["shuffle_seed"],
        "enumeration_batch_size": resolved["enumeration_batch_size"],
        "inverse_crime": resolved["inverse_crime"],
        "measured_geophysics": resolved["measured_geophysics"],
        "formal_training_authorized": False,
    }


def _save_case(output_dir: Path, case) -> dict[str, object]:
    tensors = {
        "truth_labels.pt": case.truth_labels,
        "baseline_labels.pt": case.baseline_labels,
        "condition_mask.pt": case.condition_mask,
        "subsurface_mask.pt": case.subsurface_mask,
        "fixed_target_mask.pt": case.fixed_target_mask,
        "candidate_masks.pt": case.candidate_masks,
    }
    records: dict[str, object] = {}
    for name, tensor in tensors.items():
        path = output_dir / "tensors" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor.detach().cpu(), path)
        records[name] = {
            "path": str(path.relative_to(output_dir)),
            "file_sha256": _file_sha256(path),
            "tensor_sha256": _tensor_sha256(tensor),
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
    return records


def _report_markdown(summary: Mapping[str, object]) -> str:
    lines = [
        "# Phase 6Q Q0/Q1 machine report",
        "",
        f"Completed: {summary['completed_at_utc']}",
        "",
        "This is a generator-free analytic inverse-crime diagnostic. No flow checkpoint was loaded and no training was performed.",
        "",
        "## Q0 hard enumeration",
        "",
        "| Mode | Truth rank | Truth RMSE | Second nontruth RMSE | Zero-count | Baseline RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, result in summary["enumeration"].items():
        lines.append(
            "| {mode} | {rank} | {truth:.8g} | {second:.8g} | {zero} | {baseline:.8g} |".format(
                mode=mode,
                rank=result["truth_pair_rank"],
                truth=result["truth_pair_rmse"],
                second=result["second_best_nontruth_rmse"],
                zero=result["near_numerical_zero_count"],
                baseline=result["baseline_rmse"],
            )
        )
    lines.extend(
        [
            "",
            "## Q1 best hard checkpoints",
            "",
            "| Mode | Control | Method | Hard attainment | Selected | Body P/R | Best step |",
            "|---|---|---|---:|---|---|---:|",
        ]
    )
    for result in summary.get("optimization", []):
        best = result["best_metrics"]
        lines.append(
            "| {mode} | {control} | {method} | {attain:.6f} | {selected} | {precision:.3f}/{recall:.3f} | {step} |".format(
                mode=result["mode"],
                control=result["control"],
                method=result["method"],
                attain=float(best["hard_attainment"]),
                selected=best["selected_indices"],
                precision=float(best["body_precision"]),
                recall=float(best["body_recall"]),
                step=int(best["best_step"]),
            )
        )
    lines.extend(["", "See `summary.json`, full enumeration CSVs and optimization traces for authoritative values.", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = _read_json(config_path)
    resolved = validate_simple_causality_config(config)
    modes = _parse_csv_choices(
        args.modes,
        default=resolved["observation_modes"],
        allowed=OBSERVATION_MODES,
        name="modes",
    )
    methods = _parse_csv_choices(
        args.methods,
        default=resolved["optimization"]["methods"],
        allowed=OPTIMIZATION_METHODS,
        name="methods",
    )
    seismic_controls = _parse_csv_choices(
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
    acoustic_path = _resolve_repo_path(config.get("acoustic_config"), "acoustic_config")
    seismic_path = _resolve_repo_path(config.get("seismic_config"), "seismic_config")
    density_path = _resolve_repo_path(config.get("density_config"), "density_config")
    gravity_path = _resolve_repo_path(config.get("gravity_config"), "gravity_config")
    source_path = Path(__file__).resolve()
    module_path = (PROJECT_DIR / "guidance/simple_causality.py").resolve()
    asset_paths = {
        "config": config_path,
        "acoustic_config": acoustic_path,
        "seismic_config": seismic_path,
        "density_config": density_path,
        "gravity_config": gravity_path,
        "runner_source": source_path,
        "causality_source": module_path,
    }

    case = build_simple_causal_case(config)
    acoustic_tables, acoustic_metadata = acoustic_tables_from_config(
        _read_json(acoustic_path), 15
    )
    density_table, density_metadata = density_table_from_config(
        _read_json(density_path), 15
    )
    seismic_operator, seismic_metadata = seismic_operator_from_config(
        _read_json(seismic_path), grid_shape=resolved["grid_shape"]
    )
    gravity_operator, gravity_metadata = gravity_operator_from_config(
        _read_json(gravity_path), grid_shape=resolved["grid_shape"]
    )
    suite = AnalyticObservationSuite(
        case,
        acoustic_property_table=acoustic_tables.property_table.to(device),
        density_table=density_table.to(device),
        seismic_operator=seismic_operator,
        gravity_operator=gravity_operator,
        blur_sigma_voxels=float(resolved["blur_sigma_voxels"]),
    )

    _write_json(output_dir / "config_input.json", config)
    public = _public_config(config, resolved)
    public.update(
        {
            "implementation_version": PHASE6Q_IMPLEMENTATION_VERSION,
            "selected_modes": modes,
            "selected_methods": methods,
            "selected_seismic_controls": seismic_controls,
            "enumeration_only": bool(args.enumeration_only),
            "device": str(device),
            "asset_hashes": {name: _file_sha256(path) for name, path in asset_paths.items()},
            "started_at_utc": started_at,
        }
    )
    _write_json(output_dir / "config_resolved.json", public)
    tensor_records = _save_case(output_dir, case)
    _write_json(
        output_dir / "input_validation.json",
        {**case.validation, "tensor_records": tensor_records},
    )

    enumeration_summary: dict[str, object] = {}
    for mode in modes:
        result = enumerate_hard_pairs(
            suite,
            mode,
            batch_size=int(resolved["enumeration_batch_size"]),
        )
        _write_rows(output_dir / "enumeration" / f"{mode}.csv", result["rows"])
        for tensor_name in ("observed", "baseline_field"):
            tensor_path = output_dir / "fields" / mode / f"{tensor_name}.pt"
            tensor_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(result[tensor_name], tensor_path)
        enumeration_summary[mode] = {
            key: value
            for key, value in result.items()
            if key not in {"rows", "observed", "baseline_field"}
        }

    optimization_summary: list[dict[str, object]] = []
    if not args.enumeration_only:
        optimization = resolved["optimization"]
        for mode in modes:
            controls = seismic_controls if mode == "seismic" else ["correct"]
            for control in controls:
                for method in methods:
                    result = optimize_candidate_logits(
                        suite,
                        mode,
                        control=control,
                        method=method,
                        temperatures=optimization["temperatures"],
                        learning_rate=float(optimization["learning_rate"]),
                        weight_decay=float(optimization["weight_decay"]),
                        initial_logit=float(optimization["initial_logit"]),
                        cardinality_weight=float(optimization["cardinality_weight"]),
                        hard_check_interval=int(optimization["hard_check_interval"]),
                        shuffle_seed=int(resolved["shuffle_seed"]),
                    )
                    base = output_dir / "optimization" / mode / control / method
                    _write_rows(base / "trace.csv", result["trace"])
                    torch.save(result["best_logits"], base / "best_logits.pt")
                    torch.save(
                        result["best_hard_coefficients"],
                        base / "best_hard_coefficients.pt",
                    )
                    optimization_summary.append(
                        {
                            "mode": mode,
                            "control": control,
                            "method": method,
                            "baseline_rmse": result["baseline_rmse"],
                            "best_metrics": result["best_metrics"],
                            "final_metrics": result["final_metrics"],
                            "trace_path": str((base / "trace.csv").relative_to(output_dir)),
                        }
                    )

    completed_at = datetime.now(timezone.utc).isoformat()
    runtime = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "pid": os.getpid(),
    }
    summary = {
        "stage": "phase6q_q0_q1_simple_causality",
        "implementation_version": PHASE6Q_IMPLEMENTATION_VERSION,
        "status": "completed",
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "case_validation": case.validation,
        "truth_candidate_indices": list(case.truth_candidate_indices),
        "enumeration": enumeration_summary,
        "optimization": optimization_summary,
        "acoustic_metadata": acoustic_metadata,
        "density_metadata": density_metadata,
        "seismic_metadata": seismic_metadata,
        "gravity_metadata": gravity_metadata,
        "runtime": runtime,
        "formal_training_performed": False,
        "flow_checkpoint_loaded": False,
    }
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "REPORT.md").write_text(
        _report_markdown(summary), encoding="utf-8"
    )
    print(json.dumps({
        "status": "completed",
        "output_dir": str(output_dir),
        "enumeration_modes": modes,
        "optimization_runs": len(optimization_summary),
        "gpu_peak_allocated_bytes": runtime["gpu_peak_allocated_bytes"],
    }, indent=2))


if __name__ == "__main__":
    main()
