#!/usr/bin/env python3
"""Run the Stage6Q D1 forward/observation closure gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shlex
import socket
import subprocess
import sys

import torch
import torch.nn.functional as F


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from guidance.gravity import density_table_from_config, gravity_operator_from_config
from guidance.observation_closure_audit import audit_five_body_observation_closure
from guidance.seismic import acoustic_tables_from_config, seismic_operator_from_config
from guidance.simple_causality import AnalyticObservationSuite, build_simple_causal_case
from guided_geophysical_sampling import soft_decode_to_probs
from scripts.stage6.run_embedding_endpoint_causality import _checkpoint_embedding
from scripts.stage6.run_simple_causality import (
    _file_sha256,
    _read_json,
    _resolve_repo_path,
    _tensor_sha256,
    _write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage6Q observation closure audit")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    return parser.parse_args()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPOSITORY_ROOT, text=True
    ).strip()


def _report(summary: dict[str, object]) -> str:
    lines = [
        "# Phase 6Q D1 observation closure report",
        "",
        f"Verdict: **{'PASS' if summary['closure_pass'] else 'FAIL'}**",
        "",
        "| Operator | Truth loss | Relative L2 difference | Baseline hard RMSE |",
        "|---|---:|---:|---:|",
    ]
    for mode in ("property", "reflectivity_spikes", "seismic", "gravity"):
        result = summary[mode]
        lines.append(
            f"| {mode} | {result['closure']['truth_loss']:.8g} | "
            f"{result['closure']['relative_difference_l2']:.8g} | "
            f"{result['baseline_hard_to_truth']['raw_rmse']:.8g} |"
        )
    lines.extend(
        [
            "",
            "The inversion observation and independently recomputed hard response use the same canonical single-source operators. Soft and hard baseline responses are stored separately.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    config_path = args.config.resolve()
    config = _read_json(config_path)
    if config.get("schema") != "phase6q_observation_closure_config_v1":
        raise ValueError("unexpected observation closure config schema")
    if not bool(config.get("noiseless_inverse_crime")):
        raise ValueError("D1 requires the noiseless inverse-crime case")
    if bool(config.get("flow_unet_loaded")) or bool(config.get("checkpoint_used_for_flow")):
        raise ValueError("D1 forbids loading the flow U-Net")
    if bool(config.get("formal_training_authorized")):
        raise ValueError("D1 forbids training")
    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    base_path = _resolve_repo_path(config.get("base_config"), "base_config")
    base_config = _read_json(base_path)
    checkpoint_path = _resolve_repo_path(config.get("checkpoint"), "checkpoint")
    embedding = _checkpoint_embedding(
        checkpoint_path, str(config.get("checkpoint_embedding_key"))
    ).to(device=device, dtype=dtype)
    acoustic_path = _resolve_repo_path(base_config.get("acoustic_config"), "acoustic_config")
    seismic_path = _resolve_repo_path(base_config.get("seismic_config"), "seismic_config")
    density_path = _resolve_repo_path(base_config.get("density_config"), "density_config")
    gravity_path = _resolve_repo_path(base_config.get("gravity_config"), "gravity_config")
    acoustic_tables, acoustic_metadata = acoustic_tables_from_config(
        _read_json(acoustic_path), embedding.shape[0]
    )
    density_table, density_metadata = density_table_from_config(
        _read_json(density_path), embedding.shape[0]
    )
    case = build_simple_causal_case(base_config)
    seismic_operator, seismic_metadata = seismic_operator_from_config(
        _read_json(seismic_path), grid_shape=case.truth_labels.shape[2:]
    )
    gravity_operator, gravity_metadata = gravity_operator_from_config(
        _read_json(gravity_path), grid_shape=case.truth_labels.shape[2:]
    )
    suite = AnalyticObservationSuite(
        case,
        acoustic_property_table=acoustic_tables.property_table.to(device=device, dtype=dtype),
        density_table=density_table.to(device=device, dtype=dtype),
        seismic_operator=seismic_operator,
        gravity_operator=gravity_operator,
        blur_sigma_voxels=float(base_config["blur_sigma_voxels"]),
    )
    baseline_categories = case.baseline_labels.to(device).long()[:, 0] + 1
    baseline_state = embedding[baseline_categories].permute(0, 4, 1, 2, 3).contiguous()
    soft_probabilities = soft_decode_to_probs(
        baseline_state,
        embedding,
        tau=float(config["soft_reference_temperature"]),
    )
    summary, tensors = audit_five_body_observation_closure(
        case=case,
        suite=suite,
        soft_baseline_probabilities=soft_probabilities,
    )
    started = datetime.now(timezone.utc).isoformat()
    git_status = _git("status", "--short")
    summary.update(
        {
            "stage": "phase6q_d1_observation_closure",
            "completed_at_utc": started,
            "exact_command": shlex.join([sys.executable, *sys.argv]),
            "git_sha": _git("rev-parse", "HEAD"),
            "git_branch": _git("branch", "--show-current"),
            "git_status": "clean" if not git_status else "dirty",
            "git_status_short": git_status.splitlines(),
            "seed": int(config["seed"]),
            "device": str(device),
            "dtype": str(dtype),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _file_sha256(checkpoint_path),
            "checkpoint_embedding_sha256": _tensor_sha256(embedding),
            "checkpoint_used_for_flow": False,
            "flow_unet_loaded": False,
            "ema_raw_policy": "checkpoint raw frozen embedding only; U-Net not loaded",
            "base_config_sha256": _file_sha256(base_path),
            "property_table_sha256": _tensor_sha256(density_table),
            "acoustic_table_sha256": _tensor_sha256(acoustic_tables.property_table),
            "wavelet_sha256": seismic_operator.metadata()["wavelet"]["sha256"],
            "observation_sha256": {
                name: values["closure"]["truth_observation_hash"]
                for name, values in summary.items()
                if isinstance(values, dict) and "closure" in values
            },
            "source_hashes": {
                "runner": _file_sha256(Path(__file__).resolve()),
                "audit": _file_sha256(PROJECT_DIR / "guidance/observation_closure_audit.py"),
                "simple_causality": _file_sha256(PROJECT_DIR / "guidance/simple_causality.py"),
                "seismic": _file_sha256(PROJECT_DIR / "guidance/seismic.py"),
                "gravity": _file_sha256(PROJECT_DIR / "guidance/gravity.py"),
            },
            "runtime": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version,
                "torch": torch.__version__,
            },
            "acoustic_metadata": acoustic_metadata,
            "density_metadata": density_metadata,
            "seismic_metadata": seismic_metadata,
            "gravity_metadata": gravity_metadata,
        }
    )
    output_dir.mkdir(parents=True)
    _write_json(output_dir / "config_input.json", config)
    _write_json(
        output_dir / "config_resolved.json",
        {
            **config,
            "device": str(device),
            "dtype": str(dtype),
            "base_config_sha256": _file_sha256(base_path),
            "checkpoint_sha256": _file_sha256(checkpoint_path),
        },
    )
    for name, value in tensors.items():
        path = output_dir / "tensors" / f"{name}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(value, path)
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    print(json.dumps({"closure_pass": summary["closure_pass"], "output_dir": str(output_dir)}, indent=2))
    if not summary["closure_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
