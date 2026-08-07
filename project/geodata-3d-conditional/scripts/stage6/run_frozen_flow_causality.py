#!/usr/bin/env python3
"""Run Stage6Q D4 paired frozen-flow trajectory isolation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shlex
import socket
import subprocess
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import inference_runtime as runtime
from guidance.frozen_flow_causality import (
    FROZEN_FLOW_CAUSALITY_VERSION,
    run_base_trajectory,
    run_paired_trajectory,
)
from guidance.physics_attainment import optimize_endpoint_state
from guidance.seismic import (
    acoustic_tables_from_config,
    hard_labels_to_acoustic,
    overwrite_exact_condition_acoustic,
    probabilities_to_subsurface_acoustic,
    seismic_operator_from_config,
)
from guidance.simple_causality import (
    AnalyticObservationSuite,
    build_simple_causal_case,
    controlled_observation,
)
from guided_geophysical_sampling import soft_decode_to_probs
from scripts.stage6.run_simple_causality import (
    _file_sha256, _read_json, _resolve_repo_path, _tensor_sha256, _write_json, _write_rows
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage6Q D4 frozen-flow audit")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPOSITORY_ROOT, text=True).strip()


def _model_hash(model) -> str:
    import hashlib
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _report(summary: dict[str, object]) -> str:
    lines = ["# Phase 6Q D4 frozen-flow trajectory isolation", "", "| Control | Mode | Best/final hard attainment | Best hard step | Final soft attainment |", "|---|---|---:|---:|---:|"]
    for run in summary["runs"]:
        lines.append(f"| {run['control']} | {run['mode']} | {run['best_hard_attainment']:.4f}/{run['final_hard_attainment']:.4f} | {run['best_hard_step']} | {run['final_soft_attainment']:.4f} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to reuse output: {out}")
    config_path = args.config.resolve()
    config = _read_json(config_path)
    if config.get("schema") != "phase6q_frozen_flow_trajectory_config_v1":
        raise ValueError("unexpected D4 schema")
    if bool(config.get("formal_training_authorized")) or bool(config.get("base_model_trainable")):
        raise ValueError("D4 requires a frozen model and forbids training")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("D4 formal run requires CUDA")
    base_path = _resolve_repo_path(config["base_config"], "base_config")
    checkpoint = _resolve_repo_path(config["checkpoint"], "checkpoint")
    if _file_sha256(checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("checkpoint hash mismatch")
    base_config = _read_json(base_path)
    case = build_simple_causal_case(base_config)
    from model_train_sh_inference_cond import Geo3DStochInterp
    model, load_report = runtime.load_model_with_weight_policy(
        Geo3DStochInterp, checkpoint, device, "ema"
    )
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    model_hash_before = _model_hash(model)
    truth = case.truth_labels.to(device)
    condition = case.condition_mask.to(device)
    subsurface = case.subsurface_mask.to(device)
    embedded_truth = model.embed(truth).detach()
    conditioning = embedded_truth * condition.expand_as(embedded_truth)
    acoustic_path = _resolve_repo_path(base_config["acoustic_config"], "acoustic_config")
    seismic_path = _resolve_repo_path(base_config["seismic_config"], "seismic_config")
    tables, acoustic_meta = acoustic_tables_from_config(_read_json(acoustic_path), model.num_categories)
    table = tables.property_table.to(device)
    operator, seismic_meta = seismic_operator_from_config(_read_json(seismic_path), grid_shape=truth.shape[2:])
    target_acoustic = hard_labels_to_acoustic(truth, table)
    truth_response = operator(target_acoustic[:, 0:1], target_acoustic[:, 1:2], subsurface).detach()
    suite = AnalyticObservationSuite(
        case, acoustic_property_table=table, density_table=torch.ones(model.num_categories, device=device),
        seismic_operator=operator, gravity_operator=object(), blur_sigma_voxels=float(base_config["blur_sigma_voxels"])
    )
    generator = torch.Generator(device="cpu").manual_seed(int(config["seed"]))
    initial_cpu = torch.randn((1, model.embedding_dim, *model.data_shape), generator=generator)
    initial_hash = _tensor_sha256(initial_cpu)
    base_schedule = run_base_trajectory(
        model=model, initial_state=initial_cpu.to(device), conditioning=conditioning,
        embedded_conditions=embedded_truth, condition_mask=condition, n_steps=int(config["n_steps"])
    )
    controller = config["controller"]
    results = []
    all_trace = []
    for control in config["controls"]:
        observation = controlled_observation(truth_response, control, shuffle_seed=int(base_config["shuffle_seed"]))
        def soft_response(state: torch.Tensor, tau: float) -> torch.Tensor:
            probs = soft_decode_to_probs(state, model.embedding.weight, tau=tau)
            acoustic = probabilities_to_subsurface_acoustic(probs, table, subsurface)
            known = overwrite_exact_condition_acoustic(acoustic, target_acoustic, condition)
            return operator(known[:, 0:1], known[:, 1:2], subsurface)
        def soft_loss(state: torch.Tensor, tau: float) -> torch.Tensor:
            return (soft_response(state, tau) - observation).square().mean()
        def hard_response(labels: torch.Tensor) -> torch.Tensor:
            return suite.field_from_labels(labels, "seismic")
        for mode in [m for m in config["modes"] if m != "ENDPOINT_REFERENCE"]:
            result = run_paired_trajectory(
                mode=mode, model=model, initial_state=initial_cpu.to(device), conditioning=conditioning,
                embedded_conditions=embedded_truth, condition_mask=condition,
                embedding_weight=model.embedding.weight, soft_loss=soft_loss,
                soft_response=soft_response, hard_response=hard_response,
                observation=observation, truth_response=truth_response,
                target_label=case.target_label, base_schedule=base_schedule,
                n_steps=int(config["n_steps"]), alpha=float(controller["alpha"]),
                max_ratio=float(controller["max_ratio"]), tau_start=float(controller["tau_start"]),
                tau_end=float(controller["tau_end"]), tau_schedule=controller["tau_schedule"],
                guidance_start=float(controller["guidance_start"]), guidance_schedule=controller["guidance_schedule"],
                gradient_clip_norm=float(controller["gradient_clip_norm"]), scaling_mode=controller["scaling_mode"],
                late_start=float(controller["late_start"]),
            )
            trace = result["trace"]
            for row in trace: row.update({"control": control})
            all_trace.extend(trace)
            hard_values = [float(row["hard_attainment"]) for row in trace if math.isfinite(float(row["hard_attainment"]))]
            result_summary = {
                "control": control, "mode": mode, "initial_state_sha256": initial_hash,
                "best_hard_step": result["best_hard_step"], "best_soft_step": result["best_soft_step"],
                "best_hard_attainment": max(hard_values) if hard_values else float("nan"),
                "final_hard_attainment": float(trace[-1]["hard_attainment"]),
                "final_soft_attainment": float(trace[-1]["soft_attainment"]),
            }
            results.append(result_summary)
            path = out / "traces" / control / f"{mode}.csv"; path.parent.mkdir(parents=True, exist_ok=True); _write_rows(path, trace)
            state_dir = out / "states" / control / mode; state_dir.mkdir(parents=True, exist_ok=True)
            torch.save(result["best_hard_state"], state_dir / "best_hard_state.pt")
            torch.save(result["final_state"], state_dir / "final_state.pt")
        base_endpoint = base_schedule["states"][-1].to(device)
        endpoint_cfg = config["endpoint"]
        def endpoint_hard(state: torch.Tensor):
            labels = (model.decode(state) - 1).unsqueeze(1)
            response = hard_response(labels)
            loss = (response - observation).square().mean()
            return {"hard_loss": loss}, {"hard_response": response, "labels": labels}
        endpoint = optimize_endpoint_state(
            initial_state=base_endpoint, embedded_conditions=embedded_truth, condition_mask=condition,
            soft_loss=lambda state, tau: (soft_loss(state, tau), {}), hard_evaluate=endpoint_hard,
            temperature_schedule=endpoint_cfg["temperature_schedule"], learning_rate=float(endpoint_cfg["learning_rate"]),
            weight_decay=float(endpoint_cfg["weight_decay"]), gradient_clip_norm=float(endpoint_cfg["gradient_clip_norm"]),
            hard_check_interval=int(endpoint_cfg["hard_check_interval"]),
            max_voxel_norm=float(torch.linalg.vector_norm(model.embedding.weight, dim=1).max()) * float(endpoint_cfg["max_state_norm_to_embedding_norm"]),
        )
        endpoint_trace = [{"control": control, "mode": "ENDPOINT_REFERENCE", **row} for row in endpoint["trace"]]
        _write_rows(out / "traces" / control / "ENDPOINT_REFERENCE.csv", endpoint_trace)
        baseline_endpoint_loss = float(endpoint["initial_metrics"]["hard_loss"])
        best_endpoint_loss = float(endpoint["best_metrics"]["hard_loss"])
        final_endpoint_loss = float(endpoint_trace[-1]["hard_loss"])
        best_attainment = 1.0 - best_endpoint_loss / baseline_endpoint_loss if baseline_endpoint_loss > 0 else float("nan")
        final_attainment = 1.0 - final_endpoint_loss / baseline_endpoint_loss if baseline_endpoint_loss > 0 else float("nan")
        results.append({
            "control": control, "mode": "ENDPOINT_REFERENCE", "initial_state_sha256": initial_hash,
            "base_endpoint_sha256": _tensor_sha256(base_schedule["states"][-1]), "best_hard_step": endpoint["best_step"],
            "best_soft_step": -1, "best_hard_attainment": best_attainment,
            "final_hard_attainment": final_attainment,
            "final_soft_attainment": float("nan"),
        })
        state_dir = out / "states" / control / "ENDPOINT_REFERENCE"; state_dir.mkdir(parents=True, exist_ok=True)
        torch.save(endpoint["best_state"], state_dir / "best_hard_state.pt")
        torch.save(endpoint["final_state"], state_dir / "final_state.pt")
    model_hash_after = _model_hash(model)
    git_status = _git("status", "--short")
    summary = {
        "stage": "phase6q_d4_frozen_flow", "version": FROZEN_FLOW_CAUSALITY_VERSION,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "runs": results,
        "initial_state_sha256": initial_hash, "all_arms_share_initial_state": len({r["initial_state_sha256"] for r in results}) == 1,
        "base_endpoint_sha256": _tensor_sha256(base_schedule["states"][-1]),
        "base_model_hash_before": model_hash_before, "base_model_hash_after": model_hash_after,
        "base_model_hash_unchanged": model_hash_before == model_hash_after,
        "base_model_gradients_absent": all(p.grad is None for p in model.parameters()),
        "budget_extension_run": False,
        "exact_command": shlex.join([sys.executable, *sys.argv]), "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"), "git_status": "clean" if not git_status else "dirty",
        "git_status_short": git_status.splitlines(), "seed": config["seed"], "device": str(device), "dtype": "torch.float32",
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": _file_sha256(checkpoint), "ema_raw_policy": load_report,
        "flow_unet_loaded": True, "checkpoint_used_for_flow": True,
        "property_table_sha256": _tensor_sha256(tables.property_table),
        "acoustic_table_sha256": _tensor_sha256(tables.property_table), "wavelet_sha256": operator.metadata()["wavelet"]["sha256"],
        "observation_sha256": _tensor_sha256(truth_response),
        "source_hashes": {
            "runner": _file_sha256(Path(__file__).resolve()),
            "sampler": _file_sha256(PROJECT_DIR / "guidance/frozen_flow_causality.py"),
            "runtime": _file_sha256(PROJECT_DIR / "inference_runtime.py"),
            "controller": _file_sha256(PROJECT_DIR / "guided_geophysical_sampling.py"),
            "seismic": _file_sha256(PROJECT_DIR / "guidance/seismic.py"),
            "model_loader": _file_sha256(PROJECT_DIR / "model_train_sh_inference_cond.py"),
        },
        "runtime": {"hostname": socket.gethostname(), "torch": torch.__version__, "gpu_name": torch.cuda.get_device_name(device)},
        "acoustic_metadata": acoustic_meta, "seismic_metadata": seismic_meta,
    }
    _write_json(out / "config_input.json", config); _write_json(out / "config_resolved.json", {**config, "git_sha": summary["git_sha"], "device": str(device)})
    _write_rows(out / "trajectory_trace.csv", all_trace); _write_json(out / "summary.json", summary)
    (out / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    print(json.dumps({"runs": len(results), "output_dir": str(out)}, indent=2))


if __name__ == "__main__": main()
