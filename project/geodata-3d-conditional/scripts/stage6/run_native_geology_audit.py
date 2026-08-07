#!/usr/bin/env python3
"""Run Stage6Q D5 StructuralGeo-native support and frozen-flow audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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
STRUCTURALGEO_ROOT = REPOSITORY_ROOT / "StructuralGeo-main"
for path in (PROJECT_DIR, STRUCTURALGEO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import inference_runtime as runtime
from guidance.frozen_flow_causality import run_base_trajectory, run_paired_trajectory
from guidance.native_geology_audit import (
    NATIVE_GEOLOGY_AUDIT_VERSION,
    build_structuralgeo_native_case,
    connected_target_statistics,
)
from guidance.seismic import (
    acoustic_tables_from_config,
    hard_labels_to_acoustic,
    overwrite_exact_condition_acoustic,
    probabilities_to_subsurface_acoustic,
    seismic_operator_from_config,
)
from guidance.simple_causality import controlled_observation
from guided_geophysical_sampling import soft_decode_to_probs
from scripts.stage6.run_simple_causality import (
    _file_sha256, _read_json, _resolve_repo_path, _tensor_sha256, _write_json, _write_rows
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage6Q D5 native geology audit")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPOSITORY_ROOT, text=True).strip()


def _model_hash(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _truth_overlap(labels: torch.Tensor, truth: torch.Tensor, target_label: int) -> dict[str, float]:
    predicted = labels == target_label
    actual = truth == target_label
    intersection = int((predicted & actual).sum())
    union = int((predicted | actual).sum())
    return {
        "target_iou": intersection / union if union else 1.0,
        "target_recall": intersection / int(actual.sum()) if int(actual.sum()) else 1.0,
        "target_precision": intersection / int(predicted.sum()) if int(predicted.sum()) else 0.0,
    }


def _report(summary: dict[str, object]) -> str:
    support = summary["prior_support"]
    lines = [
        "# Phase 6Q D5 StructuralGeo-native audit", "",
        f"Native history: five IntrusionSpec(kind=hemisphere) events; audit label 9 is their recorded union, not a dike assumption.",
        f"Prior support: {support['samples_with_target_beyond_conditions']}/{support['sample_count']} samples contain target voxels beyond exact conditions; size-compatible component fraction {support['size_compatible_sample_fraction']:.3f}.",
        "", "| Control | Mode | Best/final hard attainment | Final soft attainment | Final target IoU |", "|---|---|---:|---:|---:|",
    ]
    for run in summary["runs"]:
        lines.append(
            f"| {run['control']} | {run['mode']} | {run['best_hard_attainment']:.4f}/{run['final_hard_attainment']:.4f} | "
            f"{run['final_soft_attainment']:.4f} | {run['final_target_iou']:.4f} |"
        )
    lines.extend(["", f"Correct-control hard-specificity margin over the strongest control: {summary['correct_control_specificity']['hard_margin']:.6f}."])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to reuse output: {out}")
    config_path = args.config.resolve()
    config = _read_json(config_path)
    if config.get("schema") != "phase6q_native_geology_audit_config_v1":
        raise ValueError("unexpected D5 config schema")
    if bool(config.get("formal_training_authorized")) or bool(config.get("base_model_trainable")):
        raise ValueError("D5 forbids training and requires a frozen model")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("D5 formal run requires CUDA")

    case, native_metadata = build_structuralgeo_native_case(seed=int(config["native_generator_seed"]))
    truth = case.truth_labels.to(device)
    condition = case.condition_mask.to(device)
    subsurface = case.subsurface_mask.to(device)
    checkpoint = _resolve_repo_path(config["checkpoint"], "checkpoint")
    if _file_sha256(checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("checkpoint hash mismatch")
    base_config_path = _resolve_repo_path(config["base_config"], "base_config")
    base_config = _read_json(base_config_path)
    acoustic_path = _resolve_repo_path(base_config["acoustic_config"], "acoustic_config")
    seismic_path = _resolve_repo_path(base_config["seismic_config"], "seismic_config")

    from model_train_sh_inference_cond import Geo3DStochInterp
    model, load_report = runtime.load_model_with_weight_policy(Geo3DStochInterp, checkpoint, device, "ema")
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    model_hash_before = _model_hash(model)
    embedded_truth = model.embed(truth).detach()
    conditioning = embedded_truth * condition.expand_as(embedded_truth)
    tables, acoustic_meta = acoustic_tables_from_config(_read_json(acoustic_path), model.num_categories)
    table = tables.property_table.to(device)
    operator, seismic_meta = seismic_operator_from_config(_read_json(seismic_path), grid_shape=truth.shape[2:])
    target_acoustic = hard_labels_to_acoustic(truth, table)
    truth_response = operator(target_acoustic[:, 0:1], target_acoustic[:, 1:2], subsurface).detach()

    out.mkdir(parents=True)
    artifact_dir = out / "native_case"
    artifact_dir.mkdir()
    torch.save(case.truth_labels, artifact_dir / "truth_labels.pt")
    torch.save(case.condition_mask, artifact_dir / "condition_mask.pt")
    torch.save(case.body_masks, artifact_dir / "body_masks.pt")
    _write_json(artifact_dir / "event_history.json", native_metadata)

    prior_rows = []
    prior_state_dir = out / "prior_samples"
    prior_state_dir.mkdir()
    native_sizes = [int(v) for v in native_metadata["body_voxel_counts"]]
    ratio_low, ratio_high = (float(v) for v in config["target_support_size_ratio_interval"])
    size_low, size_high = min(native_sizes) * ratio_low, max(native_sizes) * ratio_high
    for seed in config["prior_seeds"]:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        initial_cpu = torch.randn((1, model.embedding_dim, *model.data_shape), generator=generator)
        schedule = run_base_trajectory(
            model=model, initial_state=initial_cpu.to(device), conditioning=conditioning,
            embedded_conditions=embedded_truth, condition_mask=condition, n_steps=int(config["n_steps"]),
        )
        final_state = schedule["states"][-1]
        labels = (model.decode(final_state.to(device)) - 1).unsqueeze(1).cpu()
        if not torch.equal(labels[case.condition_mask], case.truth_labels[case.condition_mask]):
            raise RuntimeError("prior sample violated exact hard conditions")
        stats = connected_target_statistics(labels, target_label=case.target_label, condition_mask=case.condition_mask)
        stats.update({
            "seed": int(seed), "initial_state_sha256": _tensor_sha256(initial_cpu),
            "final_state_sha256": _tensor_sha256(final_state), "labels_sha256": _tensor_sha256(labels),
            "condition_exact": True,
            "size_compatible_component": any(size_low <= row["voxel_count"] <= size_high for row in stats["components"]),
        })
        prior_rows.append(stats)
        torch.save(labels, prior_state_dir / f"labels_seed_{seed}.pt")
    support_summary = {
        "sample_count": len(prior_rows),
        "samples_with_target_beyond_conditions": sum(row["target_outside_condition_voxels"] > 0 for row in prior_rows),
        "target_beyond_condition_fraction": sum(row["target_outside_condition_voxels"] > 0 for row in prior_rows) / len(prior_rows),
        "size_compatible_sample_fraction": sum(bool(row["size_compatible_component"]) for row in prior_rows) / len(prior_rows),
        "native_body_size_range": [min(native_sizes), max(native_sizes)],
        "compatible_component_size_interval": [size_low, size_high],
        "target_voxel_counts": [row["target_voxel_count"] for row in prior_rows],
        "component_counts_6": [row["component_count_6"] for row in prior_rows],
        "unconditioned_component_counts": [row["unconditioned_component_count"] for row in prior_rows],
        "component_centroids_xyz": [component["centroid_xyz"] for row in prior_rows for component in row["components"]],
        "condition_exact_all": all(bool(row["condition_exact"]) for row in prior_rows),
    }
    _write_json(out / "prior_support_samples.json", prior_rows)

    trajectory_generator = torch.Generator(device="cpu").manual_seed(int(config["trajectory_seed"]))
    trajectory_initial = torch.randn((1, model.embedding_dim, *model.data_shape), generator=trajectory_generator)
    trajectory_initial_hash = _tensor_sha256(trajectory_initial)
    base_schedule = run_base_trajectory(
        model=model, initial_state=trajectory_initial.to(device), conditioning=conditioning,
        embedded_conditions=embedded_truth, condition_mask=condition, n_steps=int(config["n_steps"]),
    )
    controller = config["controller"]
    results = []
    all_trace = []
    for control in config["controls"]:
        observation = controlled_observation(truth_response, control, shuffle_seed=int(config["shuffle_seed"]))

        def soft_response(state: torch.Tensor, tau: float) -> torch.Tensor:
            probs = soft_decode_to_probs(state, model.embedding.weight, tau=tau)
            acoustic = probabilities_to_subsurface_acoustic(probs, table, subsurface)
            known = overwrite_exact_condition_acoustic(acoustic, target_acoustic, condition)
            return operator(known[:, 0:1], known[:, 1:2], subsurface)

        def soft_loss(state: torch.Tensor, tau: float) -> torch.Tensor:
            return (soft_response(state, tau) - observation).square().mean()

        def hard_response(labels: torch.Tensor) -> torch.Tensor:
            acoustic = hard_labels_to_acoustic(labels, table)
            return operator(acoustic[:, 0:1], acoustic[:, 1:2], subsurface)

        for mode in config["modes"]:
            result = run_paired_trajectory(
                mode=mode, model=model, initial_state=trajectory_initial.to(device), conditioning=conditioning,
                embedded_conditions=embedded_truth, condition_mask=condition, embedding_weight=model.embedding.weight,
                soft_loss=soft_loss, soft_response=soft_response, hard_response=hard_response,
                observation=observation, truth_response=truth_response, target_label=case.target_label,
                base_schedule=base_schedule, n_steps=int(config["n_steps"]), alpha=float(controller["alpha"]),
                max_ratio=float(controller["max_ratio"]), tau_start=float(controller["tau_start"]),
                tau_end=float(controller["tau_end"]), tau_schedule=controller["tau_schedule"],
                guidance_start=float(controller["guidance_start"]), guidance_schedule=controller["guidance_schedule"],
                gradient_clip_norm=float(controller["gradient_clip_norm"]), scaling_mode=controller["scaling_mode"],
                late_start=float(controller["late_start"]),
            )
            trace = result["trace"]
            for row in trace:
                row["control"] = control
            all_trace.extend(trace)
            hard_attainments = [float(row["hard_attainment"]) for row in trace if math.isfinite(float(row["hard_attainment"]))]
            final_labels = (model.decode(result["final_state"].to(device)) - 1).unsqueeze(1)
            best_labels = (model.decode(result["best_hard_state"].to(device)) - 1).unsqueeze(1)
            final_overlap = _truth_overlap(final_labels, truth, case.target_label)
            best_overlap = _truth_overlap(best_labels, truth, case.target_label)
            run_summary = {
                "control": control, "mode": mode, "initial_state_sha256": trajectory_initial_hash,
                "best_hard_step": result["best_hard_step"], "best_soft_step": result["best_soft_step"],
                "best_hard_attainment": max(hard_attainments) if hard_attainments else float("nan"),
                "final_hard_attainment": float(trace[-1]["hard_attainment"]),
                "final_soft_attainment": float(trace[-1]["soft_attainment"]),
                "best_target_iou": best_overlap["target_iou"], "final_target_iou": final_overlap["target_iou"],
                "final_target_recall": final_overlap["target_recall"], "final_target_precision": final_overlap["target_precision"],
            }
            results.append(run_summary)
            trace_path = out / "traces" / control / f"{mode}.csv"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            _write_rows(trace_path, trace)
            state_dir = out / "states" / control / mode
            state_dir.mkdir(parents=True, exist_ok=True)
            torch.save(result["best_hard_state"], state_dir / "best_hard_state.pt")
            torch.save(result["final_state"], state_dir / "final_state.pt")

    correct = next(row for row in results if row["control"] == "correct" and row["mode"] == "BASE_PLUS_PHYSICS")
    controls = [row for row in results if row["control"] != "correct" and row["mode"] == "BASE_PLUS_PHYSICS"]
    specificity = {
        "correct_final_hard_attainment": correct["final_hard_attainment"],
        "strongest_control_final_hard_attainment": max(row["final_hard_attainment"] for row in controls),
    }
    specificity["hard_margin"] = specificity["correct_final_hard_attainment"] - specificity["strongest_control_final_hard_attainment"]
    model_hash_after = _model_hash(model)
    git_status = _git("status", "--short")
    native_hashes = {
        "truth_labels": _tensor_sha256(case.truth_labels), "condition_mask": _tensor_sha256(case.condition_mask),
        "subsurface_mask": _tensor_sha256(case.subsurface_mask), "body_masks": _tensor_sha256(case.body_masks),
        "event_history": _text_hash(native_metadata["event_history"]),
    }
    summary = {
        "stage": "phase6q_d5_native_geology_audit", "version": NATIVE_GEOLOGY_AUDIT_VERSION,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "native_metadata": native_metadata,
        "native_artifact_hashes": native_hashes, "prior_support": support_summary, "runs": results,
        "correct_control_specificity": specificity, "trajectory_initial_state_sha256": trajectory_initial_hash,
        "all_trajectory_arms_share_initial_state": len({row["initial_state_sha256"] for row in results}) == 1,
        "base_endpoint_sha256": _tensor_sha256(base_schedule["states"][-1]),
        "base_model_hash_before": model_hash_before, "base_model_hash_after": model_hash_after,
        "base_model_hash_unchanged": model_hash_before == model_hash_after,
        "base_model_gradients_absent": all(parameter.grad is None for parameter in model.parameters()),
        "exact_command": shlex.join([sys.executable, *sys.argv]), "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"), "git_status": "clean" if not git_status else "dirty",
        "git_status_short": git_status.splitlines(), "seed": config["native_generator_seed"],
        "device": str(device), "dtype": "torch.float32", "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _file_sha256(checkpoint), "ema_raw_policy": load_report,
        "flow_unet_loaded": True, "checkpoint_used_for_flow": True,
        "property_table_sha256": _tensor_sha256(tables.property_table),
        "acoustic_table_sha256": _tensor_sha256(tables.property_table),
        "wavelet_sha256": operator.metadata()["wavelet"]["sha256"], "observation_sha256": _tensor_sha256(truth_response),
        "source_hashes": {
            "runner": _file_sha256(Path(__file__).resolve()),
            "native_audit": _file_sha256(PROJECT_DIR / "guidance/native_geology_audit.py"),
            "frozen_flow": _file_sha256(PROJECT_DIR / "guidance/frozen_flow_causality.py"),
            "runtime": _file_sha256(PROJECT_DIR / "inference_runtime.py"),
            "controller": _file_sha256(PROJECT_DIR / "guided_geophysical_sampling.py"),
            "seismic": _file_sha256(PROJECT_DIR / "guidance/seismic.py"),
            "structuralgeo_parametric": _file_sha256(STRUCTURALGEO_ROOT / "src/geogen/engine/parametric.py"),
            "structuralgeo_geoprocess": _file_sha256(STRUCTURALGEO_ROOT / "src/geogen/model/geoprocess.py"),
            "structuralgeo_geomodel": _file_sha256(STRUCTURALGEO_ROOT / "src/geogen/model/geomodel.py"),
        },
        "runtime": {"hostname": socket.gethostname(), "torch": torch.__version__, "gpu_name": torch.cuda.get_device_name(device)},
        "acoustic_metadata": acoustic_meta, "seismic_metadata": seismic_meta,
    }
    _write_json(out / "config_input.json", config)
    _write_json(out / "config_resolved.json", {**config, "git_sha": summary["git_sha"], "device": str(device)})
    _write_rows(out / "trajectory_trace.csv", all_trace)
    _write_json(out / "summary.json", summary)
    (out / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    print(json.dumps({"prior_samples": len(prior_rows), "trajectory_runs": len(results), "output_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()
