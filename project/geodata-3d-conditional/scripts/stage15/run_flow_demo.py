#!/usr/bin/env python3
"""Truth-blind paired Flow demo using the frozen Stage15-G binary evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime
from guidance.probability_volume import tensor_sha256
from scripts.stage14.run_gansim_style_geo_guidance import _sample_one
from scripts.stage15.common import base_manifest, normalize_volume, read_json, refuse_nonempty, write_csv, write_json

ROOT = PROJECT_DIR / "experiments/stage15_binary_seismic_consensus"
DEFAULT_CONFIG = ROOT / "configs/flow_demo_coarse_occupancy_v1.json"
DEFAULT_OBSERVATION = ROOT / "observations/cond_generation_0"
DEFAULT_PROBABILITY = ROOT / "binary_logistic/cond_generation_0_8x8x8_v2/coarse_label9_probability.pt"
DEFAULT_OUTPUT = ROOT / "flow_demo/coarse_occupancy_seed_screen_n8_v2"
DEFAULT_CHECKPOINT = PROJECT_DIR / "demo_model/conditional-weights.ckpt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--observation-dir", type=Path, default=DEFAULT_OBSERVATION)
    p.add_argument("--probability", type=Path, default=DEFAULT_PROBABILITY)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def evidence_transform(probability: torch.Tensor, subsurface: torch.Tensor) -> torch.Tensor:
    maximum = probability[subsurface.bool()].max()
    if float(maximum) <= 0:
        raise ValueError("binary probability has no positive evidence")
    evidence = (probability / maximum).clamp(0.0, 1.0)
    return torch.where(subsurface.bool(), evidence, torch.zeros_like(evidence)).contiguous()


def main() -> None:
    args = parse_args()
    refuse_nonempty(args.output_dir)
    config = read_json(args.config)
    if config.get("schema") != "stage15_flow_demo_coarse_occupancy_v1" or config.get("status") != "frozen_before_flow_demo":
        raise ValueError("invalid Stage15 flow demo config")
    paths = {
        "checkpoint": args.checkpoint,
        "raw_probability": args.probability,
        "condition_values": args.observation_dir / "flow_condition_values.pt",
        "condition_mask": args.observation_dir / "flow_condition_mask.pt",
        "subsurface_mask": args.observation_dir / "subsurface_mask.pt",
    }
    for name, path in paths.items():
        if runtime.file_sha256(path) != config[f"{name}_sha256"]:
            raise ValueError(f"frozen flow-demo input changed: {name}")
    device = torch.device(args.device)
    from model_train_sh_inference_cond import Geo3DStochInterp
    model, model_report = runtime.load_model_with_weight_policy(
        model_class=Geo3DStochInterp,
        checkpoint_path=args.checkpoint,
        map_location=device,
        weight_source="ema",
    )
    model = model.to(device)
    condition_values_cpu = normalize_volume(runtime.load_tensor(paths["condition_values"]), "condition_values").long()
    condition_mask_cpu = normalize_volume(runtime.load_tensor(paths["condition_mask"]), "condition_mask").bool()
    subsurface_cpu = normalize_volume(runtime.load_tensor(paths["subsurface_mask"]), "subsurface").bool()
    raw_probability = runtime.load_tensor(args.probability).float()
    if tuple(raw_probability.shape) != (1, 1, 8, 8, 8):
        raise ValueError("coarse Flow guidance target must have shape [1,1,8,8,8]")
    if not torch.isfinite(raw_probability).all() or bool(((raw_probability < 0) | (raw_probability > 1)).any()):
        raise ValueError("coarse Flow guidance target must be finite [0,1]")
    # Trilinear volume is saved for visualization only. The loss consumes the
    # original 8^3 target and averages the Flow probability within each cell.
    evidence_cpu = F.interpolate(raw_probability, size=(64, 64, 64), mode="trilinear", align_corners=False)
    evidence_cpu = torch.where(subsurface_cpu, evidence_cpu, torch.zeros_like(evidence_cpu))
    roi_cpu = subsurface_cpu & ~condition_mask_cpu
    core_cpu = torch.zeros_like(subsurface_cpu)
    condition_values = condition_values_cpu.to(device)
    condition_mask = condition_mask_cpu.to(device)
    embedded = model.embed(condition_values)
    expanded_mask = condition_mask.expand(-1, embedded.shape[1], -1, -1, -1)
    conditioning = embedded * expanded_mask
    settings = config

    args.output_dir.mkdir(parents=True)
    torch.save(evidence_cpu, args.output_dir / "guidance_evidence.pt")
    manifest = base_manifest("stage15_flow_demo_truth_blind_v1", Path(__file__), args.config)
    write_json(args.output_dir / "run_manifest.json", manifest)
    pair_rows = []
    traces = {"FLOW_ONLY": [], "GEO_EVIDENCE_GUIDED": []}
    try:
        for sample_id, seed_value in enumerate(config["source_seeds"]):
            seed = int(seed_value)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            initial = torch.randn(1, model.embedding_dim, *model.data_shape, generator=generator, dtype=embedded.dtype)
            outputs = {}
            for arm, alpha in (("FLOW_ONLY", 0.0), ("GEO_EVIDENCE_GUIDED", float(config["alpha"]))):
                decoded, trace = _sample_one(
                    model=model,
                    initial_cpu=initial,
                    conditioning=conditioning,
                    embedded_conditions=embedded,
                    condition_values=condition_values,
                    condition_mask=condition_mask,
                    target_probability=raw_probability.to(device),
                    target_core=core_cpu.to(device),
                    guidance_roi=roi_cpu.to(device),
                    settings=settings,
                    alpha=alpha,
                    sample_id=sample_id,
                )
                arm_dir = args.output_dir / arm
                arm_dir.mkdir(exist_ok=True)
                torch.save(decoded, arm_dir / f"sample_{sample_id}.pt")
                for row in trace:
                    row.update({"arm": arm, "source_seed": seed})
                traces[arm].extend(trace)
                outputs[arm] = decoded
            violations = {
                arm: int(((output != condition_values_cpu[0]) & condition_mask_cpu[0]).sum())
                for arm, output in outputs.items()
            }
            if any(violations.values()):
                raise RuntimeError(f"hard condition violation at seed {seed}")
            pair_rows.append({
                "sample_id": sample_id,
                "source_seed": seed,
                "initial_noise_sha256": tensor_sha256(initial),
                "baseline_sha256": tensor_sha256(outputs["FLOW_ONLY"]),
                "guided_sha256": tensor_sha256(outputs["GEO_EVIDENCE_GUIDED"]),
                "baseline_condition_violations": violations["FLOW_ONLY"],
                "guided_condition_violations": violations["GEO_EVIDENCE_GUIDED"],
            })
            print(f"Flow demo pair {sample_id + 1}/8 seed={seed}", flush=True)
        write_csv(args.output_dir / "pair_manifest.csv", pair_rows)
        for arm, rows in traces.items():
            write_csv(args.output_dir / arm / "guidance_trace.csv", rows)
        manifest.update({
            "run_status": "completed",
            "pair_count": len(pair_rows),
            "source_seeds": config["source_seeds"],
            "input_assets": {name: runtime.asset_record(path) for name, path in paths.items()},
            "model_load_report": model_report,
            "raw_probability_tensor_sha256": tensor_sha256(raw_probability),
            "coarse_guidance_target_shape": list(raw_probability.shape),
            "coarse_guidance_target_range": [float(raw_probability.min()), float(raw_probability.max())],
            "guidance_evidence": runtime.asset_record(args.output_dir / "guidance_evidence.pt"),
            "guidance_evidence_tensor_sha256": tensor_sha256(evidence_cpu),
            "guidance_evidence_range_subsurface": [float(evidence_cpu[subsurface_cpu].min()), float(evidence_cpu[subsurface_cpu].max())],
            "internal_dice_core_voxels": 0,
            "fine_voxel_repeat_used_in_loss": False,
            "hard_dice_core_used": False,
            "truth_loaded_by_flow_runner": False,
            "sample_selection_performed_by_flow_runner": False,
            "parameter_sweep_performed": False,
            "training_performed": False,
        })
        write_json(args.output_dir / "run_manifest.json", manifest)
    except Exception as exc:
        manifest.update({"run_status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        write_json(args.output_dir / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
