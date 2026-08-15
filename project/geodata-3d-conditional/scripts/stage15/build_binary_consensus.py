#!/usr/bin/env python3
"""Build fixed-threshold consensus from completed hard inversion members."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime
from guidance.seismic import tensor_sha256
from scripts.stage15.common import (
    base_manifest,
    normalize_volume,
    read_json,
    refuse_nonempty,
    write_json,
)


EXPERIMENT_ROOT = PROJECT_DIR / "experiments/stage15_binary_seismic_consensus"
DEFAULT_OBSERVATION = EXPERIMENT_ROOT / "observations/cond_generation_0"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs/consensus_v1.json"


def build_consensus(
    members: torch.Tensor,
    subsurface_mask: torch.Tensor,
    flow_condition_mask: torch.Tensor,
    positive_threshold: float = 0.8,
    negative_threshold: float = 0.2,
) -> dict[str, torch.Tensor]:
    """Return exact strong consensus tensors for ``members [N,1,X,Y,Z]``."""
    if members.ndim != 5 or members.shape[1] != 1:
        raise ValueError("members must have shape [N,1,X,Y,Z]")
    if members.shape[0] < 1 or not torch.isfinite(members).all():
        raise ValueError("members must be a non-empty finite tensor")
    if not torch.equal(members, members.round()) or bool(((members < 0) | (members > 1)).any()):
        raise ValueError("members must be hard binary")
    for value, name in ((subsurface_mask, "subsurface_mask"), (flow_condition_mask, "flow_condition_mask")):
        if value.shape != (1, *members.shape[1:]):
            raise ValueError(f"{name} must have shape [1,1,X,Y,Z]")
    if not (0 <= negative_threshold < positive_threshold <= 1):
        raise ValueError("consensus thresholds must satisfy 0 <= negative < positive <= 1")
    subsurface = subsurface_mask.bool()
    conditioned = flow_condition_mask.bool()
    frequency = members.float().mean(dim=0, keepdim=True)
    positive = (frequency >= float(positive_threshold)) & subsurface
    negative = (frequency <= float(negative_threshold)) & subsurface
    confidence = positive | negative
    unknown = subsurface & ~confidence
    target = positive.float()
    guidance_roi = confidence & subsurface & ~conditioned
    return {
        "occupancy_frequency": frequency,
        "consensus_target": target,
        "positive_mask": positive,
        "negative_mask": negative,
        "unknown_mask": unknown,
        "confidence_mask": confidence,
        "guidance_roi": guidance_roi,
        "target_core": positive & guidance_roi,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inversion-dir", type=Path, required=True)
    parser.add_argument("--observation-dir", type=Path, default=DEFAULT_OBSERVATION)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    refuse_nonempty(args.output_dir)
    config = read_json(args.config)
    if config.get("schema") != "stage15_binary_consensus_config_v1":
        raise ValueError("invalid consensus config")
    if config.get("threshold_sweep") is not False or config.get("preprocessing") is not None:
        raise ValueError("Stage15 consensus forbids preprocessing and threshold sweeps")
    manifest = base_manifest("stage15_binary_consensus_v1", Path(__file__), args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "consensus_manifest.json", manifest)
    try:
        inversion_manifest = read_json(args.inversion_dir / "manifest.json")
        if inversion_manifest.get("run_status") != "completed":
            raise ValueError("inversion ensemble is not completed")
        member_dirs = sorted(args.inversion_dir.glob("member_*"))
        if len(member_dirs) != int(inversion_manifest["n_members"]):
            raise ValueError("member count differs from inversion manifest")
        members: list[torch.Tensor] = []
        member_hashes: list[str] = []
        member_manifest_hashes: list[str] = []
        for member_dir in member_dirs:
            member_manifest = read_json(member_dir / "manifest.json")
            if member_manifest.get("run_status") != "completed":
                raise ValueError(f"incomplete member: {member_dir.name}")
            hard = normalize_volume(runtime.load_tensor(member_dir / "hard_binary.pt"), member_dir.name, torch.float32)
            expected = member_manifest["output_tensor_sha256"]["hard_binary.pt"]
            if tensor_sha256(hard) != expected:
                raise ValueError(f"member tensor hash changed: {member_dir.name}")
            members.append(hard)
            member_hashes.append(expected)
            member_manifest_hashes.append(runtime.file_sha256(member_dir / "manifest.json"))
        observation_manifest = read_json(args.observation_dir / "manifest.json")
        if observation_manifest.get("run_status") != "completed":
            raise ValueError("observation is not completed")
        subsurface = normalize_volume(runtime.load_tensor(args.observation_dir / "subsurface_mask.pt"), "subsurface_mask").bool()
        flow_mask = normalize_volume(runtime.load_tensor(args.observation_dir / "flow_condition_mask.pt"), "flow_condition_mask").bool()
        result = build_consensus(
            torch.cat(members, dim=0),
            subsurface,
            flow_mask,
            float(config["positive_threshold"]),
            float(config["negative_threshold"]),
        )
        output_names = (
            "occupancy_frequency",
            "consensus_target",
            "positive_mask",
            "negative_mask",
            "unknown_mask",
            "guidance_roi",
        )
        for name in output_names:
            torch.save(result[name].cpu(), args.output_dir / f"{name}.pt")
        subsurface_count = int(subsurface.sum())
        positive_count = int(result["positive_mask"].sum())
        negative_count = int(result["negative_mask"].sum())
        unknown_count = int(result["unknown_mask"].sum())
        manifest.update(
            {
                "run_status": "completed",
                "N": len(members),
                "member_hashes": member_hashes,
                "member_manifest_hashes": member_manifest_hashes,
                "thresholds": {
                    "positive": float(config["positive_threshold"]),
                    "negative": float(config["negative_threshold"]),
                },
                "positive_voxel_count": positive_count,
                "negative_voxel_count": negative_count,
                "unknown_voxel_count": unknown_count,
                "confidence_coverage": (positive_count + negative_count) / subsurface_count,
                "guidance_roi_voxel_count": int(result["guidance_roi"].sum()),
                "all_tensor_hashes": {name: tensor_sha256(result[name]) for name in output_names},
                "input_assets": {
                    "inversion_manifest": runtime.asset_record(args.inversion_dir / "manifest.json"),
                    "observation_manifest": runtime.asset_record(args.observation_dir / "manifest.json"),
                    "subsurface_mask": runtime.asset_record(args.observation_dir / "subsurface_mask.pt"),
                    "flow_condition_mask": runtime.asset_record(args.observation_dir / "flow_condition_mask.pt"),
                },
                "truth_loaded_by_runner": False,
                "unknown_is_lithology": False,
                "preprocessing": None,
            }
        )
        write_json(args.output_dir / "consensus_manifest.json", manifest)
    except Exception as exc:
        manifest.update({"run_status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        write_json(args.output_dir / "consensus_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
