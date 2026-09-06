#!/usr/bin/env python3
"""Engineering/provenance guard for the frozen Stage20 formal checkpoint."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime
from scripts.stage15.common import base_manifest, read_json, refuse_nonempty, write_json
from scripts.stage20.common import CONFIG_DIR, ROOT, asset, require_config
from scripts.stage20.run_inference import load_validated_adapter_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", type=Path, default=ROOT / "checkpoints/formal_v1")
    parser.add_argument("--training-config", type=Path, default=CONFIG_DIR / "training_v1.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints/formal_v1_audit")
    return parser.parse_args()


def main() -> None:
    args = parse_args(); refuse_nonempty(args.output_dir)
    cfg = require_config(args.training_config, "stage20_continuous_impedance_adapter_training_v1")
    manifest_path = args.training_dir / "training_manifest.json"
    checkpoint_path = args.training_dir / "adapter_checkpoint.pt"
    manifest = read_json(manifest_path)
    payload = load_validated_adapter_checkpoint(
        manifest_path, checkpoint_path, args.training_config, cfg,
        require_formal=True, map_location="cpu",
    )
    trace_path = args.training_dir / "training_trace.csv"
    validation_path = args.training_dir / "validation_trace.csv"
    if not trace_path.is_file() or not validation_path.is_file():
        raise RuntimeError("FORMAL_ADAPTER_CHECKPOINT_INVALID: missing training traces")
    numeric_fields = ("total_loss", "flow_loss", "cross_entropy_loss", "dice_loss", "residual_regularizer", "gradient_norm")
    import csv
    with trace_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1024 or any(not math.isfinite(float(row[field])) for row in rows for field in numeric_fields):
        raise RuntimeError("FORMAL_ADAPTER_CHECKPOINT_INVALID: trace count or finite check")
    if any(row.get("gradients_finite") != "True" or row.get("base_gradients_absent") != "True" for row in rows):
        raise RuntimeError("FORMAL_ADAPTER_CHECKPOINT_INVALID: per-update gradient guard")
    state_dict = payload.get("adapter_state_dict")
    if not isinstance(state_dict, dict) or not state_dict or any(not torch.is_tensor(value) or not bool(torch.isfinite(value).all()) for value in state_dict.values()):
        raise RuntimeError("FORMAL_ADAPTER_CHECKPOINT_INVALID: adapter state_dict")
    result = base_manifest("stage20_formal_adapter_checkpoint_audit_v1", Path(__file__), args.training_config)
    result.update({
        "run_status": "completed",
        "machine_decision": "STAGE20_FORMAL_ADAPTER_TRAINING_VALIDATED",
        "formal_checkpoint_valid": True,
        "training_manifest": asset(manifest_path),
        "adapter_checkpoint": asset(checkpoint_path),
        "training_trace": asset(trace_path),
        "validation_trace": asset(validation_path),
        "optimizer_updates": len(rows),
        "adapter_parameter_count": payload["adapter_parameter_count"],
        "checkpoint_epoch": payload["epoch"],
        "base_model_hash_before": manifest["base_model_hash_before"],
        "base_model_hash_after": manifest["base_model_hash_after"],
        "base_state_dict_hash_before": manifest["base_state_dict_hash_before"],
        "base_state_dict_hash_after": manifest["base_state_dict_hash_after"],
        "no_formal_test_inference_executed": True,
    })
    args.output_dir.mkdir(parents=True); write_json(args.output_dir / "summary.json", result)
    print(result["machine_decision"])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if "FORMAL_ADAPTER_CHECKPOINT_INVALID" in str(exc):
            raise
        raise RuntimeError(f"FORMAL_ADAPTER_CHECKPOINT_INVALID: {exc}") from exc
