#!/usr/bin/env python3
"""Build truth-blind Phase2-style property assets from Stage15-H scores."""

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
from guidance.property_volume import property_table_from_config
from guidance.probability_volume import tensor_sha256
from scripts.stage15.common import read_json, refuse_nonempty, write_json

ROOT = PROJECT_DIR / "experiments/stage15_binary_seismic_consensus"
DEFAULT_INVERSION = ROOT / "trace_boundary/cond_generation_0_v1"
DEFAULT_OBSERVATION = ROOT / "observations/cond_generation_0"
DEFAULT_CONFIG = ROOT / "configs/binary_trace_property_indicator_v1.json"
DEFAULT_OUTPUT = ROOT / "trace_boundary/property_assets_cond_generation_0_v4"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inversion-dir", type=Path, default=DEFAULT_INVERSION)
    parser.add_argument("--observation-dir", type=Path, default=DEFAULT_OBSERVATION)
    parser.add_argument("--property-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    refuse_nonempty(args.output_dir)
    inversion_manifest = read_json(args.inversion_dir / "manifest.json")
    if inversion_manifest.get("truth_loaded_by_runner") is not False:
        raise ValueError("source inversion is not truth-blind")
    score = runtime.load_tensor(args.inversion_dir / "binary_impedance_score.pt").float()
    support = runtime.load_tensor(args.observation_dir / "subsurface_mask.pt").bool()
    condition_values = runtime.load_tensor(args.observation_dir / "flow_condition_values.pt").long()
    condition_mask = runtime.load_tensor(args.observation_dir / "flow_condition_mask.pt").bool()
    table, _, _ = property_table_from_config(read_json(args.property_config), 15)
    channels = table.shape[0]
    background = table[:, 1].view(1, channels, 1, 1, 1)
    target = table[:, 10].view(1, channels, 1, 1, 1)
    air = table[:, 0].view(1, channels, 1, 1, 1)
    # Seismic observes interfaces, not a calibrated interior occupancy. Treat
    # every detected anomaly as target-endpoint property evidence and retain
    # the continuous inversion score solely as its confidence weight.
    properties = target.expand_as(score.expand(-1, channels, -1, -1, -1)).clone()
    properties = torch.where(support.expand_as(properties), properties, air.expand_as(properties))
    # The inversion score is both the continuous binary property estimate and
    # its positive-evidence confidence. This prevents the overwhelmingly large
    # unresolved background from suppressing Flow target generation.
    confidence = score * (support & ~condition_mask).float()
    args.output_dir.mkdir(parents=True)
    tensors = {
        "property_table.pt": table,
        "target_properties.pt": properties.contiguous(),
        "property_confidence.pt": confidence.contiguous(),
        "condition_mask.pt": condition_mask.contiguous(),
    }
    records = {}
    for name, value in tensors.items():
        path = args.output_dir / name
        torch.save(value, path)
        records[name] = {
            "sha256": runtime.file_sha256(path),
            "tensor_sha256": tensor_sha256(value),
            "shape": list(value.shape),
        }
    write_json(args.output_dir / "manifest.json", {
        "schema": "stage15_binary_trace_property_assets_v1",
        "status": "complete",
        "truth_geology_loaded": False,
        "truth_acoustic_loaded": False,
        "truth_metrics_used_for_construction": False,
        "trace_boundary_evaluation_used_as_stop_gate": True,
        "binary_target": "raw_label9_vs_identical_background",
        "score_transform": "known_binary_log_impedance_endpoint_anomaly_strength_no_empirical_rescaling",
        "target_property_policy": "label9_binary_endpoint_where_continuous_confidence_is_nonzero_v1",
        "confidence_policy": "continuous_positive_binary_impedance_score_no_threshold_v1",
        "property_config": runtime.asset_record(args.property_config),
        "source_inversion_manifest": runtime.asset_record(args.inversion_dir / "manifest.json"),
        "generated_tensors": records,
    })


if __name__ == "__main__":
    main()
