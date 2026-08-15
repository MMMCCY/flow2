#!/usr/bin/env python3
"""Build a checkpoint-compatible Stage7-style five-label9-body benchmark."""

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
from guidance.binary_seismic_inversion import binary_acoustic_properties_from_configs, binary_occupancy_to_acoustic
from guidance.seismic import build_seismic_observation, seismic_operator_from_config, tensor_sha256
from scripts.stage15.common import base_manifest, normalize_volume, read_json, refuse_nonempty, write_json

ROOT = PROJECT_DIR / "experiments/stage15_five_body_flow"


def _path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def body_masks(protocol: dict[str, object]) -> tuple[torch.Tensor, list[dict[str, object]]]:
    masks = []
    records = []
    for body in protocol["bodies"]:
        mask = torch.zeros((64, 64, 64), dtype=torch.bool)
        start, stop = body["start"], body["stop"]
        mask[start[0] : stop[0], start[1] : stop[1], start[2] : stop[2]] = True
        masks.append(mask)
        records.append(dict(body))
    return torch.stack(masks), records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/frozen_protocol_v1.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "cases_v1/FIVE_BODY")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    refuse_nonempty(args.output_dir)
    protocol = read_json(args.protocol)
    if protocol.get("schema") != "stage15_five_body_flow_protocol_v1" or protocol.get("status") != "frozen_before_execution":
        raise ValueError("five-body protocol is not frozen")
    base = _path(protocol["base_case"])
    original = normalize_volume(runtime.load_tensor(base / "true_model.pt"), "truth").long()
    original_boreholes = normalize_volume(runtime.load_tensor(base / "boreholes.pt"), "boreholes").long()
    support = original != -1
    truth = original.clone()
    truth[truth == 9] = int(protocol["background_replacement_label"])
    boreholes = original_boreholes.clone()
    boreholes[boreholes == 9] = int(protocol["background_replacement_label"])
    masks, records = body_masks(protocol)
    if int(masks.sum(0).max()) != 1 or len(set(int(mask.sum()) for mask in masks)) != 1:
        raise ValueError("five bodies must be disjoint and equal-volume")
    if bool((masks & ~support[0, 0]).any()):
        raise ValueError("a registered body leaves the observed subsurface support")
    target = masks.any(0).view(1, 1, 64, 64, 64)
    truth[target] = 9
    full_well_columns = []
    for x in range(64):
        for y in range(64):
            if int(((original_boreholes[0, 0, x, y] != -1) & support[0, 0, x, y]).sum()) > 3:
                full_well_columns.append((x, y))
    drilled_ids, hidden_ids = [], []
    for index, body in enumerate(records):
        intersections = [(x, y) for x, y in full_well_columns if bool(masks[index, x, y].any())]
        if body["role"] == "drilled":
            if intersections != [tuple(body["well_xy"])]:
                raise ValueError(f"drilled body has unexpected well intersections: {body['id']}={intersections}")
            drilled_ids.append(body["id"])
            x, y = body["well_xy"]
            boreholes[0, 0, x, y, masks[index, x, y]] = 9
        else:
            if intersections:
                raise ValueError(f"hidden body intersects a well: {body['id']}={intersections}")
            hidden_ids.append(body["id"])
    condition_mask = (~support) | (boreholes != -1)
    validation = runtime.validate_conditioning_pair(truth, boreholes, num_categories=15, target_label=9)
    binary_config = read_json(_path(protocol["binary_acoustic_config"]))
    source_path = _path(binary_config["source_acoustic_config"]["path"])
    properties = binary_acoustic_properties_from_configs(binary_config, read_json(source_path))
    operator, seismic_metadata = seismic_operator_from_config(read_json(_path(protocol["seismic_config"])), grid_shape=(64, 64, 64))
    impedance, slowness = binary_occupancy_to_acoustic(target.float().to(args.device), support.to(args.device), properties)
    observation = build_seismic_observation(
        torch.cat((impedance, slowness), dim=1), support.to(args.device), operator,
        uncertainty_amplitude=float(seismic_metadata["uncertainty_amplitude"]),
        noise_std_amplitude=float(seismic_metadata["noise"]["std_amplitude"]),
        noise_seed=int(seismic_metadata["noise"]["seed"]),
    )
    hidden = masks[[index for index, body in enumerate(records) if body["role"] == "hidden"]].any(0).view(1, 1, 64, 64, 64)
    args.output_dir.mkdir(parents=True)
    truth_dir = args.output_dir / "truth_restricted"
    truth_dir.mkdir()
    inference = {
        "condition_values.pt": boreholes,
        "condition_mask.pt": condition_mask,
        "subsurface_mask.pt": support,
        "observed_seismic.pt": observation.values.cpu(),
    }
    restricted = {"true_model.pt": truth, "binary_truth.pt": target, "hidden_binary_truth.pt": hidden, "body_masks.pt": masks}
    for name, tensor in inference.items():
        torch.save(tensor, args.output_dir / name)
    for name, tensor in restricted.items():
        torch.save(tensor, truth_dir / name)
    manifest = base_manifest("stage15_five_body_flow_case_v1", Path(__file__), args.protocol)
    manifest.update({
        "run_status": "completed", "risk_gate_passed": True,
        "risk_gate": {
            "grid_shape_64_cubed": list(truth.shape[2:]) == [64, 64, 64],
            "checkpoint_compatible_background": True,
            "all_five_targets_raw_label9": True,
            "body_count": len(records), "equal_body_voxels": int(masks[0].sum()),
            "disjoint": True, "drilled_body_ids": drilled_ids, "hidden_body_ids": hidden_ids,
            "hidden_well_intersection_voxels": int((hidden & condition_mask).sum()),
            "condition_mismatches": validation["nonair_borehole_truth_mismatches"],
            "same_stage15_binary_physics": True,
        },
        "bodies": records, "conditioning_report": validation,
        "inference_tensor_sha256": {name: tensor_sha256(value) for name, value in inference.items()},
        "truth_tensor_sha256": {name: tensor_sha256(value) for name, value in restricted.items()},
        "observation_metadata": observation.metadata,
        "truth_role": "observation_generation_and_retrospective_evaluation_only",
    })
    write_json(args.output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
