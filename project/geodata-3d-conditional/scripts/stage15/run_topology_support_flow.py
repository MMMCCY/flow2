#!/usr/bin/env python3
"""Run fixed-seed Flow-only, seismic-bridge, or oracle topology arms."""

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
from guidance.binary_trace_boundary import refine_binary_trace_volume
from guidance.property_sampling import fixed_euler_property_sample
from guidance.property_volume import property_table_from_config
from guidance.probability_volume import tensor_sha256
from guidance.seismic import seismic_operator_from_config
from guidance.seismic_inversion import ModelBasedInversionConfig
from scripts.stage14.run_gansim_style_geo_guidance import _sample_one
from scripts.stage15.common import base_manifest, normalize_volume, read_json, refuse_nonempty, write_csv, write_json

ROOT = PROJECT_DIR / "experiments/stage15_topology_support"
ARMS = ("FLOW_ONLY", "SEISMIC_GUIDED", "ORACLE_GUIDED")


def _path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/frozen_protocol_v1.json")
    parser.add_argument("--cases-dir", type=Path, default=ROOT / "cases_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs_v1")
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol = read_json(args.protocol)
    if protocol.get("status") != "frozen_before_execution" or args.arm not in protocol["arms"]:
        raise ValueError("invalid frozen topology protocol/arm")
    arm_root = args.output_dir / args.arm
    refuse_nonempty(arm_root)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    from model_train_sh_inference_cond import Geo3DStochInterp
    checkpoint = _path(protocol["checkpoint"])
    model, model_report = runtime.load_model_with_weight_policy(
        model_class=Geo3DStochInterp, checkpoint_path=checkpoint, map_location=device, weight_source="ema"
    )
    model = model.to(device)
    property_config_path = _path(protocol["property_config"])
    property_table, channel_weights, _ = property_table_from_config(read_json(property_config_path), model.num_categories)
    binary_config_path = _path(protocol["binary_acoustic_config"])
    binary_config = read_json(binary_config_path)
    source_path = _path(binary_config["source_acoustic_config"]["path"])
    properties = binary_acoustic_properties_from_configs(binary_config, read_json(source_path))
    seismic_config_path = _path(protocol["seismic_config"])
    trace_config_path = _path(protocol["trace_inversion_config"])
    trace_config = read_json(trace_config_path)
    manifest = base_manifest("stage15_topology_support_flow_run_v1", Path(__file__), args.protocol)
    arm_root.mkdir(parents=True)
    write_json(arm_root / "manifest.json", manifest)
    records = []
    try:
        for case_id in protocol["cases"]:
            case_dir = args.cases_dir / case_id
            condition_cpu = normalize_volume(runtime.load_tensor(case_dir / "condition_values.pt"), "condition").long()
            condition_mask_cpu = normalize_volume(runtime.load_tensor(case_dir / "condition_mask.pt"), "condition_mask").bool()
            support_cpu = normalize_volume(runtime.load_tensor(case_dir / "subsurface_mask.pt"), "support").bool()
            condition = condition_cpu.to(device)
            condition_mask = condition_mask_cpu.to(device)
            embedded = model.embed(condition)
            conditioning = embedded * condition_mask.expand(-1, embedded.shape[1], -1, -1, -1)
            roi = support_cpu & ~condition_mask_cpu
            seismic_score = None
            if args.arm == "SEISMIC_GUIDED":
                observed = runtime.load_tensor(case_dir / "observed_seismic.pt").float().to(device)
                operator, _ = seismic_operator_from_config(read_json(seismic_config_path), grid_shape=(64, 64, 64))
                inversion_config = ModelBasedInversionConfig(
                    "stage15_topology_trace_bridge_v1",
                    float(trace_config["prior_relative_weight"]),
                    float(trace_config["vertical_smoothness_relative_weight"]),
                )
                seismic_score, _, _, inversion_trace = refine_binary_trace_volume(
                    observed, support_cpu.to(device), operator, properties, inversion_config,
                    int(trace_config["refinement_passes"]),
                )
                case_output = arm_root / case_id
                case_output.mkdir(parents=True, exist_ok=True)
                torch.save(seismic_score, case_output / "seismic_inversion_score.pt")
                write_csv(case_output / "inversion_trace.csv", inversion_trace)
            oracle = None
            if args.arm == "ORACLE_GUIDED":
                oracle = normalize_volume(runtime.load_tensor(case_dir / "truth_restricted/binary_truth.pt"), "oracle").float()
            traces = []
            for sample_id, seed in enumerate(protocol["source_seeds"]):
                generator = torch.Generator(device="cpu").manual_seed(int(seed))
                initial = torch.randn(1, model.embedding_dim, *model.data_shape, generator=generator, dtype=embedded.dtype)
                if args.arm in {"FLOW_ONLY", "ORACLE_GUIDED"}:
                    target = torch.zeros_like(support_cpu, dtype=torch.float32) if oracle is None else oracle
                    target_core = target >= 0.5
                    decoded, trace = _sample_one(
                        model=model, initial_cpu=initial, conditioning=conditioning,
                        embedded_conditions=embedded, condition_values=condition,
                        condition_mask=condition_mask, target_probability=target.to(device),
                        target_core=target_core.to(device), guidance_roi=roi.to(device),
                        settings={**protocol["flow"], **protocol["guidance"]},
                        alpha=0.0 if args.arm == "FLOW_ONLY" else float(protocol["guidance"]["alpha"]),
                        sample_id=sample_id,
                    )
                else:
                    target_properties = torch.ones((1, 1, 64, 64, 64), dtype=torch.float32)
                    confidence = seismic_score.float() * roi.float()
                    final, trace = fixed_euler_property_sample(
                        model=model, initial_state=initial.to(device), conditioning=conditioning,
                        embedded_truth=embedded, truth_model=condition, condition_mask=condition_mask,
                        target_properties=target_properties, property_table=property_table,
                        confidence=confidence, property_sigmas=protocol["guidance"]["property_sigmas"],
                        property_scale_weights=protocol["guidance"]["property_scale_weights"],
                        property_channel_weights=channel_weights, n_steps=int(protocol["flow"]["n_euler_steps"]),
                        alpha=float(protocol["guidance"]["alpha"]),
                        max_guidance_ratio=float(protocol["guidance"]["max_guidance_ratio"]),
                        tau_start=float(protocol["guidance"]["tau_start"]), tau_end=float(protocol["guidance"]["tau_end"]),
                        tau_schedule=protocol["guidance"]["tau_schedule"], guidance_start=float(protocol["guidance"]["guidance_start"]),
                        guidance_schedule=protocol["guidance"]["guidance_schedule"], grad_clip_norm=float(protocol["guidance"]["grad_clip_norm"]),
                        guidance_scaling_mode=protocol["guidance"]["guidance_scaling_mode"], sample_id=sample_id,
                    )
                    decoded = (model.decode(final).detach().cpu() - 1)[0].long()
                violations = int(((decoded != condition_cpu[0]) & condition_mask_cpu[0]).sum())
                if violations:
                    raise RuntimeError(f"condition violation: {case_id}/{seed}")
                output = arm_root / case_id / f"seed_{seed}.pt"
                output.parent.mkdir(parents=True, exist_ok=True)
                torch.save(decoded, output)
                for row in trace:
                    row.update({"case_id": case_id, "arm": args.arm, "source_seed": int(seed)})
                traces.extend(trace)
                records.append({
                    "case_id": case_id, "arm": args.arm, "source_seed": int(seed),
                    "initial_noise_sha256": tensor_sha256(initial), "sample_sha256": tensor_sha256(decoded),
                    "condition_violations": violations,
                })
            write_csv(arm_root / case_id / "guidance_trace.csv", traces)
        write_csv(arm_root / "sample_manifest.csv", records)
        manifest.update({
            "run_status": "completed", "arm": args.arm, "sample_count": len(records),
            "source_seeds": protocol["source_seeds"], "model_load_report": model_report,
            "truth_loaded_by_runner": args.arm == "ORACLE_GUIDED",
            "observed_seismic_loaded_by_runner": args.arm == "SEISMIC_GUIDED",
            "sample_selection_performed": False, "training_performed": False, "parameter_sweep": False,
        })
        write_json(arm_root / "manifest.json", manifest)
    except Exception as exc:
        manifest.update({"run_status": "failed", "arm": args.arm, "error": f"{type(exc).__name__}: {exc}"})
        write_json(arm_root / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
