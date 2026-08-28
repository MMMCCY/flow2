#!/usr/bin/env python3
"""Truth-blind Stage18B continuous-property-target single-arm runner."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime
from guidance.dflow import model_state_dict_hashes
from guidance.property_sampling import fixed_euler_property_sample
from guidance.property_volume import property_table_from_config
from guided_geophysical_sampling import soft_decode_to_probs
from guidance.probability_volume import tensor_sha256
from scripts.stage15.common import base_manifest, normalize_volume, read_json, refuse_nonempty, write_csv, write_json
from scripts.stage17.common import validate_asset_record
from scripts.stage18.common import NEW_ARM, load_case_tensors, load_stage18_references, reference_index


ROOT = PROJECT_DIR / "experiments/stage18_evidence_semantics"
DEFAULT_CONFIG = ROOT / "configs/continuous_property_target_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    refuse_nonempty(args.output_dir)
    stage18 = read_json(args.config)
    stage17, _, reference_records = load_stage18_references(stage18)
    reference = reference_index(reference_records)
    checkpoint = validate_asset_record(stage17["checkpoint"], "checkpoint")
    property_path = validate_asset_record(stage17["property_config"], "property config")
    device = torch.device(args.device or str(stage17["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    cases = stage17["cases"][:1] if args.smoke else stage17["cases"]
    seeds = stage17["source_seeds"][:1] if args.smoke else stage17["source_seeds"]

    from model_train_sh_inference_cond import Geo3DStochInterp

    model, model_report = runtime.load_model_with_weight_policy(
        model_class=Geo3DStochInterp,
        checkpoint_path=checkpoint,
        map_location=device,
        weight_source="ema",
    )
    model = model.to(device).eval()
    model_hash_before, tensor_hash_before = model_state_dict_hashes(model)
    table, channel_weights, property_metadata = property_table_from_config(
        read_json(property_path), model.num_categories
    )
    if table.shape[0] != 1 or float(table[0, 10]) != 1.0 or int(torch.count_nonzero(table)) != 1:
        raise ValueError("Stage18 requires the frozen single-channel label9=1 binary table")
    table = table.to(device)
    channel_weights = channel_weights.to(device)
    guidance = stage17["guidance"]

    args.output_dir.mkdir(parents=True)
    shutil.copyfile(args.config, args.output_dir / "config.json")
    manifest = base_manifest("stage18_continuous_property_target_run_v1", Path(__file__), args.config)
    manifest.update(
        {
            "run_status": "running",
            "run_class": "engineering_smoke" if args.smoke else "formal_semantics_attribution",
            "smoke_subset": bool(args.smoke),
            "truth_loaded_by_runner": False,
            "authorized_arm": NEW_ARM,
            "target_semantics": stage18["target_semantics"],
            "confidence_policy": stage18["confidence_policy"],
            "spatial_confidence_weighting_held_fixed": True,
        }
    )
    write_json(args.output_dir / "run_manifest.json", manifest)
    records: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []
    try:
        for case in cases:
            case_id = str(case["case_id"])
            tensors = load_case_tensors(case)
            condition_cpu = tensors["condition_values"]
            condition_mask_cpu = tensors["condition_mask"]
            support_cpu = tensors["support"]
            evidence_path = validate_asset_record(case["evidence"], f"{case_id}/evidence")
            validate_asset_record(case["evidence_case_manifest"], f"{case_id}/evidence manifest")
            score = normalize_volume(runtime.load_tensor(evidence_path), "evidence", torch.float32)
            score_sha = tensor_sha256(score)
            condition_values_sha = tensor_sha256(condition_cpu)
            condition_mask_sha = tensor_sha256(condition_mask_cpu)
            support_sha = tensor_sha256(support_cpu)
            free_subsurface_cpu = support_cpu & ~condition_mask_cpu
            condition_values = condition_cpu.to(device)
            condition_mask = condition_mask_cpu.to(device)
            embedded = model.embed(condition_values)
            conditioning = embedded * condition_mask.expand(-1, embedded.shape[1], -1, -1, -1)

            # Stage18's only scientific intervention: score becomes the target.
            # Stage17C's score-weighted spatial confidence remains byte-for-byte
            # semantically fixed, so low-score unresolved background is not
            # re-expanded into a dominant loss region.
            target_properties = score.to(device=device, dtype=table.dtype)
            confidence = (score * free_subsurface_cpu.float()).to(
                device=device, dtype=table.dtype
            )
            if not torch.equal(target_properties.cpu(), score.to(dtype=table.dtype)):
                raise RuntimeError("Stage18 target semantics construction failed")
            expected_confidence = score * free_subsurface_cpu.float()
            if not torch.equal(confidence.cpu(), expected_confidence.to(dtype=table.dtype)):
                raise RuntimeError("Stage17 spatial confidence weighting changed")

            for sample_id, seed in enumerate(seeds):
                baseline_reference = reference[(case_id, int(seed), "FLOW_ONLY")]
                guided_reference = reference[(case_id, int(seed), "TRAJECTORY_EVIDENCE")]
                if baseline_reference["evidence_tensor_sha256"] != score_sha or guided_reference[
                    "evidence_tensor_sha256"
                ] != score_sha:
                    raise ValueError(f"Stage17 evidence tensor mismatch: {case_id}/{seed}")
                generator = torch.Generator(device="cpu").manual_seed(int(seed))
                initial_cpu = torch.randn(
                    1,
                    model.embedding_dim,
                    *model.data_shape,
                    generator=generator,
                    dtype=embedded.dtype,
                )
                initial_sha = tensor_sha256(initial_cpu)
                if initial_sha != baseline_reference["initial_noise_sha256"] or initial_sha != guided_reference[
                    "initial_noise_sha256"
                ]:
                    raise ValueError(f"Stage17 initial-noise pairing failed: {case_id}/{seed}")
                final, trace = fixed_euler_property_sample(
                    model=model,
                    initial_state=initial_cpu.to(device),
                    conditioning=conditioning,
                    embedded_truth=embedded,
                    truth_model=condition_values,
                    condition_mask=condition_mask,
                    target_properties=target_properties,
                    property_table=table,
                    confidence=confidence,
                    property_sigmas=guidance["property_sigmas"],
                    property_scale_weights=guidance["property_scale_weights"],
                    property_channel_weights=channel_weights,
                    n_steps=int(stage17["n_steps"]),
                    alpha=float(guidance["alpha"]),
                    max_guidance_ratio=float(guidance["max_guidance_ratio"]),
                    tau_start=float(guidance["tau_start"]),
                    tau_end=float(guidance["tau_end"]),
                    tau_schedule=str(guidance["tau_schedule"]),
                    guidance_start=float(guidance["guidance_start"]),
                    guidance_schedule=str(guidance["guidance_schedule"]),
                    grad_clip_norm=float(guidance["grad_clip_norm"]),
                    guidance_scaling_mode=str(guidance["guidance_scaling_mode"]),
                    sample_id=sample_id,
                    loss_mode=str(guidance["property_loss_mode"]),
                )
                with torch.no_grad():
                    probabilities = soft_decode_to_probs(
                        final, model.embedding.weight, tau=float(guidance["tau_end"])
                    )
                    soft_label9 = probabilities[:, int(stage17["target_label"]) + 1]
                    decoded = (model.decode(final).detach().cpu() - 1).unsqueeze(1).long()
                violations = int(((decoded != condition_cpu) & condition_mask_cpu).sum())
                if violations:
                    raise RuntimeError(f"condition violation: {case_id}/{seed}/{NEW_ARM}")
                free = free_subsurface_cpu[:, 0]
                soft_mass = float(soft_label9.detach().cpu()[free].sum())
                hard_mass = int(((decoded[:, 0] == int(stage17["target_label"])) & free).sum())
                output_dir = args.output_dir / case_id / f"seed_{seed}" / NEW_ARM
                output_dir.mkdir(parents=True)
                final_cpu = final.detach().cpu().contiguous()
                soft_cpu = soft_label9.detach().cpu().unsqueeze(1).contiguous()
                torch.save(decoded, output_dir / "decoded_geology.pt")
                torch.save(final_cpu, output_dir / "final_state.pt")
                torch.save(soft_cpu, output_dir / "final_soft_label9_probability.pt")
                for row in trace:
                    traces.append({"case_id": case_id, "source_seed": int(seed), "arm": NEW_ARM, **row})
                records.append(
                    {
                        "case_id": case_id,
                        "source_seed": int(seed),
                        "sample_id": sample_id,
                        "arm": NEW_ARM,
                        "initial_noise_sha256": initial_sha,
                        "evidence_tensor_sha256": score_sha,
                        "condition_values_tensor_sha256": condition_values_sha,
                        "condition_mask_tensor_sha256": condition_mask_sha,
                        "subsurface_tensor_sha256": support_sha,
                        "checkpoint_sha256": stage17["checkpoint"]["sha256"],
                        "decoded_geology_sha256": tensor_sha256(decoded),
                        "final_state_sha256": tensor_sha256(final_cpu),
                        "final_soft_label9_probability_sha256": tensor_sha256(soft_cpu),
                        "condition_violations": violations,
                        "soft_label9_mass_free_subsurface": soft_mass,
                        "hard_label9_mass_free_subsurface": hard_mass,
                        "hard_minus_soft_mass_free_subsurface": hard_mass - soft_mass,
                        "decoded_path": str((output_dir / "decoded_geology.pt").relative_to(args.output_dir)),
                        "final_state_path": str((output_dir / "final_state.pt").relative_to(args.output_dir)),
                        "soft_label9_path": str(
                            (output_dir / "final_soft_label9_probability.pt").relative_to(args.output_dir)
                        ),
                    }
                )
                print(f"Stage18B {case_id} seed={seed} completed", flush=True)

        model_hash_after, tensor_hash_after = model_state_dict_hashes(model)
        if model_hash_before != model_hash_after or tensor_hash_before != tensor_hash_after:
            raise RuntimeError("frozen model changed during Stage18B")
        write_csv(args.output_dir / "sample_manifest.csv", records)
        write_csv(args.output_dir / "guidance_traces.csv", traces)
        manifest.update(
            {
                "run_status": "completed",
                "sample_arm_count": len(records),
                "executed_case_count": len(cases),
                "executed_source_seed_count_per_case": len(seeds),
                "model_load_report": model_report,
                "property_metadata": property_metadata,
                "model_state_dict_sha256_before": model_hash_before,
                "model_state_dict_sha256_after": model_hash_after,
                "model_state_dict_unchanged": True,
                "all_condition_violations_zero": True,
                "same_evidence_as_stage17c": True,
                "same_initial_noise_as_stage17c": True,
                "scientific_conclusion_allowed": not args.smoke,
                "training_performed": False,
                "parameter_sweep_performed": False,
                "thresholding_performed": False,
            }
        )
        write_json(args.output_dir / "run_manifest.json", manifest)
    except Exception as exc:
        manifest.update({"run_status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        write_json(args.output_dir / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
