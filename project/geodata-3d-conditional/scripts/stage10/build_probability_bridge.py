#!/usr/bin/env python3
"""Build and freeze the three truth-blind Stage 10 probability bridges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Mapping

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import inference_runtime as runtime
from guidance.geophysical_probability_bridge import (
    scalar_gaussian_sample_bridge,
    validate_class_model,
    validate_grid_alignment,
    validate_probabilities,
)
from guidance.prior_ensemble import file_sha256, load_tensor_gzip, plain_rmse
from guidance.seismic import (
    hard_labels_to_acoustic,
    seismic_operator_from_config,
    tensor_sha256,
)
from guidance.seismic_inversion import (
    invert_acoustic_member,
    labels_to_clean_prior_acoustic,
    parse_inversion_config,
)
from scripts.stage10.common import (
    BRIDGE_CASE_SCHEMA,
    BRIDGE_COLLECTION_SCHEMA,
    EXPERIMENT_DIR,
    bridge_case_dir,
    inference_case_dir,
    load_frozen_config,
    load_stage10_inference_case,
    resolve_repository_path,
    stage9_pool_dir,
    target_probability_channel,
)
from scripts.stage9.common import (
    create_staging_directory,
    file_record,
    publish_staging_directory,
    read_csv,
    read_json,
    save_tensor_x,
    utc_now,
    write_csv_x,
    write_json_x,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPOSITORY_ROOT, text=True
    ).strip()


def _load_fixed_prior_members(
    pool_dir: Path,
    indices: list[int],
) -> tuple[torch.Tensor, list[dict[str, object]], dict[str, object]]:
    manifest = read_json(pool_dir / "manifest.json")
    if manifest.get("status") != "complete" or manifest.get("mode") != "formal":
        raise RuntimeError("Stage9A formal pool is incomplete")
    if manifest.get("truth_tensor_received") is not False:
        raise RuntimeError("Stage9A source pool violated the truth firewall")
    rows = read_csv(pool_dir / "candidate_pool.csv")
    row_index = {int(row["candidate_index"]): row for row in rows}
    chunks = manifest.get("model_chunks")
    if not isinstance(chunks, list):
        raise ValueError("Stage9A pool lacks model chunks")
    members: list[torch.Tensor] = []
    records: list[dict[str, object]] = []
    for index in indices:
        matches = [
            record
            for record in chunks
            if int(record["candidate_start"]) <= index
            < int(record["candidate_stop_exclusive"])
        ]
        if len(matches) != 1:
            raise ValueError(f"candidate {index} is not uniquely cached")
        record = matches[0]
        chunk = load_tensor_gzip(pool_dir / str(record["path"]), expected=record)
        offset = index - int(record["candidate_start"])
        member = chunk[offset : offset + 1].long().contiguous()
        row = row_index[index]
        if tensor_sha256(member.to(torch.int8)) != row["hard_model_sha256"]:
            raise ValueError(f"candidate model hash mismatch: {index}")
        if int(row["condition_violations"]) != 0:
            raise ValueError(f"source candidate violates hard conditions: {index}")
        members.append(member)
        records.append(
            {
                "candidate_id": row["candidate_id"],
                "candidate_index": index,
                "source_seed": int(row["source_seed"]),
                "source_noise_sha256": row["source_noise_sha256"],
                "hard_model_sha256": row["hard_model_sha256"],
                "model_chunk": file_record(pool_dir / str(record["path"]), relative_to=REPOSITORY_ROOT),
                "chunk_offset": offset,
            }
        )
    return torch.cat(members, dim=0), records, manifest


def _write_provenance_audit(
    config: Mapping[str, object],
    inversion_config_path: Path,
) -> None:
    cases = []
    for case_id in config["case_ids"]:
        case_dir = inference_case_dir(config, case_id)
        manifest = read_json(case_dir / "manifest.json")
        observation = manifest["tensors"]["observation_correct"]
        cases.append(
            {
                "case_id": case_id,
                "observed_seismic": {
                    "path": str((case_dir / observation["path"]).relative_to(REPOSITORY_ROOT)),
                    "file_sha256": observation["file_sha256"],
                    "tensor_sha256": observation["tensor_sha256"],
                    "shape": observation["shape"],
                    "dtype": observation["dtype"],
                },
                "forward_operator_config": "project/geodata-3d-conditional/experiments/stage4_seismic/configs/full_cube_noiseless_inverse_crime_v1.json",
                "inversion_method": "linearized post-stack log-impedance Tikhonov inversion reused from Phase 5a",
                "inversion_input_data": [
                    "observation_correct",
                    "surface_plus_three_fixed_native_boreholes_v1",
                    "registered acoustic codebook",
                    "fixed unranked Stage9A prior candidates 100-111"
                ],
                "initialization": "each of 12 fixed hard Flow members mapped through the registered acoustic codebook",
                "regularization": read_json(inversion_config_path),
                "output_posterior_representation": "12 log-impedance samples plus population mean/std",
                "property_units": "natural log of kg m^-2 s^-1",
                "grid_coordinates": {
                    "axis_order": config["axis_order"],
                    "grid_shape": config["grid_shape"],
                    "cell_size_m": [100.0, 100.0, 50.0]
                },
                "posterior_samples_available": True,
                "posterior_mean_std_available": True,
                "truth_geology_input": False,
                "truth_property_input": False,
                "truth_derived_measurement_only": True
            }
        )
    audit = {
            "schema": "stage10_property_inversion_provenance_v1",
            "status": "registered_before_bridge_truth_evaluation",
            "phase5a_method_source": file_record(PROJECT_DIR / "guidance/seismic_inversion.py", relative_to=REPOSITORY_ROOT),
            "inversion_config": file_record(inversion_config_path, relative_to=REPOSITORY_ROOT),
            "cases": cases,
            "only_observation_constrained_property_entering_bridge": "log_acoustic_impedance",
            "slowness_entering_bridge": False,
            "susceptibility_entering_bridge": False,
            "current_case_truth_used": False,
            "qualification": "truth-blind synthetic inverse-crime property posterior; not measured geophysics"
    }
    path = EXPERIMENT_DIR / "audit/property_inversion_provenance.json"
    if path.exists():
        if read_json(path) != audit:
            raise ValueError("existing property-inversion provenance audit differs")
    else:
        write_json_x(path, audit)


def _build_case(
    *,
    config: Mapping[str, object],
    class_model: Mapping[str, object],
    inversion_config: object,
    inversion_config_path: Path,
    case_id: str,
    device: torch.device,
) -> dict[str, object]:
    case_manifest, tensors = load_stage10_inference_case(config, case_id)
    pool_dir = stage9_pool_dir(config, case_id)
    prior_indices = [int(value) for value in config["property_inversion"]["prior_candidate_indices"]]
    prior_labels, candidate_records, pool_manifest = _load_fixed_prior_members(pool_dir, prior_indices)
    condition_values = tensors["condition_values"].long()
    condition_mask = tensors["condition_mask"].bool()
    subsurface = tensors["subsurface_mask"].bool()
    property_table = tensors["acoustic_property_table"].float()
    observed = tensors["observation_correct"].float()
    validate_grid_alignment(condition_values, condition_mask, subsurface, expected_shape=config["grid_shape"])
    if prior_labels.shape != (len(prior_indices), 1, *config["grid_shape"]):
        raise ValueError("fixed prior member stack has the wrong shape")
    expected_means = property_table[0].double().log()
    registered_means = torch.tensor(class_model["log_impedance_mean"], dtype=torch.float64)
    if not torch.allclose(expected_means, registered_means, rtol=0.0, atol=2e-12):
        raise ValueError("case property table differs from registered class means")
    seismic_config = read_json(
        PROJECT_DIR / "experiments/stage4_seismic/configs/full_cube_noiseless_inverse_crime_v1.json"
    )
    operator, seismic_metadata = seismic_operator_from_config(
        seismic_config, grid_shape=tuple(config["grid_shape"])
    )
    # Stage9A stores a complete background-filled condition-value tensor and a
    # separate authoritative sparse mask.  Values outside that mask are not
    # observations and must never be promoted to conditions by sentinel logic.
    condition_acoustic = hard_labels_to_acoustic(condition_values, property_table)
    if bool((~subsurface & ~condition_mask).any()):
        raise ValueError("known exterior air is not fully hard-conditioned")
    observed_device = observed.to(device)
    subsurface_device = subsurface.to(device)
    table_device = property_table.to(device)
    condition_acoustic_device = condition_acoustic.to(device)
    condition_mask_device = condition_mask.to(device)
    property_samples = []
    rows = []
    for local_index, (candidate_index, label) in enumerate(zip(prior_indices, prior_labels)):
        prior, cleanup = labels_to_clean_prior_acoustic(
            label.unsqueeze(0).to(device), table_device, subsurface_device
        )
        prior_exact, inverted, fields, diagnostics = invert_acoustic_member(
            prior,
            observed_seismic=observed_device,
            subsurface_mask=subsurface_device,
            condition_target=condition_acoustic_device,
            condition_mask=condition_mask_device,
            property_table=table_device,
            forward_operator=operator,
            config=inversion_config,
        )
        condition_expanded = condition_mask_device.expand_as(inverted[:, 0:1])
        property_condition_error = int(
            ((inverted[:, 0:1] != condition_acoustic_device[:, 0:1]) & condition_expanded).sum().item()
        )
        if property_condition_error:
            raise RuntimeError("property inversion changed an exact condition")
        property_samples.append(inverted[:, 0:1].log().detach().cpu().float())
        rows.append(
            {
                "member_id": local_index,
                "candidate_index": candidate_index,
                "source_seed": candidate_records[local_index]["source_seed"],
                "prior_seismic_rmse": float(plain_rmse(fields[0], observed_device).item()),
                "inverted_seismic_rmse": float(plain_rmse(fields[1], observed_device).item()),
                "delta_seismic_rmse": float((plain_rmse(fields[1], observed_device) - plain_rmse(fields[0], observed_device)).item()),
                "condition_violations": property_condition_error,
                **cleanup,
                **diagnostics,
            }
        )
    samples = torch.cat(property_samples, dim=0).contiguous()
    mean = samples.mean(dim=0, keepdim=True).contiguous()
    uncertainty = samples.std(dim=0, unbiased=False, keepdim=True).contiguous()
    probabilities, entropy = scalar_gaussian_sample_bridge(samples, class_model)
    air_channel = target_probability_channel(class_model, -1)
    outside = ~subsurface.bool()
    probabilities = torch.where(
        outside.expand_as(probabilities), torch.zeros_like(probabilities), probabilities
    )
    probabilities[:, air_channel : air_channel + 1] = torch.where(
        outside, torch.ones_like(probabilities[:, air_channel : air_channel + 1]), probabilities[:, air_channel : air_channel + 1]
    )
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
    probability_checks = validate_probabilities(probabilities)
    entropy = -(
        probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
        * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
    ).sum(dim=1, keepdim=True)
    label9_channel = target_probability_channel(class_model, int(config["target_label"]))
    label9 = probabilities[:, label9_channel : label9_channel + 1].contiguous()
    final_dir = bridge_case_dir(config, case_id)
    staging = create_staging_directory(final_dir)
    try:
        generated = {
            "property_samples": save_tensor_x(staging / "property_samples.pt", samples),
            "property_mean": save_tensor_x(staging / "property_mean.pt", mean),
            "property_uncertainty": save_tensor_x(staging / "property_uncertainty.pt", uncertainty),
            "probability_all_classes": save_tensor_x(staging / "probability_all_classes.pt", probabilities),
            "probability_label9": save_tensor_x(staging / "probability_label9.pt", label9),
            "entropy": save_tensor_x(staging / "entropy.pt", entropy),
        }
        write_csv_x(staging / "member_inversion_metrics.csv", rows)
        manifest = {
            "schema": BRIDGE_CASE_SCHEMA,
            "status": "complete_frozen_before_truth_evaluation",
            "case_id": case_id,
            "axis_order": config["axis_order"],
            "grid_shape": config["grid_shape"],
            "property": "natural_log_acoustic_impedance",
            "property_units": "ln(kg m^-2 s^-1)",
            "posterior_member_count": len(property_samples),
            "prior_candidate_indices": prior_indices,
            "prior_selection_used_ranking": False,
            "target_label": int(config["target_label"]),
            "target_probability_channel": label9_channel,
            "probability_checks": probability_checks,
            "property_uncertainty_minimum": float(uncertainty.min().item()),
            "property_uncertainty_maximum": float(uncertainty.max().item()),
            "truth_tensor_received": False,
            "truth_property_received": False,
            "truth_visible_selection": False,
            "frozen_config_sha256": file_sha256(EXPERIMENT_DIR / "configs/frozen_experiment_config.json"),
            "class_model_sha256": file_sha256(EXPERIMENT_DIR / "configs/petrophysical_class_model.json"),
            "source_assets": {
                "inference_case_manifest": file_record(inference_case_dir(config, case_id) / "manifest.json", relative_to=REPOSITORY_ROOT),
                "observed_seismic": file_record(inference_case_dir(config, case_id) / "observation_correct.pt", relative_to=REPOSITORY_ROOT),
                "stage9_pool_manifest": file_record(pool_dir / "manifest.json", relative_to=REPOSITORY_ROOT),
                "stage9_candidate_pool": file_record(pool_dir / "candidate_pool.csv", relative_to=REPOSITORY_ROOT),
                "inversion_config": file_record(inversion_config_path, relative_to=REPOSITORY_ROOT),
                "class_model": file_record(EXPERIMENT_DIR / "configs/petrophysical_class_model.json", relative_to=REPOSITORY_ROOT),
                "builder_source": file_record(Path(__file__), relative_to=REPOSITORY_ROOT),
                "bridge_source": file_record(PROJECT_DIR / "guidance/geophysical_probability_bridge.py", relative_to=REPOSITORY_ROOT),
            },
            "candidate_records": candidate_records,
            "seismic_metadata": seismic_metadata,
            "stage9_pool_truth_tensor_received": pool_manifest["truth_tensor_received"],
            "generated_tensors": generated,
            "member_metrics": file_record(
                staging / "member_inversion_metrics.csv", relative_to=staging
            ),
            "completed_at_utc": utc_now(),
        }
        write_json_x(staging / "manifest.json", manifest)
        publish_staging_directory(staging, final_dir)
    except Exception:
        # Keep non-empty staging for forensic inspection; never overwrite it.
        raise
    return {
        "manifest": file_record(final_dir / "manifest.json", relative_to=REPOSITORY_ROOT),
        "probability_label9_tensor_sha256": tensor_sha256(label9),
    }


def main() -> None:
    args = parse_args()
    config = load_frozen_config()
    class_model_path = EXPERIMENT_DIR / "configs/petrophysical_class_model.json"
    class_model = read_json(class_model_path)
    validate_class_model(class_model)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    bridge_root = EXPERIMENT_DIR / "bridge"
    if bridge_root.exists():
        raise FileExistsError(f"refusing to reuse immutable Stage10 bridge: {bridge_root}")
    inversion_config_path = resolve_repository_path(config["property_inversion"]["source_config"])
    inversion_config = parse_inversion_config(read_json(inversion_config_path))
    _write_provenance_audit(config, inversion_config_path)
    case_records = {}
    for case_id in config["case_ids"]:
        case_records[case_id] = _build_case(
            config=config,
            class_model=class_model,
            inversion_config=inversion_config,
            inversion_config_path=inversion_config_path,
            case_id=case_id,
            device=device,
        )
        print(json.dumps({"status": "bridge_frozen", "case_id": case_id}), flush=True)
    write_json_x(
        bridge_root / "manifest.json",
        {
            "schema": BRIDGE_COLLECTION_SCHEMA,
            "status": "complete_frozen_before_truth_evaluation",
            "case_ids": config["case_ids"],
            "case_manifests": {case_id: record["manifest"] for case_id, record in case_records.items()},
            "probability_label9_tensor_sha256": {
                case_id: record["probability_label9_tensor_sha256"]
                for case_id, record in case_records.items()
            },
            "truth_tensor_received": False,
            "truth_property_received": False,
            "git_head": _git("rev-parse", "HEAD"),
            "git_branch": _git("branch", "--show-current"),
            "git_status": _git("status", "--short"),
            "completed_at_utc": utc_now(),
        },
    )
    write_json_x(
        EXPERIMENT_DIR / "audit/leakage_audit.json",
        {
            "schema": "stage10_truth_firewall_audit_v1",
            "status": "bridge_frozen_before_retrospective_truth_load",
            "bridge_builder_accepts_truth_argument": False,
            "bridge_builder_loaded_truth_geology": False,
            "bridge_builder_loaded_truth_property": False,
            "truth_derived_input": "synthetic observed seismic only",
            "prior_member_selection": "fixed indices 100-111 without ranking",
            "case_truth_proportions_used": False,
            "truth_smoothing_or_bbox_used": False,
            "class_model_truth_tuned": False,
            "bridge_collection_manifest": file_record(bridge_root / "manifest.json", relative_to=REPOSITORY_ROOT),
        },
    )
    print(json.dumps({"status": "PASS_BRIDGE_FREEZE", "cases": config["case_ids"]}))


if __name__ == "__main__":
    main()
