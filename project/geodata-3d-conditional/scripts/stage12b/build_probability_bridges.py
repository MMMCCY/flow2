#!/usr/bin/env python3
"""Run the frozen truth-blind inversion and prior/post categorical bridges."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from guidance.geophysical_probability_bridge import (
    class_channel,
    scalar_gaussian_sample_bridge,
    validate_class_model,
    validate_grid_alignment,
    validate_probabilities,
)
from guidance.prior_ensemble import file_sha256, plain_rmse
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
from scripts.stage12b.common import (
    CASE_IDS,
    CONFIG_PATH,
    EXPERIMENT_DIR,
    bridge_case_dir,
    inference_case_dir,
    load_config,
    load_inference_case,
    load_prior_bank,
    resolve_path,
)
from scripts.stage9.common import (
    create_staging_directory,
    file_record,
    publish_staging_directory,
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


def _bridge_with_air(
    samples: torch.Tensor,
    class_model: dict[str, object],
    subsurface: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    probability, _ = scalar_gaussian_sample_bridge(samples, class_model)
    air = class_channel(class_model["raw_labels"], -1)
    outside = ~subsurface.cpu().bool()
    probability = torch.where(
        outside.expand_as(probability), torch.zeros_like(probability), probability
    )
    probability[:, air : air + 1] = torch.where(
        outside,
        torch.ones_like(probability[:, air : air + 1]),
        probability[:, air : air + 1],
    )
    probability = probability / probability.sum(dim=1, keepdim=True)
    checks = validate_probabilities(probability)
    clipped = probability.clamp_min(torch.finfo(probability.dtype).tiny)
    entropy = -(clipped * clipped.log()).sum(dim=1, keepdim=True)
    return probability.contiguous(), entropy.contiguous(), checks


def _build_case(
    case_id: str,
    *,
    config: dict[str, object],
    prior_labels: torch.Tensor,
    prior_manifest: dict[str, object],
    class_model: dict[str, object],
    inversion_config: object,
    seismic_config: dict[str, object],
    device: torch.device,
) -> dict[str, object]:
    inference_manifest, tensors = load_inference_case(case_id)
    condition_values = tensors["condition_values"].long()
    condition_mask = tensors["condition_mask"].bool()
    subsurface = tensors["subsurface_mask"].bool()
    property_table = tensors["acoustic_property_table"].float()
    observed = tensors["observation_correct"].float()
    validate_grid_alignment(
        condition_values, condition_mask, subsurface, expected_shape=config["grid_shape"]
    )
    if prior_labels.shape != (12, 1, *config["grid_shape"]):
        raise ValueError("common property prior bank has the wrong shape")
    expected_means = property_table[0].double().log()
    registered_means = torch.tensor(class_model["log_impedance_mean"], dtype=torch.float64)
    if not torch.allclose(expected_means, registered_means, rtol=0.0, atol=2e-12):
        raise ValueError("acoustic codebook differs from frozen class-model means")
    operator, seismic_metadata = seismic_operator_from_config(
        seismic_config, grid_shape=tuple(config["grid_shape"])
    )
    condition_acoustic = hard_labels_to_acoustic(condition_values, property_table)
    if bool((~subsurface & ~condition_mask).any()):
        raise ValueError("known exterior air is not fully hard-conditioned")

    observed_device = observed.to(device)
    subsurface_device = subsurface.to(device)
    table_device = property_table.to(device)
    condition_acoustic_device = condition_acoustic.to(device)
    condition_mask_device = condition_mask.to(device)
    prior_samples: list[torch.Tensor] = []
    posterior_samples: list[torch.Tensor] = []
    rows: list[dict[str, object]] = []
    member_records = prior_manifest["members"]
    for local_index, label in enumerate(prior_labels):
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
        active = condition_mask_device.expand_as(inverted[:, 0:1])
        violations = int(
            ((inverted[:, 0:1] != condition_acoustic_device[:, 0:1]) & active).sum().item()
        )
        if violations:
            raise RuntimeError("property inversion changed an exact condition")
        prior_samples.append(prior_exact[:, 0:1].log().detach().cpu().float())
        posterior_samples.append(inverted[:, 0:1].log().detach().cpu().float())
        before = plain_rmse(fields[0], observed_device)
        after = plain_rmse(fields[1], observed_device)
        rows.append(
            {
                "member_id": local_index,
                "candidate_index": member_records[local_index]["candidate_index"],
                "source_seed": member_records[local_index]["source_seed"],
                "prior_seismic_rmse": float(before.item()),
                "inverted_seismic_rmse": float(after.item()),
                "delta_seismic_rmse": float((after - before).item()),
                "condition_violations": violations,
                **cleanup,
                **diagnostics,
            }
        )
        print(f"{case_id}: inverted prior member {local_index + 1}/12", flush=True)

    prior_stack = torch.cat(prior_samples, dim=0).contiguous()
    post_stack = torch.cat(posterior_samples, dim=0).contiguous()
    prior_probability, prior_entropy, prior_checks = _bridge_with_air(
        prior_stack, class_model, subsurface
    )
    post_probability, post_entropy, post_checks = _bridge_with_air(
        post_stack, class_model, subsurface
    )
    label9_channel = class_channel(class_model["raw_labels"], int(config["target_label"]))
    final = bridge_case_dir(case_id)
    staging = create_staging_directory(final)
    generated = {
        "property_samples_prior": save_tensor_x(
            staging / "property_samples_prior.pt", prior_stack
        ),
        "property_mean_prior": save_tensor_x(
            staging / "property_mean_prior.pt", prior_stack.mean(0, keepdim=True)
        ),
        "property_spread_prior": save_tensor_x(
            staging / "property_spread_prior.pt",
            prior_stack.std(0, unbiased=False, keepdim=True),
        ),
        "property_samples_post": save_tensor_x(
            staging / "property_samples_post.pt", post_stack
        ),
        "property_mean_post": save_tensor_x(
            staging / "property_mean_post.pt", post_stack.mean(0, keepdim=True)
        ),
        "property_spread_post": save_tensor_x(
            staging / "property_spread_post.pt",
            post_stack.std(0, unbiased=False, keepdim=True),
        ),
        "probability_all_classes_prior": save_tensor_x(
            staging / "probability_all_classes_prior.pt", prior_probability
        ),
        "probability_label9_prior": save_tensor_x(
            staging / "probability_label9_prior.pt",
            prior_probability[:, label9_channel : label9_channel + 1],
        ),
        "entropy_prior": save_tensor_x(staging / "entropy_prior.pt", prior_entropy),
        "probability_all_classes_post": save_tensor_x(
            staging / "probability_all_classes_post.pt", post_probability
        ),
        "probability_label9_post": save_tensor_x(
            staging / "probability_label9_post.pt",
            post_probability[:, label9_channel : label9_channel + 1],
        ),
        "entropy_post": save_tensor_x(staging / "entropy_post.pt", post_entropy),
    }
    write_csv_x(staging / "member_inversion_metrics.csv", rows)
    write_json_x(
        staging / "manifest.json",
        {
            "schema": "stage12b_truth_blind_bridge_case_v1",
            "status": "complete_frozen_before_truth_evaluation",
            "created_at_utc": utc_now(),
            "case_id": case_id,
            "axis_order": config["axis_order"],
            "grid_shape": config["grid_shape"],
            "truth_tensor_received": False,
            "truth_property_received": False,
            "truth_visible_selection": False,
            "prior_member_count": 12,
            "prior_candidate_indices": prior_manifest["candidate_indices"],
            "prior_selection_used_ranking": False,
            "target_label": config["target_label"],
            "target_probability_channel": label9_channel,
            "property": "natural_log_acoustic_impedance",
            "property_units": "ln(kg m^-2 s^-1)",
            "bridge_schema": config["bridge"]["schema"],
            "prior_probability_checks": prior_checks,
            "post_probability_checks": post_checks,
            "seismic_metadata": seismic_metadata,
            "source_assets": {
                "protocol": file_record(CONFIG_PATH, relative_to=REPOSITORY_ROOT),
                "inference_case_manifest": file_record(
                    inference_case_dir(case_id) / "manifest.json", relative_to=REPOSITORY_ROOT
                ),
                "observation_correct": file_record(
                    inference_case_dir(case_id) / "observation_correct.pt",
                    relative_to=REPOSITORY_ROOT,
                ),
                "prior_bank_manifest": file_record(
                    EXPERIMENT_DIR / "prior_bank/manifest.json", relative_to=REPOSITORY_ROOT
                ),
                "inversion_config": file_record(
                    resolve_path(config["inversion_config"]["path"]),
                    relative_to=REPOSITORY_ROOT,
                ),
                "class_model": file_record(
                    resolve_path(config["class_model"]["path"]), relative_to=REPOSITORY_ROOT
                ),
                "builder_source": file_record(Path(__file__), relative_to=REPOSITORY_ROOT),
            },
            "inference_manifest_truth_available": inference_manifest[
                "geological_truth_available_to_inference"
            ],
            "member_records": member_records,
            "generated_tensors": generated,
            "member_metrics": file_record(
                staging / "member_inversion_metrics.csv", relative_to=staging
            ),
        },
    )
    publish_staging_directory(staging, final)
    return {
        "manifest": file_record(final / "manifest.json", relative_to=REPOSITORY_ROOT),
        "prior_label9_tensor_sha256": tensor_sha256(
            prior_probability[:, label9_channel : label9_channel + 1]
        ),
        "post_label9_tensor_sha256": tensor_sha256(
            post_probability[:, label9_channel : label9_channel + 1]
        ),
    }


def main() -> None:
    args = parse_args()
    config = load_config()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    class_model = read_json(resolve_path(config["class_model"]["path"]))
    validate_class_model(class_model)
    inversion_config = parse_inversion_config(
        read_json(resolve_path(config["inversion_config"]["path"]))
    )
    seismic_config = read_json(resolve_path(config["seismic_config"]["path"]))
    prior_manifest, prior_labels = load_prior_bank()
    bridge_root = EXPERIMENT_DIR / "bridge"
    if bridge_root.exists():
        raise FileExistsError(f"refusing to reuse immutable output: {bridge_root}")
    bridge_root.mkdir(parents=True)
    records = {
        case_id: _build_case(
            case_id,
            config=config,
            prior_labels=prior_labels,
            prior_manifest=prior_manifest,
            class_model=class_model,
            inversion_config=inversion_config,
            seismic_config=seismic_config,
            device=device,
        )
        for case_id in CASE_IDS
    }
    write_json_x(
        bridge_root / "manifest.json",
        {
            "schema": "stage12b_bridge_collection_v1",
            "status": "complete_frozen_before_truth_evaluation",
            "created_at_utc": utc_now(),
            "case_ids": list(CASE_IDS),
            "truth_tensor_received": False,
            "case_manifests": {key: value["manifest"] for key, value in records.items()},
            "prior_label9_tensor_sha256": {
                key: value["prior_label9_tensor_sha256"] for key, value in records.items()
            },
            "post_label9_tensor_sha256": {
                key: value["post_label9_tensor_sha256"] for key, value in records.items()
            },
            "protocol_sha256": file_sha256(CONFIG_PATH),
        },
    )
    print("Stage12B prior/post bridges frozen before retrospective truth evaluation")


if __name__ == "__main__":
    main()
