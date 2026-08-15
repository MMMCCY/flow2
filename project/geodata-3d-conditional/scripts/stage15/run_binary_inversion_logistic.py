#!/usr/bin/env python3
"""Train/apply the fixed binary Stage15-G linear inversion mapper."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime
from guidance.binary_inversion_logistic import (
    FEATURE_NAMES,
    apply_linear_probability,
    binary_inversion_features,
    coarse_support_count_8,
    weighted_mean_std,
)
from guidance.inversion_score_probability import (
    coarse_truth_occupancy_8,
    upsample_inversion_score,
)
from guidance.seismic import tensor_sha256
from scripts.stage10.evaluate_bridge_information import average_precision
from scripts.stage15.common import (
    base_manifest,
    normalize_volume,
    read_json,
    refuse_nonempty,
    write_csv,
    write_json,
)
from scripts.stage15.run_inversion_score_probability import generate_case


EXPERIMENT_ROOT = PROJECT_DIR / "experiments/stage15_binary_seismic_consensus"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs/binary_inversion_logistic_8x8x8_v1.json"
DEFAULT_SOURCE = EXPERIMENT_ROOT / "inversion_probability/calibration_n128_8x8x8_v1"
DEFAULT_HELDOUT_SOURCE = EXPERIMENT_ROOT / "inversion_probability/cond_generation_0_8x8x8_v1"
DEFAULT_OBSERVATION_DIR = EXPERIMENT_ROOT / "observations/cond_generation_0"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "binary_logistic/calibration_n128_8x8x8_v2"
DEFAULT_HELDOUT_OUTPUT = EXPERIMENT_ROOT / "binary_logistic/cond_generation_0_8x8x8_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--heldout-source-dir", type=Path, default=DEFAULT_HELDOUT_SOURCE)
    parser.add_argument("--observation-dir", type=Path, default=DEFAULT_OBSERVATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--heldout-output-dir", type=Path, default=DEFAULT_HELDOUT_OUTPUT)
    return parser.parse_args()


def validate_protocol(config: Mapping[str, object]) -> None:
    expected = {
        "schema": "stage15_binary_inversion_logistic_8x8x8_v1",
        "status": "frozen_before_training",
        "source_calibration_case_count": 128,
        "training_case_indices": [0, 95],
        "validation_case_indices": [96, 127],
        "target": "binary_raw_label9_vs_every_other_subsurface_voxel",
        "coarse_grid_shape": [8, 8, 8],
        "feature_names": list(FEATURE_NAMES),
        "model": "single_linear_layer_4_to_1",
        "loss": "natural_prevalence_subsurface_voxel_weighted_BCEWithLogits",
        "class_balancing": False,
        "optimizer": "Adam",
        "learning_rate": 0.05,
        "training_steps": 1000,
        "parameter_sweep": False,
        "flow_used": False,
        "seismic_forward_rerun": False,
        "seismic_inversion_rerun": False,
        "heldout_truth_loaded_by_runner": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"frozen Stage15-G config mismatch: {key}")


def case_probability_metrics(
    probability_coarse: torch.Tensor,
    binary_truth: torch.Tensor,
    subsurface: torch.Tensor,
) -> dict[str, object]:
    probability = upsample_inversion_score(probability_coarse)
    active = subsurface.bool()
    scores = probability[active]
    labels = binary_truth.bool()[active]
    positive = int(labels.sum())
    return {
        "subsurface_voxels": int(labels.numel()),
        "label9_voxels": positive,
        "prevalence": float(labels.float().mean()),
        "auprc": average_precision(scores, labels) if positive else None,
        "brier": float((scores - labels.float()).square().mean()),
        "truth_mean_probability": float(scores[labels].mean()) if positive else None,
        "background_mean_probability": float(scores[~labels].mean()),
    }


def main() -> None:
    args = parse_args()
    refuse_nonempty(args.output_dir)
    refuse_nonempty(args.heldout_output_dir)
    config = read_json(args.config)
    validate_protocol(config)
    source_manifest_path = args.source_dir / "calibration_manifest.json"
    if runtime.file_sha256(source_manifest_path) != config["source_calibration_manifest_sha256"]:
        raise ValueError("frozen Stage15-F source manifest changed")
    source_manifest = read_json(source_manifest_path)
    records = source_manifest["case_records"]
    if len(records) != 128:
        raise ValueError("Stage15-G requires all 128 frozen Stage15-F cases")

    args.output_dir.mkdir(parents=True)
    manifest = base_manifest(
        "stage15_g_binary_inversion_logistic_calibration_v1", Path(__file__), args.config
    )
    write_json(args.output_dir / "manifest.json", manifest)
    started = time.perf_counter()
    case_data: list[dict[str, object]] = []
    source_hashes = {str(source_manifest_path.resolve()): runtime.file_sha256(source_manifest_path)}
    try:
        for index, record in enumerate(records):
            case_dir = args.source_dir / "cases" / f"case_{index:03d}"
            q_path = case_dir / "coarse_inversion_score.pt"
            q_true_path = case_dir / "coarse_truth_occupancy.pt"
            source_hashes[str(q_path.resolve())] = runtime.file_sha256(q_path)
            source_hashes[str(q_true_path.resolve())] = runtime.file_sha256(q_true_path)
            q = runtime.load_tensor(q_path).float()
            saved_q_true = runtime.load_tensor(q_true_path).float()
            if tensor_sha256(q) != record["coarse_inversion_score_tensor_sha256"]:
                raise ValueError(f"case {index} inversion score changed")
            geology, subsurface, _ = generate_case(int(record["root_seed"]))
            if tensor_sha256(geology) != record["geology_tensor_sha256"]:
                raise ValueError(f"case {index} deterministic geology replay changed")
            if tensor_sha256(subsurface) != record["subsurface_tensor_sha256"]:
                raise ValueError(f"case {index} deterministic subsurface replay changed")
            binary_truth = ((geology == 9) & subsurface).bool()
            q_true, _, support_count = coarse_truth_occupancy_8(binary_truth, subsurface)
            if tensor_sha256(q_true) != record["coarse_truth_occupancy_tensor_sha256"]:
                raise ValueError(f"case {index} binary coarse truth replay changed")
            if not torch.equal(q_true, saved_q_true):
                raise ValueError(f"case {index} saved/replayed binary target mismatch")
            features = binary_inversion_features(q, support_count)
            domain = support_count[0, 0] > 0
            case_data.append(
                {
                    "index": index,
                    "seed": int(record["root_seed"]),
                    "split": str(record["split"]),
                    "features": features[domain],
                    "target": q_true[0, 0][domain],
                    "weight": support_count[0, 0][domain].float(),
                    "binary_truth": binary_truth,
                    "subsurface": subsurface,
                    "full_features": features,
                    "domain": domain,
                }
            )
            if (index + 1) % 16 == 0:
                print(f"binary calibration replay: {index + 1}/128", flush=True)

        train = [case for case in case_data if case["split"] == "train"]
        validation = [case for case in case_data if case["split"] == "validation"]
        train_features = torch.cat([case["features"] for case in train])
        train_targets = torch.cat([case["target"] for case in train])
        train_weights = torch.cat([case["weight"] for case in train])
        feature_mean, feature_std = weighted_mean_std(train_features, train_weights)
        normalized = (train_features - feature_mean) / feature_std
        torch.manual_seed(int(config["random_seed"]))
        model = torch.nn.Linear(len(FEATURE_NAMES), 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
        trace: list[dict[str, object]] = []
        for step in range(int(config["training_steps"])):
            optimizer.zero_grad(set_to_none=True)
            logits = model(normalized).squeeze(-1)
            loss_values = F.binary_cross_entropy_with_logits(
                logits, train_targets, reduction="none"
            )
            loss = (loss_values * train_weights).sum() / train_weights.sum()
            loss.backward()
            optimizer.step()
            if step % 50 == 0 or step == int(config["training_steps"]) - 1:
                trace.append({"step": step, "training_bce": float(loss.detach())})
        write_csv(args.output_dir / "training_trace.csv", trace)
        weight = model.weight.detach().reshape(-1).cpu()
        bias = model.bias.detach().reshape(1).cpu()
        checkpoint = {
            "feature_names": list(FEATURE_NAMES),
            "feature_mean": feature_mean.cpu(),
            "feature_std": feature_std.cpu(),
            "linear_weight": weight,
            "linear_bias": bias,
        }
        torch.save(checkpoint, args.output_dir / "binary_logistic_checkpoint.pt")

        validation_rows: list[dict[str, object]] = []
        pooled_scores: list[torch.Tensor] = []
        pooled_labels: list[torch.Tensor] = []
        for case in validation:
            probability_values = apply_linear_probability(
                case["full_features"], feature_mean, feature_std, weight, bias
            )
            probability_coarse = torch.zeros((1, 1, 8, 8, 8))
            probability_coarse[0, 0][case["domain"]] = probability_values[case["domain"]]
            metrics = case_probability_metrics(
                probability_coarse, case["binary_truth"], case["subsurface"]
            )
            validation_rows.append(
                {"case_index": case["index"], "root_seed": case["seed"], **metrics}
            )
            fine_probability = upsample_inversion_score(probability_coarse)
            active = case["subsurface"].bool()
            pooled_scores.append(fine_probability[active])
            pooled_labels.append(case["binary_truth"][active])
        write_csv(args.output_dir / "validation_metrics.csv", validation_rows)
        pooled_score = torch.cat(pooled_scores)
        pooled_label = torch.cat(pooled_labels).bool()
        validation_summary = {
            "case_count": len(validation_rows),
            "label9_positive_case_count": sum(row["label9_voxels"] > 0 for row in validation_rows),
            "voxel_count": int(pooled_label.numel()),
            "label9_voxels": int(pooled_label.sum()),
            "prevalence": float(pooled_label.float().mean()),
            "auprc": average_precision(pooled_score, pooled_label),
            "brier": float((pooled_score - pooled_label.float()).square().mean()),
            "truth_mean_probability": float(pooled_score[pooled_label].mean()),
            "background_mean_probability": float(pooled_score[~pooled_label].mean()),
        }
        write_json(args.output_dir / "validation_summary.json", validation_summary)
        manifest.update(
            {
                "run_status": "completed",
                "training_case_count": len(train),
                "validation_case_count": len(validation),
                "binary_target_only": True,
                "natural_prevalence_weighting": True,
                "class_balancing_performed": False,
                "feature_names": list(FEATURE_NAMES),
                "final_training_bce": trace[-1]["training_bce"],
                "validation_summary": validation_summary,
                "checkpoint": runtime.asset_record(
                    args.output_dir / "binary_logistic_checkpoint.pt"
                ),
                "checkpoint_tensor_sha256": {
                    key: tensor_sha256(value) for key, value in checkpoint.items() if torch.is_tensor(value)
                },
                "source_file_sha256_before_and_after": source_hashes,
                "source_geology_replayed_for_binary_truth_and_support_only": True,
                "seismic_forward_rerun": False,
                "seismic_inversion_rerun": False,
                "flow_used": False,
                "parameter_sweep_performed": False,
                "runtime_seconds": time.perf_counter() - started,
            }
        )
        for path, expected in source_hashes.items():
            if runtime.file_sha256(Path(path)) != expected:
                raise RuntimeError(f"Stage15-G source changed: {path}")
        write_json(args.output_dir / "manifest.json", manifest)
    except Exception as exc:
        manifest.update({"run_status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        write_json(args.output_dir / "manifest.json", manifest)
        raise

    # Truth-blind held-out application: frozen score + subsurface + checkpoint only.
    args.heldout_output_dir.mkdir(parents=True)
    heldout_manifest = base_manifest(
        "stage15_g_binary_inversion_logistic_heldout_v1", Path(__file__), args.config
    )
    write_json(args.heldout_output_dir / "manifest.json", heldout_manifest)
    try:
        q_path = args.heldout_source_dir / "coarse_inversion_score.pt"
        subsurface_path = args.observation_dir / "subsurface_mask.pt"
        if runtime.file_sha256(q_path) != config["heldout_coarse_score_sha256"]:
            raise ValueError("held-out coarse inversion score changed")
        if runtime.file_sha256(subsurface_path) != config["heldout_subsurface_mask_sha256"]:
            raise ValueError("held-out subsurface mask changed")
        q = runtime.load_tensor(q_path).float()
        subsurface = normalize_volume(
            runtime.load_tensor(subsurface_path), "subsurface_mask"
        ).bool()
        support_count = coarse_support_count_8(subsurface)
        features = binary_inversion_features(q, support_count)
        probability_values = apply_linear_probability(
            features, feature_mean, feature_std, weight, bias
        )
        domain = support_count[0, 0] > 0
        probability_coarse = torch.zeros((1, 1, 8, 8, 8))
        probability_coarse[0, 0][domain] = probability_values[domain]
        probability_fine = upsample_inversion_score(probability_coarse)
        probability_fine = torch.where(subsurface, probability_fine, torch.zeros_like(probability_fine))
        torch.save(probability_coarse, args.heldout_output_dir / "coarse_label9_probability.pt")
        torch.save(probability_fine, args.heldout_output_dir / "label9_probability_volume.pt")
        heldout_manifest.update(
            {
                "run_status": "completed",
                "case_id": "cond_generation_0",
                "truth_loaded_by_runner": False,
                "binary_target_only": True,
                "coarse_inversion_score": runtime.asset_record(q_path),
                "subsurface_mask": runtime.asset_record(subsurface_path),
                "checkpoint": runtime.asset_record(
                    args.output_dir / "binary_logistic_checkpoint.pt"
                ),
                "coarse_probability": runtime.asset_record(
                    args.heldout_output_dir / "coarse_label9_probability.pt"
                ),
                "probability_volume": runtime.asset_record(
                    args.heldout_output_dir / "label9_probability_volume.pt"
                ),
                "probability_volume_tensor_sha256": tensor_sha256(probability_fine),
                "subsurface_probability_range": [
                    float(probability_fine[subsurface].min()),
                    float(probability_fine[subsurface].max()),
                ],
                "seismic_forward_rerun": False,
                "seismic_inversion_rerun": False,
                "flow_used": False,
            }
        )
        write_json(args.heldout_output_dir / "manifest.json", heldout_manifest)
    except Exception as exc:
        heldout_manifest.update(
            {"run_status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        )
        write_json(args.heldout_output_dir / "manifest.json", heldout_manifest)
        raise


if __name__ == "__main__":
    main()
