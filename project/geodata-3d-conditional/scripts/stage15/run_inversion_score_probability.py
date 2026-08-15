#!/usr/bin/env python3
"""Fit the frozen Stage15-F 8^3 inversion-score empirical P9 lookup."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
STRUCTURALGEO_SRC = REPOSITORY_ROOT / "StructuralGeo-main/src"
for root in (PROJECT_DIR, REPOSITORY_ROOT, STRUCTURALGEO_SRC):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime
from geogen.generation.model_generators import MarkovGeostoryGenerator
from geogen.generation.rng_contract import RNG_CONTRACT_VERSION
from guidance.binary_seismic_inversion import (
    binary_acoustic_properties_from_configs,
    binary_occupancy_to_acoustic,
)
from guidance.full_structuralgeo_benchmark import MODEL_BOUNDS, MODEL_RESOLUTION
from guidance.inversion_score_probability import (
    coarse_truth_occupancy_8,
    upsample_inversion_score,
)
from guidance.seismic import seismic_operator_from_config, tensor_sha256
from guidance.seismic_attribute_probability import (
    apply_probability_lookup,
    fit_empirical_probability_lookup,
    quantile_bin_edges,
)
from scripts.stage10.evaluate_bridge_information import average_precision
from scripts.stage15.common import (
    base_manifest,
    normalize_volume,
    read_json,
    refuse_nonempty,
    validate_asset,
    write_csv,
    write_json,
)
from scripts.stage15.run_direct_seismic_attribute_probability import (
    contiguous_subsurface_from_geology,
)


EXPERIMENT_ROOT = PROJECT_DIR / "experiments/stage15_binary_seismic_consensus"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs/inversion_score_probability_8x8x8_v1.json"
DEFAULT_BINARY_CONFIG = EXPERIMENT_ROOT / "configs/binary_acoustic_upper_bound_v1.json"
DEFAULT_SEISMIC_CONFIG = (
    PROJECT_DIR / "experiments/stage4_seismic/configs/full_cube_noiseless_inverse_crime_v1.json"
)
DEFAULT_OBSERVATION_DIR = EXPERIMENT_ROOT / "observations/cond_generation_0"
DEFAULT_CALIBRATION_OUTPUT = (
    EXPERIMENT_ROOT / "inversion_probability/calibration_n128_8x8x8_v1"
)
DEFAULT_HELDOUT_OUTPUT = (
    EXPERIMENT_ROOT / "inversion_probability/cond_generation_0_8x8x8_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--binary-acoustic-config", type=Path, default=DEFAULT_BINARY_CONFIG)
    parser.add_argument("--seismic-config", type=Path, default=DEFAULT_SEISMIC_CONFIG)
    parser.add_argument("--observation-dir", type=Path, default=DEFAULT_OBSERVATION_DIR)
    parser.add_argument("--calibration-output", type=Path, default=DEFAULT_CALIBRATION_OUTPUT)
    parser.add_argument("--heldout-output", type=Path, default=DEFAULT_HELDOUT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def validate_protocol(config: Mapping[str, object]) -> list[int]:
    expected = {
        "schema": "stage15_inversion_score_probability_8x8x8_v1",
        "status": "frozen_before_run",
        "grid_shape": [64, 64, 64],
        "coarse_grid_shape": [8, 8, 8],
        "fine_voxels_per_coarse_cell": [8, 8, 8],
        "target_label": 9,
        "calibration_case_count": 128,
        "training_case_count": 96,
        "validation_case_count": 32,
        "optimizer": "Adam",
        "learning_rate": 0.1,
        "number_of_iterations": 300,
        "initialization": "all_logits_zero",
        "loss": "raw_mean_squared_seismic_misfit",
        "regularization": None,
        "quantile_bin_count": 64,
        "class_balancing": False,
        "parameter_sweep": False,
        "flow_used": False,
        "heldout_truth_loaded_by_runner": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"frozen Stage15-F config mismatch: {key}")
    start = int(config["calibration_seed_start"])
    return list(range(start, start + int(config["calibration_case_count"])))


def generate_case(seed: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    generator = MarkovGeostoryGenerator(
        model_bounds=MODEL_BOUNDS,
        model_resolution=MODEL_RESOLUTION,
        config=None,
        root_seed=int(seed),
    )
    model, metadata = generator.generate_model_with_metadata()
    model.fill_nans()
    geology = torch.from_numpy(np.asarray(model.get_data_grid())).view(1, 1, 64, 64, 64)
    if not torch.isfinite(geology).all() or not torch.equal(geology, geology.round()):
        raise ValueError(f"StructuralGeo seed {seed} produced invalid geology")
    geology = geology.long()
    if int(geology.min()) < -1 or int(geology.max()) > 13:
        raise ValueError(f"StructuralGeo seed {seed} produced invalid raw labels")
    subsurface = contiguous_subsurface_from_geology(geology)
    return geology, subsurface, metadata


def invert_observation(
    observed: torch.Tensor,
    subsurface: torch.Tensor,
    *,
    operator: object,
    properties: object,
    learning_rate: float,
    iterations: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Deterministic inversion API; it receives observation and support only."""
    device = observed.device
    u = torch.zeros((1, 1, 8, 8, 8), device=device, dtype=observed.dtype, requires_grad=True)
    optimizer = torch.optim.Adam([u], lr=float(learning_rate))
    started = time.perf_counter()
    with torch.no_grad():
        q = torch.sigmoid(u)
        impedance, slowness = binary_occupancy_to_acoustic(
            upsample_inversion_score(q), subsurface, properties
        )
        initial_loss = float((operator(impedance, slowness, subsurface) - observed).square().mean())
    final_gradient_norm = 0.0
    for _ in range(int(iterations)):
        optimizer.zero_grad(set_to_none=True)
        q = torch.sigmoid(u)
        impedance, slowness = binary_occupancy_to_acoustic(
            upsample_inversion_score(q), subsurface, properties
        )
        loss = (operator(impedance, slowness, subsurface) - observed).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite Stage15-F inversion loss")
        loss.backward()
        final_gradient_norm = float(u.grad.norm().detach().cpu())
        if not math.isfinite(final_gradient_norm):
            raise FloatingPointError("non-finite Stage15-F gradient")
        optimizer.step()
    with torch.no_grad():
        q_final = torch.sigmoid(u).cpu().contiguous()
        impedance, slowness = binary_occupancy_to_acoustic(
            upsample_inversion_score(q_final.to(device)), subsurface, properties
        )
        final_loss = float((operator(impedance, slowness, subsurface) - observed).square().mean())
    return q_final, {
        "initial_seismic_mse": initial_loss,
        "final_seismic_mse": final_loss,
        "loss_reduction_fraction": (
            (initial_loss - final_loss) / initial_loss if initial_loss > 0 else 0.0
        ),
        "final_gradient_norm": final_gradient_norm,
        "runtime_seconds": time.perf_counter() - started,
    }


def probability_metrics(
    probability: torch.Tensor, binary_label: torch.Tensor, subsurface: torch.Tensor
) -> dict[str, object]:
    scores = probability[subsurface.bool()].float()
    labels = binary_label[subsurface.bool()].bool()
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
    refuse_nonempty(args.calibration_output)
    refuse_nonempty(args.heldout_output)
    config = read_json(args.config)
    seeds = validate_protocol(config)
    train_count = int(config["training_case_count"])

    input_paths = {
        "binary_acoustic_config": args.binary_acoustic_config,
        "seismic_config": args.seismic_config,
    }
    for name, path in input_paths.items():
        expected = str(config[f"{name}_sha256"])
        if runtime.file_sha256(path) != expected:
            raise ValueError(f"frozen Stage15-F input changed: {name}")
    binary_config = read_json(args.binary_acoustic_config)
    source_record = binary_config["source_acoustic_config"]
    source_path = REPOSITORY_ROOT / str(source_record["path"])
    validate_asset(source_path, str(source_record["sha256"]))
    properties = binary_acoustic_properties_from_configs(binary_config, read_json(source_path))
    operator, seismic_parameters = seismic_operator_from_config(
        read_json(args.seismic_config), grid_shape=(64, 64, 64)
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    args.calibration_output.mkdir(parents=True)
    manifest = base_manifest(
        "stage15_f_inversion_score_calibration_v1", Path(__file__), args.config
    )
    write_json(args.calibration_output / "calibration_manifest.json", manifest)
    started = time.perf_counter()
    train_scores: list[torch.Tensor] = []
    train_labels: list[torch.Tensor] = []
    validation_cases: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    case_records: list[dict[str, object]] = []
    try:
        for index, seed in enumerate(seeds):
            split = "train" if index < train_count else "validation"
            geology, subsurface, metadata = generate_case(seed)
            binary_label = ((geology == int(config["target_label"])) & subsurface).float()
            impedance, slowness = binary_occupancy_to_acoustic(
                binary_label.to(device), subsurface.to(device), properties
            )
            observed = operator(
                impedance, slowness, subsurface.to(device)
            ).detach()
            q_coarse, inversion_metrics = invert_observation(
                observed,
                subsurface.to(device),
                operator=operator,
                properties=properties,
                learning_rate=float(config["learning_rate"]),
                iterations=int(config["number_of_iterations"]),
            )
            q_fine = upsample_inversion_score(q_coarse)
            active = subsurface.bool()
            if split == "train":
                train_scores.append(q_fine[active])
                train_labels.append(binary_label.bool()[active])
            else:
                validation_cases.append((seed, q_fine, binary_label.bool(), subsurface.bool()))
            q_true, presence, support = coarse_truth_occupancy_8(
                binary_label.bool(), subsurface.bool()
            )
            case_dir = args.calibration_output / "cases" / f"case_{index:03d}"
            case_dir.mkdir(parents=True)
            torch.save(q_coarse, case_dir / "coarse_inversion_score.pt")
            torch.save(q_true, case_dir / "coarse_truth_occupancy.pt")
            record = {
                "case_index": index,
                "root_seed": seed,
                "split": split,
                "label9_voxels": int(binary_label.sum()),
                "target_containing_coarse_cells": int(presence[support > 0].sum()),
                "geology_tensor_sha256": tensor_sha256(geology),
                "subsurface_tensor_sha256": tensor_sha256(subsurface),
                "synthetic_seismic_tensor_sha256": tensor_sha256(observed.cpu()),
                "coarse_inversion_score_tensor_sha256": tensor_sha256(q_coarse),
                "coarse_truth_occupancy_tensor_sha256": tensor_sha256(q_true),
                "markov_sequence": metadata["markov_sequence"],
                **inversion_metrics,
            }
            write_json(case_dir / "metrics.json", record)
            case_records.append(record)
            print(
                f"calibration {index + 1}/128 seed={seed} split={split} "
                f"label9={record['label9_voxels']} mse={record['final_seismic_mse']:.8g}",
                flush=True,
            )

        pooled_score = torch.cat(train_scores)
        pooled_label = torch.cat(train_labels)
        edges = quantile_bin_edges(pooled_score, int(config["quantile_bin_count"]))
        lookup, bin_totals, bin_positives = fit_empirical_probability_lookup(
            pooled_score, pooled_label, edges
        )
        torch.save(edges, args.calibration_output / "inversion_score_bin_edges.pt")
        torch.save(lookup, args.calibration_output / "inversion_score_probability_lookup.pt")

        validation_rows: list[dict[str, object]] = []
        pooled_validation_scores: list[torch.Tensor] = []
        pooled_validation_labels: list[torch.Tensor] = []
        for seed, q_fine, binary_label, subsurface in validation_cases:
            probability = apply_probability_lookup(q_fine, subsurface, edges, lookup)
            row = {"root_seed": seed, **probability_metrics(probability, binary_label, subsurface)}
            validation_rows.append(row)
            pooled_validation_scores.append(probability[subsurface])
            pooled_validation_labels.append(binary_label[subsurface])
        write_csv(args.calibration_output / "validation_metrics.csv", validation_rows)
        pooled_validation_score = torch.cat(pooled_validation_scores)
        pooled_validation_label = torch.cat(pooled_validation_labels)
        validation_summary = {
            "case_count": len(validation_rows),
            "label9_positive_case_count": sum(row["label9_voxels"] > 0 for row in validation_rows),
            "subsurface_voxels": int(pooled_validation_label.numel()),
            "label9_voxels": int(pooled_validation_label.sum()),
            "prevalence": float(pooled_validation_label.float().mean()),
            "auprc": average_precision(pooled_validation_score, pooled_validation_label),
            "brier": float(
                (pooled_validation_score - pooled_validation_label.float()).square().mean()
            ),
            "truth_mean_probability": float(
                pooled_validation_score[pooled_validation_label].mean()
            ),
            "background_mean_probability": float(
                pooled_validation_score[~pooled_validation_label].mean()
            ),
        }
        write_json(args.calibration_output / "validation_summary.json", validation_summary)
        manifest.update(
            {
                "run_status": "completed",
                "calibration_seeds": seeds,
                "training_seeds": seeds[:train_count],
                "validation_seeds": seeds[train_count:],
                "case_records": case_records,
                "training_label9_positive_case_count": sum(
                    record["label9_voxels"] > 0 for record in case_records[:train_count]
                ),
                "training_subsurface_voxels": int(pooled_label.numel()),
                "training_label9_voxels": int(pooled_label.sum()),
                "training_prevalence": float(pooled_label.float().mean()),
                "validation_summary": validation_summary,
                "bin_total_counts": bin_totals.tolist(),
                "bin_label9_counts": bin_positives.tolist(),
                "empty_bins_due_to_ties": int((bin_totals == 0).sum()),
                "inversion_score_bin_edges": runtime.asset_record(
                    args.calibration_output / "inversion_score_bin_edges.pt"
                ),
                "inversion_score_probability_lookup": runtime.asset_record(
                    args.calibration_output / "inversion_score_probability_lookup.pt"
                ),
                "lookup_tensor_sha256": tensor_sha256(lookup),
                "binary_acoustic_config": runtime.asset_record(args.binary_acoustic_config),
                "source_acoustic_config": runtime.asset_record(source_path),
                "seismic_config": runtime.asset_record(args.seismic_config),
                "seismic_parameters": seismic_parameters,
                "generator": {
                    "name": "StructuralGeo MarkovGeostoryGenerator",
                    "rng_contract_version": RNG_CONTRACT_VERSION,
                    "model_bounds": [list(row) for row in MODEL_BOUNDS],
                    "model_resolution": list(MODEL_RESOLUTION),
                },
                "cond_generation_0_excluded_from_calibration": True,
                "class_balancing_performed": False,
                "parameter_sweep_performed": False,
                "flow_used": False,
                "runtime_seconds": time.perf_counter() - started,
            }
        )
        write_json(args.calibration_output / "calibration_manifest.json", manifest)
    except Exception as exc:
        manifest.update({"run_status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        write_json(args.calibration_output / "calibration_manifest.json", manifest)
        raise

    args.heldout_output.mkdir(parents=True)
    heldout_manifest = base_manifest(
        "stage15_f_heldout_inversion_score_probability_v1", Path(__file__), args.config
    )
    write_json(args.heldout_output / "manifest.json", heldout_manifest)
    try:
        observed_path = args.observation_dir / "observed_seismic.pt"
        subsurface_path = args.observation_dir / "subsurface_mask.pt"
        if runtime.file_sha256(observed_path) != config["observed_seismic_file_sha256"]:
            raise ValueError("held-out observed seismic file changed")
        if runtime.file_sha256(subsurface_path) != config["subsurface_mask_file_sha256"]:
            raise ValueError("held-out subsurface file changed")
        observed_cpu = runtime.load_tensor(observed_path).float()
        subsurface_cpu = normalize_volume(
            runtime.load_tensor(subsurface_path), "subsurface_mask"
        ).bool()
        if tensor_sha256(observed_cpu) != config["observed_seismic_tensor_sha256"]:
            raise ValueError("held-out observed seismic tensor changed")
        if tensor_sha256(subsurface_cpu) != config["subsurface_mask_tensor_sha256"]:
            raise ValueError("held-out subsurface tensor changed")
        heldout_q, heldout_inversion_metrics = invert_observation(
            observed_cpu.to(device),
            subsurface_cpu.to(device),
            operator=operator,
            properties=properties,
            learning_rate=float(config["learning_rate"]),
            iterations=int(config["number_of_iterations"]),
        )
        heldout_score = upsample_inversion_score(heldout_q)
        heldout_probability = apply_probability_lookup(
            heldout_score, subsurface_cpu, edges, lookup
        )
        torch.save(heldout_q, args.heldout_output / "coarse_inversion_score.pt")
        torch.save(
            heldout_probability,
            args.heldout_output / "inversion_score_probability_volume.pt",
        )
        heldout_manifest.update(
            {
                "run_status": "completed",
                "case_id": "cond_generation_0",
                "truth_loaded_by_runner": False,
                "observed_seismic": runtime.asset_record(observed_path),
                "subsurface_mask": runtime.asset_record(subsurface_path),
                "calibration_manifest": runtime.asset_record(
                    args.calibration_output / "calibration_manifest.json"
                ),
                "coarse_inversion_score": runtime.asset_record(
                    args.heldout_output / "coarse_inversion_score.pt"
                ),
                "probability_volume": runtime.asset_record(
                    args.heldout_output / "inversion_score_probability_volume.pt"
                ),
                "coarse_inversion_score_tensor_sha256": tensor_sha256(heldout_q),
                "probability_volume_tensor_sha256": tensor_sha256(heldout_probability),
                "inversion_metrics": heldout_inversion_metrics,
                "flow_used": False,
                "thresholding_performed": False,
                "parameter_sweep_performed": False,
            }
        )
        write_json(args.heldout_output / "manifest.json", heldout_manifest)
    except Exception as exc:
        heldout_manifest.update(
            {"run_status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        )
        write_json(args.heldout_output / "manifest.json", heldout_manifest)
        raise


if __name__ == "__main__":
    main()
