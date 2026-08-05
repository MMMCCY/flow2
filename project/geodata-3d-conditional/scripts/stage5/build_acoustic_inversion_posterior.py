#!/usr/bin/env python3
"""Build the truth-blind Phase-5a fixed-12 acoustic inversion posterior."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Mapping, Sequence

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import inference_runtime as runtime
from guidance.seismic import (
    seismic_field_loss,
    seismic_operator_from_config,
    tensor_sha256,
    validate_contiguous_subsurface_mask,
)
from guidance.seismic_inversion import (
    build_exact_condition_acoustic,
    invert_acoustic_member,
    labels_to_clean_prior_acoustic,
    parse_inversion_config,
    posterior_statistics,
)
from scripts.stage4.audit_seismic_identifiability import (
    _default_baseline_dirs,
    _validate_source_pool,
    validate_output_directory,
)
from scripts.stage4.run_seismic_guidance import read_json, write_json, write_rows


PHASE5A_BUILD_SCHEMA = "phase5a_acoustic_inversion_posterior_v1"
TRUTH_BLIND_OBSERVATION_FILES = (
    "acoustic_property_table.pt",
    "observed_seismic.pt",
    "noiseless_seismic.pt",
    "seismic_noise.pt",
    "sample_mask.pt",
    "subsurface_mask.pt",
    "uncertainty_amplitude.pt",
    "wavelet.pt",
)
OUTPUT_TENSOR_FILES = (
    "prior_acoustic_members.pt",
    "inverted_acoustic_members.pt",
    "prior_log_impedance_mean.pt",
    "prior_log_impedance_std.pt",
    "prior_slowness_mean.pt",
    "prior_slowness_std.pt",
    "prior_acoustic_mean.pt",
    "posterior_log_impedance_mean.pt",
    "posterior_log_impedance_std.pt",
    "posterior_slowness_mean.pt",
    "posterior_slowness_std.pt",
    "posterior_acoustic_mean.pt",
    "condition_mask.pt",
    "condition_acoustic.pt",
)


def parse_args() -> argparse.Namespace:
    experiment = PROJECT_DIR / "experiments/stage5_acoustic_inversion"
    parser = argparse.ArgumentParser(
        description="Build the fixed-12 model-based log-impedance posterior.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline-dir", action="append", type=Path, default=None)
    parser.add_argument(
        "--phase4c-anchor-dir",
        type=Path,
        default=PROJECT_DIR
        / "experiments/stage4_seismic/runs/cond_generation_0/phase4c_seismic_v1"
        / "seed42_n1_s32_a025_c025/baseline",
    )
    parser.add_argument(
        "--observation-dir",
        type=Path,
        default=PROJECT_DIR
        / "experiments/stage4_seismic/observations/cond_generation_0"
        / "distinct_upper_bound_v1_fix2",
    )
    parser.add_argument(
        "--boreholes",
        type=Path,
        default=PROJECT_DIR / "samples/jupyter-demo/cond_generation_0/boreholes.pt",
    )
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        default=PROJECT_DIR / "demo_model/conditional-weights.ckpt",
    )
    parser.add_argument(
        "--inversion-config",
        type=Path,
        default=experiment / "configs/model_based_log_impedance_v1.json",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=experiment
        / "outputs/cond_generation_0/model_based_fixed12_v1",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _recorded_path(record: Mapping[str, object]) -> Path:
    path = Path(str(record.get("path", "")))
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path


def _validate_record(record: Mapping[str, object], *, expected: Path | None = None) -> Path:
    path = expected or _recorded_path(record)
    if runtime.file_sha256(path) != record.get("sha256"):
        raise ValueError(f"source asset hash mismatch: {path}")
    return path


def load_truth_blind_observation_assets(
    observation_dir: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, object], object, dict[str, object]]:
    """Load Phase-4c inputs while explicitly refusing truth acoustic/geology."""
    manifest = read_json(observation_dir / "manifest.json")
    for field, expected in {
        "status": "complete",
        "truth_derived": True,
        "measured_geophysics": False,
        "inverse_crime": True,
    }.items():
        if manifest.get(field) != expected:
            raise ValueError(f"observation manifest {field} must be {expected!r}")
    source_assets = manifest.get("source_assets")
    if not isinstance(source_assets, Mapping):
        raise ValueError("observation manifest lacks source_assets")
    for name in ("truth_model", "seismic_source", "acoustic_config", "observation_config"):
        record = source_assets.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"observation manifest lacks source record: {name}")
        expected = PROJECT_DIR / "guidance/seismic.py" if name == "seismic_source" else None
        _validate_record(record, expected=expected)

    records = manifest.get("generated_tensors")
    if not isinstance(records, Mapping):
        raise ValueError("observation manifest lacks generated_tensors")
    tensors: dict[str, torch.Tensor] = {}
    for filename in TRUTH_BLIND_OBSERVATION_FILES:
        record = records.get(filename)
        if not isinstance(record, Mapping):
            raise ValueError(f"observation manifest lacks tensor: {filename}")
        value = runtime.load_tensor(observation_dir / filename, map_location="cpu")
        if list(value.shape) != record.get("shape") or str(value.dtype) != record.get("dtype"):
            raise ValueError(f"observation tensor shape/dtype mismatch: {filename}")
        if tensor_sha256(value) != record.get("sha256"):
            raise ValueError(f"observation tensor hash mismatch: {filename}")
        tensors[filename] = value

    resolved = manifest.get("observation_config_resolved")
    if not isinstance(resolved, Mapping):
        raise ValueError("observation manifest lacks resolved configuration")
    operator, validated = seismic_operator_from_config(
        resolved, grid_shape=tensors["subsurface_mask.pt"].shape[2:]
    )
    validate_contiguous_subsurface_mask(tensors["subsurface_mask.pt"])
    if not torch.equal(
        tensors["wavelet.pt"], operator.wavelet(torch.device("cpu"), torch.float64)
    ):
        raise ValueError("saved wavelet does not match the forward operator")
    if not torch.equal(
        tensors["observed_seismic.pt"],
        tensors["noiseless_seismic.pt"] + tensors["seismic_noise.pt"],
    ):
        raise ValueError("observed seismic does not equal noiseless plus noise")
    if bool(tensors["seismic_noise.pt"].any()):
        raise ValueError("Phase-5a v1 requires the frozen noiseless upper bound")
    if not bool(torch.all(tensors["sample_mask.pt"] == 1)):
        raise ValueError("Phase-5a v1 requires the frozen full sample mask")
    uncertainty = tensors["uncertainty_amplitude.pt"]
    if bool((uncertainty <= 0).any()) or not torch.allclose(
        uncertainty, uncertainty.reshape(-1)[0]
    ):
        raise ValueError("Phase-5a v1 requires finite constant uncertainty")
    table = tensors["acoustic_property_table.pt"]
    if table.ndim != 2 or table.shape[0] != 2 or bool((table <= 0).any()):
        raise ValueError("invalid acoustic property table")
    # Deliberately never load truth_acoustic.pt here.
    return tensors, manifest, operator, validated


def _condition_violation_count(
    acoustic: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> int:
    expanded = mask.to(acoustic.device).expand(acoustic.shape[0], 2, *mask.shape[2:])
    exact = target.to(acoustic).expand_as(acoustic)
    return int(((acoustic != exact) & expanded).any(dim=1).sum().item())


def _field_metrics(
    field: torch.Tensor,
    observed: torch.Tensor,
    sample_mask: torch.Tensor,
    uncertainty: torch.Tensor,
) -> dict[str, float]:
    loss, diagnostics = seismic_field_loss(field, observed, sample_mask, uncertainty)
    return {
        "loss": float(loss.detach().cpu()),
        "rmse": float(diagnostics["seismic_rmse_amplitude"].detach().cpu()),
        "mae": float(diagnostics["seismic_mae_amplitude"].detach().cpu()),
    }


def _tensor_record(path: Path, value: torch.Tensor) -> dict[str, object]:
    return {
        "path": str(path),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": runtime.file_sha256(path),
        "tensor_sha256": tensor_sha256(value),
        "size_bytes": path.stat().st_size,
    }


def _save_tensor(path: Path, value: torch.Tensor) -> dict[str, object]:
    cpu = value.detach().cpu().contiguous()
    torch.save(cpu, path)
    return _tensor_record(path, cpu)


def _save_statistics(
    output_dir: Path,
    prefix: str,
    statistics: Mapping[str, torch.Tensor],
    generated: dict[str, dict[str, object]],
) -> None:
    names = {
        "log_impedance_mean": f"{prefix}_log_impedance_mean.pt",
        "log_impedance_std": f"{prefix}_log_impedance_std.pt",
        "slowness_mean": f"{prefix}_slowness_mean.pt",
        "slowness_std": f"{prefix}_slowness_std.pt",
        "acoustic_mean": f"{prefix}_acoustic_mean.pt",
    }
    for key, filename in names.items():
        generated[filename] = _save_tensor(output_dir / filename, statistics[key])


def main() -> None:
    args = parse_args()
    validate_output_directory(args.output_dir, overwrite=args.overwrite)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu")

    assets, observation_manifest, operator, resolved_observation = (
        load_truth_blind_observation_assets(args.observation_dir)
    )
    inversion_config_raw = read_json(args.inversion_config)
    inversion_config = parse_inversion_config(inversion_config_raw)
    boreholes = runtime.normalize_single_geology(
        runtime.load_tensor(args.boreholes), str(args.boreholes)
    ).long()
    subsurface = assets["subsurface_mask.pt"].bool()
    if boreholes.shape != subsurface.shape:
        raise ValueError("boreholes and subsurface mask must have matching shapes")
    table = assets["acoustic_property_table.pt"]
    if int(boreholes.min()) < -1 or int(boreholes.max()) > table.shape[1] - 2:
        raise ValueError("boreholes contain labels outside the acoustic codebook")

    source_assets = observation_manifest["source_assets"]
    truth_hash = str(source_assets["truth_model"]["sha256"])
    boreholes_hash = runtime.file_sha256(args.boreholes)
    checkpoint_hash = runtime.file_sha256(args.ckpt_path)
    baseline_dirs: Sequence[Path] = args.baseline_dir or _default_baseline_dirs()
    candidates, source_records = _validate_source_pool(
        baseline_dirs,
        truth_hash=truth_hash,
        boreholes_hash=boreholes_hash,
        checkpoint_hash=checkpoint_hash,
        anchor_dir=args.phase4c_anchor_dir,
    )
    if len(candidates) != 12:
        raise ValueError("Phase-5a requires exactly the frozen 12-member pool")

    condition_target, condition_mask = build_exact_condition_acoustic(
        boreholes, subsurface, table
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    observed = assets["observed_seismic.pt"].to(device)
    sample_mask = assets["sample_mask.pt"].to(device)
    uncertainty = assets["uncertainty_amplitude.pt"].to(device)
    subsurface_device = subsurface.to(device)
    condition_target_device = condition_target.to(device)
    condition_mask_device = condition_mask.to(device)
    table_device = table.to(device)

    prior_members: list[torch.Tensor] = []
    inverted_members: list[torch.Tensor] = []
    rows: list[dict[str, object]] = []
    candidate_records: list[dict[str, object]] = []
    for global_id, candidate in enumerate(candidates):
        sample_path = Path(candidate["sample_path"])
        labels = runtime.normalize_single_geology(
            runtime.load_tensor(sample_path), str(sample_path)
        ).long().to(device)
        prior, cleanup = labels_to_clean_prior_acoustic(
            labels, table_device, subsurface_device
        )
        prior_exact, inverted, fields, diagnostics = invert_acoustic_member(
            prior,
            observed_seismic=observed,
            subsurface_mask=subsurface_device,
            condition_target=condition_target_device,
            condition_mask=condition_mask_device,
            property_table=table_device,
            forward_operator=operator,
            config=inversion_config,
        )
        before = _field_metrics(fields[0], observed, sample_mask, uncertainty)
        after = _field_metrics(fields[1], observed, sample_mask, uncertainty)
        candidate_id = str(candidate["candidate_id"])
        row: dict[str, object] = {
            "member_id": global_id,
            "candidate_id": candidate_id,
            "seed": int(candidate["seed"]),
            "local_sample_id": int(candidate["local_sample_id"]),
            "sample_sha256": str(candidate["sample_sha256"]),
            **cleanup,
            "prior_condition_violation_count": _condition_violation_count(
                prior_exact, condition_target_device, condition_mask_device
            ),
            "inverted_condition_violation_count": _condition_violation_count(
                inverted, condition_target_device, condition_mask_device
            ),
            "prior_seismic_loss": before["loss"],
            "prior_seismic_rmse": before["rmse"],
            "prior_seismic_mae": before["mae"],
            "inverted_seismic_loss": after["loss"],
            "inverted_seismic_rmse": after["rmse"],
            "inverted_seismic_mae": after["mae"],
            "delta_seismic_rmse": after["rmse"] - before["rmse"],
            **diagnostics,
        }
        rows.append(row)
        prior_members.append(prior_exact[0].detach().cpu())
        inverted_members.append(inverted[0].detach().cpu())
        candidate_records.append(
            {
                "member_id": global_id,
                "candidate_id": candidate_id,
                "seed": int(candidate["seed"]),
                "local_sample_id": int(candidate["local_sample_id"]),
                "sample": runtime.asset_record(sample_path),
            }
        )

    prior_stack = torch.stack(prior_members, dim=0)
    inverted_stack = torch.stack(inverted_members, dim=0)
    prior_stats = posterior_statistics(prior_stack)
    posterior_stats = posterior_statistics(inverted_stack)
    # Force exact mean/zero-spread conditions rather than relying on log/exp
    # roundoff at voxels that are identical in all members.
    mean_mask = condition_mask.expand(1, 2, *condition_mask.shape[2:])
    exact = condition_target.to(posterior_stats["acoustic_mean"])
    for stats in (prior_stats, posterior_stats):
        stats["acoustic_mean"] = torch.where(mean_mask, exact, stats["acoustic_mean"])
        stats["log_impedance_mean"] = torch.where(
            condition_mask,
            exact[:, 0:1].log(),
            stats["log_impedance_mean"],
        )
        stats["slowness_mean"] = torch.where(
            condition_mask, exact[:, 1:2], stats["slowness_mean"]
        )
        stats["log_impedance_std"] = torch.where(
            condition_mask,
            torch.zeros_like(stats["log_impedance_std"]),
            stats["log_impedance_std"],
        )
        stats["slowness_std"] = torch.where(
            condition_mask,
            torch.zeros_like(stats["slowness_std"]),
            stats["slowness_std"],
        )

    generated: dict[str, dict[str, object]] = {}
    generated["prior_acoustic_members.pt"] = _save_tensor(
        args.output_dir / "prior_acoustic_members.pt", prior_stack
    )
    generated["inverted_acoustic_members.pt"] = _save_tensor(
        args.output_dir / "inverted_acoustic_members.pt", inverted_stack
    )
    _save_statistics(args.output_dir, "prior", prior_stats, generated)
    _save_statistics(args.output_dir, "posterior", posterior_stats, generated)
    generated["condition_mask.pt"] = _save_tensor(
        args.output_dir / "condition_mask.pt", condition_mask
    )
    generated["condition_acoustic.pt"] = _save_tensor(
        args.output_dir / "condition_acoustic.pt", condition_target
    )
    if set(generated) != set(OUTPUT_TENSOR_FILES):
        raise AssertionError("internal Phase-5a output tensor list mismatch")
    write_rows(args.output_dir / "member_inversion_metrics.csv", rows)

    build_config = {
        "schema": PHASE5A_BUILD_SCHEMA,
        "status": "complete",
        "stage": "phase5a_no_training_acoustic_inversion_bridge",
        "description": "Truth-blind fixed-12 model-based log-impedance posterior.",
        "device": str(device),
        "member_count": len(candidates),
        "candidate_order": [record["candidate_id"] for record in candidate_records],
        "inversion_config": inversion_config_raw,
        "observation_config_resolved": resolved_observation,
        "truth_geology_loaded_by_builder": False,
        "truth_acoustic_loaded_by_builder": False,
        "unconstrained_truth_used_by_builder": False,
        "training_modified": False,
        "unet_modified": False,
        "checkpoint_loaded": False,
        "checkpoint_sha256": checkpoint_hash,
        "boreholes_sha256": boreholes_hash,
        "truth_model_expected_sha256": truth_hash,
        "condition_mask_sha256": tensor_sha256(condition_mask),
        "condition_acoustic_sha256": tensor_sha256(condition_target),
    }
    write_json(args.output_dir / "config.json", build_config)
    manifest = {
        "schema": PHASE5A_BUILD_SCHEMA,
        "status": "complete",
        "anti_leakage": {
            "truth_geology_loaded": False,
            "truth_acoustic_loaded": False,
            "unconstrained_truth_used": False,
            "builder_allowed_truth_information": (
                "truth file hash recorded by immutable observation manifest only"
            ),
        },
        "source_assets": {
            "observation_manifest": runtime.asset_record(
                args.observation_dir / "manifest.json"
            ),
            "boreholes": runtime.asset_record(args.boreholes),
            "checkpoint_not_loaded": runtime.asset_record(args.ckpt_path),
            "inversion_config": runtime.asset_record(args.inversion_config),
            "builder_source": runtime.asset_record(Path(__file__)),
            "inversion_source": runtime.asset_record(
                PROJECT_DIR / "guidance/seismic_inversion.py"
            ),
            "seismic_source": runtime.asset_record(PROJECT_DIR / "guidance/seismic.py"),
            "phase4d_source_records": source_records,
        },
        "candidate_records": candidate_records,
        "generated_tensors": generated,
        "member_metrics": runtime.asset_record(
            args.output_dir / "member_inversion_metrics.csv"
        ),
        "build_config": runtime.asset_record(args.output_dir / "config.json"),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(f"Phase-5a posterior complete: {args.output_dir}")
    print("Truth geology/acoustic loaded by builder: false/false")
    print(f"Members: {len(candidates)}")


if __name__ == "__main__":
    main()
