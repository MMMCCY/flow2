"""Shared runtime checks for conditional geology inference.

This module centralizes checkpoint weight selection, immutable asset
fingerprints, and validation of the truth/borehole conditioning pair.  Keeping
these operations in one place prevents the reference and guided inference
paths from silently using different model weights or inputs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch


PROTOCOL_VERSION = 1
PAIRED_INTEGRATOR = "fixed_euler_midpoint_v1"
INITIAL_NOISE_POLICY = "single_cpu_generator_sequential_samples_v1"


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 digest for one immutable experiment asset."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"asset not found: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset_record(path: Optional[Path]) -> Optional[Dict[str, object]]:
    """Return a JSON-safe path, size, and SHA-256 record."""
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"asset not found: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _torch_load_checkpoint(path: Path, map_location: object) -> Mapping[str, Any]:
    """Load a trusted Lightning checkpoint across supported PyTorch versions."""
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"checkpoint must contain a mapping, got {type(checkpoint).__name__}: {path}"
        )
    return checkpoint


def load_tensor(path: Path, map_location: object = "cpu") -> torch.Tensor:
    """Load a tensor-only ``.pt`` asset without enabling arbitrary globals."""
    try:
        value = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        value = torch.load(path, map_location=map_location)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected a torch.Tensor in {path}, got {type(value).__name__}")
    return value


def load_model_with_weight_policy(
    model_class,
    checkpoint_path: Path,
    map_location: object,
    weight_source: str = "ema",
):
    """Load a Lightning model and apply a validated raw/EMA weight policy.

    The training callback tracks only trainable parameters.  In this model the
    categorical embedding is frozen, so ``embedding.weight`` is intentionally
    absent from ``ema_shadow`` and remains at its checkpoint value.
    """
    checkpoint_path = Path(checkpoint_path)
    if weight_source not in {"ema", "raw"}:
        raise ValueError("weight_source must be 'ema' or 'raw'")

    model = model_class.load_from_checkpoint(
        str(checkpoint_path),
        map_location=map_location,
    )
    checkpoint = _torch_load_checkpoint(checkpoint_path, map_location="cpu")

    parameters = dict(model.named_parameters())
    trainable_names = {
        name for name, parameter in parameters.items() if parameter.requires_grad
    }
    frozen_names = set(parameters) - trainable_names
    ema_shadow = checkpoint.get("ema_shadow")
    ema_applied = False
    ema_missing_trainable: list[str] = []
    ema_unexpected: list[str] = []
    ema_shape_mismatches: list[str] = []

    if weight_source == "ema":
        if not isinstance(ema_shadow, Mapping) or not ema_shadow:
            raise ValueError(
                "EMA weights were requested but checkpoint has no non-empty ema_shadow"
            )
        ema_names = set(ema_shadow)
        ema_missing_trainable = sorted(trainable_names - ema_names)
        ema_unexpected = sorted(ema_names - set(parameters))
        for name in sorted(ema_names & set(parameters)):
            shadow = ema_shadow[name]
            if not torch.is_tensor(shadow):
                raise TypeError(f"ema_shadow[{name!r}] is not a tensor")
            if tuple(shadow.shape) != tuple(parameters[name].shape):
                ema_shape_mismatches.append(name)

        if ema_missing_trainable or ema_shape_mismatches:
            raise ValueError(
                "EMA coverage is invalid: "
                f"missing_trainable={ema_missing_trainable[:10]}, "
                f"shape_mismatches={ema_shape_mismatches[:10]}"
            )

        with torch.no_grad():
            for name, parameter in parameters.items():
                if name in ema_shadow:
                    parameter.copy_(
                        ema_shadow[name].to(
                            device=parameter.device,
                            dtype=parameter.dtype,
                        )
                    )
        ema_applied = True

    report = {
        "checkpoint": asset_record(checkpoint_path),
        "weight_source": weight_source,
        "ema_applied": ema_applied,
        "model_parameter_count": len(parameters),
        "trainable_parameter_count": len(trainable_names),
        "frozen_parameters": sorted(frozen_names),
        "ema_entry_count": len(ema_shadow) if isinstance(ema_shadow, Mapping) else 0,
        "ema_missing_trainable": ema_missing_trainable,
        "ema_unexpected": ema_unexpected,
        "ema_shape_mismatches": ema_shape_mismatches,
        "ema_excluded_frozen_parameters": (
            sorted(frozen_names - set(ema_shadow))
            if isinstance(ema_shadow, Mapping)
            else sorted(frozen_names)
        ),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_global_step": checkpoint.get("global_step"),
    }
    del checkpoint
    model.eval()
    return model, report


def normalize_single_geology(volume: torch.Tensor, name: str) -> torch.Tensor:
    """Normalize a single categorical volume to ``[1, 1, X, Y, Z]``."""
    if not isinstance(volume, torch.Tensor):
        raise TypeError(f"{name} must contain a torch.Tensor")
    if volume.ndim == 3:
        volume = volume.unsqueeze(0).unsqueeze(0)
    elif volume.ndim == 4:
        if volume.shape[0] != 1:
            raise ValueError(f"{name} must contain one geology volume")
        volume = volume.unsqueeze(0)
    elif volume.ndim != 5 or volume.shape[:2] != (1, 1):
        raise ValueError(
            f"{name} must have shape [X,Y,Z], [1,X,Y,Z], or [1,1,X,Y,Z]"
        )
    return volume


def _label_counts(volume: torch.Tensor) -> Dict[str, int]:
    labels, counts = torch.unique(volume.detach().cpu().long(), return_counts=True)
    return {
        str(int(label.item())): int(count.item())
        for label, count in zip(labels, counts)
    }


def validate_conditioning_pair(
    truth: torch.Tensor,
    boreholes: torch.Tensor,
    num_categories: int,
    target_label: Optional[int] = None,
) -> Dict[str, object]:
    """Validate categorical ranges and exact truth/borehole agreement."""
    truth = normalize_single_geology(truth, "truth_model")
    boreholes = normalize_single_geology(boreholes, "boreholes")
    if truth.shape != boreholes.shape:
        raise ValueError(
            "truth_model and boreholes must have matching shapes, got "
            f"{tuple(truth.shape)} and {tuple(boreholes.shape)}"
        )
    if not torch.isfinite(truth).all() or not torch.isfinite(boreholes).all():
        raise ValueError("truth_model and boreholes must contain only finite values")
    if not torch.equal(truth, truth.round()) or not torch.equal(
        boreholes, boreholes.round()
    ):
        raise ValueError("truth_model and boreholes must be integer-valued")

    min_label = -1
    max_label = int(num_categories) - 2
    truth_min = int(truth.min().item())
    truth_max = int(truth.max().item())
    boreholes_min = int(boreholes.min().item())
    boreholes_max = int(boreholes.max().item())
    if truth_min < min_label or truth_max > max_label:
        raise ValueError(
            f"truth labels must be in [{min_label}, {max_label}], "
            f"got [{truth_min}, {truth_max}]"
        )
    if boreholes_min < min_label or boreholes_max > max_label:
        raise ValueError(
            f"borehole labels must be in [{min_label}, {max_label}], "
            f"got [{boreholes_min}, {boreholes_max}]"
        )

    observed_nonair = boreholes != -1
    mismatches = observed_nonair & (boreholes != truth)
    mismatch_count = int(mismatches.sum().item())
    if mismatch_count:
        raise ValueError(
            f"boreholes disagree with truth at {mismatch_count} non-air voxels"
        )

    truth_air = truth == -1
    effective_condition = observed_nonair | truth_air
    spatial_voxels = int(truth.numel())

    truth_3d = truth[0, 0]
    boreholes_3d = boreholes[0, 0]
    full_columns = torch.all(boreholes_3d == truth_3d, dim=-1)
    full_columns &= torch.any(truth_3d != -1, dim=-1)
    borehole_xy = [
        [int(value) for value in coordinate.tolist()]
        for coordinate in torch.nonzero(full_columns, as_tuple=False).cpu()
    ]

    report: Dict[str, object] = {
        "shape": list(truth.shape),
        "dtype": str(truth.dtype),
        "valid_label_range": [min_label, max_label],
        "truth_label_counts": _label_counts(truth),
        "borehole_label_counts": _label_counts(boreholes),
        "nonair_observed_voxels": int(observed_nonair.sum().item()),
        "nonair_observed_fraction": float(observed_nonair.float().mean().item()),
        "truth_air_voxels": int(truth_air.sum().item()),
        "effective_condition_voxels": int(effective_condition.sum().item()),
        "effective_condition_fraction": float(
            effective_condition.float().mean().item()
        ),
        "unconstrained_subsurface_voxels": spatial_voxels
        - int(effective_condition.sum().item()),
        "full_borehole_count": len(borehole_xy),
        "full_borehole_xy": borehole_xy,
        "nonair_borehole_truth_mismatches": mismatch_count,
    }

    if target_label is not None:
        target = truth == int(target_label)
        target_count = int(target.sum().item())
        target_conditioned = int((target & effective_condition).sum().item())
        full_columns_5d = full_columns[None, None, :, :, None].expand_as(truth)
        report.update(
            {
                "target_label": int(target_label),
                "target_voxels": target_count,
                "target_fraction": target_count / spatial_voxels,
                "target_conditioned_voxels": target_conditioned,
                "target_conditioned_fraction": (
                    target_conditioned / target_count if target_count else None
                ),
                "target_hits_in_full_boreholes": int(
                    (target & full_columns_5d).sum().item()
                ),
                "borehole_columns_hitting_target": int(
                    torch.any(
                        target[0, 0] & full_columns[:, :, None],
                        dim=-1,
                    )
                    .sum()
                    .item()
                ),
            }
        )
    return report


def experiment_asset_records(
    **paths: Optional[Path],
) -> Dict[str, Optional[Dict[str, object]]]:
    """Fingerprint a named set of immutable experiment inputs."""
    return {name: asset_record(path) for name, path in paths.items()}


def flatten_asset_hashes(
    records: Mapping[str, Optional[Mapping[str, object]]],
) -> Dict[str, Optional[str]]:
    """Return compact ``<name>_sha256`` fields for pairing checks."""
    return {
        f"{name}_sha256": (
            str(record["sha256"]) if record is not None else None
        )
        for name, record in records.items()
    }


def require_equal_fields(
    baseline: Mapping[str, object],
    guided: Mapping[str, object],
    fields: Iterable[str],
) -> tuple[bool, str]:
    """Compare a list of protocol fields with a useful first failure."""
    for field in fields:
        if field not in baseline or field not in guided:
            return False, f"missing strict pairing field: {field}"
        if baseline[field] != guided[field]:
            return False, f"strict pairing field differs: {field}"
    return True, "all strict pairing fields match"
