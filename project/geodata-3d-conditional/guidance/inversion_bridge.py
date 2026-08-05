"""Convert a Phase-5a acoustic posterior into Phase-5b property assets."""

from __future__ import annotations

import math
from typing import Mapping

import torch


PHASE5B_BRIDGE_CONFIG_SCHEMA = "phase5b_inversion_property_bridge_config_v1"
PHASE5B_BRIDGE_MANIFEST_SCHEMA = "phase5b_inversion_property_assets_v1"
PHASE5B_CONFIDENCE_MODE = (
    "inverse_quadratic_relative_to_active_positive_median_v1"
)


def validate_bridge_config(config: Mapping[str, object]) -> None:
    required = {
        "schema": PHASE5B_BRIDGE_CONFIG_SCHEMA,
        "property_channel": "log_acoustic_impedance",
        "target": "posterior_mean_log_impedance",
        "uncertainty": "population_std_log_impedance",
        "confidence": PHASE5B_CONFIDENCE_MODE,
        "active_region": "unconstrained_subsurface",
        "condition_confidence": 0.0,
        "spatial_sigma": 0.0,
        "truth_tuned": False,
    }
    for field, expected in required.items():
        if config.get(field) != expected:
            raise ValueError(f"bridge config {field} must be {expected!r}")
    if not str(config.get("id", "")).strip():
        raise ValueError("bridge config requires a non-empty id")


def log_impedance_property_table(acoustic_table: torch.Tensor) -> torch.Tensor:
    """Return a one-channel full-category log-impedance codebook."""
    if acoustic_table.ndim != 2 or acoustic_table.shape[0] != 2:
        raise ValueError("acoustic_table must have shape [2,C]")
    impedance = acoustic_table[0]
    if not torch.isfinite(impedance).all() or bool((impedance <= 0).any()):
        raise ValueError("acoustic impedance codebook must be finite and positive")
    return impedance.log().reshape(1, -1).contiguous()


def posterior_spread_confidence(
    log_impedance_std: torch.Tensor,
    subsurface_mask: torch.Tensor,
    condition_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    """Build truth-blind confidence from fixed-ensemble posterior spread."""
    if log_impedance_std.ndim != 5 or log_impedance_std.shape[1] != 1:
        raise ValueError("log_impedance_std must have shape [1,1,X,Y,Z]")
    if subsurface_mask.shape != log_impedance_std.shape:
        raise ValueError("subsurface_mask must match log_impedance_std")
    if condition_mask.shape != log_impedance_std.shape:
        raise ValueError("condition_mask must match log_impedance_std")
    if not torch.isfinite(log_impedance_std).all() or bool(
        (log_impedance_std < 0).any()
    ):
        raise ValueError("posterior spread must be finite and non-negative")
    active = subsurface_mask.bool() & (~condition_mask.bool())
    positive = log_impedance_std[active & (log_impedance_std > 0)]
    if positive.numel() == 0:
        raise ValueError("active posterior spread contains no positive values")
    reference = positive.median()
    if not torch.isfinite(reference) or float(reference) <= 0:
        raise ValueError("posterior spread reference must be finite and positive")
    ratio = log_impedance_std / reference
    confidence = torch.where(
        active,
        1.0 / (1.0 + ratio.square()),
        torch.zeros_like(log_impedance_std),
    )
    if bool((confidence < 0).any()) or bool((confidence > 1).any()):
        raise FloatingPointError("posterior confidence left [0,1]")
    active_values = confidence[active]
    return confidence.contiguous(), {
        "mode": PHASE5B_CONFIDENCE_MODE,
        "active_voxels": int(active.sum().item()),
        "positive_spread_voxels": int(positive.numel()),
        "spread_reference_median": float(reference.detach().cpu()),
        "active_confidence_min": float(active_values.min().detach().cpu()),
        "active_confidence_mean": float(active_values.mean().detach().cpu()),
        "active_confidence_max": float(active_values.max().detach().cpu()),
    }


def property_config_from_table(
    table: torch.Tensor,
    *,
    description: str,
) -> dict[str, object]:
    """Serialize a one-channel property config in the existing Phase-2 schema."""
    if table.ndim != 2 or table.shape[0] != 1:
        raise ValueError("property table must have shape [1,C]")
    values = {
        str(category - 1): float(table[0, category].item())
        for category in range(table.shape[1])
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("property table contains non-finite values")
    return {
        "schema": "full_lithology_property_channels_v1",
        "description": description,
        "channels": [
            {
                "name": "log_acoustic_impedance",
                "unit": "log(kg m^-2 s^-1)",
                "weight": 1.0,
                "values": values,
            }
        ],
    }
