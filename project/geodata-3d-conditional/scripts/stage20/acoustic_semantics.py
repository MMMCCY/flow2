"""Unique Stage20 geology-to-acoustics semantic contract."""

from __future__ import annotations

import torch

from guidance.seismic import hard_labels_to_acoustic
from guidance.seismic_inversion import neutral_rock_category


def enforce_stage20_acoustic_contract(
    acoustic: torch.Tensor,
    subsurface_mask: torch.Tensor,
    property_table: torch.Tensor,
    *,
    context: str,
) -> None:
    """Strict finite/positive/rock-bound checks; no tolerance or clamping."""
    if acoustic.ndim != 5 or acoustic.shape[1] != 2:
        raise ValueError(f"{context}: acoustic must have shape [B,2,X,Y,Z]")
    support = subsurface_mask.to(device=acoustic.device, dtype=torch.bool)
    if support.shape != acoustic[:, 0:1].shape:
        raise ValueError(f"{context}: subsurface_mask shape mismatch")
    if not bool(torch.isfinite(acoustic).all()):
        raise FloatingPointError(f"{context}: non-finite acoustic value")
    if not bool((acoustic[:, 1:2] > 0).all()):
        raise RuntimeError(f"{context}: slowness must be strictly positive")
    rock = property_table[0, 1:].to(acoustic)
    impedance = acoustic[:, 0:1]
    if bool(support.any()) and not bool(
        ((impedance[support] >= rock.min()) & (impedance[support] <= rock.max())).all()
    ):
        raise RuntimeError(f"{context}: subsurface impedance outside frozen rock bounds")


def stage20_labels_to_acoustic(
    labels: torch.Tensor,
    subsurface_mask: torch.Tensor,
    property_table: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Map labels without mutating geology; subsurface raw air becomes neutral rock."""
    if labels.ndim != 5 or labels.shape[1] != 1:
        raise ValueError("labels must have shape [B,1,X,Y,Z]")
    if subsurface_mask.shape != labels.shape:
        raise ValueError("subsurface_mask must match labels")
    if not torch.isfinite(labels).all() or not torch.equal(labels, labels.round()):
        raise ValueError("labels must be finite integers")
    if int(labels.min()) < -1 or int(labels.max()) > property_table.shape[1] - 2:
        raise ValueError("labels are outside the frozen codebook")
    cleaned = labels.long().clone()
    support = subsurface_mask.to(device=labels.device, dtype=torch.bool)
    underground_air = support & (cleaned == -1)
    neutral_category = neutral_rock_category(property_table)
    cleaned[underground_air] = neutral_category - 1
    acoustic = hard_labels_to_acoustic(cleaned, property_table)
    enforce_stage20_acoustic_contract(acoustic, support, property_table, context="Stage20 label mapper")
    return acoustic, {
        "underground_air_voxels_replaced": int(underground_air.sum().item()),
        "neutral_category": neutral_category,
        "neutral_raw_label": neutral_category - 1,
    }
