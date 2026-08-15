"""Fixed local-energy attribute and empirical label-9 calibration for Stage15-C."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def local_seismic_energy(seismic: torch.Tensor, window_num_samples: int) -> torch.Tensor:
    """Centered trace-local mean squared amplitude with fixed zero padding."""
    if seismic.ndim != 5 or seismic.shape[1] != 1:
        raise ValueError("seismic must have shape [B,1,X,Y,T]")
    if not seismic.is_floating_point() or not torch.isfinite(seismic).all():
        raise ValueError("seismic must be finite floating point")
    window = int(window_num_samples)
    if window <= 0 or window % 2 != 1:
        raise ValueError("window_num_samples must be a positive odd integer")
    batch, _, nx, ny, nt = seismic.shape
    traces = seismic.square().reshape(batch * nx * ny, 1, nt)
    kernel = torch.full(
        (1, 1, window),
        1.0 / window,
        device=seismic.device,
        dtype=seismic.dtype,
    )
    energy = F.conv1d(traces, kernel, padding=window // 2)
    return energy.reshape(batch, 1, nx, ny, nt).contiguous()


def depth_resample_local_energy(
    energy: torch.Tensor,
    subsurface_mask: torch.Tensor,
    *,
    sample_interval_ms: float,
    vertical_cell_size_m: float,
    background_velocity_m_s: float,
) -> torch.Tensor:
    """Map local-datum TWT energy to voxel centers using one fixed velocity."""
    if energy.ndim != 5 or energy.shape[1] != 1:
        raise ValueError("energy must have shape [B,1,X,Y,T]")
    if subsurface_mask.ndim != 5 or subsurface_mask.shape[1] != 1:
        raise ValueError("subsurface_mask must have shape [B,1,X,Y,Z]")
    if energy.shape[:4] != subsurface_mask.shape[:4]:
        raise ValueError("energy and subsurface mask batch/lateral shapes must match")
    if not energy.is_floating_point() or not torch.isfinite(energy).all():
        raise ValueError("energy must be finite floating point")
    interval = float(sample_interval_ms)
    dz = float(vertical_cell_size_m)
    velocity = float(background_velocity_m_s)
    if not all(math.isfinite(value) and value > 0 for value in (interval, dz, velocity)):
        raise ValueError("time-depth parameters must be finite and positive")
    rock = subsurface_mask.to(device=energy.device, dtype=torch.bool)
    if bool((~rock).all(dim=-1).any()):
        raise ValueError("each column must contain subsurface")
    if bool(((~rock[..., :-1]) & rock[..., 1:]).any()):
        raise ValueError("subsurface must be contiguous below the local surface")

    nz = rock.shape[-1]
    z = torch.arange(nz, device=energy.device, dtype=torch.long).view(1, 1, 1, 1, nz)
    surface_z = torch.where(rock, z, z.new_full((), -1)).amax(dim=-1, keepdim=True)
    depth_cell_index = surface_z - z
    cell_twt_ms = 2.0 * dz * 1000.0 / velocity
    voxel_center_twt_ms = (depth_cell_index.to(energy.dtype) + 0.5) * cell_twt_ms
    sample_position = voxel_center_twt_ms / interval
    nt = energy.shape[-1]
    if bool((sample_position[rock] < 0).any()) or bool((sample_position[rock] > nt - 1).any()):
        raise ValueError("fixed-velocity voxel-center time falls outside seismic recording")
    left = torch.floor(sample_position).long().clamp(0, nt - 1)
    right = (left + 1).clamp(0, nt - 1)
    fraction = sample_position - left.to(sample_position.dtype)
    left_value = torch.gather(energy, -1, left)
    right_value = torch.gather(energy, -1, right)
    attribute = left_value * (1.0 - fraction) + right_value * fraction
    return torch.where(rock, attribute, torch.zeros_like(attribute)).contiguous()


def quantile_bin_edges(values: torch.Tensor, bin_count: int) -> torch.Tensor:
    """Return replayable float32 edges for fixed pooled quantile bins."""
    flattened = values.detach().cpu().float().reshape(-1)
    if flattened.numel() == 0 or not torch.isfinite(flattened).all():
        raise ValueError("calibration attributes must be finite and non-empty")
    count = int(bin_count)
    if count <= 1:
        raise ValueError("bin_count must exceed one")
    edges = np.quantile(
        flattened.numpy(), np.linspace(0.0, 1.0, count + 1), method="linear"
    )
    result = torch.from_numpy(np.asarray(edges)).float().contiguous()
    if bool((result[1:] < result[:-1]).any()):
        raise RuntimeError("quantile edges are not nondecreasing")
    return result


def attribute_bin_indices(values: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    if edges.ndim != 1 or edges.numel() < 3:
        raise ValueError("edges must be a one-dimensional bin_count+1 tensor")
    if not torch.isfinite(values).all() or not torch.isfinite(edges).all():
        raise ValueError("values and edges must be finite")
    if bool((edges[1:] < edges[:-1]).any()):
        raise ValueError("edges must be nondecreasing")
    return torch.bucketize(values, edges[1:-1].to(values), right=True)


def fit_empirical_probability_lookup(
    values: torch.Tensor,
    labels: torch.Tensor,
    edges: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit natural-prevalence Laplace-smoothed probabilities in fixed bins."""
    score = values.detach().cpu().float().reshape(-1)
    target = labels.detach().cpu().bool().reshape(-1)
    if score.numel() != target.numel() or score.numel() == 0:
        raise ValueError("values and labels must be non-empty and matching")
    indices = attribute_bin_indices(score, edges.detach().cpu())
    bin_count = edges.numel() - 1
    total = torch.bincount(indices, minlength=bin_count).long()
    positive = torch.bincount(indices[target], minlength=bin_count).long()
    probability = (positive.float() + 1.0) / (total.float() + 2.0)
    if not torch.isfinite(probability).all() or bool(((probability <= 0) | (probability >= 1)).any()):
        raise RuntimeError("Laplace-smoothed lookup must lie strictly inside (0,1)")
    return probability.contiguous(), total, positive


def apply_probability_lookup(
    attribute: torch.Tensor,
    subsurface_mask: torch.Tensor,
    edges: torch.Tensor,
    probability_lookup: torch.Tensor,
) -> torch.Tensor:
    if attribute.shape != subsurface_mask.shape:
        raise ValueError("attribute and subsurface_mask must match")
    if probability_lookup.ndim != 1 or probability_lookup.numel() != edges.numel() - 1:
        raise ValueError("lookup length must equal edge count minus one")
    if bool(((probability_lookup < 0) | (probability_lookup > 1)).any()):
        raise ValueError("lookup probabilities must lie in [0,1]")
    indices = attribute_bin_indices(attribute, edges.to(attribute))
    probability = probability_lookup.to(attribute)[indices]
    return torch.where(
        subsurface_mask.to(device=attribute.device, dtype=torch.bool),
        probability,
        torch.zeros_like(probability),
    ).contiguous()
