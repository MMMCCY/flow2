"""Truth-blind acoustic-property to categorical-probability bridge.

This module deliberately contains no geological-truth input.  It converts
inference-visible scalar property posterior samples into normalized class
probabilities using a globally registered Gaussian class model.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch


BRIDGE_SCHEMA = "stage10_scalar_gaussian_sample_bridge_v1"
AXIS_ORDER = ("x", "y", "z")


def stable_logsumexp(value: torch.Tensor, dim: int) -> torch.Tensor:
    """Numerically stable log-sum-exp with explicit finite validation."""
    if not torch.isfinite(value).all():
        raise ValueError("log weights must be finite")
    maximum = value.max(dim=dim, keepdim=True).values
    result = maximum + torch.log(torch.exp(value - maximum).sum(dim=dim, keepdim=True))
    if not torch.isfinite(result).all():
        raise FloatingPointError("logsumexp produced NaN/Inf")
    return result


def validate_grid_alignment(
    *values: torch.Tensor,
    expected_shape: Sequence[int] = (64, 64, 64),
) -> tuple[int, int, int]:
    """Require the final three axes of every tensor to share one xyz grid."""
    shape = tuple(int(item) for item in expected_shape)
    if len(shape) != 3 or any(item <= 0 for item in shape):
        raise ValueError("expected_shape must contain three positive dimensions")
    for value in values:
        if value.ndim < 3 or tuple(value.shape[-3:]) != shape:
            raise ValueError(
                f"tensor grid {tuple(value.shape[-3:])} does not match xyz grid {shape}"
            )
    return shape


def validate_class_model(model: Mapping[str, object]) -> dict[str, torch.Tensor]:
    """Validate a frozen scalar Gaussian class model."""
    labels = torch.as_tensor(model.get("raw_labels"), dtype=torch.int64)
    means = torch.as_tensor(model.get("log_impedance_mean"), dtype=torch.float64)
    sigmas = torch.as_tensor(model.get("log_impedance_sigma"), dtype=torch.float64)
    priors = torch.as_tensor(model.get("class_prior"), dtype=torch.float64)
    if labels.ndim != 1 or labels.numel() < 2:
        raise ValueError("class model requires at least two raw labels")
    if not (means.shape == sigmas.shape == priors.shape == labels.shape):
        raise ValueError("class model arrays must have identical one-dimensional shapes")
    if len(set(int(value) for value in labels.tolist())) != labels.numel():
        raise ValueError("raw labels must be unique")
    if not torch.isfinite(means).all():
        raise ValueError("class means must be finite")
    if not torch.isfinite(sigmas).all() or bool((sigmas <= 0).any()):
        raise ValueError("class sigmas must be finite and positive")
    if not torch.isfinite(priors).all() or bool((priors <= 0).any()):
        raise ValueError("class priors must be finite and positive")
    priors = priors / priors.sum()
    return {
        "raw_labels": labels,
        "means": means,
        "sigmas": sigmas,
        "priors": priors,
    }


def scalar_gaussian_log_weights(
    property_values: torch.Tensor,
    class_means: torch.Tensor,
    class_sigmas: torch.Tensor,
    class_priors: torch.Tensor,
) -> torch.Tensor:
    """Return ``log pi_k + log N(q; mu_k, sigma_k^2)`` for every sample/class."""
    if property_values.ndim != 5 or property_values.shape[1] != 1:
        raise ValueError("property_values must have shape [S,1,X,Y,Z]")
    means = class_means.to(device=property_values.device, dtype=property_values.dtype)
    sigmas = class_sigmas.to(device=property_values.device, dtype=property_values.dtype)
    priors = class_priors.to(device=property_values.device, dtype=property_values.dtype)
    if means.ndim != 1 or not (means.shape == sigmas.shape == priors.shape):
        raise ValueError("class parameters must be matching one-dimensional tensors")
    if bool((sigmas <= 0).any()) or bool((priors <= 0).any()):
        raise ValueError("class sigmas and priors must be positive")
    if not torch.isfinite(property_values).all():
        raise ValueError("property samples contain NaN/Inf")
    q = property_values[:, 0:1]
    shape = (1, means.numel(), 1, 1, 1)
    standardized = (q - means.view(shape)) / sigmas.view(shape)
    return (
        priors.log().view(shape)
        - sigmas.log().view(shape)
        - 0.5 * standardized.square()
        - 0.5 * math.log(2.0 * math.pi)
    )


def validate_probabilities(
    probabilities: torch.Tensor,
    *,
    class_dim: int = 1,
    tolerance: float = 2e-6,
) -> dict[str, float]:
    """Require finite, bounded probabilities normalized along ``class_dim``."""
    if probabilities.ndim < 2:
        raise ValueError("probability tensor must have at least two dimensions")
    if not torch.isfinite(probabilities).all():
        raise FloatingPointError("probabilities contain NaN/Inf")
    minimum = float(probabilities.min().item())
    maximum = float(probabilities.max().item())
    if minimum < -tolerance or maximum > 1.0 + tolerance:
        raise ValueError("probabilities lie outside [0,1]")
    error = float((probabilities.sum(dim=class_dim) - 1.0).abs().max().item())
    if error > tolerance:
        raise ValueError(f"class probabilities are not normalized: max error {error}")
    return {"minimum": minimum, "maximum": maximum, "normalization_max_error": error}


def scalar_gaussian_sample_bridge(
    property_samples: torch.Tensor,
    class_model: Mapping[str, object],
    *,
    output_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average per-sample categorical posteriors and return probability/entropy."""
    parsed = validate_class_model(class_model)
    work = property_samples.to(dtype=torch.float64)
    log_weights = scalar_gaussian_log_weights(
        work,
        parsed["means"],
        parsed["sigmas"],
        parsed["priors"],
    )
    probabilities_per_sample = torch.exp(
        log_weights - stable_logsumexp(log_weights, dim=1)
    )
    probabilities = probabilities_per_sample.mean(dim=0, keepdim=True)
    probabilities = probabilities.to(dtype=output_dtype)
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
    validate_probabilities(probabilities)
    entropy = -(
        probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
        * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
    ).sum(dim=1, keepdim=True)
    if not torch.isfinite(entropy).all():
        raise FloatingPointError("categorical entropy contains NaN/Inf")
    return probabilities.contiguous(), entropy.contiguous()


def class_channel(raw_labels: Sequence[int], target_label: int) -> int:
    """Return the unique probability channel for one raw geological label."""
    matches = [index for index, value in enumerate(raw_labels) if int(value) == int(target_label)]
    if len(matches) != 1:
        raise ValueError(f"target label {target_label} is not unique in class model")
    return matches[0]


def shuffle_xy_probability(
    probabilities: torch.Tensor,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Permute complete xy columns while preserving all probability values."""
    if probabilities.ndim != 5 or probabilities.shape[0] != 1:
        raise ValueError("probabilities must have shape [1,C,X,Y,Z]")
    validate_probabilities(probabilities)
    x_size, y_size, z_size = probabilities.shape[-3:]
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(x_size * y_size, generator=generator)
    flattened = probabilities.cpu().reshape(1, probabilities.shape[1], x_size * y_size, z_size)
    shuffled = flattened[:, :, permutation, :].reshape_as(probabilities.cpu()).contiguous()
    if not torch.equal(
        probabilities.cpu().reshape(-1).sort().values,
        shuffled.reshape(-1).sort().values,
    ):
        raise AssertionError("shuffle changed the probability values")
    validate_probabilities(shuffled)
    return shuffled, permutation


def inference_visible_guidance_masks(
    probability_label9: torch.Tensor,
    subsurface_mask: torch.Tensor,
    condition_mask: torch.Tensor,
    *,
    core_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build Phase-1-compatible core/ROI without geological truth."""
    if not 0.0 < float(core_threshold) < 1.0:
        raise ValueError("core_threshold must lie strictly inside (0,1)")
    validate_grid_alignment(probability_label9, subsurface_mask, condition_mask)
    if probability_label9.shape != subsurface_mask.shape or condition_mask.shape != subsurface_mask.shape:
        raise ValueError("probability and masks must have identical shapes")
    active = subsurface_mask.bool() & ~condition_mask.bool()
    core = (probability_label9 >= float(core_threshold)) & active
    roi = active
    return core, roi


def hard_condition_violation_count(
    labels: torch.Tensor,
    condition_values: torch.Tensor,
    condition_mask: torch.Tensor,
) -> int:
    """Count categorical disagreements at hard observed voxels."""
    if labels.ndim != 5 or labels.shape[1] != 1:
        raise ValueError("labels must have shape [B,1,X,Y,Z]")
    if condition_values.shape != condition_mask.shape or condition_values.shape[2:] != labels.shape[2:]:
        raise ValueError("hard condition grid does not match labels")
    values = condition_values.to(device=labels.device, dtype=labels.dtype)
    mask = condition_mask.to(device=labels.device).bool()
    return int(((labels != values) & mask).sum().item())
