"""Forward/observation closure diagnostics for the Stage6Q five-body case."""

from __future__ import annotations

import math
from typing import Mapping

import torch

from .gravity import hard_labels_to_density, tensor_sha256 as gravity_tensor_sha256
from .seismic import hard_labels_to_acoustic, tensor_sha256 as seismic_tensor_sha256
from .simple_causality import AnalyticObservationSuite, SimpleCausalCase


OBSERVATION_CLOSURE_VERSION = "phase6q_observation_closure_v1"


def epsilon_safe_attainment(
    baseline_loss: float,
    candidate_loss: float,
    truth_loss: float,
    *,
    eps: float = 1e-12,
) -> float:
    """Return reference-consistent attainment with a guarded denominator."""
    denominator = float(baseline_loss) - float(truth_loss)
    if not all(math.isfinite(value) for value in (baseline_loss, candidate_loss, truth_loss)):
        raise ValueError("attainment inputs must be finite")
    if denominator <= float(eps):
        return float("nan")
    return (float(baseline_loss) - float(candidate_loss)) / denominator


def response_geometry(
    response: torch.Tensor,
    truth: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Return raw response separation without hiding scale in normalization."""
    if response.shape != truth.shape:
        raise ValueError("response and truth must have matching shapes")
    left = response.detach().double().reshape(-1)
    right = truth.detach().double().reshape(-1)
    difference = left - right
    mse = difference.square().mean()
    rmse = mse.sqrt()
    l2 = torch.linalg.vector_norm(difference)
    truth_energy = right.square().mean()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    cosine = torch.dot(left, right) / denominator.clamp_min(eps)
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    correlation_denominator = (
        torch.linalg.vector_norm(left_centered)
        * torch.linalg.vector_norm(right_centered)
    )
    correlation = torch.dot(left_centered, right_centered) / correlation_denominator.clamp_min(eps)
    return {
        "raw_mse": float(mse),
        "raw_rmse": float(rmse),
        "normalized_mse_by_truth_energy": float(mse / truth_energy.clamp_min(eps)),
        "response_l2_distance": float(l2),
        "response_cosine": float(cosine),
        "response_correlation": float(correlation),
    }


def closure_metrics(
    observation: torch.Tensor,
    recomputed: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> dict[str, object]:
    """Measure exact tensor closure and retain both absolute and relative errors."""
    if observation.shape != recomputed.shape:
        raise ValueError("observation and recomputed response must match")
    observed = observation.detach().double()
    repeated = recomputed.detach().double()
    difference = repeated - observed
    absolute_l2 = torch.linalg.vector_norm(difference)
    reference_l2 = torch.linalg.vector_norm(observed)
    truth_loss = difference.square().mean()
    return {
        "truth_observation_hash": seismic_tensor_sha256(observation),
        "recomputed_truth_observation_hash": seismic_tensor_sha256(recomputed),
        "exact_tensor_equal": bool(torch.equal(observation, recomputed)),
        "absolute_difference_max": float(difference.abs().max()),
        "absolute_difference_l2": float(absolute_l2),
        "relative_difference_l2": float(absolute_l2 / reference_l2.clamp_min(eps)),
        "truth_loss": float(truth_loss),
    }


def audit_five_body_observation_closure(
    *,
    case: SimpleCausalCase,
    suite: AnalyticObservationSuite,
    soft_baseline_probabilities: torch.Tensor | None = None,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    """Rebuild all Stage6Q hard observations through independent public mappings."""
    device = suite.acoustic_property_table.device
    truth_labels = case.truth_labels.to(device=device)
    baseline_labels = case.baseline_labels.to(device=device)
    truth_coefficients = case.truth_coefficients(device=device)
    baseline_coefficients = torch.zeros_like(truth_coefficients)

    inversion_observations = {
        mode: suite.field(truth_coefficients, mode).detach()
        for mode in ("property", "reflectivity_spikes", "seismic", "gravity")
    }
    recomputed = {
        mode: suite.field_from_labels(truth_labels, mode).detach()
        for mode in inversion_observations
    }
    baseline_hard = {
        mode: suite.field_from_labels(baseline_labels, mode).detach()
        for mode in inversion_observations
    }
    reports: dict[str, object] = {}
    tensors: dict[str, torch.Tensor] = {}
    for mode, observation in inversion_observations.items():
        closure = closure_metrics(observation, recomputed[mode])
        baseline = response_geometry(baseline_hard[mode], observation)
        if baseline["raw_rmse"] <= 0:
            raise RuntimeError(f"{mode} baseline response does not differ from truth")
        reports[mode] = {
            "closure": closure,
            "baseline_hard_to_truth": baseline,
        }
        tensors[f"{mode}_truth_observation"] = observation.cpu()
        tensors[f"{mode}_recomputed_truth"] = recomputed[mode].cpu()
        tensors[f"{mode}_baseline_hard"] = baseline_hard[mode].cpu()

    truth_acoustic = hard_labels_to_acoustic(
        truth_labels, suite.acoustic_property_table
    )
    truth_density = hard_labels_to_density(truth_labels, suite.density_table)
    tensors["truth_acoustic"] = truth_acoustic.cpu()
    tensors["truth_density"] = truth_density.cpu()
    reports["intermediate_hashes"] = {
        "truth_acoustic_sha256": seismic_tensor_sha256(truth_acoustic),
        "truth_density_sha256": gravity_tensor_sha256(truth_density),
        "truth_reflectivity_sha256": seismic_tensor_sha256(
            recomputed["reflectivity_spikes"]
        ),
    }

    if soft_baseline_probabilities is not None:
        probabilities = soft_baseline_probabilities.to(device=device)
        soft_reports: dict[str, object] = {}
        for mode, observation in inversion_observations.items():
            soft = suite.field_from_probabilities(probabilities, mode).detach()
            soft_reports[mode] = response_geometry(soft, observation)
            tensors[f"{mode}_baseline_soft"] = soft.cpu()
        reports["baseline_soft_to_truth"] = soft_reports

    closure_pass = all(
        float(reports[mode]["closure"]["truth_loss"]) <= 1e-20
        and float(reports[mode]["closure"]["relative_difference_l2"]) <= 1e-12
        for mode in ("property", "reflectivity_spikes", "seismic", "gravity")
    )
    reports.update(
        {
            "version": OBSERVATION_CLOSURE_VERSION,
            "closure_pass": closure_pass,
            "baseline_truth_separated": True,
            "checkpoint_used_for_flow": False,
            "flow_unet_loaded": False,
        }
    )
    return reports, tensors
