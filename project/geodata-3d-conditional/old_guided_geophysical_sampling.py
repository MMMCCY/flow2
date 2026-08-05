"""Weak geophysical guidance for the trained conditional geology CFM.

This module only changes sampling.  The categorical model, its training
objective, and the existing ODE solver are intentionally left untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F

from geophysics import (
    GravityGradientForward,
    LithologyPropertyMap,
    MagneticTMIForward,
    SimpleGravityForward,
    normalized_misfit,
)
from geology_io_utils import (
    load_density_config,
    load_susceptibility_config,
    property_map_from_density_config,
    property_map_from_susceptibility_config,
)


def soft_decode_to_probs(
    x: torch.Tensor,
    embedding_weight: torch.Tensor,
    tau: float = 0.1,
) -> torch.Tensor:
    """Soft-decode embedding volumes to category probabilities.

    Args:
        x: Continuous embedding state with shape ``[B, E, X, Y, Z]``.
        embedding_weight: Category embeddings with shape ``[C, E]``.
        tau: Positive softmax temperature.
    """
    if x.ndim != 5:
        raise ValueError("x must have shape [B, E, X, Y, Z]")
    if embedding_weight.ndim != 2:
        raise ValueError("embedding_weight must have shape [C, E]")
    if x.shape[1] != embedding_weight.shape[1]:
        raise ValueError("x and embedding_weight must have the same embedding dimension")
    if tau <= 0:
        raise ValueError("tau must be positive")

    embeddings = embedding_weight.to(device=x.device, dtype=x.dtype)
    x_normalized = F.normalize(x, dim=1)
    embeddings_normalized = F.normalize(embeddings, dim=1)
    similarities = torch.einsum("bexyz,ce->bcxyz", x_normalized, embeddings_normalized)
    return torch.softmax(similarities / tau, dim=1)


def probs_to_density(
    probs: torch.Tensor,
    property_map: LithologyPropertyMap,
) -> torch.Tensor:
    """Map soft category probabilities to expected density contrast.

    Category index zero represents lithology label ``-1``; category index
    one represents label ``0``, and so on.
    """
    if probs.ndim != 5:
        raise ValueError("probs must have shape [B, C, X, Y, Z]")
    if not hasattr(property_map, "properties") or not hasattr(
        property_map, "default_value"
    ):
        raise TypeError("property_map must expose properties and default_value")

    values = [
        float(property_map.properties.get(category - 1, property_map.default_value))
        for category in range(probs.shape[1])
    ]
    density_values = torch.as_tensor(values, device=probs.device, dtype=probs.dtype)
    return torch.einsum("bcxyz,c->bxyz", probs, density_values).unsqueeze(1)


def _proxy_loss(predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    return normalized_misfit(predicted, observed, reduction="mean")


def geophysical_guidance_loss(
    x: torch.Tensor,
    embedding_weight: torch.Tensor,
    observed_gravity: torch.Tensor,
    property_map: LithologyPropertyMap,
    forward_model: SimpleGravityForward,
    tau: float = 0.1,
) -> torch.Tensor:
    """Return differentiable normalized gravity misfit for a continuous state."""
    probs = soft_decode_to_probs(x, embedding_weight, tau=tau)
    density = probs_to_density(probs, property_map)
    predicted_gravity = forward_model(density)
    return _proxy_loss(predicted_gravity, observed_gravity)


def multi_physics_guidance_loss(
    x: torch.Tensor,
    embedding_weight: torch.Tensor,
    property_map: LithologyPropertyMap,
    forward_model: SimpleGravityForward,
    observed_gravity: torch.Tensor | None = None,
    susceptibility_map: LithologyPropertyMap | None = None,
    magnetic_forward_model: MagneticTMIForward | None = None,
    observed_magnetic: torch.Tensor | None = None,
    gravity_gradient_forward_model: GravityGradientForward | None = None,
    observed_gravity_gradient: torch.Tensor | None = None,
    physics_mode: str = "gravity",
    gravity_weight: float = 1.0,
    magnetic_weight: float = 1.0,
    gravity_gradient_weight: float = 0.0,
    tau: float = 0.1,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Return weighted differentiable lightweight geophysical proxy misfit."""
    probs = soft_decode_to_probs(x, embedding_weight, tau=tau)
    density = probs_to_density(probs, property_map)
    total = torch.zeros((), device=x.device, dtype=x.dtype)
    diagnostics: Dict[str, float] = {}

    use_gravity = physics_mode in {"gravity", "joint"}
    use_magnetic = physics_mode in {"magnetic", "joint"}
    use_gradient = physics_mode == "gravity_gradient" or (
        physics_mode == "joint" and float(gravity_gradient_weight) > 0.0
    )

    if use_gravity:
        if observed_gravity is None:
            raise ValueError("observed_gravity is required for gravity guidance")
        gravity_loss = _proxy_loss(forward_model(density), observed_gravity)
        total = total + float(gravity_weight) * gravity_loss
        diagnostics["gravity_loss"] = float(gravity_loss.detach().cpu())
    else:
        diagnostics["gravity_loss"] = float("nan")

    if use_gradient:
        if observed_gravity_gradient is None:
            raise ValueError("observed_gravity_gradient is required for gravity-gradient guidance")
        if gravity_gradient_forward_model is None:
            raise ValueError("gravity_gradient_forward_model is required")
        gradient_loss = _proxy_loss(
            gravity_gradient_forward_model(density),
            observed_gravity_gradient,
        )
        total = total + float(gravity_gradient_weight) * gradient_loss
        diagnostics["gravity_gradient_loss"] = float(gradient_loss.detach().cpu())
    else:
        diagnostics["gravity_gradient_loss"] = float("nan")

    if use_magnetic:
        if observed_magnetic is None:
            raise ValueError("observed_magnetic is required for magnetic guidance")
        if susceptibility_map is None or magnetic_forward_model is None:
            raise ValueError("susceptibility_map and magnetic_forward_model are required")
        susceptibility = probs_to_density(probs, susceptibility_map)
        magnetic_loss = _proxy_loss(
            magnetic_forward_model(susceptibility),
            observed_magnetic,
        )
        total = total + float(magnetic_weight) * magnetic_loss
        diagnostics["magnetic_loss"] = float(magnetic_loss.detach().cpu())
    else:
        diagnostics["magnetic_loss"] = float("nan")

    diagnostics["total_geo_loss"] = float(total.detach().cpu())
    return total, diagnostics


def guidance_weight(
    t: Union[float, torch.Tensor],
    schedule: str = "late_quadratic",
    start: float = 0.5,
) -> Union[float, torch.Tensor]:
    """Evaluate a scalar or tensor guidance schedule at time ``t``."""
    if not 0.0 <= start < 1.0:
        raise ValueError("start must satisfy 0 <= start < 1")

    if isinstance(t, torch.Tensor):
        zero = torch.zeros_like(t)
        if schedule == "late_quadratic":
            active = ((t - start) / (1.0 - start)).square()
            return torch.where(t < start, zero, active)
        if schedule == "quadratic":
            return t.square()
        if schedule == "constant_after_start":
            return torch.where(t < start, zero, torch.ones_like(t))
    else:
        t_float = float(t)
        if schedule == "late_quadratic":
            return 0.0 if t_float < start else ((t_float - start) / (1.0 - start)) ** 2
        if schedule == "quadratic":
            return t_float**2
        if schedule == "constant_after_start":
            return 0.0 if t_float < start else 1.0

    raise ValueError(
        "schedule must be 'late_quadratic', 'quadratic', or "
        "'constant_after_start'"
    )


def clip_gradient_by_norm(grad: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Clip each sample over all non-batch dimensions without changing direction."""
    if grad.ndim < 1:
        raise ValueError("grad must include a batch dimension")
    if max_norm < 0:
        raise ValueError("max_norm must be non-negative")

    norms = grad.flatten(1).norm(dim=1)
    scale = (float(max_norm) / norms.clamp_min(torch.finfo(grad.dtype).eps)).clamp(max=1.0)
    return grad * scale.view(-1, *([1] * (grad.ndim - 1)))


def _batch_l2_norm(value: torch.Tensor) -> float:
    return float(value.detach().flatten(1).norm(dim=1).mean().cpu())


def build_guidance_velocity(
    grad_geo: torch.Tensor,
    v_prior: torch.Tensor,
    w_t: float,
    mode: str = "absolute",
    mu: float = 0.0,
    alpha: float = 0.01,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    Return the geophysical guidance velocity term and diagnostic statistics.

    For absolute mode:
        guidance = mu * w_t * grad_geo

    For relative mode:
        guidance = alpha * w_t * ||v_prior|| * grad_geo / ||grad_geo||

    Return:
        guidance_velocity: same shape as grad_geo
        diagnostics:
            raw_grad_norm
            prior_velocity_norm
            guidance_velocity_norm
            effective_guidance_ratio
    """
    if grad_geo.shape != v_prior.shape:
        raise ValueError("grad_geo and v_prior must have the same shape")
    if grad_geo.ndim < 2:
        raise ValueError("grad_geo and v_prior must include batch and feature dimensions")
    if eps <= 0:
        raise ValueError("eps must be positive")

    grad_norm = grad_geo.flatten(1).norm(dim=1)
    prior_norm = v_prior.flatten(1).norm(dim=1)
    expand_shape = (-1, *([1] * (grad_geo.ndim - 1)))

    if mode == "absolute":
        guidance_velocity = float(mu) * float(w_t) * grad_geo
    elif mode == "relative":
        grad_unit = grad_geo / (grad_norm + eps).view(expand_shape)
        grad_scaled = grad_unit * prior_norm.view(expand_shape)
        guidance_velocity = float(alpha) * float(w_t) * grad_scaled
    else:
        raise ValueError("mode must be 'absolute' or 'relative'")

    guidance_norm = guidance_velocity.detach().flatten(1).norm(dim=1)
    effective_ratio = guidance_norm / (prior_norm.detach() + eps)
    diagnostics = {
        "raw_grad_norm": float(grad_norm.detach().mean().cpu()),
        "prior_velocity_norm": float(prior_norm.detach().mean().cpu()),
        "guidance_velocity_norm": float(guidance_norm.mean().cpu()),
        "effective_guidance_ratio": float(effective_ratio.mean().cpu()),
    }
    return guidance_velocity, diagnostics


def guided_euler_sample(
    model,
    X0: torch.Tensor,
    ATb_lith: torch.Tensor,
    observed_gravity: Optional[torch.Tensor],
    property_map: LithologyPropertyMap,
    forward_model: SimpleGravityForward,
    n_steps: int,
    mu: float,
    tau: float,
    guidance_start: float,
    guidance_schedule: str,
    grad_clip_norm: float,
    alpha: float = 0.01,
    guidance_mode: str = "absolute",
    sample_id: int = 0,
    susceptibility_map: Optional[LithologyPropertyMap] = None,
    magnetic_forward_model: Optional[MagneticTMIForward] = None,
    observed_magnetic: Optional[torch.Tensor] = None,
    gravity_gradient_forward_model: Optional[GravityGradientForward] = None,
    observed_gravity_gradient: Optional[torch.Tensor] = None,
    physics_mode: str = "gravity",
    gravity_weight: float = 1.0,
    magnetic_weight: float = 1.0,
    gravity_gradient_weight: float = 0.25,
):
    """Integrate the conditional velocity with lightweight geophysical guidance."""
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if X0.ndim != 5:
        raise ValueError("X0 must have shape [B, E, X, Y, Z]")
    if ATb_lith.shape[1:] != X0.shape[1:]:
        raise ValueError("ATb_lith and X0 must match in non-batch dimensions")
    if ATb_lith.shape[0] not in (1, X0.shape[0]):
        raise ValueError("ATb_lith batch size must be 1 or match X0")

    x = X0
    conditioning = ATb_lith.to(device=x.device, dtype=x.dtype).expand(
        x.shape[0], -1, -1, -1, -1
    )
    observed = observed_gravity.to(device=x.device) if observed_gravity is not None else None
    observed_magnetic_device = (
        observed_magnetic.to(device=x.device) if observed_magnetic is not None else None
    )
    observed_gravity_gradient_device = (
        observed_gravity_gradient.to(device=x.device)
        if observed_gravity_gradient is not None
        else None
    )
    embedding_weight = model.embedding.weight
    dt = 1.0 / n_steps
    trace: List[Dict[str, object]] = []

    for step in range(n_steps):
        t_value = (step + 0.5) / n_steps
        t = torch.full((x.shape[0],), t_value, device=x.device, dtype=x.dtype)
        w_t = float(guidance_weight(t_value, guidance_schedule, guidance_start))

        x = x.detach().requires_grad_(True)
        with torch.no_grad():
            v_prior = model.net(x, conditioning, t)
        loss_geo, loss_diagnostics = multi_physics_guidance_loss(
            x,
            embedding_weight,
            property_map,
            forward_model,
            observed_gravity=observed,
            susceptibility_map=susceptibility_map,
            magnetic_forward_model=magnetic_forward_model,
            observed_magnetic=observed_magnetic_device,
            gravity_gradient_forward_model=gravity_gradient_forward_model,
            observed_gravity_gradient=observed_gravity_gradient_device,
            physics_mode=physics_mode,
            gravity_weight=gravity_weight,
            magnetic_weight=magnetic_weight,
            gravity_gradient_weight=gravity_gradient_weight,
            tau=tau,
        )
        grad_geo = torch.autograd.grad(loss_geo, x)[0]
        grad_geo = clip_gradient_by_norm(grad_geo, grad_clip_norm)
        guidance_velocity, guidance_diagnostics = build_guidance_velocity(
            grad_geo=grad_geo,
            v_prior=v_prior,
            w_t=w_t,
            mode=guidance_mode,
            mu=mu,
            alpha=alpha,
        )
        v = v_prior - guidance_velocity

        trace.append(
            {
                "sample_id": sample_id,
                "step": step,
                "t": t_value,
                "w_t": w_t,
                "geo_loss": float(loss_geo.detach().cpu()),
                "gravity_loss": loss_diagnostics["gravity_loss"],
                "gravity_gradient_loss": loss_diagnostics["gravity_gradient_loss"],
                "magnetic_loss": loss_diagnostics["magnetic_loss"],
                "grad_norm": guidance_diagnostics["raw_grad_norm"],
                "prior_velocity_norm": guidance_diagnostics["prior_velocity_norm"],
                "guidance_velocity_norm": guidance_diagnostics[
                    "guidance_velocity_norm"
                ],
                "guided_velocity_norm": _batch_l2_norm(v),
                "effective_guidance_ratio": guidance_diagnostics[
                    "effective_guidance_ratio"
                ],
                "guidance_mode": guidance_mode,
                "physics_mode": physics_mode,
                "mu": float(mu),
                "alpha": float(alpha),
                "gravity_weight": float(gravity_weight),
                "gravity_gradient_weight": float(gravity_gradient_weight),
                "magnetic_weight": float(magnetic_weight),
            }
        )
        x = x.detach() + dt * v.detach()

    return x, trace


def _normalize_geology(volume: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(volume, torch.Tensor):
        raise TypeError(f"{name} must contain a torch.Tensor")
    if volume.ndim == 3:
        volume = volume.unsqueeze(0).unsqueeze(0)
    elif volume.ndim == 4:
        if volume.shape[0] != 1:
            raise ValueError(f"{name} must contain one geology volume")
        volume = volume.unsqueeze(0)
    elif volume.ndim != 5 or volume.shape[:2] != (1, 1):
        raise ValueError(f"{name} must have shape [X,Y,Z], [1,X,Y,Z], or [1,1,X,Y,Z]")
    return volume


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample a trained conditional geology CFM with inference-time "
            "lightweight geophysical proxy guidance."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--truth-model", type=Path, default=None)
    parser.add_argument("--boreholes", type=Path, default=None)
    parser.add_argument("--density-config", type=Path, default=None)
    parser.add_argument("--susceptibility-config", type=Path, default=None)
    parser.add_argument(
        "--physics-mode",
        choices=("gravity", "gravity_gradient", "magnetic", "joint"),
        default="gravity",
        help=(
            "Geophysical proxy used for guidance. joint combines gravity, "
            "magnetic, and gravity-gradient terms with the supplied weights."
        ),
    )
    parser.add_argument(
        "--observed-gravity",
        type=Path,
        default=None,
        help="Optional observed lightweight gravity-proxy tensor generated from the same density_config.",
    )
    parser.add_argument(
        "--observed-magnetic",
        type=Path,
        default=None,
        help="Optional observed lightweight magnetic-proxy tensor generated from the same susceptibility_config.",
    )
    parser.add_argument(
        "--observed-gravity-gradient",
        type=Path,
        default=None,
        help="Optional observed lightweight gravity-gradient-proxy tensor generated from the same density_config.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=32)
    parser.add_argument(
        "--guidance-mode",
        choices=("absolute", "relative"),
        default="absolute",
    )
    parser.add_argument("--mu", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--gravity-weight", type=float, default=1.0)
    parser.add_argument("--magnetic-weight", type=float, default=1.0)
    parser.add_argument("--gravity-gradient-weight", type=float, default=0.25)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--guidance-start", type=float, default=0.5)
    parser.add_argument(
        "--guidance-schedule",
        choices=("late_quadratic", "quadratic", "constant_after_start"),
        default="late_quadratic",
    )
    parser.add_argument("--kernel-size", type=int, default=9)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing baseline sample_i.pt files for "
            "decoded category-change diagnostics."
        ),
    )
    return parser.parse_args()


def _validate_cli_args(args: argparse.Namespace) -> None:
    if args.n_samples <= 0:
        raise ValueError("--n-samples must be positive")
    if args.n_steps <= 0:
        raise ValueError("--n-steps must be positive")
    if args.tau <= 0:
        raise ValueError("--tau must be positive")
    if args.grad_clip_norm < 0:
        raise ValueError("--grad-clip-norm must be non-negative")
    if args.alpha < 0:
        raise ValueError("--alpha must be non-negative")
    if args.gravity_weight < 0:
        raise ValueError("--gravity-weight must be non-negative")
    if args.magnetic_weight < 0:
        raise ValueError("--magnetic-weight must be non-negative")
    if args.gravity_gradient_weight < 0:
        raise ValueError("--gravity-gradient-weight must be non-negative")
    if not 0.0 <= args.guidance_start < 1.0:
        raise ValueError("--guidance-start must satisfy 0 <= value < 1")
    if args.physics_mode in {"magnetic", "joint"} and args.magnetic_weight == 0:
        raise ValueError("--magnetic-weight must be positive when magnetic guidance is active")
    if args.physics_mode in {"gravity", "joint"} and args.gravity_weight == 0:
        raise ValueError("--gravity-weight must be positive when gravity guidance is active")
    if args.physics_mode == "gravity_gradient" and args.gravity_gradient_weight == 0:
        raise ValueError(
            "--gravity-gradient-weight must be positive when physics-mode is gravity_gradient"
        )


def main() -> None:
    args = _parse_args()
    _validate_cli_args(args)
    device = torch.device(args.device)

    # Keep the heavyweight training-module import out of utility/test imports.
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    from model_train_sh_inference_cond import Geo3DStochInterp

    truth_path = args.truth_model or args.samples_dir / "true_model.pt"
    boreholes_path = args.boreholes or args.samples_dir / "boreholes.pt"
    truth = _normalize_geology(
        torch.load(truth_path, map_location=device), str(truth_path)
    ).to(device)
    boreholes = _normalize_geology(
        torch.load(boreholes_path, map_location=device), str(boreholes_path)
    ).to(device)
    if truth.shape != boreholes.shape:
        raise ValueError("truth model and boreholes must have matching shapes")

    model = Geo3DStochInterp.load_from_checkpoint(
        str(args.ckpt_path), map_location=device
    ).to(device)
    model.eval()

    # This exactly follows populate_solutions in the existing conditional inference.
    boreholes_mask = (boreholes != -1) | (truth == -1)
    embedded_truth = model.embed(truth)
    embedded_mask = boreholes_mask.expand(-1, embedded_truth.shape[1], -1, -1, -1)
    ATb_lith = embedded_truth * embedded_mask

    density_config = load_density_config(args.density_config)
    susceptibility_config = load_susceptibility_config(args.susceptibility_config)
    property_map = property_map_from_density_config(density_config)
    susceptibility_map = property_map_from_susceptibility_config(susceptibility_config)
    forward_model = SimpleGravityForward(kernel_size=args.kernel_size)
    magnetic_forward_model = MagneticTMIForward(kernel_size=args.kernel_size)
    gravity_gradient_forward_model = GravityGradientForward(kernel_size=args.kernel_size)

    use_gravity = args.physics_mode in {"gravity", "joint"}
    use_magnetic = args.physics_mode in {"magnetic", "joint"}
    use_gravity_gradient = args.physics_mode == "gravity_gradient" or (
        args.physics_mode == "joint" and args.gravity_gradient_weight > 0.0
    )

    if args.observed_gravity is not None:
        observed_gravity = torch.load(args.observed_gravity, map_location=device).detach()
    elif use_gravity:
        observed_gravity = forward_model(property_map(truth)).detach()
    else:
        observed_gravity = None

    if args.observed_magnetic is not None:
        observed_magnetic = torch.load(args.observed_magnetic, map_location=device).detach()
    elif use_magnetic:
        observed_magnetic = magnetic_forward_model(susceptibility_map(truth)).detach()
    else:
        observed_magnetic = None

    if args.observed_gravity_gradient is not None:
        observed_gravity_gradient = torch.load(
            args.observed_gravity_gradient,
            map_location=device,
        ).detach()
    elif use_gravity_gradient:
        observed_gravity_gradient = gravity_gradient_forward_model(property_map(truth)).detach()
    else:
        observed_gravity_gradient = None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if observed_gravity is not None:
        torch.save(observed_gravity.cpu(), args.output_dir / "observed_gravity.pt")
    if observed_magnetic is not None:
        torch.save(observed_magnetic.cpu(), args.output_dir / "observed_magnetic.pt")
    if observed_gravity_gradient is not None:
        torch.save(
            observed_gravity_gradient.cpu(),
            args.output_dir / "observed_gravity_gradient.pt",
        )

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    all_trace: List[Dict[str, object]] = []
    decoded_change_records: List[Dict[str, float]] = []
    for sample_index in range(args.n_samples):
        X0 = torch.randn(
            1,
            model.embedding_dim,
            *model.data_shape,
            generator=generator,
            dtype=embedded_truth.dtype,
        ).to(device)
        final_state, trace = guided_euler_sample(
            model=model,
            X0=X0,
            ATb_lith=ATb_lith,
            observed_gravity=observed_gravity,
            property_map=property_map,
            forward_model=forward_model,
            susceptibility_map=susceptibility_map,
            magnetic_forward_model=magnetic_forward_model,
            observed_magnetic=observed_magnetic,
            gravity_gradient_forward_model=gravity_gradient_forward_model,
            observed_gravity_gradient=observed_gravity_gradient,
            physics_mode=args.physics_mode,
            gravity_weight=args.gravity_weight,
            magnetic_weight=args.magnetic_weight,
            gravity_gradient_weight=args.gravity_gradient_weight,
            n_steps=args.n_steps,
            mu=args.mu,
            alpha=args.alpha,
            guidance_mode=args.guidance_mode,
            tau=args.tau,
            guidance_start=args.guidance_start,
            guidance_schedule=args.guidance_schedule,
            grad_clip_norm=args.grad_clip_norm,
            sample_id=sample_index,
        )
        if not torch.isfinite(final_state).all():
            raise FloatingPointError(f"sample {sample_index} final state contains NaN or Inf")

        decoded = (model.decode(final_state).detach().cpu() - 1)[0]
        torch.save(decoded, args.output_dir / f"sample_{sample_index}.pt")
        if args.baseline_dir is not None:
            baseline_path = args.baseline_dir / f"sample_{sample_index}.pt"
            baseline_decoded = _normalize_geology(
                torch.load(baseline_path, map_location="cpu"),
                str(baseline_path),
            )
            current_decoded = _normalize_geology(decoded, f"sample_{sample_index}.pt")
            if baseline_decoded.shape != current_decoded.shape:
                raise ValueError(
                    f"baseline {baseline_path} shape {tuple(baseline_decoded.shape)} "
                    f"does not match generated sample shape {tuple(current_decoded.shape)}"
                )
            changed = baseline_decoded != current_decoded
            decoded_change_records.append(
                {
                    "sample_id": sample_index,
                    "changed_voxel_fraction": float(changed.float().mean().item()),
                }
            )
        all_trace.extend(trace)

    trace_fields = (
        "sample_id",
        "step",
        "t",
        "w_t",
        "geo_loss",
        "gravity_loss",
        "gravity_gradient_loss",
        "magnetic_loss",
        "grad_norm",
        "prior_velocity_norm",
        "guidance_velocity_norm",
        "guided_velocity_norm",
        "effective_guidance_ratio",
        "guidance_mode",
        "physics_mode",
        "mu",
        "alpha",
        "gravity_weight",
        "gravity_gradient_weight",
        "magnetic_weight",
    )
    with (args.output_dir / "guided_trace.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=trace_fields)
        writer.writeheader()
        writer.writerows(all_trace)

    if args.baseline_dir is not None:
        with (args.output_dir / "decoded_change_ratio.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("sample_id", "changed_voxel_fraction"),
            )
            writer.writeheader()
            writer.writerows(decoded_change_records)

    config = {
        "ckpt_path": str(args.ckpt_path),
        "samples_dir": str(args.samples_dir),
        "truth_model": str(truth_path),
        "boreholes": str(boreholes_path),
        "output_dir": str(args.output_dir),
        "n_samples": args.n_samples,
        "n_steps": args.n_steps,
        "guidance_mode": args.guidance_mode,
        "physics_mode": args.physics_mode,
        "mu": args.mu,
        "alpha": args.alpha,
        "gravity_weight": args.gravity_weight,
        "magnetic_weight": args.magnetic_weight,
        "gravity_gradient_weight": args.gravity_gradient_weight,
        "effective_parameter_explanation": (
            "absolute mode uses mu * w_t * grad_geo; relative mode scales "
            "geophysical guidance to alpha * w_t of the prior velocity norm"
        ),
        "description": (
            "relative mode scales geophysical guidance to alpha * w_t of "
            "prior velocity norm"
            if args.guidance_mode == "relative"
            else "absolute mode preserves the original mu * w_t * grad_geo guidance"
        ),
        "tau": args.tau,
        "guidance_start": args.guidance_start,
        "guidance_schedule": args.guidance_schedule,
        "kernel_size": args.kernel_size,
        "density_config": str(args.density_config) if args.density_config else None,
        "susceptibility_config": (
            str(args.susceptibility_config) if args.susceptibility_config else None
        ),
        "observed_gravity": (
            str(args.observed_gravity)
            if args.observed_gravity
            else str(args.output_dir / "observed_gravity.pt")
            if observed_gravity is not None
            else None
        ),
        "observed_magnetic": (
            str(args.observed_magnetic)
            if args.observed_magnetic
            else str(args.output_dir / "observed_magnetic.pt")
            if observed_magnetic is not None
            else None
        ),
        "observed_gravity_gradient": (
            str(args.observed_gravity_gradient)
            if args.observed_gravity_gradient
            else str(args.output_dir / "observed_gravity_gradient.pt")
            if observed_gravity_gradient is not None
            else None
        ),
        "geophysical_proxy_description": (
            "Inference-time lightweight geophysical proxy guidance only; "
            "not quantitative gravity, magnetic, or gravity-gradient inversion."
        ),
        "grad_clip_norm": args.grad_clip_norm,
        "device": str(device),
        "seed": args.seed,
        "baseline_dir": str(args.baseline_dir) if args.baseline_dir else None,
    }
    with (args.output_dir / "config.json").open("w") as handle:
        json.dump(config, handle, indent=2)


if __name__ == "__main__":
    main()
