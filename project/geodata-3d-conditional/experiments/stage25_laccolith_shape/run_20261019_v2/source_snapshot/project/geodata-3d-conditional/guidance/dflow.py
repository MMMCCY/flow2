"""D-Flow source-noise optimization through the frozen conditional Flow.

This module implements the mechanism from Ben-Hamu et al. (ICML 2024): the
only optimized quantity is the source point.  The endpoint loss is
differentiated through every fixed-Euler Flow step.  Model parameters are
frozen, but the state-to-velocity graph is deliberately retained.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time
from typing import Callable, Mapping

import torch
from torch.utils.checkpoint import checkpoint

from guidance.probability_sampling import _condition_violations, _project_conditions
from guidance.probability_volume import tensor_sha256


DFLOW_ENGINE_VERSION = "dflow_source_optimization_v1"
DFLOW_SOLVERS = ("fixed_euler",)
DFLOW_OPTIMIZERS = ("adam", "lbfgs")
CHECKPOINT_POLICY = "model_net_per_euler_step_nonreentrant_v1"

EndpointLoss = Callable[
    [torch.Tensor], tuple[torch.Tensor, Mapping[str, object]]
]
TraceDiagnostics = Callable[[torch.Tensor], Mapping[str, object]]


@dataclass(frozen=True)
class DFlowOptimizationResult:
    """Detached result of one source-point optimization."""

    initial_source: torch.Tensor
    optimized_source: torch.Tensor
    final_state: torch.Tensor
    trace: list[dict[str, object]]
    diagnostics: dict[str, object]


def freeze_model_for_dflow(model) -> int:
    """Put the model in inference mode and freeze every model parameter."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if count != 0:
        raise RuntimeError("D-Flow requires zero trainable model parameters")
    return count


def model_state_dict_hashes(model) -> tuple[str, dict[str, str]]:
    """Hash every state entry and return a deterministic aggregate digest."""
    entries: dict[str, str] = {}
    aggregate = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if not torch.is_tensor(value):
            raise TypeError(f"model state entry is not a tensor: {name}")
        # The authoritative tensor helper predates scalar state entries and
        # cannot byte-view a 0-D tensor.  Preserve the original shape in the
        # aggregate digest while hashing one scalar as a one-element buffer.
        hashable = value if value.ndim > 0 else value.reshape(1)
        entry_hash = tensor_sha256(hashable)
        entries[name] = entry_hash
        aggregate.update(name.encode("utf-8"))
        aggregate.update(str(tuple(value.shape)).encode("ascii"))
        aggregate.update(entry_hash.encode("ascii"))
    return aggregate.hexdigest(), entries


def _expanded_condition_mask(
    condition_mask: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    if condition_mask.ndim != 5 or condition_mask.shape[1] != 1:
        raise ValueError("condition_mask must have shape [B,1,X,Y,Z]")
    mask = condition_mask.to(device=reference.device, dtype=torch.bool)
    if mask.shape[0] == 1 and reference.shape[0] > 1:
        mask = mask.expand(reference.shape[0], -1, -1, -1, -1)
    if mask.shape[0] != reference.shape[0] or mask.shape[2:] != reference.shape[2:]:
        raise ValueError("condition_mask must match the source batch/spatial shape")
    return mask.expand(-1, reference.shape[1], -1, -1, -1)


def _condition_embedding_error(
    state: torch.Tensor,
    embedded_conditions: torch.Tensor,
    condition_mask: torch.Tensor,
) -> float:
    mask = _expanded_condition_mask(condition_mask, state)
    values = embedded_conditions.to(device=state.device, dtype=state.dtype)
    if values.shape[0] == 1 and state.shape[0] > 1:
        values = values.expand(state.shape[0], -1, -1, -1, -1)
    if values.shape != state.shape:
        raise ValueError("embedded_conditions must broadcast to the state shape")
    if not bool(mask.any()):
        return 0.0
    return float((state[mask] - values[mask]).abs().max().detach().cpu())


def differentiable_flow_solve(
    model,
    initial_state: torch.Tensor,
    conditioning: torch.Tensor,
    embedded_conditions: torch.Tensor,
    condition_mask: torch.Tensor,
    n_steps: int,
    solver: str = "fixed_euler",
    *,
    use_gradient_checkpointing: bool = True,
    condition_values: torch.Tensor | None = None,
    collect_step_diagnostics: bool = False,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Integrate the authoritative projected midpoint-grid fixed Euler Flow.

    No ``torch.no_grad`` context is used around ``model.net``.  Freezing model
    parameters removes weight gradients while preserving the full derivative
    of every velocity evaluation with respect to the evolving state.
    """
    if solver not in DFLOW_SOLVERS:
        raise ValueError(f"solver must be one of {DFLOW_SOLVERS}")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if initial_state.ndim != 5:
        raise ValueError("initial_state must have shape [B,E,X,Y,Z]")
    if conditioning.ndim != 5 or conditioning.shape[1:] != initial_state.shape[1:]:
        raise ValueError("conditioning and initial_state channel/spatial shapes must match")

    # Conditioning and projected values are immutable experiment inputs.  A
    # caller may have constructed them before freezing the embedding; detach
    # them here so repeated source-optimization iterations never reuse such a
    # stale embedding graph.  This does not detach the evolving source state.
    embedded_conditions = embedded_conditions.detach()
    conditioning = conditioning.detach()
    state, initial_projection_norm = _project_conditions(
        initial_state,
        embedded_conditions,
        condition_mask,
    )
    conditioning_device = conditioning.to(device=state.device, dtype=state.dtype)
    if conditioning_device.shape[0] == 1 and state.shape[0] > 1:
        conditioning_device = conditioning_device.expand(
            state.shape[0], -1, -1, -1, -1
        )
    if conditioning_device.shape != state.shape:
        raise ValueError("conditioning must broadcast to the full state shape")

    dt = 1.0 / int(n_steps)
    step_rows: list[dict[str, object]] = []
    for step in range(int(n_steps)):
        t_value = (step + 0.5) / int(n_steps)
        time_value = torch.full(
            (state.shape[0],),
            t_value,
            device=state.device,
            dtype=state.dtype,
        )

        if use_gradient_checkpointing and torch.is_grad_enabled() and state.requires_grad:
            velocity = checkpoint(
                model.net,
                state,
                conditioning_device,
                time_value,
                use_reentrant=False,
            )
        else:
            # This call must remain outside torch.no_grad: the source-state VJP
            # depends on the graph from state through model.net to velocity.
            velocity = model.net(state, conditioning_device, time_value)

        candidate = state + dt * velocity
        next_state, projection_norm = _project_conditions(
            candidate,
            embedded_conditions,
            condition_mask,
        )
        condition_error = _condition_embedding_error(
            next_state,
            embedded_conditions,
            condition_mask,
        )
        if condition_error != 0.0:
            raise RuntimeError(
                "hard-condition embedding projection failed: "
                f"max_abs_error={condition_error}"
            )
        if collect_step_diagnostics:
            step_rows.append(
                {
                    "step": step,
                    "t": t_value,
                    "dt": dt,
                    "state_norm": float(next_state.detach().norm().cpu()),
                    "velocity_norm": float(velocity.detach().norm().cpu()),
                    "condition_projection_norm": projection_norm,
                    "condition_embedding_max_abs_error": condition_error,
                }
            )
        state = next_state

    decoded_violations: int | None = None
    if condition_values is not None:
        decoded_violations = _condition_violations(
            model,
            state,
            condition_values,
            condition_mask,
        )
        if decoded_violations != 0:
            raise RuntimeError(
                f"hard-condition projection produced {decoded_violations} decoded violations"
            )
    diagnostics: dict[str, object] = {
        "solver": solver,
        "integrator_semantics": "fixed_euler_midpoint_v1",
        "n_steps": int(n_steps),
        "dt": dt,
        "use_gradient_checkpointing": bool(use_gradient_checkpointing),
        "gradient_checkpoint_policy": (
            CHECKPOINT_POLICY if use_gradient_checkpointing else None
        ),
        "initial_condition_projection_norm": initial_projection_norm,
        "final_condition_embedding_max_abs_error": _condition_embedding_error(
            state,
            embedded_conditions,
            condition_mask,
        ),
        "hard_condition_violations": decoded_violations,
        "step_diagnostics": step_rows,
    }
    return state, diagnostics


def _source_shell_terms(
    source: torch.Tensor,
    condition_mask: torch.Tensor,
) -> tuple[torch.Tensor, float, int]:
    free = ~_expanded_condition_mask(condition_mask, source)
    d_free = int(free.sum().item())
    if d_free <= 0:
        raise ValueError("source optimization requires at least one free value")
    free_norm = source[free].square().sum().sqrt()
    expected = math.sqrt(d_free)
    relative_deviation = (free_norm - expected) / expected
    regularization = relative_deviation.square()
    return regularization, expected, d_free


def _scalar_diagnostics(values: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in values.items():
        if torch.is_tensor(value):
            if value.numel() == 1:
                result[key] = float(value.detach().cpu())
        elif isinstance(value, (bool, int, float, str)) or value is None:
            result[key] = value
    return result


def optimize_source_noise(
    model,
    initial_source: torch.Tensor,
    conditioning: torch.Tensor,
    embedded_conditions: torch.Tensor,
    condition_mask: torch.Tensor,
    endpoint_loss: EndpointLoss,
    *,
    n_steps: int,
    solver: str = "fixed_euler",
    optimizer_name: str = "adam",
    learning_rate: float = 1e-2,
    optimization_iterations: int = 20,
    lbfgs_max_iter: int = 5,
    source_regularization_weight: float = 0.0,
    use_gradient_checkpointing: bool = True,
    condition_values: torch.Tensor | None = None,
    trace_diagnostics: TraceDiagnostics | None = None,
) -> DFlowOptimizationResult:
    """Optimize only the initial Flow state against an endpoint objective."""
    if optimizer_name not in DFLOW_OPTIMIZERS:
        raise ValueError(f"optimizer_name must be one of {DFLOW_OPTIMIZERS}")
    if learning_rate <= 0 or not math.isfinite(float(learning_rate)):
        raise ValueError("learning_rate must be finite and positive")
    if optimization_iterations < 0:
        raise ValueError("optimization_iterations must be non-negative")
    if lbfgs_max_iter <= 0:
        raise ValueError("lbfgs_max_iter must be positive")
    if source_regularization_weight < 0:
        raise ValueError("source_regularization_weight must be non-negative")

    freeze_model_for_dflow(model)
    model_hash_before, state_hashes_before = model_state_dict_hashes(model)
    initial = initial_source.detach().clone()
    source = torch.nn.Parameter(initial.clone(), requires_grad=True)
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam([source], lr=float(learning_rate))
    else:
        optimizer = torch.optim.LBFGS(
            [source],
            lr=float(learning_rate),
            max_iter=int(lbfgs_max_iter),
            line_search_fn="strong_wolfe",
        )
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_parameter_ids != {id(source)}:
        raise RuntimeError("D-Flow optimizer must contain only source_state")

    trace: list[dict[str, object]] = []
    closure_evaluations = 0
    started = time.perf_counter()

    def evaluate(backward: bool) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, object],
        dict[str, object],
    ]:
        final_state, flow_diagnostics = differentiable_flow_solve(
            model=model,
            initial_state=source,
            conditioning=conditioning,
            embedded_conditions=embedded_conditions,
            condition_mask=condition_mask,
            n_steps=n_steps,
            solver=solver,
            use_gradient_checkpointing=use_gradient_checkpointing,
            condition_values=condition_values,
        )
        primary, objective_diagnostics = endpoint_loss(final_state)
        if primary.ndim != 0 or not torch.isfinite(primary):
            raise FloatingPointError("endpoint primary loss must be one finite scalar")
        shell_regularization, _, _ = _source_shell_terms(source, condition_mask)
        total = primary + float(source_regularization_weight) * shell_regularization
        if not torch.isfinite(total):
            raise FloatingPointError("D-Flow total loss is non-finite")
        if backward:
            total.backward()
        return (
            total,
            primary,
            final_state,
            _scalar_diagnostics(objective_diagnostics),
            flow_diagnostics,
        )

    for iteration in range(1, int(optimization_iterations) + 1):
        previous = source.detach().clone()
        closure_before = closure_evaluations
        gradient_norm = float("nan")

        if optimizer_name == "adam":
            optimizer.zero_grad(set_to_none=True)
            evaluate(backward=True)
            if source.grad is None or not torch.isfinite(source.grad).all():
                raise FloatingPointError("source gradient is missing or non-finite")
            gradient_norm = float(source.grad.detach().norm().cpu())
            optimizer.step()
            closure_evaluations += 1
        else:
            last_gradient_norm = [float("nan")]

            def closure() -> torch.Tensor:
                nonlocal closure_evaluations
                optimizer.zero_grad(set_to_none=True)
                total_value, _, _, _, _ = evaluate(backward=True)
                if source.grad is None or not torch.isfinite(source.grad).all():
                    raise FloatingPointError("source gradient is missing or non-finite")
                last_gradient_norm[0] = float(source.grad.detach().norm().cpu())
                closure_evaluations += 1
                return total_value

            optimizer.step(closure)
            gradient_norm = last_gradient_norm[0]

        update_norm = float((source.detach() - previous).norm().cpu())
        total, primary, final_state, objective_values, flow_values = evaluate(
            backward=False
        )
        shell_regularization, expected_shell_norm, d_free = _source_shell_terms(
            source,
            condition_mask,
        )
        source_norm = float(source.detach().norm().cpu())
        free_mask = ~_expanded_condition_mask(condition_mask, source)
        free_norm = float(source[free_mask].detach().norm().cpu())
        shell_deviation = free_norm - expected_shell_norm
        extra_values = (
            _scalar_diagnostics(trace_diagnostics(final_state))
            if trace_diagnostics is not None
            else {}
        )
        hard_violations = flow_values["hard_condition_violations"]
        trace.append(
            {
                "iteration": iteration,
                "optimizer": optimizer_name,
                "closure_evaluation_count": closure_evaluations - closure_before,
                "closure_evaluation_count_cumulative": closure_evaluations,
                "endpoint_total_loss": float(total.detach().cpu()),
                "endpoint_primary_loss": float(primary.detach().cpu()),
                "source_regularization_loss": float(shell_regularization.detach().cpu()),
                "source_regularization_weight": float(source_regularization_weight),
                "source_gradient_norm": gradient_norm,
                "source_update_norm": update_norm,
                "source_norm": source_norm,
                "source_free_norm": free_norm,
                "source_free_dimension": d_free,
                "source_shell_expected_norm": expected_shell_norm,
                "source_shell_deviation": shell_deviation,
                "endpoint_state_norm": float(final_state.detach().norm().cpu()),
                "hard_condition_violations": hard_violations,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "runtime_seconds": time.perf_counter() - started,
                **objective_values,
                **extra_values,
            }
        )
        del total, primary, final_state

    final_state, final_flow_diagnostics = differentiable_flow_solve(
        model=model,
        initial_state=source,
        conditioning=conditioning,
        embedded_conditions=embedded_conditions,
        condition_mask=condition_mask,
        n_steps=n_steps,
        solver=solver,
        use_gradient_checkpointing=use_gradient_checkpointing,
        condition_values=condition_values,
    )
    model_hash_after, state_hashes_after = model_state_dict_hashes(model)
    if model_hash_before != model_hash_after or state_hashes_before != state_hashes_after:
        raise RuntimeError("frozen model state_dict changed during D-Flow optimization")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("a frozen model parameter accumulated a gradient")

    diagnostics = {
        "engine_version": DFLOW_ENGINE_VERSION,
        "model_state_dict_sha256_before": model_hash_before,
        "model_state_dict_sha256_after": model_hash_after,
        "model_state_dict_unchanged": True,
        "model_state_tensor_hashes_before": state_hashes_before,
        "model_state_tensor_hashes_after": state_hashes_after,
        "trainable_model_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "optimizer_parameter_tensor_count": 1,
        "optimizer_parameter_element_count": source.numel(),
        "optimizer_contains_only_source_state": True,
        "initial_noise_sha256": tensor_sha256(initial),
        "optimized_noise_sha256": tensor_sha256(source.detach()),
        "final_state_sha256": tensor_sha256(final_state.detach()),
        "optimization_iterations": int(optimization_iterations),
        "closure_evaluation_count": closure_evaluations,
        "runtime_seconds": time.perf_counter() - started,
        "final_flow_diagnostics": final_flow_diagnostics,
    }
    return DFlowOptimizationResult(
        initial_source=initial,
        optimized_source=source.detach().clone(),
        final_state=final_state.detach().clone(),
        trace=trace,
        diagnostics=diagnostics,
    )
