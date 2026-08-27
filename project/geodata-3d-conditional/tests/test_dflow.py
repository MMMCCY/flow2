from __future__ import annotations

import copy

import torch
from torch import nn
import torch.nn.functional as F

from guidance.dflow import (
    differentiable_flow_solve,
    model_state_dict_hashes,
    optimize_source_noise,
)
from scripts.stage16.run_dflow import (
    objective_required_asset_keys,
    seismic_runner_requires_truth,
)


class TinyVelocity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.15))
        self.condition_gain = nn.Parameter(torch.tensor(0.05))

    def forward(
        self,
        state: torch.Tensor,
        conditioning: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        time_term = time.view(-1, 1, 1, 1, 1) * 0.01
        return self.gain * state + self.condition_gain * conditioning + time_term


class TinyFlow(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(2, 2)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.tensor([[-1.0, 0.0], [1.0, 0.0]]))
        self.net = TinyVelocity()
        self.embedding_dim = 2
        self.data_shape = (2, 2, 2)

    def embed(self, labels: torch.Tensor) -> torch.Tensor:
        indices = labels.squeeze(1).long() + 1
        return self.embedding(indices).permute(0, 4, 1, 2, 3).contiguous()

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        state_normalized = F.normalize(state, dim=1)
        embeddings = F.normalize(self.embedding.weight, dim=1)
        logits = torch.einsum("bexyz,ce->bcxyz", state_normalized, embeddings)
        return logits.argmax(dim=1)


def tiny_inputs(seed: int = 7):
    model = TinyFlow()
    condition_values = torch.full((1, 1, 2, 2, 2), -1, dtype=torch.long)
    condition_mask = torch.zeros_like(condition_values, dtype=torch.bool)
    condition_mask[..., 0, 0, 0] = True
    embedded = model.embed(condition_values)
    conditioning = embedded * condition_mask.expand_as(embedded)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    source = torch.randn((1, 2, 2, 2, 2), generator=generator)
    return model, source, conditioning, embedded, condition_mask, condition_values


def endpoint_target_loss(target: torch.Tensor):
    def loss(state: torch.Tensor):
        value = (state - target.to(state)).square().mean()
        return value, {"mock_loss": value}

    return loss


def solve(model, source, conditioning, embedded, mask, values, checkpointing):
    return differentiable_flow_solve(
        model,
        source,
        conditioning,
        embedded,
        mask,
        n_steps=4,
        solver="fixed_euler",
        use_gradient_checkpointing=checkpointing,
        condition_values=values,
    )


def test_optimizer_changes_only_source_not_model_parameters():
    model, source, conditioning, embedded, mask, values = tiny_inputs()
    before, _ = model_state_dict_hashes(model)
    result = optimize_source_noise(
        model,
        source,
        conditioning,
        embedded,
        mask,
        endpoint_target_loss(torch.zeros_like(source)),
        n_steps=4,
        optimization_iterations=3,
        optimizer_name="adam",
        learning_rate=0.05,
        condition_values=values,
    )
    after, _ = model_state_dict_hashes(model)
    assert not torch.equal(result.initial_source, result.optimized_source)
    assert before == after
    assert result.diagnostics["trainable_model_parameter_count"] == 0
    assert result.diagnostics["optimizer_parameter_tensor_count"] == 1


def test_through_flow_gradient_is_nonzero_and_finite():
    model, source, conditioning, embedded, mask, values = tiny_inputs()
    source = source.requires_grad_(True)
    final, _ = solve(model, source, conditioning, embedded, mask, values, True)
    gradient = torch.autograd.grad(final.square().mean(), source)[0]
    assert torch.isfinite(gradient).all()
    assert float(gradient.norm()) > 0


def test_hard_condition_projection_is_exact_after_flow():
    model, source, conditioning, embedded, mask, values = tiny_inputs()
    final, diagnostics = solve(
        model, source, conditioning, embedded, mask, values, False
    )
    expanded = mask.expand_as(final)
    assert torch.equal(final[expanded], embedded[expanded])
    assert diagnostics["hard_condition_violations"] == 0
    assert diagnostics["final_condition_embedding_max_abs_error"] == 0.0


def test_zero_optimization_matches_paired_baseline():
    model, source, conditioning, embedded, mask, values = tiny_inputs()
    baseline, _ = solve(model, source, conditioning, embedded, mask, values, True)
    result = optimize_source_noise(
        model,
        source,
        conditioning,
        embedded,
        mask,
        endpoint_target_loss(torch.zeros_like(source)),
        n_steps=4,
        optimization_iterations=0,
        condition_values=values,
    )
    assert torch.equal(source, result.optimized_source)
    assert torch.allclose(baseline, result.final_state, atol=0.0, rtol=0.0)


def test_same_seed_is_deterministic():
    inputs_a = tiny_inputs(seed=123)
    inputs_b = tiny_inputs(seed=123)
    results = []
    for model, source, conditioning, embedded, mask, values in (inputs_a, inputs_b):
        results.append(
            optimize_source_noise(
                model,
                source,
                conditioning,
                embedded,
                mask,
                endpoint_target_loss(torch.zeros_like(source)),
                n_steps=4,
                optimization_iterations=3,
                optimizer_name="adam",
                learning_rate=0.05,
                condition_values=values,
            )
        )
    assert torch.equal(results[0].optimized_source, results[1].optimized_source)
    assert torch.equal(results[0].final_state, results[1].final_state)


def test_mock_inverse_endpoint_loss_decreases():
    model, source, conditioning, embedded, mask, values = tiny_inputs(seed=99)
    target = torch.zeros_like(source)
    baseline, _ = solve(model, source, conditioning, embedded, mask, values, True)
    initial_loss = float((baseline - target).square().mean().detach())
    result = optimize_source_noise(
        model,
        source,
        conditioning,
        embedded,
        mask,
        endpoint_target_loss(target),
        n_steps=4,
        optimization_iterations=10,
        optimizer_name="adam",
        learning_rate=0.1,
        condition_values=values,
    )
    final_loss = float((result.final_state - target).square().mean())
    assert final_loss < initial_loss


def test_gradient_checkpointing_on_off_are_numerically_consistent():
    model, source, conditioning, embedded, mask, values = tiny_inputs(seed=13)
    model_without = copy.deepcopy(model)
    source_on = source.clone().requires_grad_(True)
    source_off = source.clone().requires_grad_(True)
    final_on, _ = solve(
        model, source_on, conditioning, embedded, mask, values, True
    )
    final_off, _ = solve(
        model_without, source_off, conditioning, embedded, mask, values, False
    )
    grad_on = torch.autograd.grad(final_on.square().mean(), source_on)[0]
    grad_off = torch.autograd.grad(final_off.square().mean(), source_off)[0]
    assert torch.allclose(final_on, final_off, atol=1e-7, rtol=1e-6)
    assert torch.allclose(grad_on, grad_off, atol=1e-7, rtol=1e-6)


def test_seismic_runner_api_does_not_require_truth_geology():
    keys = objective_required_asset_keys("seismic")
    assert not seismic_runner_requires_truth()
    assert all("truth" not in key for key in keys)
    assert "observed_seismic" in keys
    assert "condition_values" in keys


def test_model_state_dict_hash_unchanged_before_after_optimization():
    model, source, conditioning, embedded, mask, values = tiny_inputs(seed=81)
    before, entries_before = model_state_dict_hashes(model)
    result = optimize_source_noise(
        model,
        source,
        conditioning,
        embedded,
        mask,
        endpoint_target_loss(torch.zeros_like(source)),
        n_steps=4,
        optimization_iterations=2,
        condition_values=values,
    )
    after, entries_after = model_state_dict_hashes(model)
    assert before == after
    assert entries_before == entries_after
    assert result.diagnostics["model_state_dict_unchanged"] is True
