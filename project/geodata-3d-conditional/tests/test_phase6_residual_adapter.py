from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn.functional as F


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from guidance.generator_posterior import projected_fixed_euler_prior_sample
from guidance.residual_velocity_adapter import (
    ResidualVelocityAdapter,
    cap_residual_velocity,
    class_balancing_weights,
    fixed_euler_adapter_sample,
    residual_adapter_losses,
)
from scripts.stage6.train_oracle_adapter_smoke import validate_config


class _ConstantVelocityNet(torch.nn.Module):
    def forward(
        self,
        state: torch.Tensor,
        conditioning: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        del conditioning, time
        return torch.full_like(state, 0.03)


class _DummyConditionalModel:
    def __init__(self) -> None:
        self.embedding = torch.nn.Embedding(3, 3)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.eye(3))
        self.embedding.weight.requires_grad = False
        self.net = _ConstantVelocityNet()

    def embed(self, labels: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(labels.squeeze(1).long() + 1)
        return embedded.permute(0, 4, 1, 2, 3).contiguous()

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(state, dim=1)
        embeddings = F.normalize(self.embedding.weight, dim=1)
        return torch.einsum("bexyz,ce->bcxyz", normalized, embeddings).argmax(dim=1)


def test_frozen_phase6_smoke_config_matches_implementation() -> None:
    import json

    path = (
        PROJECT_DIR
        / "experiments/stage6_geo_adapter/configs/oracle_acoustic_tiny_overfit_v1.json"
    )
    config = json.loads(path.read_text(encoding="utf-8"))
    resolved = validate_config(config)
    assert resolved["base_width"] == 12
    assert resolved["training_steps"] == 80
    assert resolved["state_seeds"] == [6100, 6101, 6102, 6103]
    config["allow_hyperparameter_sweep_on_case"] = True
    try:
        validate_config(config)
    except ValueError as error:
        assert "hyperparameter sweeps" in str(error)
    else:
        raise AssertionError("truth-audited smoke unexpectedly allowed a sweep")


def _inputs():
    torch.manual_seed(31)
    shape = (3, 3, 3)
    state = torch.randn(1, 3, *shape)
    base = torch.randn_like(state)
    conditioning = torch.zeros_like(state)
    condition = torch.zeros(1, 1, *shape, dtype=torch.bool)
    condition[..., 0, 0, 0] = True
    geophysics = torch.randn(1, 2, *shape)
    time = torch.tensor([0.4])
    return state, base, conditioning, condition, geophysics, time


def test_adapter_is_small_zero_initialized_and_condition_zero() -> None:
    adapter = ResidualVelocityAdapter(
        3, geophysics_channels=2, base_width=8, dilations=(1, 2)
    )
    state, base, conditioning, condition, geophysics, time = _inputs()
    correction = adapter(
        state, base, conditioning, condition, geophysics, time
    )
    assert adapter.parameter_count() < 2_000_000
    assert torch.count_nonzero(correction) == 0
    assert torch.count_nonzero(correction[condition.expand_as(correction)]) == 0


def test_cap_residual_velocity_obeys_ratio_and_conditions() -> None:
    state, base, _, condition, _, _ = _inputs()
    correction = 10.0 * state
    capped, used = cap_residual_velocity(
        correction, base, condition, max_ratio=0.25
    )
    assert float(used.max()) <= 0.250001
    assert torch.count_nonzero(capped[condition.expand_as(capped)]) == 0


def test_adapter_losses_backpropagate_into_zero_output_layer() -> None:
    adapter = ResidualVelocityAdapter(
        3, geophysics_channels=2, base_width=8, dilations=(1,)
    )
    state, base, conditioning, condition, geophysics, time = _inputs()
    truth = torch.zeros(1, 1, 3, 3, 3, dtype=torch.long)
    truth[..., 1:, 1:, 1:] = 1
    active = (~condition) & (truth != -1)
    weights = class_balancing_weights(truth, active, 3)
    correction = adapter(
        state, base.detach(), conditioning, condition, geophysics, time
    )
    target_velocity = torch.randn_like(state)
    loss, diagnostics = residual_adapter_losses(
        state=state,
        target_velocity=target_velocity,
        base_velocity=base.detach(),
        correction=correction,
        truth=truth,
        condition_mask=condition,
        embedding_weight=torch.eye(3),
        time=time,
        class_weights=weights,
        logit_temperature=0.2,
        flow_weight=1.0,
        cross_entropy_weight=0.25,
        dice_weight=0.25,
        residual_regularizer_weight=1e-4,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert 0 <= float(diagnostics["endpoint_accuracy"]) <= 1
    assert adapter.output_conv.weight.grad is not None
    assert float(adapter.output_conv.weight.grad.norm()) > 0
    assert torch.count_nonzero(correction[condition.expand_as(correction)]) == 0


def test_adapter_scale_zero_exactly_matches_projected_baseline() -> None:
    torch.manual_seed(41)
    model = _DummyConditionalModel()
    adapter = ResidualVelocityAdapter(
        3, geophysics_channels=2, base_width=8, dilations=(1,)
    )
    truth = torch.tensor(
        [[[[[-1, 0, 1], [0, 1, 0]], [[1, 0, 1], [0, 1, 0]]]]],
        dtype=torch.long,
    )
    condition = torch.zeros_like(truth, dtype=torch.bool)
    condition[..., 0] = True
    embedded = model.embed(truth)
    conditioning = torch.where(
        condition.expand_as(embedded), embedded, torch.zeros_like(embedded)
    )
    initial = torch.randn_like(embedded)
    geophysics = torch.randn(1, 2, *truth.shape[2:])
    expected = projected_fixed_euler_prior_sample(
        model,
        initial,
        conditioning,
        embedded,
        condition,
        n_steps=4,
    )
    actual, trace = fixed_euler_adapter_sample(
        model=model,
        adapter=adapter,
        initial_state=initial,
        conditioning=conditioning,
        embedded_conditions=embedded,
        condition_mask=condition,
        geophysics=geophysics,
        n_steps=4,
        adapter_scale=0.0,
        max_residual_ratio=0.25,
    )
    assert torch.equal(actual, expected)
    assert all(row["used_residual_ratio"] == 0.0 for row in trace)
