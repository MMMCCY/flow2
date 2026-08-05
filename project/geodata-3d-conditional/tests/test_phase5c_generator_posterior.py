from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from guidance.generator_posterior import (
    CONDITION_PROJECTION_POLICY,
    GENERATOR_POSTERIOR_VERSION,
    latent_diagnostics,
    metropolis_decision,
    pcn_proposal,
    posterior_energy,
    projected_fixed_euler_prior_sample,
)
import inference_runtime as runtime
from scripts.stage5.run_generator_posterior import (
    _validate_historical_baseline,
    validate_protocol_config,
)


class _ConstantVelocityNet(torch.nn.Module):
    def forward(
        self,
        state: torch.Tensor,
        conditioning: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        del conditioning, time
        return torch.full_like(state, 0.04)


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


def _protocol() -> dict[str, object]:
    return {
        "schema": "phase5c_generator_posterior_config_v1",
        "status": "frozen_before_cuda_output",
        "sampler": GENERATOR_POSTERIOR_VERSION,
        "condition_policy": CONDITION_PROJECTION_POLICY,
        "truth_used_by_sampler": False,
        "allow_parameter_sweep_on_case": False,
        "initial_seed": 42,
        "proposal_seed": 5501,
        "n_euler_steps": 32,
        "pcn_beta": 0.1,
        "likelihood_weight": 1.0,
        "performance_smoke_proposals": 1,
        "primary_pilot_proposals": 8,
    }


def test_frozen_protocol_file_matches_implementation() -> None:
    path = (
        PROJECT_DIR
        / "experiments/stage5_generator_posterior/configs"
        / "cond0_hard_pcn_mechanism_v1.json"
    )
    config = json.loads(path.read_text(encoding="utf-8"))
    smoke = validate_protocol_config(config, "performance_smoke")
    pilot = validate_protocol_config(config, "primary_pilot")
    assert smoke["chain_proposals"] == 1
    assert pilot["chain_proposals"] == 8
    assert smoke["n_steps"] == 32
    assert smoke["beta"] == pytest.approx(0.1)


def test_protocol_rejects_truth_use_and_parameter_sweep() -> None:
    config = _protocol()
    config["truth_used_by_sampler"] = True
    with pytest.raises(ValueError, match="truth metrics"):
        validate_protocol_config(config, "performance_smoke")
    config = _protocol()
    config["allow_parameter_sweep_on_case"] = True
    with pytest.raises(ValueError, match="parameter sweeps"):
        validate_protocol_config(config, "performance_smoke")


def test_beta_zero_is_exact_and_does_not_advance_generator() -> None:
    current = torch.randn(2, 3, 2, 2, 2)
    generator = torch.Generator(device="cpu").manual_seed(77)
    before = generator.get_state().clone()
    proposal, innovation = pcn_proposal(current, beta=0.0, generator=generator)
    assert torch.equal(proposal, current)
    assert torch.count_nonzero(innovation) == 0
    assert torch.equal(generator.get_state(), before)


def test_pcn_proposal_matches_prior_preserving_formula() -> None:
    current = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 1, 2)
    generator = torch.Generator(device="cpu").manual_seed(13)
    proposal, innovation = pcn_proposal(current, beta=0.25, generator=generator)
    expected = (1.0 - 0.25**2) ** 0.5 * current + 0.25 * innovation
    assert torch.equal(proposal, expected)
    diagnostics = latent_diagnostics(proposal)
    assert diagnostics["latent_l2"] > 0
    assert diagnostics["latent_std"] > 0
    with pytest.raises(ValueError, match="CPU"):
        pcn_proposal(current.to("meta"), beta=0.1, generator=generator)


def test_metropolis_decision_and_energy() -> None:
    assert posterior_energy(8.0, 0.5) == pytest.approx(2.0)
    better = metropolis_decision(4.0, 3.0, 0.999)
    assert better["accepted"] is True
    worse_rejected = metropolis_decision(3.0, 5.0, 0.5)
    assert worse_rejected["accepted"] is False
    worse_accepted = metropolis_decision(3.0, 3.1, 0.5)
    assert worse_accepted["accepted"] is True
    with pytest.raises(ValueError, match="strictly"):
        metropolis_decision(1.0, 1.0, 0.0)


def test_projected_fixed_euler_matches_manual_no_guidance_path() -> None:
    torch.manual_seed(9)
    model = _DummyConditionalModel()
    truth = torch.tensor([[[[[-1, 0, 1, 0]]]]], dtype=torch.long)
    condition = torch.zeros_like(truth, dtype=torch.bool)
    condition[..., 0] = True
    condition[..., 2] = True
    embedded = model.embed(truth)
    conditioning = torch.where(
        condition.expand_as(embedded), embedded, torch.zeros_like(embedded)
    )
    initial = torch.randn(1, 3, 1, 1, 4)
    actual = projected_fixed_euler_prior_sample(
        model,
        initial,
        conditioning,
        embedded,
        condition,
        n_steps=4,
    )
    expected = torch.where(condition.expand_as(initial), embedded, initial)
    for _ in range(4):
        expected = expected + 0.04 / 4
        expected = torch.where(condition.expand_as(expected), embedded, expected)
    assert torch.equal(actual, expected)
    decoded = (model.decode(actual) - 1).unsqueeze(1)
    assert int(((decoded != truth) & condition).sum()) == 0


def test_historical_regression_normalizes_saved_sample_shape(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.bin"
    truth = tmp_path / "truth.pt"
    boreholes = tmp_path / "boreholes.pt"
    observation_dir = tmp_path / "observation"
    baseline_dir = tmp_path / "baseline"
    observation_dir.mkdir()
    baseline_dir.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    torch.save(torch.zeros(1), truth)
    torch.save(torch.zeros(1), boreholes)
    (observation_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    saved = torch.zeros((2, 2, 2), dtype=torch.long)
    torch.save(saved, baseline_dir / "sample_0.pt")
    (baseline_dir / "config.json").write_text("{}\n", encoding="utf-8")
    baseline_config = {
        "run_status": "completed",
        "model_weight_source": "ema",
        "ema_applied": True,
        "integrator": runtime.PAIRED_INTEGRATOR,
        "seed": 42,
        "n_steps": 32,
        "alpha": 0.0,
        "checkpoint_sha256": runtime.file_sha256(checkpoint),
        "truth_model_sha256": runtime.file_sha256(truth),
        "boreholes_sha256": runtime.file_sha256(boreholes),
        "observation_manifest_sha256": runtime.file_sha256(
            observation_dir / "manifest.json"
        ),
        "initial_noise_sha256": ["noise"],
    }
    report = _validate_historical_baseline(
        baseline_dir=baseline_dir,
        baseline_config=baseline_config,
        initial_decoded=saved.unsqueeze(0),
        initial_noise_sha256="noise",
        ckpt_path=checkpoint,
        truth_path=truth,
        boreholes_path=boreholes,
        observation_dir=observation_dir,
        initial_seed=42,
        n_steps=32,
    )
    assert report["exact_initial_hard_regression"] is True
