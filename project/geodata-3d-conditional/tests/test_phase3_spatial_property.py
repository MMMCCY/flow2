from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from guidance.property_sampling import fixed_euler_property_sample
from guidance.property_volume import hard_labels_to_properties
from guidance.spatial_property import (
    SPATIAL_PROPERTY_CONFIG_SCHEMA,
    SPATIAL_PROPERTY_LOSS_MODE,
    apply_spatial_property_operator,
    build_spatial_property_observation,
    depth_exponential_confidence,
    hard_spatial_observation_loss,
    spatial_property_volume_loss,
    validate_spatial_property_config,
)
from scripts.stage3.run_spatial_property_guidance import (
    PHASE3_PAIR_FIELDS,
    _add_hard_observation_metrics,
    paired_spatial_property_config_verdict,
    validate_args,
)
from scripts.stage3.audit_spatial_screen import _classification
from scripts.stage3.summarize_gaussian_screen import screen_decision


def _config(
    operator: dict[str, object] | None = None,
    confidence: dict[str, object] | None = None,
    noise: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema": SPATIAL_PROPERTY_CONFIG_SCHEMA,
        "id": "phase3_test",
        "description": "small deterministic Phase-3 test",
        "operator": operator or {"type": "identity"},
        "confidence": confidence or {"type": "base"},
        "noise": noise or {"type": "none"},
    }


def _target_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    table = torch.tensor([[0.0, 1.0, 3.0], [0.0, 4.0, 2.0]])
    truth = torch.zeros((1, 1, 4, 4, 4), dtype=torch.long)
    truth[..., 2:] = 1
    target = hard_labels_to_properties(truth, table)
    condition = torch.zeros_like(truth, dtype=torch.bool)
    condition[..., 0, 0, 0] = True
    confidence = ((truth != -1) & ~condition).float()
    return truth, table, target, confidence


def test_spatial_config_validates_operator_shape_and_schema() -> None:
    metadata = validate_spatial_property_config(
        _config({"type": "average_pool", "factor": 2}),
        spatial_shape=(4, 6, 8),
    )
    assert metadata["operator"]["factor"] == [2, 2, 2]
    assert metadata["truth_derived"] is True
    assert metadata["is_measured_geophysics"] is False

    with pytest.raises(ValueError, match="divide"):
        validate_spatial_property_config(
            _config({"type": "average_pool", "factor": 3}),
            spatial_shape=(4, 6, 8),
        )
    with pytest.raises(ValueError, match="schema"):
        validate_spatial_property_config(
            {**_config(), "schema": "wrong"},
            spatial_shape=(4, 4, 4),
        )


def test_identity_observation_and_exact_hard_target_have_zero_loss() -> None:
    truth, table, target, confidence = _target_inputs()
    condition = ~confidence.bool()
    observation = build_spatial_property_observation(
        target,
        confidence,
        truth != -1,
        _config(),
    )
    predicted = hard_labels_to_properties(truth, table)
    loss, _ = hard_spatial_observation_loss(
        predicted,
        target,
        condition,
        observation,
        _config(),
        sigmas=(0.0,),
        scale_weights=(1.0,),
        channel_weights=torch.tensor([0.5, 0.5]),
    )

    assert torch.equal(observation.values, target)
    assert torch.count_nonzero(observation.noise) == 0
    assert loss == pytest.approx(0.0, abs=1e-8)


def test_gaussian_and_average_pool_are_matched_channelwise_operators() -> None:
    volume = torch.zeros((1, 2, 9, 9, 9))
    volume[:, 0, 4, 4, 4] = 1.0
    blurred = apply_spatial_property_operator(
        volume,
        {"type": "gaussian_blur", "sigma_voxels": 1.0},
    )
    coarse_volume = torch.zeros((1, 2, 4, 4, 4))
    coarse_volume[:, 0, 1:3, 1:3, 1:3] = 8.0
    pooled = apply_spatial_property_operator(
        coarse_volume,
        {"type": "average_pool", "factor": [2, 2, 2]},
    )

    assert blurred.shape == volume.shape
    assert blurred[:, 0].sum() == pytest.approx(1.0, rel=1e-5)
    assert torch.count_nonzero(blurred[:, 1]) == 0
    assert pooled.shape == (1, 2, 2, 2, 2)
    assert pooled[:, 0].sum() == pytest.approx(8.0)
    assert torch.count_nonzero(pooled[:, 1]) == 0


def test_depth_confidence_uses_last_axis_and_decays_downward() -> None:
    nonair = torch.zeros((1, 1, 1, 1, 6), dtype=torch.bool)
    nonair[..., :5] = True
    weights = depth_exponential_confidence(
        nonair,
        e_folding_depth_voxels=2.0,
        floor=0.0,
    )

    assert weights[..., 5].item() == 0.0
    assert weights[..., 4].item() == pytest.approx(1.0)
    assert weights[..., 2].item() == pytest.approx(torch.exp(torch.tensor(-1.0)).item())
    assert weights[..., 0].item() < weights[..., 2].item()


def test_missing_support_is_explicit_on_resolved_observation_grid() -> None:
    truth, _, target, confidence = _target_inputs()
    config = _config(
        confidence={
            "type": "axis_aligned_missing",
            "blocks": [{"start": [0, 0, 0], "stop": [2, 4, 4]}],
        }
    )
    observation = build_spatial_property_observation(
        target,
        confidence,
        truth != -1,
        config,
    )

    assert torch.count_nonzero(observation.confidence[..., :2, :, :]) == 0
    assert torch.count_nonzero(observation.confidence[..., 2:, :, :]) > 0


def test_relative_noise_is_deterministic_and_only_changes_observation() -> None:
    truth, _, target, confidence = _target_inputs()
    config = _config(noise={"type": "relative_gaussian", "relative_std": 0.05, "seed": 17})
    first = build_spatial_property_observation(target, confidence, truth != -1, config)
    second = build_spatial_property_observation(target, confidence, truth != -1, config)

    assert torch.equal(first.values, second.values)
    assert torch.equal(first.noise, second.noise)
    assert torch.equal(first.noiseless_values, target)
    assert torch.count_nonzero(first.noise) > 0
    assert not torch.equal(first.values, first.noiseless_values)
    assert first.metadata["observation_values_sha256"] == second.metadata[
        "observation_values_sha256"
    ]


def test_soft_observation_gradient_is_finite_and_zero_at_conditions() -> None:
    truth, table, target, confidence = _target_inputs()
    condition = ~confidence.bool()
    observation = build_spatial_property_observation(
        target,
        confidence,
        truth != -1,
        _config({"type": "gaussian_blur", "sigma_voxels": 1.0}),
    )
    state = torch.randn((1, 3, 4, 4, 4), requires_grad=True)
    loss, diagnostics = spatial_property_volume_loss(
        state,
        torch.eye(3),
        target,
        table,
        confidence,
        tau=0.25,
        sigmas=(0.0,),
        scale_weights=(1.0,),
        channel_weights=torch.tensor([0.5, 0.5]),
        observed_properties=observation.values,
        observation_confidence=observation.confidence,
        observation_config=_config({"type": "gaussian_blur", "sigma_voxels": 1.0}),
        condition_mask=condition,
    )
    gradient = torch.autograd.grad(loss, state)[0]
    expanded_condition = condition.expand_as(gradient)

    assert torch.isfinite(loss)
    assert torch.isfinite(gradient).all()
    assert gradient.norm() > 0
    assert torch.count_nonzero(gradient[expanded_condition]) == 0
    assert diagnostics["observation_loss"] == loss


class _ConstantVelocityNet(torch.nn.Module):
    def forward(
        self,
        state: torch.Tensor,
        conditioning: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        del conditioning, time
        return torch.full_like(state, 0.025)


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
        logits = torch.einsum("bexyz,ce->bcxyz", normalized, embeddings)
        return logits.argmax(dim=1)


def test_phase3_alpha_zero_uses_same_projected_euler_trajectory() -> None:
    truth, table, target, confidence = _target_inputs()
    condition = ~confidence.bool()
    model = _DummyConditionalModel()
    embedded_truth = model.embed(truth)
    embedded_mask = condition.expand_as(embedded_truth)
    conditioning = embedded_truth * embedded_mask
    initial = torch.randn((1, 3, 4, 4, 4), generator=torch.Generator().manual_seed(9))
    common = {
        "model": model,
        "initial_state": initial,
        "conditioning": conditioning,
        "embedded_truth": embedded_truth,
        "truth_model": truth,
        "condition_mask": condition,
        "target_properties": target,
        "property_table": table,
        "confidence": confidence,
        "property_sigmas": (0.0,),
        "property_scale_weights": (1.0,),
        "property_channel_weights": torch.tensor([0.5, 0.5]),
        "n_steps": 4,
        "alpha": 0.0,
        "max_guidance_ratio": 0.25,
        "tau_start": 0.5,
        "tau_end": 0.1,
        "tau_schedule": "cosine",
        "guidance_start": 0.0,
        "guidance_schedule": "windowed_sine",
        "grad_clip_norm": 1.0,
    }
    phase2_state, _ = fixed_euler_property_sample(**common)
    observation = build_spatial_property_observation(
        target,
        confidence,
        truth != -1,
        _config(),
    )
    phase3_state, trace = fixed_euler_property_sample(
        **common,
        loss_function=spatial_property_volume_loss,
        loss_extra_kwargs={
            "observed_properties": observation.values,
            "observation_confidence": observation.confidence,
            "observation_config": _config(),
            "condition_mask": condition,
        },
        loss_mode=SPATIAL_PROPERTY_LOSS_MODE,
    )

    assert torch.equal(phase3_state, phase2_state)
    assert all(row["raw_grad_norm"] == 0 for row in trace)
    assert all(row["post_projection_condition_violations"] == 0 for row in trace)
    assert all(row["property_loss_mode"] == SPATIAL_PROPERTY_LOSS_MODE for row in trace)


def test_phase3_pairing_includes_observation_assets() -> None:
    baseline = {field: f"same-{field}" for field in PHASE3_PAIR_FIELDS}
    baseline.update(
        {
            "alpha": 0.0,
            "run_status": "completed",
            "samples_written": 2,
            "n_samples": 2,
        }
    )
    guided = dict(baseline)
    guided["alpha"] = 0.25

    paired, _ = paired_spatial_property_config_verdict(baseline, guided)
    assert paired

    guided["observation_values_sha256"] = "different"
    paired, reason = paired_spatial_property_config_verdict(baseline, guided)
    assert not paired
    assert "observation_values_sha256" in reason


def test_phase3_runner_refuses_unpaired_guidance_and_nonempty_output(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        model_weights="ema",
        n_samples=1,
        n_steps=4,
        alpha=0.25,
        baseline_dir=None,
        max_guidance_ratio=0.25,
        grad_clip_norm=1.0,
        guidance_start=0.25,
        target_roi_radius=6,
        output_dir=tmp_path / "new",
    )
    with pytest.raises(ValueError, match="baseline-dir"):
        validate_args(args)

    args.alpha = 0.0
    args.output_dir.mkdir()
    (args.output_dir / "existing.txt").write_text("immutable", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        validate_args(args)


def test_phase3_runner_hard_observation_metrics_accept_saved_3d_sample() -> None:
    truth, table, target, confidence = _target_inputs()
    condition = ~confidence.bool()
    observation = build_spatial_property_observation(
        target,
        confidence,
        truth != -1,
        _config(),
    )
    row: dict[str, object] = {}

    _add_hard_observation_metrics(
        row,
        prediction=truth[0, 0],
        truth=truth,
        condition_mask=condition,
        property_table=table,
        observation=observation,
        observation_config=_config(),
        property_sigmas=(0.0,),
        property_scale_weights=(1.0,),
        property_channel_weights=torch.tensor([0.5, 0.5]),
    )

    assert row["hard_observation_loss"] == pytest.approx(0.0, abs=1e-8)
    assert row["hard_observation_mae"] == pytest.approx(0.0, abs=1e-8)


def test_phase3_n4_classification_and_screen_bracket_are_frozen() -> None:
    assert _classification(
        n_samples=4,
        pass_count=4,
        baseline_regression_passed=True,
    )[0] == "confirmed_seed42_pass"
    assert _classification(
        n_samples=4,
        pass_count=2,
        baseline_regression_passed=True,
    )[0] == "seed42_transition"
    assert _classification(
        n_samples=4,
        pass_count=0,
        baseline_regression_passed=True,
    )[0] == "confirmed_seed42_failure"

    decision = screen_decision(
        [
            {
                "level": "identity_anchor_v1",
                "order": 0,
                "role": "implementation_anchor",
                "passed": True,
            },
            {
                "level": "gaussian_sigma1_v1",
                "order": 1,
                "role": "primary_screen",
                "passed": False,
            },
            {
                "level": "gaussian_sigma2_v1",
                "order": 2,
                "role": "primary_screen",
                "passed": False,
            },
        ]
    )
    assert decision == {
        "status": "no_nonzero_blur_passed",
        "selected_level": "identity_anchor_v1",
        "bracket_levels": ["identity_anchor_v1", "gaussian_sigma1_v1"],
    }
