from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from guidance.probability_evaluation import (
    class_transition_records,
    ensemble_diversity_summary,
    sample_hard_metrics,
)
from guidance.probability_sampling import (
    LEGACY_GUIDANCE_SCALING_MODE,
    REFERENCE_GUIDANCE_SCALING_MODE,
    build_probability_guidance_velocity,
    fixed_euler_probability_sample,
    probability_guidance_weight,
    temperature_at_time,
)
from guidance.probability_volume import (
    CALIBRATED_PROBABILITY_LOSS_MODE,
    LEGACY_PROBABILITY_LOSS_MODE,
    build_probability_volume,
    build_target_mask,
    compute_target_soft_fields,
    dilate_mask,
    paired_target_soft_deltas,
    probability_target_loss_terms,
    probability_volume_loss,
    spatial_gradient_matching_loss,
    target_soft_region_stats,
    tensor_sha256,
)
from scripts.stage1.run_probability_guidance import (
    PHASE1_PAIR_FIELDS,
    paired_probability_config_verdict,
)


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


def _reference_projected_euler(
    model: _DummyConditionalModel,
    initial: torch.Tensor,
    embedded_truth: torch.Tensor,
    condition_mask: torch.Tensor,
    n_steps: int,
) -> torch.Tensor:
    expanded_mask = condition_mask.expand(-1, initial.shape[1], -1, -1, -1)
    state = torch.where(expanded_mask, embedded_truth, initial)
    conditioning = embedded_truth * expanded_mask
    for step in range(n_steps):
        time = torch.full((1,), (step + 0.5) / n_steps, dtype=state.dtype)
        velocity = model.net(state, conditioning, time)
        candidate = state + velocity / n_steps
        state = torch.where(expanded_mask, embedded_truth, candidate)
    return state


def test_target_component_modes_are_volume_ranked_and_deterministic() -> None:
    truth = torch.zeros((1, 1, 8, 8, 8), dtype=torch.long)
    truth[:, :, 1:4, 1:4, 1:4] = 1
    truth[:, :, 6:8, 6:8, 6:8] = 1

    largest, largest_meta = build_target_mask(truth, 1, component_mode="largest")
    selected, selected_meta = build_target_mask(
        truth,
        1,
        component_mode="selected",
        component_rank=1,
    )

    assert int(largest.sum()) == 27
    assert int(selected.sum()) == 8
    assert largest_meta["ranked_components"][0]["voxel_count"] == 27
    assert selected_meta["selected_component"]["rank_by_volume"] == 1
    assert tensor_sha256(largest) == largest_meta["target_mask_sha256"]


def test_multiscale_probability_and_roi_are_bounded() -> None:
    target = torch.zeros((1, 1, 9, 9, 9), dtype=torch.bool)
    target[:, :, 4, 4, 4] = True

    probability, metadata = build_probability_volume(
        target,
        sigmas=(0.0, 1.0),
        scale_weights=(1.0, 1.0),
    )
    roi = dilate_mask(target, radius=2)

    assert probability.shape == target.shape
    assert 0.0 <= float(probability.min()) <= float(probability.max()) <= 1.0
    assert probability[0, 0, 4, 4, 4] > probability[0, 0, 4, 4, 3]
    assert int(roi.sum()) == 125
    assert metadata["target_scale_weights"] == [0.5, 0.5]


def test_probability_loss_has_finite_embedding_gradient() -> None:
    embedding = torch.eye(3)
    state = torch.randn((1, 3, 6, 6, 6), requires_grad=True)
    target = torch.zeros((1, 1, 6, 6, 6))
    target[:, :, 2:4, 2:4, 2:4] = 1.0
    roi = dilate_mask(target.bool(), radius=1)

    loss, diagnostics = probability_volume_loss(
        state,
        embedding,
        target,
        roi,
        target_label=1,
        tau=0.5,
    )
    gradient = torch.autograd.grad(loss, state)[0]

    assert torch.isfinite(loss)
    assert torch.isfinite(gradient).all()
    assert gradient.norm() > 0
    assert 0 <= diagnostics["probability_dice_score"] <= 1


def test_calibrated_soft_bce_does_not_promote_low_probability_halo() -> None:
    target = torch.tensor((0.8, 0.168, 0.05, 0.05, 0.05, 0.05)).reshape(
        1, 1, 1, 1, 6
    )
    core = torch.zeros_like(target, dtype=torch.bool)
    core[..., 0] = True
    roi = torch.ones_like(core)

    calibrated_prediction = target.clone().requires_grad_(True)
    calibrated = probability_target_loss_terms(
        calibrated_prediction,
        target,
        core,
        roi,
        loss_mode=CALIBRATED_PROBABILITY_LOSS_MODE,
    )
    calibrated_gradient = torch.autograd.grad(
        calibrated["probability_bce"],
        calibrated_prediction,
    )[0]

    legacy_prediction = target.clone().requires_grad_(True)
    legacy = probability_target_loss_terms(
        legacy_prediction,
        target,
        core,
        roi,
        loss_mode=LEGACY_PROBABILITY_LOSS_MODE,
    )
    legacy_gradient = torch.autograd.grad(
        legacy["probability_bce"],
        legacy_prediction,
    )[0]

    assert abs(float(calibrated_gradient[..., 1])) < 1e-6
    assert float(legacy_gradient[..., 1]) < 0
    assert calibrated["loss_positive_scale"] == 1
    assert calibrated["loss_negative_scale"] == 1
    assert legacy["loss_positive_scale"] > legacy["loss_negative_scale"]


def test_reference_relative_guidance_decays_with_gradient_and_respects_cap() -> None:
    prior = torch.ones((1, 2, 1, 1, 1))
    gradient = torch.ones_like(prior)
    first_velocity, first_diagnostics, reference = (
        build_probability_guidance_velocity(
            gradient,
            prior,
            requested_ratio=0.2,
            max_ratio=0.5,
            scaling_mode=REFERENCE_GUIDANCE_SCALING_MODE,
        )
    )
    small_velocity, small_diagnostics, _ = build_probability_guidance_velocity(
        gradient * 0.01,
        prior,
        requested_ratio=0.2,
        max_ratio=0.5,
        scaling_mode=REFERENCE_GUIDANCE_SCALING_MODE,
        reference_gradient_norm=reference,
    )
    legacy_velocity, legacy_diagnostics, _ = build_probability_guidance_velocity(
        gradient * 0.01,
        prior,
        requested_ratio=0.2,
        max_ratio=0.5,
        scaling_mode=LEGACY_GUIDANCE_SCALING_MODE,
    )
    capped_velocity, capped_diagnostics, _ = build_probability_guidance_velocity(
        gradient * 10,
        prior,
        requested_ratio=0.2,
        max_ratio=0.25,
        scaling_mode=REFERENCE_GUIDANCE_SCALING_MODE,
        reference_gradient_norm=reference,
    )

    assert first_diagnostics["effective_guidance_ratio"] == pytest.approx(0.2)
    assert small_diagnostics["effective_guidance_ratio"] == pytest.approx(0.002)
    assert legacy_diagnostics["effective_guidance_ratio"] == pytest.approx(0.2)
    assert capped_diagnostics["effective_guidance_ratio"] == pytest.approx(0.25)
    assert capped_diagnostics["guidance_cap_fraction"] == 1.0
    assert small_velocity.norm() < first_velocity.norm() < capped_velocity.norm()
    assert legacy_velocity.norm() == pytest.approx(first_velocity.norm())


def test_spatial_gradient_loss_penalizes_fragmented_probability() -> None:
    target = torch.zeros((1, 1, 7, 7, 7))
    target[:, :, 2:5, 2:5, 2:5] = 1.0
    roi = dilate_mask(target.bool(), radius=1)
    fragmented = target.clone()
    fragmented[:, :, 2:5:2, 2:5:2, 2:5:2] = 0.0

    matching_loss, matching_diagnostics = spatial_gradient_matching_loss(
        target,
        target,
        roi,
    )
    fragmented_loss, fragmented_diagnostics = spatial_gradient_matching_loss(
        fragmented,
        target,
        roi,
    )

    assert matching_loss == 0
    assert fragmented_loss > matching_loss
    assert (
        fragmented_diagnostics["roi_spatial_gradient_error_mean"]
        > matching_diagnostics["roi_spatial_gradient_error_mean"]
    )


def test_probability_loss_records_optional_spatial_term() -> None:
    embedding = torch.eye(3)
    state = torch.randn((1, 3, 6, 6, 6), requires_grad=True)
    target = torch.zeros((1, 1, 6, 6, 6))
    target[:, :, 2:4, 2:4, 2:4] = 1.0
    roi = dilate_mask(target.bool(), radius=1)

    base, base_diagnostics = probability_volume_loss(
        state,
        embedding,
        target,
        roi,
        target_label=1,
        tau=0.5,
        spatial_gradient_weight=0.0,
    )
    regularized, diagnostics = probability_volume_loss(
        state,
        embedding,
        target,
        roi,
        target_label=1,
        tau=0.5,
        spatial_gradient_weight=0.1,
    )
    gradient = torch.autograd.grad(regularized, state)[0]

    assert base_diagnostics["probability_spatial_gradient_loss"] == 0
    assert diagnostics["probability_spatial_gradient_loss"] > 0
    assert regularized > base
    assert torch.isfinite(gradient).all()


def test_final_soft_fields_expose_tau_independent_hard_boundary() -> None:
    embedding = torch.eye(3)
    state = torch.zeros((1, 3, 2, 2, 1))
    state[0, :, 0, 0, 0] = torch.tensor((0.0, 0.0, 1.0))
    state[0, :, 0, 1, 0] = torch.tensor((0.0, 1.0, 0.0))
    state[0, :, 1, 0, 0] = torch.tensor((0.0, 0.99, 1.0))
    state[0, :, 1, 1, 0] = torch.tensor((1.0, 0.0, 0.0))

    cold = compute_target_soft_fields(state, embedding, target_label=1, tau=0.1)
    warm = compute_target_soft_fields(state, embedding, target_label=1, tau=0.5)

    assert cold["target_similarity_margin"][0, 0, 0, 0, 0] > 0
    assert cold["target_similarity_margin"][0, 0, 0, 1, 0] < 0
    assert cold["soft_hard_target"][0, 0, 1, 0, 0]
    assert not cold["soft_hard_target"][0, 0, 1, 1, 0]
    assert torch.allclose(
        cold["target_similarity_margin"],
        warm["target_similarity_margin"],
    )
    assert not torch.allclose(
        cold["target_probability"],
        warm["target_probability"],
    )


def test_final_soft_region_stats_separate_conditions_and_true_background() -> None:
    embedding = torch.eye(3)
    state = torch.zeros((1, 3, 3, 2, 1))
    state[:, 2] = 1.0
    fields = compute_target_soft_fields(state, embedding, target_label=1, tau=0.2)
    truth = torch.zeros((1, 1, 3, 2, 1), dtype=torch.long)
    truth[:, :, 0, :, :] = 1
    truth[:, :, 1, 0, :] = 1
    selected = torch.zeros_like(truth, dtype=torch.bool)
    selected[:, :, 0, :, :] = True
    roi = torch.zeros_like(selected)
    roi[:, :, :2] = True
    condition = torch.zeros_like(selected)
    condition[:, :, 0, 0] = True

    rows = target_soft_region_stats(
        fields,
        truth,
        selected,
        roi,
        condition,
        target_label=1,
        sample_id=4,
    )
    by_region = {row["region"]: row for row in rows}

    assert by_region["selected_truth_target"]["voxel_count"] == 2
    assert by_region["selected_truth_target_unconditioned"]["voxel_count"] == 1
    assert by_region["truth_target_unselected"]["voxel_count"] == 1
    assert by_region["roi_true_non_target"]["voxel_count"] == 1
    assert by_region["outside_roi_true_non_target"]["voxel_count"] == 2
    assert by_region["selected_truth_target"]["soft_hard_target_fraction"] == 1.0


def test_paired_soft_deltas_distinguish_recovery_from_false_addition() -> None:
    embedding = torch.eye(3)
    baseline_state = torch.zeros((1, 3, 2, 2, 1))
    guided_state = torch.zeros_like(baseline_state)
    baseline_state[:, 1] = 1.0
    guided_state[:, 1] = 1.0
    guided_state[0, :, 0, 0, 0] = torch.tensor((0.0, 0.0, 1.0))
    guided_state[0, :, 0, 1, 0] = torch.tensor((0.0, 0.0, 1.0))
    baseline_fields = compute_target_soft_fields(
        baseline_state, embedding, target_label=1, tau=0.2
    )
    guided_fields = compute_target_soft_fields(
        guided_state, embedding, target_label=1, tau=0.2
    )
    truth = torch.zeros((1, 1, 2, 2, 1), dtype=torch.long)
    truth[:, :, 0, 0] = 1
    selected = truth == 1
    roi = torch.ones_like(selected)
    condition = torch.zeros_like(selected)
    baseline_decoded = torch.zeros_like(truth)
    guided_decoded = baseline_decoded.clone()
    guided_decoded[:, :, 0, 0] = 1
    guided_decoded[:, :, 0, 1] = 1

    rows = paired_target_soft_deltas(
        baseline_fields,
        guided_fields,
        truth,
        selected,
        roi,
        condition,
        baseline_decoded,
        guided_decoded,
        target_label=1,
        sample_id=0,
    )
    by_region = {row["region"]: row for row in rows}

    assert by_region["selected_truth_target"]["desired_hard_crossing_count"] == 1
    assert by_region["selected_truth_target"]["undesired_hard_crossing_count"] == 0
    assert by_region["roi_true_non_target"]["hard_target_entered_count"] == 1
    assert by_region["roi_true_non_target"]["undesired_hard_crossing_count"] == 1
    assert (
        by_region["selected_truth_target"][
            "delta_target_similarity_margin_mean"
        ]
        > 0
    )


def test_alpha_zero_matches_projected_fixed_euler_reference_exactly() -> None:
    model = _DummyConditionalModel()
    truth = torch.zeros((1, 1, 4, 4, 4), dtype=torch.long)
    truth[:, :, 1:3, 1:3, 1:3] = 1
    condition_mask = torch.zeros_like(truth, dtype=torch.bool)
    condition_mask[:, :, 0, 0, :] = True
    embedded_truth = model.embed(truth)
    conditioning = embedded_truth * condition_mask.expand_as(embedded_truth)
    initial = torch.randn((1, 3, 4, 4, 4), generator=torch.Generator().manual_seed(7))
    target_mask = truth == 1
    target_probability = target_mask.float()
    roi = dilate_mask(target_mask, 1)

    result, trace = fixed_euler_probability_sample(
        model=model,
        initial_state=initial,
        conditioning=conditioning,
        embedded_truth=embedded_truth,
        truth_model=truth,
        condition_mask=condition_mask,
        target_probability=target_probability,
        target_mask=target_mask,
        roi_mask=roi,
        target_label=1,
        n_steps=4,
        alpha=0.0,
        max_guidance_ratio=0.1,
        tau_start=0.5,
        tau_end=0.1,
        tau_schedule="cosine",
        guidance_start=0.25,
        guidance_schedule="windowed_sine",
        grad_clip_norm=1.0,
        bce_weight=1.0,
        dice_weight=1.0,
        spatial_gradient_weight=0.1,
        probability_loss_mode=CALIBRATED_PROBABILITY_LOSS_MODE,
        guidance_scaling_mode=REFERENCE_GUIDANCE_SCALING_MODE,
    )
    reference = _reference_projected_euler(
        model,
        initial,
        embedded_truth,
        condition_mask,
        n_steps=4,
    )

    assert torch.equal(result, reference)
    assert all(row["raw_grad_norm"] == 0.0 for row in trace)
    assert all(row["spatial_gradient_weight"] == 0.1 for row in trace)
    assert all(row["post_projection_condition_violations"] == 0 for row in trace)


def test_calibrated_reference_guidance_runs_finite_and_preserves_conditions() -> None:
    model = _DummyConditionalModel()
    truth = torch.zeros((1, 1, 4, 4, 4), dtype=torch.long)
    truth[:, :, 1:3, 1:3, 1:3] = 1
    condition_mask = torch.zeros_like(truth, dtype=torch.bool)
    condition_mask[:, :, 0, 0, :] = True
    embedded_truth = model.embed(truth)
    conditioning = embedded_truth * condition_mask.expand_as(embedded_truth)
    initial = torch.randn(
        (1, 3, 4, 4, 4),
        generator=torch.Generator().manual_seed(11),
    )
    target_mask = truth == 1
    target_probability, _ = build_probability_volume(
        target_mask,
        sigmas=(0.0, 1.0),
    )
    roi = dilate_mask(target_mask, 1)

    result, trace = fixed_euler_probability_sample(
        model=model,
        initial_state=initial,
        conditioning=conditioning,
        embedded_truth=embedded_truth,
        truth_model=truth,
        condition_mask=condition_mask,
        target_probability=target_probability,
        target_mask=target_mask,
        roi_mask=roi,
        target_label=1,
        n_steps=8,
        alpha=0.2,
        max_guidance_ratio=0.25,
        tau_start=0.5,
        tau_end=0.1,
        tau_schedule="cosine",
        guidance_start=0.25,
        guidance_schedule="windowed_sine",
        grad_clip_norm=1.0,
        bce_weight=1.0,
        dice_weight=1.0,
        probability_loss_mode=CALIBRATED_PROBABILITY_LOSS_MODE,
        guidance_scaling_mode=REFERENCE_GUIDANCE_SCALING_MODE,
    )

    active = [row for row in trace if row["w_t"] > 0]
    assert torch.isfinite(result).all()
    assert active
    assert all(row["guidance_gradient_reference_norm"] > 0 for row in active)
    assert all(row["effective_guidance_ratio"] <= 0.25 + 1e-6 for row in active)
    assert trace[-1]["w_t"] < max(row["w_t"] for row in trace)
    assert all(row["loss_positive_scale"] == 1.0 for row in trace)
    assert all(row["loss_negative_scale"] == 1.0 for row in trace)
    assert all(row["post_projection_condition_violations"] == 0 for row in trace)


def test_hard_metrics_report_zero_condition_violations_and_pair_changes() -> None:
    truth = torch.zeros((1, 1, 5, 5, 5), dtype=torch.long)
    truth[:, :, 1:4, 2:4, 1:4] = 1
    baseline = truth.clone()
    guided = truth.clone()
    guided[:, :, 1, 2, 1] = 0
    selected = truth == 1
    roi = dilate_mask(selected, 1)
    condition = torch.zeros_like(selected)
    condition[:, :, 0, 0, :] = True

    metrics = sample_hard_metrics(
        guided,
        truth,
        selected,
        roi,
        condition,
        target_label=1,
        sample_id=3,
        baseline_prediction=baseline,
    )

    assert metrics["condition_violation_count"] == 0
    assert metrics["paired_hard_change_count"] == 1
    assert metrics["target_recall"] < 1.0
    assert metrics["selected_roi_recall"] < 1.0
    assert metrics["target_absolute_volume_error"] == 1
    assert metrics["selected_absolute_volume_error"] == 1


def test_phase1_pairing_allows_only_positive_guidance_alpha() -> None:
    baseline = {field: f"value-{field}" for field in PHASE1_PAIR_FIELDS}
    baseline.update(
        {
            "alpha": 0.0,
            "run_status": "completed",
            "samples_written": 4,
            "n_samples": 4,
        }
    )
    guided = dict(baseline)
    guided.update({"alpha": 0.05, "run_status": "running"})

    paired, reason = paired_probability_config_verdict(baseline, guided)

    assert paired is True
    assert "strict Phase-1" in reason
    guided["tau_end"] = "different"
    paired, reason = paired_probability_config_verdict(baseline, guided)
    assert paired is False
    assert "tau_end" in reason

    guided = dict(baseline)
    guided.update(
        {
            "alpha": 0.05,
            "run_status": "running",
            "spatial_gradient_weight": "different",
        }
    )
    paired, reason = paired_probability_config_verdict(baseline, guided)
    assert paired is False
    assert "spatial_gradient_weight" in reason

    guided = dict(baseline)
    guided.update(
        {
            "alpha": 0.05,
            "run_status": "running",
            "probability_loss_mode": "different",
        }
    )
    paired, reason = paired_probability_config_verdict(baseline, guided)
    assert paired is False
    assert "probability_loss_mode" in reason

    guided = dict(baseline)
    guided.update(
        {
            "alpha": 0.05,
            "run_status": "running",
            "guidance_scaling_mode": "different",
        }
    )
    paired, reason = paired_probability_config_verdict(baseline, guided)
    assert paired is False
    assert "guidance_scaling_mode" in reason


def test_transition_and_diversity_artifacts_capture_paired_changes() -> None:
    first = torch.zeros((1, 4, 4, 4), dtype=torch.long)
    second = first.clone()
    second[:, 1:3, 1:3, 1:3] = 1
    selected = torch.zeros((1, 1, 4, 4, 4), dtype=torch.bool)
    selected[:, :, 1:3, 1:3, 1:3] = True
    roi = dilate_mask(selected, 1)

    transitions = class_transition_records(first, second, sample_id=2)
    diversity = ensemble_diversity_summary(
        torch.stack((first, second), dim=0),
        selected,
        roi,
        target_label=1,
        sample_hashes=(tensor_sha256(first), tensor_sha256(second)),
    )

    changed = next(row for row in transitions if row["from_label"] == 0 and row["to_label"] == 1)
    assert changed["voxel_count"] == 8
    assert diversity["unique_decoded_samples"] == 2
    assert diversity["mean_pairwise_hard_disagreement"] > 0
    assert diversity["selected_target_coverage_any_sample"] == 1.0


def test_temperature_schedule_anneals_to_requested_endpoints() -> None:
    assert temperature_at_time(0.0, 0.5, 0.1, "cosine") == 0.5
    assert temperature_at_time(1.0, 0.5, 0.1, "cosine") == 0.1
    assert temperature_at_time(0.5, 0.3, 0.3, "constant") == 0.3


def test_windowed_guidance_schedule_rises_then_returns_to_zero() -> None:
    start = 0.25
    midpoint = start + 0.5 * (1.0 - start)

    assert probability_guidance_weight(0.0, "windowed_sine", start) == 0.0
    assert probability_guidance_weight(start, "windowed_sine", start) == 0.0
    assert probability_guidance_weight(
        midpoint, "windowed_sine", start
    ) == pytest.approx(1.0)
    assert probability_guidance_weight(1.0, "windowed_sine", start) == 0.0
    assert probability_guidance_weight(
        0.9, "windowed_sine", start
    ) < probability_guidance_weight(midpoint, "windowed_sine", start)
