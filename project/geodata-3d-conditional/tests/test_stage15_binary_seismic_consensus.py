from __future__ import annotations

from pathlib import Path
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from guidance.binary_seismic_inversion import (
    BinaryAcousticProperties,
    binary_occupancy_to_acoustic,
    straight_through_binary,
)
from guidance.probability_volume import probability_volume_loss
from guidance.seismic import ConvolutionalSeismic, build_seismic_observation, tensor_sha256
from scripts.stage15.audit_b2_seismic_geology_alignment import (
    align_retained_states,
    quartile_groups,
    spearman_metrics,
)
from scripts.stage15.build_binary_consensus import build_consensus
from scripts.stage15.run_binary_flow_pcn import (
    categorical_to_binary_label9,
    condition_violation_count,
    record_current_state,
    should_retain_iteration,
    validate_config as validate_b2_config,
)
from scripts.stage15.evaluate_binary_flow_pcn_truth import (
    B2_INPUT_NAMES,
    assert_hashes_unchanged,
    binary_metrics as b3_binary_metrics,
    snapshot_hashes,
    truth_partition as b3_truth_partition,
    validate_retained_accounting,
)
from guidance.generator_posterior import (
    metropolis_decision,
    pcn_proposal,
    posterior_energy,
    projected_fixed_euler_prior_sample,
)


PROPERTIES = BinaryAcousticProperties(
    air_density=1.0,
    air_velocity=10.0,
    background_density=2.0,
    background_velocity=20.0,
    target_density=3.0,
    target_velocity=30.0,
)


def test_binary_property_mapping_uses_background_target_and_air() -> None:
    occupancy = torch.tensor([[[[[0.0, 1.0, 1.0]]]]])
    subsurface = torch.tensor([[[[[True, True, False]]]]])
    impedance, slowness = binary_occupancy_to_acoustic(
        occupancy, subsurface, PROPERTIES
    )
    assert torch.equal(impedance, torch.tensor([[[[[40.0, 90.0, 10.0]]]]]))
    assert torch.allclose(
        slowness, torch.tensor([[[[[1 / 20, 1 / 30, 1 / 10]]]]])
    )


def test_binary_conditions_and_ste_are_exact_with_nonzero_free_gradient() -> None:
    logits = torch.zeros((1, 1, 1, 1, 4), requires_grad=True)
    subsurface = torch.tensor([[[[[True, True, True, False]]]]])
    well_mask = torch.tensor([[[[[True, True, False, False]]]]])
    well_values = torch.tensor([[[[[1.0, 0.0, 0.0, 0.0]]]]])
    probability, hard, ste = straight_through_binary(
        logits, 1.0, subsurface, well_values, well_mask
    )
    assert probability[0, 0, 0, 0, 2] == 0.5
    assert torch.equal(hard, torch.tensor([[[[[1.0, 0.0, 1.0, 0.0]]]]]))
    assert torch.equal(ste.detach(), hard)
    ste.sum().backward()
    assert logits.grad[0, 0, 0, 0, 2] != 0
    assert logits.grad[0, 0, 0, 0, 0] == 0
    assert logits.grad[0, 0, 0, 0, 1] == 0
    assert logits.grad[0, 0, 0, 0, 3] == 0


def test_hard_ste_forward_and_observation_builder_close_exactly() -> None:
    operator = ConvolutionalSeismic(
        (1, 1, 4),
        cell_size_m=(10.0, 10.0, 0.01),
        num_time_samples=16,
        sample_interval_ms=2.0,
        peak_frequency_hz=25.0,
        wavelet_duration_ms=8.0,
    )
    logits = torch.tensor([[[[[-1.0, 1.0, -1.0, 1.0]]]]], requires_grad=True)
    subsurface = torch.ones_like(logits, dtype=torch.bool)
    wells = torch.zeros_like(subsurface)
    values = torch.zeros_like(logits)
    _, hard, ste = straight_through_binary(logits, 0.5, subsurface, values, wells)
    hard_acoustic = binary_occupancy_to_acoustic(hard, subsurface, PROPERTIES)
    ste_acoustic = binary_occupancy_to_acoustic(ste, subsurface, PROPERTIES)
    hard_field = operator(*hard_acoustic, subsurface)
    ste_field = operator(*ste_acoustic, subsurface)
    assert torch.equal(hard_field, ste_field.detach())
    observation = build_seismic_observation(
        torch.cat(hard_acoustic, dim=1), subsurface, operator
    )
    assert torch.equal(observation.values, hard_field)
    ste_field.square().sum().backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) > 0


def test_consensus_thresholds_are_exact_at_eight_two_and_seven_of_ten() -> None:
    members = torch.zeros((10, 1, 1, 1, 3))
    members[:8, ..., 0] = 1
    members[:2, ..., 1] = 1
    members[:7, ..., 2] = 1
    subsurface = torch.ones((1, 1, 1, 1, 3), dtype=torch.bool)
    conditioned = torch.zeros_like(subsurface)
    result = build_consensus(members, subsurface, conditioned, 0.8, 0.2)
    assert result["positive_mask"].flatten().tolist() == [True, False, False]
    assert result["negative_mask"].flatten().tolist() == [False, True, False]
    assert result["unknown_mask"].flatten().tolist() == [False, False, True]


def test_flow_probability_interface_masks_unknown_loss_and_gradient() -> None:
    shape = (1, 1, 2, 2, 2)
    target = torch.zeros(shape)
    target[..., 0, 0, 0] = 1
    positive = target.bool()
    negative = torch.zeros(shape, dtype=torch.bool)
    negative[..., 0, 0, 1] = True
    subsurface = torch.ones(shape, dtype=torch.bool)
    conditioned = torch.zeros(shape, dtype=torch.bool)
    conditioned[..., 1, 1, 1] = True
    roi = (positive | negative) & subsurface & ~conditioned
    assert target.shape == roi.shape == positive.shape
    state = torch.randn((1, 3, 2, 2, 2), requires_grad=True)
    embedding = torch.randn((15, 3))
    loss, _ = probability_volume_loss(
        state,
        embedding,
        target,
        roi,
        target_label=9,
        tau=0.5,
        loss_mode="calibrated_soft_bce_hard_dice_v2",
        target_mask=positive,
    )
    loss.backward()
    unknown = ~roi[0, 0]
    assert torch.count_nonzero(state.grad[0, :, unknown]) == 0
    assert torch.count_nonzero(state.grad[0, :, roi[0, 0]]) > 0


def test_stage15_truth_blind_runners_have_no_truth_asset_path() -> None:
    runner_names = (
        "run_binary_inversion_ensemble.py",
        "build_binary_consensus.py",
        "run_consensus_flow_guidance.py",
    )
    root = PROJECT_DIR / "scripts/stage15"
    for name in runner_names:
        source = (root / name).read_text(encoding="utf-8")
        assert "true_model.pt" not in source
        assert "truth_restricted" not in source


def test_stage15_flow_shapes_for_full_cube_are_single_channel() -> None:
    target = torch.zeros((1, 1, 64, 64, 64))
    roi = torch.ones_like(target, dtype=torch.bool)
    assert target.shape == roi.shape == (1, 1, 64, 64, 64)


def test_b2_categorical_collapse_keeps_air_separate() -> None:
    decoded = torch.tensor([[[[[-1, 9, 0, 13]]]]])
    subsurface = torch.tensor([[[[[False, True, True, True]]]]])
    binary = categorical_to_binary_label9(decoded, subsurface)
    assert binary.flatten().tolist() == [0.0, 1.0, 0.0, 0.0]
    assert not bool(subsurface[..., 0])


class _ZeroVelocityNet(torch.nn.Module):
    def forward(self, state, conditioning, time):
        del conditioning, time
        return torch.zeros_like(state)


class _ConditionProjectionModel:
    def __init__(self) -> None:
        self.net = _ZeroVelocityNet()

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        return state[:, 0].round().long()


def test_b2_pcn_flow_path_preserves_exact_categorical_conditions() -> None:
    model = _ConditionProjectionModel()
    condition_values = torch.tensor([[[[[-1, 9, 2, -1]]]]])
    condition_mask = torch.tensor([[[[[True, True, True, False]]]]])
    embedded_conditions = (condition_values + 1).float()
    conditioning = embedded_conditions * condition_mask
    current = torch.randn((1, 1, 1, 1, 4))
    proposal, _ = pcn_proposal(
        current,
        beta=0.1,
        generator=torch.Generator(device="cpu").manual_seed(515),
    )
    final = projected_fixed_euler_prior_sample(
        model,
        proposal,
        conditioning,
        embedded_conditions,
        condition_mask,
        n_steps=4,
    )
    decoded = (model.decode(final) - 1).unsqueeze(1)
    assert condition_violation_count(decoded, condition_values, condition_mask) == 0


def test_b2_binary_seismic_is_deterministic_for_same_decoded_geology() -> None:
    decoded = torch.tensor([[[[[0, 9, 9, 0]]]]])
    support = torch.ones_like(decoded, dtype=torch.bool)
    binary_a = categorical_to_binary_label9(decoded, support)
    binary_b = categorical_to_binary_label9(decoded.clone(), support)
    operator = ConvolutionalSeismic(
        (1, 1, 4),
        cell_size_m=(1.0, 1.0, 0.01),
        num_time_samples=16,
        sample_interval_ms=2.0,
        peak_frequency_hz=25.0,
        wavelet_duration_ms=8.0,
    )
    acoustic_a = binary_occupancy_to_acoustic(binary_a, support, PROPERTIES)
    acoustic_b = binary_occupancy_to_acoustic(binary_b, support, PROPERTIES)
    assert torch.equal(operator(*acoustic_a, support), operator(*acoustic_b, support))


def test_b2_reuses_pcn_and_metropolis_regression() -> None:
    current = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 2, 2)
    proposal, innovation = pcn_proposal(
        current,
        beta=0.1,
        generator=torch.Generator(device="cpu").manual_seed(7),
    )
    expected = (1.0 - 0.1**2) ** 0.5 * current + 0.1 * innovation
    assert torch.equal(proposal, expected)
    assert posterior_energy(8.0, 1.0) == 4.0
    assert metropolis_decision(3.0, 5.0, 0.5)["accepted"] is False


def test_b2_rejected_state_accounting_repeats_current_state() -> None:
    categorical_states: list[torch.Tensor] = []
    binary_states: list[torch.Tensor] = []
    current = torch.zeros((1, 1, 1, 1, 1), dtype=torch.int8)
    binary = torch.zeros_like(current, dtype=torch.uint8)
    # A rejected proposal does not replace current; recording therefore repeats it.
    for iteration in (1, 2):
        assert record_current_state(
            categorical_states,
            binary_states,
            iteration=iteration,
            burn_in=0,
            thinning=1,
            current_categorical=current,
            current_binary=binary,
        )
    assert len(categorical_states) == 2
    assert torch.equal(categorical_states[0], categorical_states[1])
    assert torch.equal(binary_states[0], binary_states[1])


def test_b2_burnin_accounting_is_exactly_96_states() -> None:
    retained = sum(
        should_retain_iteration(iteration, burn_in=8, thinning=1)
        for _chain in range(4)
        for iteration in range(1, 33)
    )
    assert retained == 96


def test_b2_config_is_frozen_and_truth_blind_runner_has_no_restricted_assets() -> None:
    config_path = (
        PROJECT_DIR
        / "experiments/stage15_binary_seismic_consensus/configs/binary_flow_pcn_pilot_v1.json"
    )
    config = __import__("json").loads(config_path.read_text(encoding="utf-8"))
    assert validate_b2_config(config)["expected_retained_states"] == 96
    source = (
        PROJECT_DIR / "scripts/stage15/run_binary_flow_pcn.py"
    ).read_text(encoding="utf-8")
    assert "true_model.pt" not in source
    assert "binary_truth.pt" not in source


def test_b3_positive_tp_fp_fn_metrics_are_exact() -> None:
    predicted = torch.tensor([True, True, False, False])
    actual = torch.tensor([True, False, True, False])
    domain = torch.ones(4, dtype=torch.bool)
    metrics = b3_binary_metrics(predicted, actual, domain)
    assert metrics["predicted_voxels"] == 2
    assert metrics["true_positive_voxels"] == 1
    assert metrics["false_positive_voxels"] == 1
    assert metrics["false_negative_voxels"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["iou"] == 1 / 3


def test_b3_truth_partition_exactly_covers_subsurface_truth() -> None:
    support = torch.tensor([[[[[True, True, True, True, False]]]]])
    positive = torch.tensor([[[[[True, False, False, False, False]]]]])
    unknown = torch.tensor([[[[[False, True, False, False, False]]]]])
    negative = torch.tensor([[[[[False, False, True, True, False]]]]])
    target = torch.tensor([[[[[True, True, True, False, False]]]]])
    result = b3_truth_partition(target, positive, unknown, negative, support)
    assert result["subsurface_partition_exact"] is True
    assert result["counts"] == {"positive": 1, "unknown": 1, "negative": 1}
    assert sum(result["fractions"].values()) == 1.0


def test_b3_retained_metric_accounting_requires_exactly_96() -> None:
    valid = [
        {"sample_group": "retained", "sample_index": index}
        for index in range(96)
    ]
    validate_retained_accounting(valid)
    with __import__("pytest").raises(ValueError, match="96"):
        validate_retained_accounting(valid[:-1])


def test_b3_evaluator_is_truth_enabled_but_b2_runner_remains_truth_blind() -> None:
    evaluator = (
        PROJECT_DIR / "scripts/stage15/evaluate_binary_flow_pcn_truth.py"
    ).read_text(encoding="utf-8")
    runner = (
        PROJECT_DIR / "scripts/stage15/run_binary_flow_pcn.py"
    ).read_text(encoding="utf-8")
    assert "true_model" in evaluator
    assert "true_model.pt" not in runner
    assert "binary_truth.pt" not in runner


def test_b3_input_hash_snapshot_detects_no_evaluator_mutation(tmp_path: Path) -> None:
    for index, name in enumerate(B2_INPUT_NAMES):
        (tmp_path / name).write_bytes(f"frozen-{index}".encode())
    before = snapshot_hashes(tmp_path)
    assert_hashes_unchanged(tmp_path, before)
    assert before == snapshot_hashes(tmp_path)


def test_b4_aligns_all_96_states_by_index_and_dtype_aware_hash() -> None:
    states = torch.zeros((96, 1, 64, 64, 64), dtype=torch.uint8)
    uint8_hash = tensor_sha256(states[0:1])
    float32_hash = tensor_sha256(states[0:1].float())
    trace_rows = []
    truth_rows = []
    for index in range(96):
        chain_id, offset = divmod(index, 24)
        trace_rows.append(
            {
                "post_burnin_recorded": "True",
                "chain_id": str(chain_id),
                "iteration": str(offset + 9),
                "current_binary_sha256": float32_hash,
                "current_hard_seismic_loss": str(index),
                "target_voxel_count": "0",
            }
        )
        truth_rows.append(
            {
                "sample_group": "retained",
                "sample_index": str(index),
                "binary_sha256": uint8_hash,
                "target_voxel_count": "0",
                "iou": "0.1",
                "precision": "0.2",
                "recall": "0.3",
                "centroid_distance": "4.0",
            }
        )
    aligned = align_retained_states(trace_rows, truth_rows, states)
    assert len(aligned) == 96
    assert [row["retained_index"] for row in aligned] == list(range(96))
    assert all(row["binary_hash_alignment_verified"] for row in aligned)


def test_b4_spearman_and_exact_quartile_accounting() -> None:
    rows = [
        {
            "retained_index": index,
            "current_hard_seismic_loss": float(index),
            "iou": float(95 - index),
            "precision": float(95 - index),
            "recall": float(95 - index),
            "centroid_distance": float(index),
        }
        for index in range(96)
    ]
    correlations = spearman_metrics(rows)
    assert correlations["iou"]["rho"] == -1.0
    assert correlations["precision"]["rho"] == -1.0
    assert correlations["recall"]["rho"] == -1.0
    assert correlations["centroid_distance"]["rho"] == 1.0
    quartiles = quartile_groups(rows)
    assert quartiles["lowest_25_percent"]["state_count"] == 24
    assert quartiles["highest_25_percent"]["state_count"] == 24
    assert set(quartiles["lowest_25_percent"]["retained_indices"]).isdisjoint(
        quartiles["highest_25_percent"]["retained_indices"]
    )
