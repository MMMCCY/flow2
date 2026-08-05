from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from guidance.seismic import (
    ACOUSTIC_CODEBOOK_SCHEMA,
    ConvolutionalSeismic,
    acoustic_tables_from_config,
    build_seismic_observation,
    hard_labels_to_acoustic,
    probabilities_to_acoustic,
    probabilities_to_subsurface_acoustic,
    seismic_field_loss,
    seismic_operator_from_config,
    seismic_volume_loss,
    validate_contiguous_subsurface_mask,
)
from guidance.seismic_sampling import fixed_euler_seismic_sample
from scripts.stage4.run_seismic_guidance import (
    PHASE4C_PAIR_FIELDS,
    load_controller_level,
    load_observation_assets,
    paired_seismic_config_verdict,
    validate_args as validate_seismic_runner_args,
)


CONFIG_DIR = PROJECT_DIR / "experiments/stage4_seismic/configs"


def _operator(
    shape: tuple[int, int, int] = (1, 1, 3),
    *,
    dz_m: float = 8.0,
    num_time_samples: int = 16,
    sample_interval_ms: float = 8.0,
) -> ConvolutionalSeismic:
    return ConvolutionalSeismic(
        shape,
        cell_size_m=(10.0, 10.0, dz_m),
        num_time_samples=num_time_samples,
        sample_interval_ms=sample_interval_ms,
        peak_frequency_hz=25.0,
        wavelet_duration_ms=16.0,
    )


def _small_codebook() -> dict[str, object]:
    return {
        "schema": ACOUSTIC_CODEBOOK_SCHEMA,
        "id": "test_acoustic",
        "density_unit": "kg m^-3",
        "velocity_unit": "m s^-1",
        "values": {
            "-1": {"density": 1.0, "vp": 1000.0},
            "0": {"density": 2.0, "vp": 2000.0},
            "1": {"density": 3.0, "vp": 2500.0},
        },
    }


def test_acoustic_codebook_and_observation_configs_are_complete() -> None:
    acoustic = json.loads(
        (CONFIG_DIR / "acoustic_distinct_label9_upper_bound_v1.json").read_text()
    )
    tables, metadata = acoustic_tables_from_config(acoustic, 15)
    assert tables.property_table.shape == (2, 15)
    assert metadata["site_calibrated_petrophysics"] is False
    assert tables.impedance_kg_m2_s[10] == tables.impedance_kg_m2_s.max()
    assert torch.allclose(tables.slowness_s_m, tables.velocity_m_s.reciprocal())

    broken = {**acoustic, "values": {**acoustic["values"]}}
    del broken["values"]["13"]
    with pytest.raises(ValueError, match="coverage"):
        acoustic_tables_from_config(broken, 15)

    config = json.loads(
        (CONFIG_DIR / "full_cube_noiseless_inverse_crime_v1.json").read_text()
    )
    operator, resolved = seismic_operator_from_config(config, grid_shape=(64, 64, 64))
    assert operator.wavelet_num_samples == 17
    assert operator.recording_end_ms == pytest.approx(2552.0)
    assert resolved["truth_derived"] is True
    assert resolved["measured_geophysics"] is False
    with pytest.raises(ValueError, match="does not match"):
        seismic_operator_from_config(config, grid_shape=(32, 32, 32))


def test_hard_and_one_hot_soft_acoustic_mappings_agree() -> None:
    tables, _ = acoustic_tables_from_config(_small_codebook(), 3)
    labels = torch.tensor([[[[[-1, 0, 1], [1, 0, -1]]]]], dtype=torch.float32)
    hard = hard_labels_to_acoustic(labels, tables.property_table)
    categories = labels.long()[:, 0] + 1
    one_hot = F.one_hot(categories, num_classes=3).permute(0, 4, 1, 2, 3).float()
    soft = probabilities_to_acoustic(one_hot, tables.property_table)
    assert torch.equal(hard, soft)


def test_known_subsurface_soft_mapping_excludes_nonphysical_air_travel_time() -> None:
    tables, _ = acoustic_tables_from_config(_small_codebook(), 3)
    probabilities = torch.tensor(
        [[[[[0.99, 0.99, 0.99]]], [[[0.005, 0.005, 0.005]]], [[[0.005, 0.005, 0.005]]]]],
        dtype=torch.float64,
    )
    subsurface = torch.ones((1, 1, 1, 1, 3), dtype=torch.bool)
    direct = probabilities_to_acoustic(probabilities, tables.property_table)
    conditioned = probabilities_to_subsurface_acoustic(
        probabilities, tables.property_table, subsurface
    )
    assert conditioned[:, 1].max() < direct[:, 1].min()
    expected_rock_slowness = 0.5 * (1.0 / 2000.0 + 1.0 / 2500.0)
    assert conditioned[:, 1].mean().item() == pytest.approx(expected_rock_slowness)


def test_subsurface_mask_requires_one_contiguous_interval_per_column() -> None:
    valid = torch.tensor([[[[[1, 1, 0, 0], [1, 1, 1, 0]]]]], dtype=torch.bool)
    report = validate_contiguous_subsurface_mask(valid)
    assert report["minimum_subsurface_cells"] == 2
    assert report["maximum_subsurface_cells"] == 3

    reentry = torch.tensor([[[[[1, 0, 1, 0]]]]], dtype=torch.bool)
    with pytest.raises(ValueError, match="contiguous"):
        validate_contiguous_subsurface_mask(reentry)
    empty = torch.zeros((1, 1, 1, 1, 4), dtype=torch.bool)
    with pytest.raises(ValueError, match="at least one"):
        validate_contiguous_subsurface_mask(empty)


def test_constant_impedance_is_zero_and_single_interface_has_correct_polarity() -> None:
    operator = _operator()
    rock = torch.ones((1, 1, 1, 1, 3), dtype=torch.bool)
    slowness = torch.full((1, 1, 1, 1, 3), 1.0 / 2000.0, dtype=torch.float64)
    constant = torch.full_like(slowness, 4.0)
    assert torch.count_nonzero(operator(constant, slowness, rock)) == 0

    # Input z is bottom-to-top: top impedance 2 over two cells of impedance 4.
    impedance = torch.tensor([[[[[4.0, 4.0, 2.0]]]]], dtype=torch.float64)
    reflectivity, time_ms, valid = operator.interface_response(
        impedance, slowness, rock
    )
    assert valid.sum().item() == 2
    assert reflectivity[..., 0].item() == pytest.approx(1.0 / 3.0)
    assert time_ms[..., 0].item() == pytest.approx(8.0)
    spikes = operator.deposit_reflectivity(reflectivity, time_ms, valid)
    assert spikes[..., 1].item() == pytest.approx(1.0 / 3.0)
    assert spikes.sum().item() == pytest.approx(1.0 / 3.0)
    trace = operator(impedance, slowness, rock)
    assert trace[..., 1].item() == pytest.approx(1.0 / 3.0)


def test_slowness_moves_reflection_later_without_lateral_crosstalk() -> None:
    operator = _operator((2, 1, 3))
    rock = torch.ones((1, 1, 2, 1, 3), dtype=torch.bool)
    impedance = torch.tensor(
        [[[[[4.0, 4.0, 2.0]], [[3.0, 3.0, 3.0]]]]], dtype=torch.float64
    )
    fast = torch.full_like(impedance, 1.0 / 2000.0)
    slow = torch.full_like(impedance, 1.0 / 1000.0)
    fast_spikes = operator.reflectivity_spikes(impedance, fast, rock)
    slow_spikes = operator.reflectivity_spikes(impedance, slow, rock)
    assert fast_spikes[0, 0, 0, 0].abs().argmax().item() == 1
    assert slow_spikes[0, 0, 0, 0].abs().argmax().item() == 2
    assert torch.count_nonzero(fast_spikes[0, 0, 1, 0]) == 0
    assert torch.count_nonzero(operator(impedance, fast, rock)[0, 0, 1, 0]) == 0


def test_linear_deposition_conserves_amplitude_and_wavelet_has_no_wraparound() -> None:
    operator = _operator(num_time_samples=12)
    reflectivity = torch.tensor([[[[[0.4, -0.1]]]]], dtype=torch.float64)
    time_ms = torch.tensor([[[[[10.0, 21.0]]]]], dtype=torch.float64)
    valid = torch.ones_like(reflectivity, dtype=torch.bool)
    spikes = operator.deposit_reflectivity(reflectivity, time_ms, valid)
    assert spikes.sum().item() == pytest.approx(0.3)

    edge = torch.zeros((1, 1, 1, 1, 12), dtype=torch.float64)
    edge[..., -1] = 1.0
    convolved = operator.convolve_reflectivity_spikes(edge)
    assert torch.count_nonzero(convolved[..., :9]) == 0
    assert convolved[..., -1].item() == pytest.approx(1.0)


def test_prediction_arrivals_are_cropped_but_truth_builder_window_is_strict() -> None:
    operator = _operator(num_time_samples=4)
    reflectivity = torch.tensor([[[[[0.4, -0.1]]]]], dtype=torch.float64)
    time_ms = torch.tensor([[[[[40.0, 48.0]]]]], dtype=torch.float64)
    valid = torch.ones_like(reflectivity, dtype=torch.bool)
    cropped = operator.deposit_reflectivity(reflectivity, time_ms, valid)
    assert torch.count_nonzero(cropped) == 0
    with pytest.raises(ValueError, match="recording window"):
        operator.deposit_reflectivity(
            reflectivity,
            time_ms,
            valid,
            require_all_interfaces_in_window=True,
        )


def test_finite_difference_directional_derivative_matches_autograd() -> None:
    operator = _operator(dz_m=10.0, num_time_samples=24)
    rock = torch.ones((1, 1, 1, 1, 3), dtype=torch.bool)
    impedance = torch.tensor(
        [[[[[5.5, 4.2, 2.1]]]]], dtype=torch.float64, requires_grad=True
    )
    slowness = torch.tensor(
        [[[[[0.00043, 0.00047, 0.00045]]]]], dtype=torch.float64, requires_grad=True
    )
    direction_impedance = torch.tensor(
        [[[[[0.3, -0.2, 0.1]]]]], dtype=torch.float64
    )
    direction_slowness = torch.tensor(
        [[[[[1e-5, -2e-5, 1e-5]]]]], dtype=torch.float64
    )
    weights = torch.linspace(-0.3, 0.7, 24, dtype=torch.float64).reshape(1, 1, 1, 1, -1)
    objective = (operator(impedance, slowness, rock) * weights).sum()
    grad_impedance, grad_slowness = torch.autograd.grad(
        objective, (impedance, slowness)
    )
    autograd_directional = (
        (grad_impedance * direction_impedance).sum()
        + (grad_slowness * direction_slowness).sum()
    )
    step = 1e-5
    plus = operator(
        (impedance + step * direction_impedance).detach(),
        (slowness + step * direction_slowness).detach(),
        rock,
    )
    minus = operator(
        (impedance - step * direction_impedance).detach(),
        (slowness - step * direction_slowness).detach(),
        rock,
    )
    finite_difference = ((plus - minus) * weights).sum() / (2.0 * step)
    assert torch.allclose(autograd_directional, finite_difference, rtol=2e-6, atol=1e-8)


def test_mask_and_uncertainty_loss_normalization() -> None:
    predicted = torch.tensor([[[[[1.0, 3.0], [5.0, 7.0]]]]])
    observed = torch.zeros_like(predicted)
    mask = torch.tensor([[[[[1.0, 0.0], [1.0, 0.0]]]]])
    uncertainty = torch.full_like(predicted, 2.0)
    loss, diagnostics = seismic_field_loss(predicted, observed, mask, uncertainty)
    assert loss.item() == pytest.approx(((1.0 / 2.0) ** 2 + (5.0 / 2.0) ** 2) / 2)
    assert diagnostics["valid_seismic_sample_count"].item() == 2
    with pytest.raises(ValueError, match="positive"):
        seismic_field_loss(predicted, observed, mask, torch.zeros_like(predicted))


def test_exact_condition_overwrite_gives_zero_condition_state_gradient() -> None:
    torch.manual_seed(8)
    operator = _operator()
    tables, _ = acoustic_tables_from_config(_small_codebook(), 3)
    table = tables.property_table.double()
    truth = torch.tensor([[[[[-1, 0, 1]]]]], dtype=torch.long)
    target = hard_labels_to_acoustic(truth, table)
    subsurface = torch.ones_like(truth, dtype=torch.bool)
    observed = operator(target[:, 0:1], target[:, 1:2], subsurface).detach()
    condition = torch.zeros_like(truth, dtype=torch.bool)
    condition[..., 0] = True
    state = torch.randn((1, 3, 1, 1, 3), dtype=torch.float64, requires_grad=True)
    loss, _ = seismic_volume_loss(
        state,
        torch.eye(3, dtype=torch.float64),
        table,
        target,
        condition,
        subsurface,
        operator,
        observed,
        torch.ones_like(observed),
        torch.full_like(observed, 0.01),
        tau=0.25,
    )
    gradient = torch.autograd.grad(loss, state)[0]
    expanded_condition = condition.expand_as(gradient)
    assert torch.isfinite(loss)
    assert torch.isfinite(gradient).all()
    assert gradient.norm() > 0
    assert torch.count_nonzero(gradient[expanded_condition]) == 0


def test_observation_is_deterministic_and_noise_changes_observation_only() -> None:
    operator = _operator()
    acoustic = torch.tensor(
        [[[[[4.0, 4.0, 2.0]]], [[[0.0005, 0.0005, 0.0005]]]]],
        dtype=torch.float64,
    )
    rock = torch.ones((1, 1, 1, 1, 3), dtype=torch.bool)
    first = build_seismic_observation(
        acoustic,
        rock,
        operator,
        uncertainty_amplitude=0.02,
        noise_std_amplitude=0.005,
        noise_seed=17,
    )
    second = build_seismic_observation(
        acoustic,
        rock,
        operator,
        uncertainty_amplitude=0.02,
        noise_std_amplitude=0.005,
        noise_seed=17,
    )
    assert torch.equal(first.values, second.values)
    assert torch.equal(first.noise, second.noise)
    assert torch.equal(first.noiseless, operator(acoustic[:, 0:1], acoustic[:, 1:2], rock))
    assert not torch.equal(first.values, first.noiseless)
    assert first.metadata["values_sha256"] == second.metadata["values_sha256"]


def test_canonical_observation_assets_validate_read_only() -> None:
    observation_dir = (
        PROJECT_DIR
        / "experiments/stage4_seismic/observations/cond_generation_0/distinct_upper_bound_v1_fix2"
    )
    truth_path = PROJECT_DIR / "samples/jupyter-demo/cond_generation_0/true_model.pt"
    truth = torch.load(truth_path, map_location="cpu", weights_only=False).long()
    tensors, manifest, operator, resolved = load_observation_assets(
        observation_dir,
        truth,
        truth_path=truth_path,
        num_categories=15,
    )
    assert manifest["status"] == "complete"
    assert tuple(tensors["observed_seismic.pt"].shape) == (1, 1, 64, 64, 320)
    assert operator.num_time_samples == 320
    assert resolved["inverse_crime"] is True


def test_seismic_runner_requires_ema_strict_baseline_and_frozen_controller(
    tmp_path: Path,
) -> None:
    common = dict(
        model_weights="ema",
        n_samples=1,
        n_steps=32,
        alpha=0.25,
        baseline_dir=tmp_path / "baseline",
        max_guidance_ratio=0.25,
        grad_clip_norm=1.0,
        guidance_start=0.25,
        target_roi_radius=6,
        output_dir=tmp_path / "guided",
    )
    validate_seismic_runner_args(SimpleNamespace(**common))
    with pytest.raises(ValueError, match="EMA"):
        validate_seismic_runner_args(SimpleNamespace(**{**common, "model_weights": "raw"}))
    with pytest.raises(ValueError, match="baseline-dir"):
        validate_seismic_runner_args(SimpleNamespace(**{**common, "baseline_dir": None}))

    controller_path = CONFIG_DIR / "seismic_controller_manifest_v1.json"
    level = load_controller_level(
        controller_path,
        "alpha025_cap025",
        run_alpha=0.25,
        max_guidance_ratio=0.25,
    )
    assert level["role"] == "primary_inverse_crime_upper_bound"
    with pytest.raises(ValueError, match="does not match"):
        load_controller_level(
            controller_path,
            "alpha025_cap025",
            run_alpha=0.10,
            max_guidance_ratio=0.25,
        )

    baseline = {field: field for field in PHASE4C_PAIR_FIELDS}
    baseline.update(alpha=0.0, run_status="completed", samples_written=1, n_samples=1)
    guided = dict(baseline)
    guided.update(alpha=0.25, run_status="running")
    paired, _ = paired_seismic_config_verdict(baseline, guided)
    assert paired
    guided["initial_noise_policy"] = "different"
    paired, reason = paired_seismic_config_verdict(baseline, guided)
    assert not paired
    assert "initial_noise_policy" in reason


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


def test_seismic_alpha_zero_is_the_projected_fixed_euler_baseline() -> None:
    torch.manual_seed(11)
    shape = (1, 1, 3)
    model = _DummyConditionalModel()
    truth = torch.tensor([[[[[-1, 0, 1]]]]], dtype=torch.long)
    condition = torch.zeros_like(truth, dtype=torch.bool)
    condition[..., 0] = True
    embedded_truth = model.embed(truth)
    embedded_mask = condition.expand_as(embedded_truth)
    conditioning = torch.where(embedded_mask, embedded_truth, torch.zeros_like(embedded_truth))
    initial = torch.randn((1, 3, *shape))
    tables, _ = acoustic_tables_from_config(_small_codebook(), 3)
    table = tables.property_table
    target = hard_labels_to_acoustic(truth, table)
    subsurface = torch.ones_like(truth, dtype=torch.bool)
    operator = _operator(shape)
    observed = operator(target[:, 0:1], target[:, 1:2], subsurface)
    final, trace = fixed_euler_seismic_sample(
        model=model,
        initial_state=initial,
        conditioning=conditioning,
        embedded_truth=embedded_truth,
        truth_model=truth,
        condition_mask=condition,
        target_acoustic=target,
        property_table=table,
        guidance_confidence=(~condition).float(),
        subsurface_mask=subsurface,
        forward_operator=operator,
        observed=observed,
        sample_mask=torch.ones_like(observed),
        uncertainty=torch.full_like(observed, 0.01),
        n_steps=4,
        alpha=0.0,
        max_guidance_ratio=0.25,
        tau_start=0.5,
        tau_end=0.1,
        tau_schedule="cosine",
        guidance_start=0.25,
        guidance_schedule="windowed_sine",
        grad_clip_norm=1.0,
    )
    expected = torch.where(embedded_mask, embedded_truth, initial)
    for _ in range(4):
        expected = expected + 0.025 / 4
        expected = torch.where(embedded_mask, embedded_truth, expected)
    assert torch.equal(final, expected)
    assert len(trace) == 4
    assert all(row["used_guidance_ratio"] == 0.0 for row in trace)
    assert all(row["post_projection_condition_violations"] == 0 for row in trace)
