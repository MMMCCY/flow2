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

from guidance.gravity import (
    GRAVITY_DENSITY_CONFIG_SCHEMA,
    RectangularPrismGravity,
    build_gravity_observation,
    density_table_from_config,
    gravity_field_loss,
    gravity_operator_from_config,
    gravity_volume_loss,
    hard_labels_to_density,
    prism_gz_kernel_mgal,
    probabilities_to_density,
)
from guidance.gravity_sampling import fixed_euler_gravity_sample
from scripts.stage4.audit_gravity_screen import _classification as gravity_classification
from scripts.stage4.rerank_gravity_ensemble import build_reranking_summary
from scripts.stage4.run_gravity_guidance import (
    PHASE4_PAIR_FIELDS,
    load_controller_level,
    load_observation_assets,
    paired_gravity_config_verdict,
    validate_args as validate_gravity_runner_args,
)


CONFIG_DIR = PROJECT_DIR / "experiments/stage4_gravity/configs"


def _operator(shape: tuple[int, int, int] = (3, 4, 2)) -> RectangularPrismGravity:
    return RectangularPrismGravity(
        shape,
        cell_size_m=(100.0, 120.0, 80.0),
        origin_m=(10.0, 20.0, -160.0),
        observation_height_m=50.0,
    )


def _small_density_config() -> dict[str, object]:
    return {
        "schema": GRAVITY_DENSITY_CONFIG_SCHEMA,
        "id": "test_density",
        "description": "test",
        "unit": "kg m^-3",
        "values": {"-1": 0.0, "0": 100.0, "1": 300.0},
    }


def test_density_configs_are_complete_and_collision_control_is_exact() -> None:
    distinct = json.loads(
        (CONFIG_DIR / "density_distinct_label9_upper_bound_v1.json").read_text()
    )
    collision = json.loads(
        (CONFIG_DIR / "density_label6_label9_collision_v1.json").read_text()
    )
    distinct_table, metadata = density_table_from_config(distinct, 15)
    collision_table, _ = density_table_from_config(collision, 15)

    assert distinct_table.shape == (15,)
    assert metadata["unit"] == "kg m^-3"
    assert metadata["site_calibrated_petrophysics"] is False
    assert distinct_table[7] != distinct_table[10]  # raw labels 6 and 9
    assert collision_table[7] == collision_table[10]

    broken = {**distinct, "values": {**distinct["values"]}}
    del broken["values"]["13"]
    with pytest.raises(ValueError, match="coverage"):
        density_table_from_config(broken, 15)


def test_fft_forward_matches_independent_direct_prism_matrix() -> None:
    operator = _operator()
    density = torch.arange(24, dtype=torch.float64).reshape(1, 1, 3, 4, 2) - 8.0
    fft_field = operator(density)[0, 0]

    stations = operator.station_coordinates(dtype=torch.float64)
    lower, upper = operator.prism_bounds(dtype=torch.float64)
    direct_kernel = prism_gz_kernel_mgal(stations, lower, upper)
    direct_field = (direct_kernel @ density.reshape(-1)).reshape(3, 4)

    assert torch.allclose(fft_field, direct_field, rtol=1e-11, atol=1e-12)
    assert fft_field.is_contiguous()


def test_linearity_positive_sign_symmetry_translation_and_no_wraparound() -> None:
    operator = RectangularPrismGravity(
        (5, 5, 1), cell_size_m=100.0, observation_height_m=50.0
    )
    first = torch.zeros((1, 1, 5, 5, 1), dtype=torch.float64)
    second = torch.zeros_like(first)
    first[..., 1, 1, 0] = 200.0
    second[..., 2, 2, 0] = 120.0
    combined = operator(2.0 * first - 0.5 * second)
    expected = 2.0 * operator(first) - 0.5 * operator(second)
    assert torch.allclose(combined, expected, rtol=1e-11, atol=1e-12)

    centre = torch.zeros_like(first)
    centre[..., 2, 2, 0] = 100.0
    centre_field = operator(centre)[0, 0]
    assert bool((centre_field > 0).all())
    assert torch.allclose(centre_field, centre_field.flip(0), rtol=1e-11, atol=1e-12)
    assert torch.allclose(centre_field, centre_field.flip(1), rtol=1e-11, atol=1e-12)

    first_field = operator(first)[0, 0]
    shifted_field = operator(second * (200.0 / 120.0))[0, 0]
    assert torch.allclose(first_field[:-1, :-1], shifted_field[1:, 1:], rtol=1e-11, atol=1e-12)
    assert first_field[-1, -1] > 0
    assert first_field[-1, -1] < first_field[1, 1]


def test_deeper_mass_is_weaker_and_far_corner_has_full_support() -> None:
    vertical = RectangularPrismGravity(
        (1, 1, 2), cell_size_m=(100.0, 100.0, 100.0), observation_height_m=50.0
    )
    bottom = torch.zeros((1, 1, 1, 1, 2), dtype=torch.float64)
    top = torch.zeros_like(bottom)
    bottom[..., 0] = 100.0
    top[..., 1] = 100.0
    assert vertical(top).item() > vertical(bottom).item() > 0

    wide = RectangularPrismGravity(
        (7, 7, 1), cell_size_m=100.0, observation_height_m=50.0
    )
    corner = torch.zeros((1, 1, 7, 7, 1), dtype=torch.float64)
    corner[..., 0, 0, 0] = 100.0
    field = wide(corner)[0, 0]
    assert field[-1, -1] > 0
    assert field[-1, -1] < field[0, 0]


def test_float32_uses_stable_float64_geometry_kernel_for_deep_cells() -> None:
    operator32 = RectangularPrismGravity(
        (8, 8, 64),
        cell_size_m=(100.0, 100.0, 50.0),
        observation_height_m=50.0,
    )
    operator64 = RectangularPrismGravity(
        (8, 8, 64),
        cell_size_m=(100.0, 100.0, 50.0),
        observation_height_m=50.0,
    )
    density32 = torch.zeros((1, 1, 8, 8, 64), dtype=torch.float32)
    density32[..., 0, 0, 0] = 100.0
    field32 = operator32(density32)
    field64 = operator64(density32.double())

    assert bool((field32 > 0).all())
    assert torch.allclose(field32.double(), field64, rtol=2e-4, atol=1e-8)


def test_finite_difference_directional_derivative_and_adjoint() -> None:
    torch.manual_seed(4)
    operator = _operator((3, 3, 2))
    density = torch.randn((1, 1, 3, 3, 2), dtype=torch.float64, requires_grad=True)
    direction = torch.randn_like(density)
    weights = torch.randn((1, 1, 3, 3), dtype=torch.float64)
    objective = (operator(density) * weights).sum()
    gradient = torch.autograd.grad(objective, density)[0]
    autograd_directional = (gradient * direction).sum()
    step = 1e-5
    finite_difference = (
        (operator((density + step * direction).detach()) * weights).sum()
        - (operator((density - step * direction).detach()) * weights).sum()
    ) / (2 * step)
    assert torch.allclose(autograd_directional, finite_difference, rtol=2e-7, atol=1e-10)

    left = (operator(density) * weights).sum()
    right = (density * gradient).sum()
    assert torch.allclose(left, right, rtol=1e-10, atol=1e-11)


def test_mask_and_uncertainty_loss_normalization() -> None:
    predicted = torch.tensor([[[[1.0, 3.0], [5.0, 7.0]]]])
    observed = torch.zeros_like(predicted)
    mask = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    uncertainty = torch.full_like(predicted, 2.0)
    loss, diagnostics = gravity_field_loss(predicted, observed, mask, uncertainty)

    assert loss.item() == pytest.approx(((1.0 / 2.0) ** 2 + (5.0 / 2.0) ** 2) / 2)
    assert diagnostics["valid_station_count"].item() == 2
    with pytest.raises(ValueError, match="positive"):
        gravity_field_loss(predicted, observed, mask, torch.zeros_like(predicted))


def test_hard_and_one_hot_soft_density_mappings_agree() -> None:
    table, _ = density_table_from_config(_small_density_config(), 3)
    labels = torch.tensor(
        [[[[[-1, 0], [1, 0]], [[1, -1], [0, 1]]]]], dtype=torch.float32
    )
    hard = hard_labels_to_density(labels, table)
    categories = labels.long()[:, 0] + 1
    one_hot = F.one_hot(categories, num_classes=3).permute(0, 4, 1, 2, 3).float()
    soft = probabilities_to_density(one_hot, table)
    assert torch.equal(hard, soft)


def test_exact_condition_overwrite_gives_zero_condition_state_gradient() -> None:
    torch.manual_seed(8)
    shape = (3, 3, 2)
    operator = _operator(shape)
    table, _ = density_table_from_config(_small_density_config(), 3)
    truth = torch.randint(-1, 2, (1, 1, *shape), dtype=torch.long)
    target_density = hard_labels_to_density(truth, table).double()
    observed = operator(target_density).detach()
    condition = torch.zeros_like(truth, dtype=torch.bool)
    condition[..., 0, 0, 0] = True
    state = torch.randn((1, 3, *shape), dtype=torch.float64, requires_grad=True)
    loss, _ = gravity_volume_loss(
        state,
        torch.eye(3, dtype=torch.float64),
        table.double(),
        target_density,
        condition,
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
    operator = _operator((2, 2, 2))
    density = torch.randn((1, 1, 2, 2, 2), dtype=torch.float64)
    first = build_gravity_observation(
        density, operator, uncertainty_mgal=0.02, noise_std_mgal=0.005, noise_seed=17
    )
    second = build_gravity_observation(
        density, operator, uncertainty_mgal=0.02, noise_std_mgal=0.005, noise_seed=17
    )
    assert torch.equal(first.values_mgal, second.values_mgal)
    assert torch.equal(first.noise_mgal, second.noise_mgal)
    assert torch.equal(first.noiseless_mgal, operator(density))
    assert not torch.equal(first.values_mgal, first.noiseless_mgal)
    assert first.metadata["values_sha256"] == second.metadata["values_sha256"]


def test_full_grid_observation_config_is_explicit_and_builds_operator() -> None:
    config = json.loads(
        (CONFIG_DIR / "full_grid_noiseless_inverse_crime_v1.json").read_text()
    )
    operator, resolved = gravity_operator_from_config(config, grid_shape=(64, 64, 64))
    assert operator.grid_shape == (64, 64, 64)
    assert resolved["truth_derived"] is True
    assert resolved["measured_geophysics"] is False
    assert resolved["inverse_crime"] is True
    assert operator.metadata()["full_support"] is True

    with pytest.raises(ValueError, match="does not match"):
        gravity_operator_from_config(config, grid_shape=(32, 32, 32))


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


def test_gravity_alpha_zero_is_the_projected_fixed_euler_baseline() -> None:
    torch.manual_seed(11)
    shape = (2, 2, 2)
    model = _DummyConditionalModel()
    truth = torch.randint(-1, 2, (1, 1, *shape), dtype=torch.long)
    condition = torch.zeros_like(truth, dtype=torch.bool)
    condition[..., 0, 0, 0] = True
    embedded_truth = model.embed(truth)
    embedded_mask = condition.expand_as(embedded_truth)
    conditioning = torch.where(embedded_mask, embedded_truth, torch.zeros_like(embedded_truth))
    initial = torch.randn((1, 3, *shape))
    table, _ = density_table_from_config(_small_density_config(), 3)
    target_density = hard_labels_to_density(truth, table)
    operator = RectangularPrismGravity(
        shape, cell_size_m=100.0, observation_height_m=50.0
    )
    observed = operator(target_density)
    final, trace = fixed_euler_gravity_sample(
        model=model,
        initial_state=initial,
        conditioning=conditioning,
        embedded_truth=embedded_truth,
        truth_model=truth,
        condition_mask=condition,
        target_density=target_density,
        density_table=table,
        guidance_confidence=(~condition).float(),
        forward_operator=operator,
        observed_mgal=observed,
        survey_mask=torch.ones_like(observed),
        uncertainty_mgal=torch.full_like(observed, 0.01),
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


def test_phase4_runner_requires_ema_strict_baseline_and_immutable_output(tmp_path) -> None:
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
    validate_gravity_runner_args(SimpleNamespace(**common))
    with pytest.raises(ValueError, match="EMA"):
        validate_gravity_runner_args(SimpleNamespace(**{**common, "model_weights": "raw"}))
    with pytest.raises(ValueError, match="baseline-dir"):
        validate_gravity_runner_args(SimpleNamespace(**{**common, "baseline_dir": None}))
    with pytest.raises(ValueError, match="takes no baseline-dir"):
        validate_gravity_runner_args(SimpleNamespace(**{**common, "alpha": 0.0}))


def test_phase4_pair_verdict_covers_all_frozen_fields() -> None:
    baseline = {field: f"same-{field}" for field in PHASE4_PAIR_FIELDS}
    baseline.update(alpha=0.0, run_status="completed", samples_written=1, n_samples=1)
    guided = dict(baseline)
    guided.update(alpha=0.25, run_status="running")
    paired, _ = paired_gravity_config_verdict(baseline, guided)
    assert paired

    guided["observed_gravity_sha256"] = "different"
    paired, reason = paired_gravity_config_verdict(baseline, guided)
    assert not paired
    assert "observed_gravity_sha256" in reason


def test_controller_manifest_freezes_alpha_and_cap() -> None:
    manifest = CONFIG_DIR / "gravity_controller_manifest_v1.json"
    baseline_level = load_controller_level(
        manifest,
        "alpha025_cap025",
        run_alpha=0.0,
        max_guidance_ratio=0.25,
    )
    guided_level = load_controller_level(
        manifest,
        "alpha025_cap025",
        run_alpha=0.25,
        max_guidance_ratio=0.25,
    )
    assert baseline_level == guided_level
    with pytest.raises(ValueError, match="does not match"):
        load_controller_level(
            manifest,
            "alpha025_cap025",
            run_alpha=0.10,
            max_guidance_ratio=0.25,
        )
    with pytest.raises(ValueError, match="cap"):
        load_controller_level(
            manifest,
            "alpha025_cap025",
            run_alpha=0.25,
            max_guidance_ratio=0.10,
        )


def test_canonical_observation_loader_validates_hashes_truth_and_operator() -> None:
    truth_path = PROJECT_DIR / "samples/jupyter-demo/cond_generation_0/true_model.pt"
    try:
        truth = torch.load(truth_path, map_location="cpu", weights_only=True)
    except TypeError:
        truth = torch.load(truth_path, map_location="cpu")
    if truth.ndim == 3:
        truth = truth[None, None]
    elif truth.ndim == 4:
        truth = truth[None]
    observation_dir = (
        PROJECT_DIR
        / "experiments/stage4_gravity/observations/cond_generation_0"
        / "distinct_upper_bound_v1_fix2"
    )
    assets, manifest, operator, resolved = load_observation_assets(
        observation_dir,
        truth.long(),
        truth_path=truth_path,
        num_categories=15,
    )
    assert manifest["status"] == "complete"
    assert resolved["id"] == "full_grid_noiseless_inverse_crime_v1"
    assert operator.metadata()["full_support"] is True
    assert torch.equal(
        assets["observed_gravity_mgal.pt"], assets["noiseless_gravity_mgal.pt"]
    )


def _rerank_row(sample_id: int, loss: float, accuracy: float) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "hard_gravity_loss": loss,
        "hard_gravity_rmse_mgal": loss**0.5,
        "global_voxel_accuracy": accuracy,
        "truth_present_mean_iou": accuracy / 2,
        "target_iou": accuracy / 3,
        "target_precision": accuracy / 2,
        "target_recall": accuracy / 2,
    }


def test_posthoc_reranking_selects_without_claiming_generation() -> None:
    baseline = [_rerank_row(0, 4.0, 0.50), _rerank_row(1, 2.0, 0.45)]
    guided = [_rerank_row(0, 1.5, 0.55), _rerank_row(1, 1.0, 0.56)]
    ranked, summary = build_reranking_summary(
        baseline, guided, pairing_reason="strict test pair"
    )
    assert [row["sample_id"] for row in ranked] == [1, 0]
    assert summary["baseline_selected_sample_id"] == 1
    assert summary["comparisons"]["guided_mean_loss_below_reranked_baseline"] is True
    assert "no baseline geology was changed" in summary["description"]


def test_gravity_screen_classification_keeps_n1_provisional() -> None:
    assert gravity_classification(1, 1, True)[0] == "single_sample_pass"
    assert gravity_classification(4, 4, True)[0] == "confirmed_seed42_pass"
    assert gravity_classification(4, 2, True)[0] == "seed42_transition"
    assert gravity_classification(4, 0, True)[0] == "confirmed_seed42_failure"
    assert gravity_classification(4, 4, False)[0] == "baseline_regression_failure"
