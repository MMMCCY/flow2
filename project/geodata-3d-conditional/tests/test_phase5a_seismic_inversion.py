from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import inference_runtime as runtime
from guidance.seismic import ConvolutionalSeismic, hard_labels_to_acoustic, ricker_wavelet
from guidance.seismic_inversion import (
    ModelBasedInversionConfig,
    build_exact_condition_acoustic,
    cell_center_twt_ms,
    invert_acoustic_member,
    labels_to_clean_prior_acoustic,
    linearized_log_impedance_operator,
    nearest_codebook_labels,
    neutral_rock_category,
    parse_inversion_config,
    posterior_statistics,
    same_length_convolution_matrix,
    sample_time_correction_to_depth,
    solve_log_impedance_correction,
)
from scripts.stage5.build_acoustic_inversion_posterior import (
    TRUTH_BLIND_OBSERVATION_FILES,
    validate_output_directory,
)


def frozen_config() -> dict[str, object]:
    return {
        "schema": "phase5a_model_based_log_impedance_config_v1",
        "id": "unit",
        "inversion_mode": "linearized_poststack_log_impedance_tikhonov_v1",
        "prior_relative_weight": 0.001,
        "vertical_smoothness_relative_weight": 0.01,
        "regularization_scale": "mean_diagonal_gtg",
        "time_difference": "forward_first_difference_last_row_zero",
        "wavelet_boundary": "zero_padding_same_length_no_wraparound",
        "time_depth_mapping": "fixed_prior_slowness_cell_center_linear_interpolation_v1",
        "slowness_update": "none_keep_prior_v1",
        "subsurface_air_policy": "codebook_rock_closest_to_median_log_impedance_v1",
        "impedance_bounds": "non_air_codebook_minmax",
        "condition_policy": "surface_and_boreholes_exact_before_and_after_inversion_v1",
        "posterior_statistics": "fixed12_population_mean_std_v1",
        "truth_tuned": False,
    }


def test_parse_frozen_config_rejects_truth_tuning() -> None:
    parsed = parse_inversion_config(frozen_config())
    assert parsed.prior_relative_weight == pytest.approx(0.001)
    invalid = frozen_config()
    invalid["truth_tuned"] = True
    with pytest.raises(ValueError, match="truth_tuned"):
        parse_inversion_config(invalid)


def test_convolution_matrix_matches_exact_zero_padded_conv1d() -> None:
    wavelet = ricker_wavelet(25.0, 4.0, 32.0, dtype=torch.float64)
    signal = torch.linspace(-1.0, 1.0, 21, dtype=torch.float64)
    matrix = same_length_convolution_matrix(wavelet, signal.numel())
    expected = F.conv1d(
        signal[None, None], wavelet[None, None], padding=wavelet.numel() // 2
    )[0, 0]
    assert torch.allclose(matrix @ signal, expected, atol=1e-12, rtol=1e-12)
    # An impulse at the first sample must not wrap into the final sample.
    impulse = torch.zeros_like(signal)
    impulse[0] = 1.0
    assert (matrix @ impulse)[-1] == 0


def test_linear_solver_reduces_a_consistent_trace_residual() -> None:
    wavelet = ricker_wavelet(20.0, 4.0, 48.0, dtype=torch.float64)
    operator, _ = linearized_log_impedance_operator(wavelet, 48)
    true_correction = 0.08 * torch.sin(
        torch.linspace(0.0, 3.0, 48, dtype=torch.float64)
    )
    residual = (operator @ true_correction).reshape(1, 1, 1, 1, -1)
    solved, diagnostics = solve_log_impedance_correction(
        residual,
        wavelet,
        ModelBasedInversionConfig("unit", 0.001, 0.01),
    )
    initial_norm = residual.norm()
    final_norm = (operator @ solved.reshape(-1) - residual.reshape(-1)).norm()
    assert final_norm < 0.25 * initial_norm
    assert diagnostics["prior_lambda"] > 0
    assert diagnostics["smoothness_lambda"] > 0


def test_cell_center_time_sampling_preserves_vertical_orientation() -> None:
    # z increases upward: two rock cells at indices 0 and 1, then air.
    slowness = torch.full((1, 1, 1, 1, 4), 0.001, dtype=torch.float64)
    subsurface = torch.tensor([[[[[True, True, False, False]]]]])
    centers = cell_center_twt_ms(slowness, subsurface, cell_size_z_m=1.0)
    correction = torch.arange(8, dtype=torch.float64).reshape(1, 1, 1, 1, 8)
    sampled = sample_time_correction_to_depth(
        correction, centers, subsurface, sample_interval_ms=1.0
    )
    assert sampled.flatten().tolist() == [3.0, 1.0, 0.0, 0.0]


def test_air_cleanup_and_conditions_use_only_mask_boreholes_and_codebook() -> None:
    table = torch.tensor(
        [[1.0, 10.0, 20.0, 40.0], [1.0, 0.5, 0.25, 0.125]],
        dtype=torch.float64,
    )
    labels = torch.tensor([[[[[-1, -1, 1, -1]]]]])
    subsurface = torch.tensor([[[[[True, True, True, False]]]]])
    acoustic, report = labels_to_clean_prior_acoustic(labels, table, subsurface)
    assert report["underground_air_voxels_replaced"] == 2
    assert report["neutral_category"] == neutral_rock_category(table)
    assert torch.equal(acoustic[..., -1], hard_labels_to_acoustic(labels, table)[..., -1])

    boreholes = torch.tensor([[[[[-1, 2, -1, -1]]]]])
    target, condition = build_exact_condition_acoustic(boreholes, subsurface, table)
    assert condition.flatten().tolist() == [False, True, False, True]
    assert target[0, 0, 0, 0, 1] == table[0, 3]
    assert target[0, 0, 0, 0, 3] == table[0, 0]


def test_member_inversion_preserves_conditions_and_reduces_simple_residual() -> None:
    table = torch.tensor(
        [[1.0, 4.0, 8.0], [1.0, 0.0005, 0.0005]], dtype=torch.float64
    )
    operator = ConvolutionalSeismic(
        (1, 1, 8),
        cell_size_m=(1.0, 1.0, 10.0),
        num_time_samples=64,
        sample_interval_ms=5.0,
        peak_frequency_hz=20.0,
        wavelet_duration_ms=100.0,
    )
    subsurface = torch.ones((1, 1, 1, 1, 8), dtype=torch.bool)
    prior_labels = torch.zeros((1, 1, 1, 1, 8), dtype=torch.long)
    truth_labels = prior_labels.clone()
    truth_labels[..., :4] = 1
    prior = hard_labels_to_acoustic(prior_labels, table)
    truth = hard_labels_to_acoustic(truth_labels, table)
    observed = operator(truth[:, 0:1], truth[:, 1:2], subsurface)
    boreholes = torch.full_like(prior_labels, -1)
    boreholes[..., 0] = truth_labels[..., 0]
    condition_target, condition_mask = build_exact_condition_acoustic(
        boreholes, subsurface, table
    )
    _, inverted, fields, _ = invert_acoustic_member(
        prior,
        observed_seismic=observed,
        subsurface_mask=subsurface,
        condition_target=condition_target,
        condition_mask=condition_mask,
        property_table=table,
        forward_operator=operator,
        config=ModelBasedInversionConfig("unit", 0.001, 0.01),
    )
    assert torch.equal(inverted[:, :, :, :, 0], condition_target[:, :, :, :, 0])
    assert torch.equal(inverted[:, 1], prior[:, 1])
    assert (fields[1] - observed).norm() < (fields[0] - observed).norm()


def test_posterior_moments_and_nearest_code_projection() -> None:
    members = torch.tensor(
        [
            [[[[10.0]]], [[[0.2]]]],
            [[[[40.0]]], [[[0.4]]]],
        ],
        dtype=torch.float64,
    )
    stats = posterior_statistics(members)
    assert stats["acoustic_mean"][0, 0, 0, 0, 0] == pytest.approx(20.0)
    assert stats["slowness_mean"][0, 0, 0, 0, 0] == pytest.approx(0.3)
    assert stats["log_impedance_std"].max() > 0
    table = torch.tensor(
        [[1.0, 10.0, 20.0, 40.0], [1.0, 0.2, 0.3, 0.4]],
        dtype=torch.float64,
    )
    mask = torch.ones((1, 1, 1, 1, 1), dtype=torch.bool)
    projected = nearest_codebook_labels(stats["acoustic_mean"], table, mask)
    assert projected.item() == 1


def test_truth_blind_inventory_and_nonempty_output_refusal(tmp_path: Path) -> None:
    assert "truth_acoustic.pt" not in TRUTH_BLIND_OBSERVATION_FILES
    assert all("truth" not in filename for filename in TRUTH_BLIND_OBSERVATION_FILES)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "evidence.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        validate_output_directory(occupied, overwrite=False)
