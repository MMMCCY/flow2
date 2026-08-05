from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from guidance.seismic import ConvolutionalSeismic, hard_labels_to_acoustic
from scripts.stage4.audit_seismic_identifiability import (
    _substitution_sensitivity,
    build_selection_summary,
    geological_support_checks,
    one_sided_negative_permutation_pvalue,
    spearman_rank_correlation,
    validate_output_directory,
    validate_source_config,
)


def _candidate(
    seed: int,
    sample_id: int,
    loss: float,
    quality: float,
    *,
    supported: bool = False,
) -> dict[str, object]:
    return {
        "candidate_id": f"seed{seed}_sample{sample_id}",
        "seed": seed,
        "local_sample_id": sample_id,
        "condition_violation_count": 0,
        "hard_seismic_loss": loss,
        "global_voxel_accuracy": quality,
        "truth_present_mean_iou": quality,
        "target_iou": 0.40 if supported else quality,
        "target_precision": 0.80 if supported else quality,
        "target_recall": 0.40 if supported else quality,
        "major_component_min_recall": 0.30 if supported else quality,
        "major_component_mean_recall": 0.45 if supported else quality,
    }


def test_spearman_and_negative_permutation_are_deterministic() -> None:
    loss = [1.0, 2.0, 3.0, 4.0, 5.0]
    quality = [5.0, 4.0, 3.0, 2.0, 1.0]
    assert spearman_rank_correlation(loss, quality) == pytest.approx(-1.0)
    first = one_sided_negative_permutation_pvalue(
        loss, quality, permutations=999, seed=17
    )
    second = one_sided_negative_permutation_pvalue(
        loss, quality, permutations=999, seed=17
    )
    assert first == second
    assert first < 0.02


def test_selection_uses_loss_tie_break_and_requires_support() -> None:
    rows = [
        _candidate(142, 0, 1.0, 0.35, supported=True),
        _candidate(42, 1, 1.0, 0.34, supported=True),
        _candidate(42, 2, 2.0, 0.20),
        _candidate(242, 0, 3.0, 0.10),
    ]
    ranking, summary = build_selection_summary(rows, permutations=199)
    assert ranking[0]["candidate_id"] == "seed42_sample1"
    assert summary["support_gate"]["passed"] is True
    assert summary["ranking_gate"]["passed"] is True
    assert summary["promoted"] is True

    unsupported = [
        _candidate(42, index, float(index + 1), 0.20 - index * 0.02)
        for index in range(4)
    ]
    _, failed = build_selection_summary(unsupported, permutations=99)
    assert failed["support_gate"]["passed"] is False
    assert failed["promoted"] is False
    assert "lacks geological support" in failed["decision"]


def test_support_gate_and_source_config_reject_invalid_invariants() -> None:
    supported = _candidate(42, 0, 1.0, 0.4, supported=True)
    assert all(geological_support_checks(supported).values())
    supported["condition_violation_count"] = 1
    assert geological_support_checks(supported)["conditions_exact"] is False

    config = {
        "seed": 42,
        "n_samples": 4,
        "samples_written": 4,
        "n_steps": 32,
        "alpha": 0.0,
        "run_status": "completed",
        "ema_applied": True,
        "model_weight_source": "ema",
        "integrator": "fixed_euler_midpoint_v1",
        "initial_noise_policy": "single_cpu_generator_sequential_samples_v1",
        "max_post_projection_condition_violations": 0,
        "truth_model_sha256": "truth",
        "boreholes_sha256": "boreholes",
        "checkpoint_sha256": "checkpoint",
        "sample_sha256": ["a", "b", "c", "d"],
        "initial_noise_sha256": ["e", "f", "g", "h"],
        "model_load_report": {
            "weight_source": "ema",
            "ema_applied": True,
            "ema_missing_trainable": [],
            "ema_shape_mismatches": [],
        },
    }
    validate_source_config(
        config,
        expected_seed=42,
        expected_truth_hash="truth",
        expected_boreholes_hash="boreholes",
        expected_checkpoint_hash="checkpoint",
    )
    config["model_weight_source"] = "raw"
    with pytest.raises(ValueError, match="model_weight_source"):
        validate_source_config(
            config,
            expected_seed=42,
            expected_truth_hash="truth",
            expected_boreholes_hash="boreholes",
            expected_checkpoint_hash="checkpoint",
        )


def test_truth_substitution_preserves_conditions() -> None:
    truth = torch.tensor([[[[[0, 0, 1]]]]], dtype=torch.long)
    condition = torch.zeros_like(truth, dtype=torch.bool)
    condition[..., 0] = True
    subsurface = torch.ones_like(truth, dtype=torch.bool)
    property_table = torch.tensor(
        [
            [1000.0, 4_000_000.0, 7_500_000.0],
            [0.001, 0.0005, 0.0004],
        ]
    )
    target = hard_labels_to_acoustic(truth, property_table)
    operator = ConvolutionalSeismic(
        (1, 1, 3),
        cell_size_m=(10.0, 10.0, 8.0),
        num_time_samples=16,
        sample_interval_ms=8.0,
        peak_frequency_hz=25.0,
        wavelet_duration_ms=16.0,
    )
    observed = operator(target[:, 0:1], target[:, 1:2], subsurface)
    rows, summary = _substitution_sensitivity(
        truth=truth,
        condition_mask=condition,
        target_acoustic=target,
        property_table=property_table,
        subsurface_mask=subsurface,
        forward_operator=operator,
        observed=observed,
        sample_mask=torch.ones_like(observed),
        uncertainty=torch.full_like(observed, 0.01),
        device=torch.device("cpu"),
    )
    assert rows
    assert all(row["condition_violation_count"] == 0 for row in rows)
    source_zero = [row for row in rows if row["source_label"] == 0]
    assert source_zero[0]["changed_voxels"] == 1
    assert summary["truth_derived_oracle_perturbation"] is True


def test_nonempty_output_directory_is_refused(tmp_path: Path) -> None:
    output = tmp_path / "report"
    output.mkdir()
    (output / "evidence.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        validate_output_directory(output, overwrite=False)
    validate_output_directory(output, overwrite=True)
