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

from guidance.property_evaluation import (
    paired_per_class_deltas,
    paired_truth_component_recovery_deltas,
    per_class_hard_metrics,
    size_stratified_component_metrics,
    truth_component_recovery_rows,
    truth_present_mean_iou,
)
from guidance.property_sampling import fixed_euler_property_sample
from guidance.property_volume import hard_labels_to_properties
from scripts.stage2.run_property_guidance import (
    PHASE2_PAIR_FIELDS,
    paired_property_config_verdict,
    validate_args,
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


def _sampling_inputs() -> dict[str, object]:
    model = _DummyConditionalModel()
    truth = torch.zeros((1, 1, 4, 4, 4), dtype=torch.long)
    truth[:, :, 2:] = 1
    condition = torch.zeros_like(truth, dtype=torch.bool)
    condition[:, :, 0, 0, 0] = True
    embedded_truth = model.embed(truth)
    expanded = condition.expand(-1, embedded_truth.shape[1], -1, -1, -1)
    property_table = torch.tensor([[0.0, 0.5, 1.0]])
    return {
        "model": model,
        "initial_state": torch.randn((1, 3, 4, 4, 4), generator=torch.Generator().manual_seed(7)),
        "conditioning": embedded_truth * expanded,
        "embedded_truth": embedded_truth,
        "truth_model": truth,
        "condition_mask": condition,
        "target_properties": hard_labels_to_properties(truth, property_table),
        "property_table": property_table,
        "confidence": ((truth != -1) & ~condition).float(),
        "property_sigmas": (0.0,),
        "property_scale_weights": (1.0,),
        "property_channel_weights": torch.ones(1),
        "n_steps": 4,
        "max_guidance_ratio": 0.25,
        "tau_start": 0.5,
        "tau_end": 0.1,
        "tau_schedule": "cosine",
        "guidance_start": 0.0,
        "guidance_schedule": "windowed_sine",
        "grad_clip_norm": 1.0,
    }


def test_property_alpha_zero_matches_reference_projected_fixed_euler() -> None:
    inputs = _sampling_inputs()
    expected = _reference_projected_euler(
        inputs["model"],
        inputs["initial_state"],
        inputs["embedded_truth"],
        inputs["condition_mask"],
        n_steps=4,
    )
    actual, trace = fixed_euler_property_sample(**inputs, alpha=0.0)

    assert torch.equal(actual, expected)
    assert len(trace) == 4
    assert all(row["raw_grad_norm"] == 0 for row in trace)
    assert all(row["guidance_velocity_norm"] == 0 for row in trace)
    assert all(row["post_projection_condition_violations"] == 0 for row in trace)


def test_positive_property_guidance_changes_state_and_preserves_conditions() -> None:
    inputs = _sampling_inputs()
    baseline, _ = fixed_euler_property_sample(**inputs, alpha=0.0)
    guided, trace = fixed_euler_property_sample(**inputs, alpha=0.20)
    condition = inputs["condition_mask"].expand(-1, 3, -1, -1, -1)

    assert torch.isfinite(guided).all()
    assert not torch.equal(guided, baseline)
    assert torch.equal(guided[condition], inputs["embedded_truth"][condition])
    assert any(row["raw_grad_norm"] > 0 for row in trace)
    assert all(row["post_projection_condition_violations"] == 0 for row in trace)


def test_phase2_strict_pairing_includes_property_assets() -> None:
    baseline = {field: f"same-{field}" for field in PHASE2_PAIR_FIELDS}
    baseline.update({"alpha": 0.0, "run_status": "completed", "samples_written": 2, "n_samples": 2})
    guided = dict(baseline)
    guided["alpha"] = 0.1

    paired, _ = paired_property_config_verdict(baseline, guided)
    assert paired

    guided["target_properties_sha256"] = "different"
    paired, reason = paired_property_config_verdict(baseline, guided)
    assert not paired
    assert "target_properties_sha256" in reason


def test_phase2_runner_refuses_unpaired_guidance_and_nonempty_output(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        model_weights="ema",
        n_samples=1,
        n_steps=4,
        alpha=0.1,
        baseline_dir=None,
        max_guidance_ratio=0.1,
        grad_clip_norm=1.0,
        guidance_start=0.25,
        target_roi_radius=2,
        output_dir=tmp_path / "new",
    )
    with pytest.raises(ValueError, match="baseline-dir"):
        validate_args(args)

    args.alpha = 0.0
    args.output_dir.mkdir()
    (args.output_dir / "existing.txt").write_text("immutable", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        validate_args(args)


def test_per_class_evaluation_and_pair_deltas_cover_absent_class() -> None:
    truth = torch.tensor([0, 0, 1, 1]).reshape(1, 1, 1, 1, 4)
    baseline = torch.tensor([0, 1, 1, 1]).reshape(1, 1, 1, 1, 4)
    guided = torch.tensor([0, 0, 1, 2]).reshape(1, 1, 1, 1, 4)
    baseline_rows = per_class_hard_metrics(baseline, truth, 0, class_ids=(0, 1, 2))
    guided_rows = per_class_hard_metrics(guided, truth, 0, class_ids=(0, 1, 2))
    deltas = paired_per_class_deltas(baseline_rows, guided_rows)

    assert guided_rows[0]["iou"] == pytest.approx(1.0)
    assert guided_rows[2]["truth_present"] is False
    assert guided_rows[2]["predicted_volume"] == 1
    assert deltas[0]["delta_iou"] > 0


def test_truth_present_mean_iou_has_fixed_paired_denominator() -> None:
    truth = torch.tensor([0, 0, 1, 1]).reshape(1, 1, 1, 1, 4)
    baseline = torch.tensor([0, 1, 1, 1]).reshape(1, 1, 1, 1, 4)
    # Class 2 is absent from truth. Its four-voxel-scale analogue should remain
    # visible in per-class/global-union metrics without changing this denominator.
    guided = torch.tensor([0, 0, 1, 2]).reshape(1, 1, 1, 1, 4)

    assert truth_present_mean_iou(baseline, truth) == pytest.approx(7 / 12)
    assert truth_present_mean_iou(guided, truth) == pytest.approx(3 / 4)


def test_size_stratified_component_metrics_separate_tiny_mass() -> None:
    mask = torch.zeros((1, 1, 6, 6, 6), dtype=torch.bool)
    mask[:, :, 0:2, 0:2, 0:2] = True
    mask[:, :, 4, 4, 4:6] = True

    metrics = size_stratified_component_metrics(mask)

    assert metrics["target_components_ge_5"] == 1
    assert metrics["target_components_ge_20"] == 0
    assert metrics["target_tiny_component_mass_le_5"] == 2
    assert metrics["target_tiny_component_mass_fraction_le_5"] == pytest.approx(
        0.2
    )
    assert metrics["target_top4_component_mass_fraction"] == pytest.approx(1.0)
    assert metrics["target_component_1_voxels"] == 8
    assert metrics["target_component_2_voxels"] == 2
    assert metrics["target_component_3_voxels"] == 0


def test_truth_component_recovery_rows_and_pair_deltas() -> None:
    truth = torch.ones((1, 1, 6, 6, 6), dtype=torch.long)
    truth[:, :, 0, 0, 0:4] = 9
    truth[:, :, 3:5, 3, 3] = 9
    baseline = torch.ones_like(truth)
    baseline[:, :, 0, 0, 0] = 9
    guided = baseline.clone()
    guided[:, :, 0, 0, 1] = 9
    guided[:, :, 3, 3, 3] = 9
    guided[:, :, 5, 5, 5] = 9

    baseline_rows = truth_component_recovery_rows(baseline, truth, 9, 0)
    guided_rows = truth_component_recovery_rows(guided, truth, 9, 0)
    deltas = paired_truth_component_recovery_deltas(
        baseline_rows,
        guided_rows,
    )

    assert [row["truth_component_voxels"] for row in guided_rows] == [4, 2]
    assert [row["recovered_voxels"] for row in guided_rows] == [2, 1]
    assert [row["recall"] for row in guided_rows] == pytest.approx([0.5, 0.5])
    assert [row["delta_recovered_voxels"] for row in deltas] == [1, 1]
