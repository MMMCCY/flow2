from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from guidance.residual_velocity_adapter import ResidualVelocityAdapter, cap_residual_velocity, fixed_euler_adapter_sample
from scripts.stage15.common import refuse_nonempty
from scripts.stage19 import audit_evidence, build_observations, evaluate, run_inference
from scripts.stage19.common import CONFIG_DIR, freeze_base_model, load_all_configs


def test_all_stage19_configs_are_frozen_and_schema_valid():
    configs = load_all_configs()
    assert set(configs) == {"cohort_v1.json", "evidence_v1.json", "training_v1.json", "inference_v1.json"}


def test_split_seed_ranges_do_not_overlap():
    config = load_all_configs()["cohort_v1.json"]
    ranges = []
    for value in config["splits"].values():
        seeds = {value["start"] + i * value["step"] for i in range(value["max_candidates"])}
        assert all(not seeds & previous for previous in ranges)
        ranges.append(seeds)


def test_historical_cases_are_excluded():
    config = load_all_configs()["cohort_v1.json"]
    assert config["historical_case_ids_excluded"] == [f"fullgeo_case{i:02d}" for i in range(1, 6)]


def test_fixed_wells_and_recipe():
    config = load_all_configs()["cohort_v1.json"]
    assert config["recipe"]["resolution"] == [64, 64, 64]
    assert config["fixed_well_xy"] == [[8,46],[9,5],[10,24],[27,17],[35,26],[39,59],[44,60],[48,6],[57,32]]


def test_stage17_evidence_parameters_frozen():
    config = load_all_configs()["evidence_v1.json"]
    inversion = json.loads((CONFIG_DIR.parent.parent / "stage15_binary_seismic_consensus/configs/binary_trace_boundary_inversion_v1.json").read_text())
    for key in ("refinement_passes", "prior_relative_weight", "vertical_smoothness_relative_weight"):
        assert config["inversion"][key] == inversion[key]


def test_evidence_is_unthresholded_single_channel():
    config = load_all_configs()["evidence_v1.json"]
    assert config["inversion"]["thresholding"] is False
    assert config["geophysics_channels"] == 1
    assert "threshold" not in config["adapter_input"]


def _adapter_inputs(channels=4):
    state = torch.randn(1, channels, 4, 4, 4)
    return state, torch.randn_like(state), torch.randn_like(state), torch.zeros(1, 1, 4, 4, 4, dtype=torch.bool), torch.rand(1, 1, 4, 4, 4), torch.tensor([0.5])


def test_single_channel_adapter_shape_and_zero_initialization():
    adapter = ResidualVelocityAdapter(4, geophysics_channels=1, base_width=12, dilations=(1,2,4,1))
    inputs = _adapter_inputs()
    assert torch.equal(adapter(*inputs), torch.zeros_like(inputs[0]))
    assert torch.count_nonzero(adapter.output_conv.weight) == 0
    assert torch.count_nonzero(adapter.output_conv.bias) == 0


def test_adapter_parameter_count_is_below_budget():
    adapter = ResidualVelocityAdapter(4, geophysics_channels=1, base_width=12, dilations=(1,2,4,1))
    assert adapter.parameter_count() < 100_000


def test_correction_is_zero_inside_condition_mask():
    adapter = ResidualVelocityAdapter(4, geophysics_channels=1)
    nn.init.ones_(adapter.output_conv.weight)
    inputs = list(_adapter_inputs())
    inputs[3][..., 1:3, 1:3, 1:3] = True
    result = adapter(*inputs)
    assert torch.count_nonzero(result[..., 1:3, 1:3, 1:3]) == 0


def test_residual_cap_never_exceeds_quarter():
    correction = torch.full((2, 4, 3, 3, 3), 100.0)
    base = torch.randn_like(correction)
    capped, used = cap_residual_velocity(correction, base, torch.zeros(2, 1, 3, 3, 3, dtype=torch.bool), max_ratio=0.25)
    assert torch.all(used <= 0.250001)


def test_freeze_base_model_disables_gradients():
    model = nn.Sequential(nn.Conv3d(2, 2, 1), nn.ReLU())
    freeze_base_model(model)
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in model.parameters())


class _DummyModel:
    def net(self, state, condition, time):
        return torch.ones_like(state) * time[:, None, None, None, None]


def test_scale_zero_sampler_is_adapter_independent_and_condition_exact():
    model = _DummyModel()
    adapter = ResidualVelocityAdapter(2, geophysics_channels=1)
    nn.init.normal_(adapter.output_conv.weight)
    initial = torch.randn(1, 2, 3, 3, 3)
    embedded = torch.zeros_like(initial)
    mask = torch.zeros(1, 1, 3, 3, 3, dtype=torch.bool); mask[..., 0, 0, 0] = True
    kwargs = dict(model=model, adapter=adapter, initial_state=initial, conditioning=embedded, embedded_conditions=embedded, condition_mask=mask, geophysics=torch.zeros(1,1,3,3,3), n_steps=4, adapter_scale=0.0, max_residual_ratio=0.25)
    first, _ = fixed_euler_adapter_sample(**kwargs)
    nn.init.uniform_(adapter.output_conv.weight, -10, 10)
    second, _ = fixed_euler_adapter_sample(**kwargs)
    assert torch.equal(first, second)
    assert torch.count_nonzero(first[..., 0, 0, 0]) == 0


def test_inference_arm_and_wrong_case_mapping_are_frozen():
    config = load_all_configs()["inference_v1.json"]
    assert tuple(config["arms"]) == run_inference.ARMS
    assert config["wrong_case_mapping"] == "next_test_case_cyclic"


def test_inference_runner_has_no_truth_tensor_load():
    source = inspect.getsource(run_inference)
    assert 'truth_assets' not in source
    assert 'true_model.pt' not in source
    assert 'binary_truth.pt' not in source


def test_output_directory_refusal(tmp_path: Path):
    (tmp_path / "existing").write_text("x")
    with pytest.raises(FileExistsError):
        refuse_nonempty(tmp_path)


def test_columnwise_support_fills_enclosed_air():
    truth = torch.full((1, 1, 2, 2, 4), -1)
    truth[..., 1:] = 2
    truth[..., 2] = -1
    support = build_observations.columnwise_support(truth)
    assert bool(support[..., 2].all())


def test_hard_seismic_uses_decoded_label9_occupancy():
    from scripts.stage18 import common
    source = inspect.getsource(common.hard_seismic_metrics)
    assert "geology == 9" in source


def test_case_first_median_helper():
    assert evaluate.med([1, 100, 2]) == 2


def test_evidence_gate_is_val_only_and_fixed():
    source = inspect.getsource(audit_evidence.main)
    assert 'case["split"] == "val"' in source
    gate = load_all_configs()["evidence_v1.json"]["reuse_gate"]
    assert gate == {"minimum_positive_ap_skill_cases": 6, "minimum_median_ap_skill": 0.30, "minimum_specificity_cases": 6}
