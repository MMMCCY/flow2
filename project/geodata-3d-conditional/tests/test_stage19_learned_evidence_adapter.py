from __future__ import annotations

import inspect
import json
import copy
from pathlib import Path

import pytest
import torch
from torch import nn

from guidance.residual_velocity_adapter import ResidualVelocityAdapter, cap_residual_velocity, fixed_euler_adapter_sample
from scripts.stage15.common import refuse_nonempty
from scripts.stage19 import audit_evidence, audit_generator_reuse, build_cohort_v2, build_observations, evaluate, run_evidence, run_inference
from scripts.stage19.common import CONFIG_DIR, asset, freeze_base_model, load_all_configs, validate_stage17a_reuse, validate_stage19_cohort_contract


def test_all_stage19_configs_are_frozen_and_schema_valid():
    configs = load_all_configs()
    assert set(configs) == {"cohort_v1.json", "cohort_v2.json", "evidence_v1.json", "evidence_v2.json", "training_v1.json", "inference_v1.json"}


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


def test_historical_seed_exact_generator_replay_hash_passes():
    result = audit_generator_reuse.run_audit(CONFIG_DIR / "cohort_v2.json")
    assert result["machine_decision"] == "GENERATOR_REUSE_VALIDATED"
    assert result["historical_replay"] == result["historical_replay_expected"]


def test_generator_critical_source_hash_mismatch_stops(monkeypatch):
    original = audit_generator_reuse.runtime.file_sha256
    target = "model_generators.py"
    monkeypatch.setattr(audit_generator_reuse.runtime, "file_sha256", lambda path: "bad" if str(path).endswith(target) else original(path))
    with pytest.raises(RuntimeError, match="STOP_GENERATOR_REUSE_MISMATCH"):
        audit_generator_reuse.run_audit(CONFIG_DIR / "cohort_v2.json")


def test_default_markov_matrix_hash_mismatch_stops(monkeypatch):
    original = audit_generator_reuse.runtime.file_sha256
    target = "default_markov_matrix.csv"
    monkeypatch.setattr(audit_generator_reuse.runtime, "file_sha256", lambda path: "bad" if str(path).endswith(target) else original(path))
    with pytest.raises(RuntimeError, match="STOP_GENERATOR_REUSE_MISMATCH"):
        audit_generator_reuse.run_audit(CONFIG_DIR / "cohort_v2.json")


def test_cohort_v2_only_changes_candidate_search_budget():
    configs = load_all_configs()
    v1, v2 = configs["cohort_v1.json"], configs["cohort_v2.json"]
    for field in ("recipe", "eligibility", "fixed_well_xy", "historical_case_ids_excluded"):
        assert v2[field] == v1[field]
    for split in ("train", "val", "test"):
        for field in ("accepted", "start", "step"):
            assert v2["splits"][split][field] == v1["splits"][split][field]
    assert [v2["splits"][name]["max_candidates"] for name in ("train", "val", "test")] == [1024, 256, 256]


def test_cohort_failure_trace_preserves_rejection_counts():
    trace = [
        {"root_seed": 1, "eligible": False, "rejection_reasons": ["missing_fold_or_fault_event"]},
        {"root_seed": 2, "eligible": True, "rejection_reasons": []},
        {"root_seed": 3, "eligible": False, "rejection_reasons": ["final_raw_label9_absent", "no_hidden_raw_label9_under_fixed_condition"]},
    ]
    payload = build_cohort_v2._trace_payload("train", {"accepted": 64}, trace, 1, "failed")
    assert payload["examined_count"] == 3
    assert payload["accepted_count"] == 1
    assert payload["rejection_reason_counts"] == {"final_raw_label9_absent": 1, "missing_fold_or_fault_event": 1, "no_hidden_raw_label9_under_fixed_condition": 1}


def test_eligibility_contract_mismatch_stops():
    configs = load_all_configs()
    changed = copy.deepcopy(configs["cohort_v2.json"])
    changed["eligibility"]["fold_or_fault_event_required"] = False
    reference = json.loads((CONFIG_DIR.parent.parent / "full_structuralgeo_benchmark/configs/full_complexity_targeted_v1.json").read_text())
    with pytest.raises(RuntimeError, match="STOP_GENERATOR_REUSE_MISMATCH"):
        validate_stage19_cohort_contract(changed, reference)


def test_stage17a_source_asset_hash_mismatch_stops():
    changed = copy.deepcopy(load_all_configs()["evidence_v2.json"])
    changed["observation"]["seismic_config"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="STOP_STAGE17A_REUSE_MISMATCH"):
        validate_stage17a_reuse(changed)


def test_stripped_test_registry_contains_no_truth_pointer():
    record = {
        "case_id": "stage19_test_case001", "split": "test", "root_seed": 1,
        "observation_assets": {name: {"path": name, "sha256": "x"} for name in ("condition_values", "condition_mask", "subsurface_mask", "observed_seismic")},
        "evidence": {"path": "score", "sha256": "x"},
        "observation_manifest": {"path": "observation", "sha256": "x"},
        "truth_assets": {"truth": {"path": "forbidden", "sha256": "x"}},
    }
    registry = run_evidence.build_test_inference_registry([record])
    serialized = json.dumps(registry).lower()
    assert all(word not in serialized for word in ("truth", "true_model", "binary_truth"))


def _formal_checkpoint_fixture(tmp_path: Path, *, epoch=4, run_class="formal_training", config_sha_override=None):
    training_config_path = tmp_path / "training.json"
    training_config = copy.deepcopy(load_all_configs()["training_v1.json"])
    training_config_path.write_text(json.dumps(training_config))
    checkpoint_path = tmp_path / "adapter.pt"
    import inference_runtime as runtime
    payload = {
        "schema": "stage19_adapter_checkpoint_v1", "epoch": epoch,
        "adapter_state_dict": {}, "adapter_parameter_count": 10,
        "base_checkpoint_sha256": training_config["base_model"]["checkpoint_sha256"],
        "training_config_sha256": config_sha_override or runtime.file_sha256(training_config_path),
    }
    torch.save(payload, checkpoint_path)
    manifest = {
        "run_status": "completed", "run_class": run_class, "smoke_subset": False,
        "optimizer_updates": 1024, "epochs": 4, "base_model_unchanged": True,
        "base_gradients_absent": True, "adapter_checkpoint": asset(checkpoint_path),
    }
    manifest_path = tmp_path / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, checkpoint_path, training_config_path, training_config


def test_formal_checkpoint_guard_accepts_exact_epoch4(tmp_path: Path):
    values = _formal_checkpoint_fixture(tmp_path)
    payload = run_inference.load_validated_adapter_checkpoint(*values, require_formal=True)
    assert payload["epoch"] == 4


def test_formal_checkpoint_guard_rejects_epoch1(tmp_path: Path):
    values = _formal_checkpoint_fixture(tmp_path, epoch=1)
    with pytest.raises(RuntimeError, match="STOP_INVALID_FORMAL_ADAPTER_CHECKPOINT"):
        run_inference.load_validated_adapter_checkpoint(*values, require_formal=True)


def test_formal_checkpoint_guard_rejects_training_config_sha(tmp_path: Path):
    values = _formal_checkpoint_fixture(tmp_path, config_sha_override="bad")
    with pytest.raises(RuntimeError, match="STOP_INVALID_FORMAL_ADAPTER_CHECKPOINT"):
        run_inference.load_validated_adapter_checkpoint(*values, require_formal=True)


def test_formal_checkpoint_guard_rejects_nonformal_manifest(tmp_path: Path):
    values = _formal_checkpoint_fixture(tmp_path, run_class="engineering_smoke")
    with pytest.raises(RuntimeError, match="STOP_INVALID_FORMAL_ADAPTER_CHECKPOINT"):
        run_inference.load_validated_adapter_checkpoint(*values, require_formal=True)
