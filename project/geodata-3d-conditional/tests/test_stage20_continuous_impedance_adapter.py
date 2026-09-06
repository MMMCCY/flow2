from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch

import inference_runtime as runtime
from guidance.residual_velocity_adapter import ResidualVelocityAdapter
from guidance.seismic import hard_labels_to_acoustic
from scripts.stage15.common import read_json
from scripts.stage20 import audit_reuse, build_continuous_evidence, build_observations, common, diagnose_acoustic_conditions, evaluate, preflight_acoustic_semantics, run_inference, train_adapter
from scripts.stage20.acoustic_semantics import stage20_labels_to_acoustic


PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "experiments/stage20_continuous_impedance_adapter/configs"


def table() -> torch.Tensor:
    # air, raw label 0, raw label 1
    return torch.tensor([[1.0, 10.0, 40.0], [0.01, 0.004, 0.002]])


def prior_fixture():
    values = torch.full((1, 1, 3, 3, 2), -1)
    known = torch.zeros_like(values, dtype=torch.bool)
    support = torch.ones_like(known)
    values[0, 0, 0, 0, :] = 0; known[0, 0, 0, 0, :] = True
    values[0, 0, 2, 0, :] = 1; known[0, 0, 2, 0, :] = True
    return values, known, support, ((0, 0), (2, 0))


def test_idw_exact_wells_and_p2_midpoint():
    values, known, support, wells = prior_fixture()
    result, diagnostics = common.build_low_frequency_acoustic(values, known, support, wells, table())
    assert result[0, 0, 0, 0, 0] == 10
    assert result[0, 0, 2, 0, 0] == 40
    assert result[0, 0, 1, 0, 0] == pytest.approx(20.0)
    assert result[0, 1, 1, 0, 0] == pytest.approx(0.003)


def test_idw_is_lateral_only_and_maps_properties_before_interpolation():
    values, known, support, wells = prior_fixture()
    values[0, 0, 0, 0, 1] = 1; values[0, 0, 2, 0, 1] = 1
    result, diagnostics = common.build_low_frequency_acoustic(values, known, support, wells, table())
    assert result[0, 0, 1, 0, 0] == pytest.approx(20.0)
    assert result[0, 0, 1, 0, 1] == pytest.approx(40.0)
    assert result[0, 0, 1, 0, 0] != pytest.approx(25.0)  # no numeric class/property arithmetic


def test_invalid_well_is_excluded_and_empty_depth_uses_nearest_valid_depth():
    values, known, support, wells = prior_fixture()
    support[0, 0, 2, 0, 0] = False
    result, diagnostics = common.build_low_frequency_acoustic(values, known, support, wells, table())
    assert result[0, 0, 1, 0, 0] == 10
    known[..., 1] = False
    result, diagnostics = common.build_low_frequency_acoustic(values, known, support, wells, table())
    assert diagnostics["fallback_map"] == [{"z": 1, "z_ref": 0, "distance": 1}]


def test_fallback_tie_chooses_deeper_smaller_z_and_copies_constructed_slice():
    values = torch.full((1, 1, 1, 1, 5), -1)
    known = torch.zeros_like(values, dtype=torch.bool); support = torch.ones_like(known)
    values[..., 1] = 0; known[..., 1] = True
    values[..., 3] = 1; known[..., 3] = True
    result, diagnostics = common.build_low_frequency_acoustic(values, known, support, ((0, 0),), table())
    mapping = {row["z"]: row for row in diagnostics["fallback_map"]}
    assert mapping[2] == {"z": 2, "z_ref": 1, "distance": 1}
    assert torch.equal(result[..., 2], result[..., 1])


def test_whole_case_without_any_valid_well_depth_stops():
    values = torch.full((1, 1, 1, 1, 2), -1)
    known = torch.zeros_like(values, dtype=torch.bool); support = torch.ones_like(known)
    with pytest.raises(RuntimeError, match="case has no valid well depth"):
        common.build_low_frequency_acoustic(values, known, support, ((0, 0),), table())


def test_air_and_exact_condition_overwrite():
    values, known, support, wells = prior_fixture()
    support[0, 0, 1, 1, 0] = False
    values[0, 0, 1, 1, 0] = -1; known[0, 0, 1, 1, 0] = True
    result, diagnostics = common.build_low_frequency_acoustic(values, known, support, wells, table())
    assert torch.equal(result[0, :, 1, 1, 0], table()[:, 0])
    exact = hard_labels_to_acoustic(values, table())
    assert torch.equal(result[known.expand_as(result)], exact[known.expand_as(exact)])


def test_shared_enclosed_air_cleanup_is_acoustic_only_and_dynamic():
    labels = torch.tensor([[[[[-1, -1]]]]])
    support = torch.tensor([[[[[True, False]]]]])
    original = labels.clone()
    acoustic, diagnostics = stage20_labels_to_acoustic(labels, support, table())
    assert torch.equal(labels, original)
    assert diagnostics == {"underground_air_voxels_replaced": 1, "neutral_category": 1, "neutral_raw_label": 0}
    assert torch.equal(acoustic[0, :, 0, 0, 0], table()[:, 1])
    assert torch.equal(acoustic[0, :, 0, 0, 1], table()[:, 0])


def test_current_phase4c_codebook_resolves_neutral_raw_label3_without_hardcoding():
    property_table, _ = common.load_codebook(PROJECT / "experiments/stage4_seismic/configs/acoustic_distinct_label9_upper_bound_v1.json")
    labels = torch.full((1, 1, 1, 1, 1), -1); support = torch.ones_like(labels, dtype=torch.bool)
    _, diagnostics = stage20_labels_to_acoustic(labels, support, property_table)
    assert diagnostics["neutral_raw_label"] == 3


def test_cleanup_helper_is_shared_by_four_required_consumers():
    consumers = (
        build_observations,
        build_continuous_evidence,
        __import__("scripts.stage20.audit_evidence", fromlist=["x"]),
        evaluate.multiclass_seismic_metrics,
    )
    assert all("stage20_labels_to_acoustic(" in inspect.getsource(consumer) for consumer in consumers)


def test_all_rock_labels_preserve_frozen_phase4c_mapping():
    property_table, _ = common.load_codebook(PROJECT / "experiments/stage4_seismic/configs/acoustic_distinct_label9_upper_bound_v1.json")
    labels = torch.arange(14).reshape(1, 1, 14, 1, 1)
    support = torch.ones_like(labels, dtype=torch.bool)
    mapped, _ = stage20_labels_to_acoustic(labels, support, property_table)
    assert torch.equal(mapped, hard_labels_to_acoustic(labels, property_table))


def test_preflight_rejects_subsurface_phase4c_air_impedance():
    support = torch.ones((1, 1, 1, 1, 1), dtype=torch.bool)
    acoustic = table()[:, 0].reshape(1, 2, 1, 1, 1)
    with pytest.raises(RuntimeError, match="outside frozen rock bounds"):
        preflight_acoustic_semantics.validate_acoustic_preflight_tensor(acoustic, support, table())


def test_val_condition_air_diagnostic_counts_match_frozen_15_58():
    registry = read_json(PROJECT / "experiments/stage20_continuous_impedance_adapter/inversion_input_registry_fix2/registry.json")
    expected = {"stage19_val_case001": 15, "stage19_val_case005": 58}
    observed = {}
    for case in registry["cases"]:
        if case["split"] != "val":
            continue
        assets = case["assets"]
        values = runtime.load_tensor(common.validate_asset(assets["condition_values"], "values"))
        condition = runtime.load_tensor(common.validate_asset(assets["condition_mask"], "condition"))
        support = runtime.load_tensor(common.validate_asset(assets["subsurface_mask"], "support"))
        count = int(diagnose_acoustic_conditions.condition_subsurface_raw_air_mask(values, condition, support).sum())
        observed[case["case_id"]] = count
    assert {case: count for case, count in observed.items() if count} == expected


def test_mapper_matches_observations_fix2_cleanup_semantics():
    case_dir = PROJECT / "experiments/stage20_continuous_impedance_adapter/observations_fix2/stage19_train_case004"
    manifest = read_json(case_dir / "manifest.json")
    property_table, _ = common.load_codebook(PROJECT / "experiments/stage4_seismic/configs/acoustic_distinct_label9_upper_bound_v1.json")
    labels = runtime.load_tensor(common.validate_asset(manifest["truth_assets"]["truth"], "truth"))
    support = runtime.load_tensor(common.validate_asset(manifest["assets"]["subsurface_mask"], "support"))
    _, current = stage20_labels_to_acoustic(labels, support, property_table)
    previous = manifest["enclosed_subsurface_air_cleanup"]
    assert current["underground_air_voxels_replaced"] == previous["enclosed_subsurface_air_voxels_replaced"]
    assert current["neutral_category"] == previous["neutral_category"]
    assert current["neutral_raw_label"] == previous["neutral_raw_label"]


def test_preflight_and_inversion_registry_are_truth_blind():
    source = inspect.getsource(preflight_acoustic_semantics)
    assert "true_model" not in source and "truth_assets" not in source
    registry = read_json(PROJECT / "experiments/stage20_continuous_impedance_adapter/inversion_input_registry_fix2/registry.json")
    common.assert_truth_firewall(registry, "focused preflight test")


def test_normalization_is_global_and_stable():
    low, high = table()[0, 1:].log().min(), table()[0, 1:].log().max()
    values = torch.tensor([[[[[low, high, (low + high) / 2]]]]])
    first, lmin, lmax = common.normalize_logz(values, table())
    second, _, _ = common.normalize_logz(values.clone(), table())
    assert first.flatten().tolist() == pytest.approx([-1, 1, 0])
    assert torch.equal(first, second)
    assert (lmin, lmax) == pytest.approx((float(low), float(high)))


def test_runtime_mask_zeros_air_and_conditions():
    q = torch.ones((1, 1, 2, 2, 1)); support = torch.ones_like(q, dtype=torch.bool); condition = torch.zeros_like(support)
    support[0, 0, 0, 0] = False; condition[0, 0, 1, 1] = True
    masked = common.runtime_evidence(q, support, condition)
    assert masked.sum() == 2 and masked[0, 0, 0, 0] == 0 and masked[0, 0, 1, 1] == 0


def test_wrong_case_values_use_current_case_mask():
    wrong = torch.arange(4.0).reshape(1, 1, 2, 2, 1)
    support = torch.ones_like(wrong, dtype=torch.bool); condition = torch.zeros_like(support); condition[0, 0, 0, 1] = True
    actual = common.wrong_case_evidence(wrong, support, condition)
    assert actual[0, 0, 0, 1] == 0 and actual[0, 0, 1, 1] == 3


def test_frozen_configs_and_arms():
    inference = read_json(CONFIG / "inference_v1.json")
    assert tuple(inference["arms"]) == run_inference.ARMS
    assert run_inference.ARMS == ("FLOW_ONLY", "ADAPTER_PRIOR_ONLY", "ADAPTER_CORRECT", "ADAPTER_WRONG_CASE")
    assert inference["source_seeds"] == [42, 142, 242] and inference["expected_outputs"] == 144
    assert "ADAPTER_ZERO" not in inference["arms"]


def test_adapter_parameter_count_exact():
    adapter = ResidualVelocityAdapter(15, geophysics_channels=1, base_width=12, dilations=(1, 2, 4, 1))
    assert adapter.parameter_count() == 54003


def test_truth_firewall_payloads():
    good = {"case_id": "case1", "normalized_inverted_logz": {"path": "q.pt", "sha256": "x"}}
    common.assert_truth_firewall(good, "test")
    with pytest.raises(RuntimeError):
        common.assert_truth_firewall({"truth": "secret.pt"}, "test")


def test_builder_and_runner_do_not_access_truth_assets():
    evidence_source = inspect.getsource(build_continuous_evidence)
    inference_source = inspect.getsource(run_inference)
    assert 'case["truth_assets"]' not in evidence_source
    assert 'case["truth_assets"]' not in inference_source
    assert "true_model.pt" not in evidence_source + inference_source


def test_observation_builder_creates_both_registry_contracts():
    source = inspect.getsource(build_observations)
    assert "stage20_observation_registry_full_v1" in source
    assert "stage20_inversion_input_registry_v1" in source
    assert "assert_truth_firewall(stripped" in source


def test_reuse_sources_are_not_reimplemented():
    source = inspect.getsource(build_continuous_evidence)
    assert "invert_acoustic_member(" in source
    assert "posterior_statistics" not in source
    assert "0.001" in source and "0.01" in source


def test_reuse_audit_rejects_base_checkpoint_hash(monkeypatch):
    original = audit_reuse.runtime.file_sha256
    monkeypatch.setattr(audit_reuse.runtime, "file_sha256", lambda path: "bad" if str(path).endswith("conditional-weights.ckpt") else original(path))
    with pytest.raises(RuntimeError, match="base checkpoint hash"):
        audit_reuse.run_audit(CONFIG / "cohort_v1.json", CONFIG / "evidence_v1.json", CONFIG / "training_v1.json")


def test_training_uses_free_mask_and_exact_formal_update_guard():
    source = inspect.getsource(train_adapter)
    assert "support & ~mask" in source
    assert "expected_updates = 2 if args.smoke else 1024" in source
    assert "epoch4_final" in source
    assert "optimizer must contain only adapter parameters" in source
    assert "frozen base received gradients" in source
    assert "git_status_at_start" in source
    assert "base_state_dict_hash_before" in source
    assert "training_gradients_finite" in source
    assert "checkpoint_save_load_validated" in source


def test_inference_has_independent_scale_zero_regression_and_conditions():
    source = inspect.getsource(run_inference)
    assert "projected_fixed_euler_prior_sample" in source
    assert "adapter_scale=0.0" in source
    assert "hard condition violation" in source
    assert "expected = 4 if args.smoke else 144" in source


def _stage20_formal_checkpoint_fixture(tmp_path: Path, **manifest_overrides):
    training_config_path = CONFIG / "training_v1.json"
    training_config = read_json(training_config_path)
    evidence_registry = PROJECT / "experiments/stage20_continuous_impedance_adapter/evidence_fix2/evidence_registry.json"
    evidence_gate = PROJECT / "experiments/stage20_continuous_impedance_adapter/evidence_audit_fix2/summary.json"
    checkpoint_path = tmp_path / "adapter.pt"
    payload = {
        "schema": "stage20_adapter_checkpoint_v1", "run_class": "formal_training",
        "smoke_subset": False, "epoch": 4, "epochs": 4, "optimizer_updates": 1024,
        "checkpoint_selection": "epoch4_final", "adapter_state_dict": {},
        "adapter_parameter_count": 54003,
        "base_checkpoint_sha256": training_config["base_model"]["checkpoint_sha256"],
        "training_config_sha256": runtime.file_sha256(training_config_path),
        "evidence_registry_sha256": runtime.file_sha256(evidence_registry),
        "evidence_gate_sha256": runtime.file_sha256(evidence_gate), "git_head": "frozen",
    }
    torch.save(payload, checkpoint_path)
    manifest = {
        "run_status": "completed", "run_class": "formal_training", "smoke_subset": False,
        "optimizer_updates": 1024, "epochs": 4, "base_model_unchanged": True,
        "base_gradients_absent": True, "base_finetuning": False, "ema_applied": True,
        "checkpoint_epoch": 4, "checkpoint_selection": "epoch4_final",
        "adapter_parameter_count": 54003, "optimizer_contains_adapter_only": True,
        "training_gradients_finite": True, "training_gradients_nonzero": True,
        "all_losses_finite": True, "residual_ratio_cap_passed": True,
        "checkpoint_save_load_validated": True, "test_truth_loaded": False,
        "base_model_hash_before": "same", "base_model_hash_after": "same",
        "base_state_dict_hash_before": "same-state", "base_state_dict_hash_after": "same-state",
        "adapter_checkpoint": common.asset(checkpoint_path),
        "evidence_registry": common.asset(evidence_registry), "evidence_gate": common.asset(evidence_gate),
        "training_config": common.asset(training_config_path),
    }
    manifest.update(manifest_overrides)
    manifest_path = tmp_path / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, checkpoint_path, training_config_path, training_config


def test_stage20_formal_checkpoint_guard_accepts_frozen_contract(tmp_path: Path):
    payload = run_inference.load_validated_adapter_checkpoint(
        *_stage20_formal_checkpoint_fixture(tmp_path), require_formal=True
    )
    assert payload["epoch"] == 4 and payload["optimizer_updates"] == 1024


def test_stage20_formal_checkpoint_guard_rejects_smoke_manifest(tmp_path: Path):
    values = _stage20_formal_checkpoint_fixture(tmp_path, run_class="engineering_smoke", smoke_subset=True)
    with pytest.raises(RuntimeError, match="STOP_INVALID_FORMAL_ADAPTER_CHECKPOINT"):
        run_inference.load_validated_adapter_checkpoint(*values, require_formal=True)


def test_stage20_formal_checkpoint_guard_rejects_base_hash_change(tmp_path: Path):
    values = _stage20_formal_checkpoint_fixture(tmp_path, base_model_hash_after="changed")
    with pytest.raises(RuntimeError, match="STOP_INVALID_FORMAL_ADAPTER_CHECKPOINT"):
        run_inference.load_validated_adapter_checkpoint(*values, require_formal=True)


def test_nearest_logz_tie_uses_first_label_and_conditions_override():
    t = table(); middle = ((t[0, 1].log() + t[0, 2].log()) / 2).reshape(1, 1, 1, 1, 1)
    support = torch.ones_like(middle, dtype=torch.bool); condition = torch.ones_like(support); values = torch.ones_like(middle, dtype=torch.long)
    assert evaluate.nearest_logz_labels(middle, t, support, torch.zeros_like(values), torch.zeros_like(condition)).item() == 0
    assert evaluate.nearest_logz_labels(middle, t, support, values, condition).item() == 1


def test_case_first_median():
    assert evaluate.med([1, 100, 2]) == 2
