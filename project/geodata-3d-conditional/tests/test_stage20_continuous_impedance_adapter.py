from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch

import inference_runtime as runtime
from guidance.residual_velocity_adapter import ResidualVelocityAdapter
from guidance.seismic import hard_labels_to_acoustic, tensor_sha256
from scripts.stage15.common import read_json, write_json
from scripts.stage20 import audit_reuse, build_continuous_evidence, build_observations, common, diagnose_acoustic_conditions, evaluate, formal_inference_provenance, preflight_acoustic_semantics, run_inference, train_adapter
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


FORMAL_ARMS = ("FLOW_ONLY", "ADAPTER_PRIOR_ONLY", "ADAPTER_CORRECT", "ADAPTER_WRONG_CASE")


def _fake_formal_inventory(tmp_path: Path):
    case_ids = [f"case{index:03d}" for index in range(1, 13)]
    seeds = [42, 142, 242]
    records = []
    for case_index, case_id in enumerate(case_ids):
        wrong_id = case_ids[(case_index + 1) % len(case_ids)]
        for seed in seeds:
            for arm_index, arm in enumerate(FORMAL_ARMS):
                output = tmp_path / case_id / str(seed) / arm / "decoded.pt"
                output.parent.mkdir(parents=True, exist_ok=True)
                tensor = torch.tensor([case_index, seed, arm_index], dtype=torch.int64)
                torch.save(tensor, output)
                source_id = wrong_id if arm == "ADAPTER_WRONG_CASE" else case_id
                records.append({
                    "case_id": case_id, "current_case_id": case_id, "source_seed": seed, "arm": arm,
                    "initial_noise_sha256": f"noise-{case_id}-{seed}", "base_checkpoint_sha256": "base",
                    "adapter_checkpoint_sha256": "adapter", "inference_config_sha256": "config",
                    "runner_source_sha256": "runner", "condition_asset_sha256": f"condition-{case_id}",
                    "evidence_asset_sha256": f"{'prior' if arm == 'ADAPTER_PRIOR_ONLY' else 'inverted'}-{source_id}",
                    "evidence_source_case_id": source_id, "mask_source_case_id": case_id,
                    "n_steps": 32, "adapter_scale": 0.0 if arm == "FLOW_ONLY" else 1.0,
                    "residual_cap": 0.25, "hard_condition_violation_count": 0,
                    "output_tensor_sha256": tensor_sha256(tensor), "output_file_sha256": runtime.file_sha256(output),
                    "output_path": str(output.relative_to(tmp_path)),
                })
    validation = {
        "expected_base_checkpoint_sha256": "base", "expected_adapter_checkpoint_sha256": "adapter",
        "expected_inference_config_sha256": "config", "expected_runner_source_sha256": "runner",
        "n_steps": 32, "residual_cap": 0.25,
    }
    return case_ids, seeds, records, validation


def test_formal_output_inventory_exact_144_and_deterministic(tmp_path: Path):
    case_ids, seeds, records, validation = _fake_formal_inventory(tmp_path)
    index_asset = formal_inference_provenance.freeze_formal_output_index(
        list(reversed(records)), run_dir=tmp_path, case_ids=case_ids, seeds=seeds,
        arms=FORMAL_ARMS, validation=validation,
    )
    index = read_json(Path(index_asset["path"]))
    assert index["formal_output_count"] == 144
    assert [(row["case_id"], row["source_seed"], row["arm"]) for row in index["records"]] == [
        (case_id, seed, arm) for case_id in case_ids for seed in seeds for arm in FORMAL_ARMS
    ]
    first = index["records"][0]
    assert first["adapter_scale"] == 0.0 and first["n_steps"] == 32 and first["residual_cap"] == 0.25
    assert first["output_file_sha256"] == runtime.file_sha256(tmp_path / first["output_path"])
    assert first["output_tensor_sha256"] == tensor_sha256(runtime.load_tensor(tmp_path / first["output_path"]))


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_formal_output_inventory_rejects_missing_and_duplicate(tmp_path: Path, mutation: str):
    case_ids, seeds, records, validation = _fake_formal_inventory(tmp_path)
    bad = records[:-1] if mutation == "missing" else records + [dict(records[0])]
    with pytest.raises(RuntimeError, match=mutation):
        formal_inference_provenance.validate_formal_output_records(
            bad, run_dir=tmp_path, case_ids=case_ids, seeds=seeds, arms=FORMAL_ARMS, **validation
        )


def test_formal_output_inventory_rejects_altered_file_before_truth(tmp_path: Path):
    case_ids, seeds, records, validation = _fake_formal_inventory(tmp_path)
    torch.save(torch.tensor([999]), tmp_path / records[0]["output_path"])
    with pytest.raises(RuntimeError, match="file SHA256 mismatch"):
        formal_inference_provenance.validate_formal_output_records(
            records, run_dir=tmp_path, case_ids=case_ids, seeds=seeds, arms=FORMAL_ARMS, **validation
        )


def test_hard_condition_violation_prevents_formal_freeze(tmp_path: Path):
    case_ids, seeds, records, validation = _fake_formal_inventory(tmp_path)
    records[0]["hard_condition_violation_count"] = 1
    with pytest.raises(RuntimeError, match="STOP_STAGE20_HARD_CONDITION_VIOLATION"):
        formal_inference_provenance.freeze_formal_output_index(
            records, run_dir=tmp_path, case_ids=case_ids, seeds=seeds,
            arms=FORMAL_ARMS, validation=validation,
        )


def test_runner_marks_complete_only_after_master_index_freeze():
    source = inspect.getsource(run_inference.main)
    assert source.index("index_asset = freeze_formal_output_index(") < source.index('"formal_inference_complete": not args.smoke')


def _fake_frozen_formal_run(tmp_path: Path):
    run_dir = tmp_path / "run"; run_dir.mkdir()
    case_ids, seeds, records, _ = _fake_formal_inventory(run_dir)
    asset_paths = {}
    for name in ("base", "adapter", "config", "runner"):
        path = tmp_path / name; path.write_text(name)
        asset_paths[name] = path
    hashes = {name: runtime.file_sha256(path) for name, path in asset_paths.items()}
    for record in records:
        record["base_checkpoint_sha256"] = hashes["base"]
        record["adapter_checkpoint_sha256"] = hashes["adapter"]
        record["inference_config_sha256"] = hashes["config"]
        record["runner_source_sha256"] = hashes["runner"]
    validation = {
        "expected_base_checkpoint_sha256": hashes["base"],
        "expected_adapter_checkpoint_sha256": hashes["adapter"],
        "expected_inference_config_sha256": hashes["config"],
        "expected_runner_source_sha256": hashes["runner"], "n_steps": 32, "residual_cap": 0.25,
    }
    index_asset = formal_inference_provenance.freeze_formal_output_index(
        records, run_dir=run_dir, case_ids=case_ids, seeds=seeds, arms=FORMAL_ARMS, validation=validation
    )
    registry = {"schema": "stage20_test_inference_registry_v1", "run_status": "completed", "case_count": 12, "cases": []}
    for case_id in case_ids:
        registry["cases"].append({
            "case_id": case_id, "condition_values": {"sha256": f"condition-{case_id}"},
            "normalized_low_frequency_logz": {"sha256": f"prior-{case_id}"},
            "normalized_inverted_logz": {"sha256": f"inverted-{case_id}"},
        })
    registry_path = tmp_path / "registry.json"; write_json(registry_path, registry)
    manifest = {
        "run_status": "completed", "formal_inference_complete": True, "formal_output_count": 144,
        "truth_loaded_by_runner": False, "formal_output_index": index_asset,
        "formal_output_index_sha256": index_asset["sha256"],
        "base_checkpoint": runtime.asset_record(asset_paths["base"]),
        "adapter_checkpoint": runtime.asset_record(asset_paths["adapter"]),
        "inference_config": runtime.asset_record(asset_paths["config"]),
        "runner_source": runtime.asset_record(asset_paths["runner"]),
    }
    write_json(run_dir / "run_manifest.json", manifest)
    return run_dir, registry_path, validation


def test_master_index_hash_mismatch_rejected_before_truth(tmp_path: Path):
    run_dir, registry_path, validation = _fake_frozen_formal_run(tmp_path)
    formal_inference_provenance.audit_frozen_formal_inference(
        run_dir=run_dir, test_registry_path=registry_path,
        expected_test_registry_sha256=runtime.file_sha256(registry_path), validation=validation,
    )
    index_path = run_dir / formal_inference_provenance.FORMAL_OUTPUT_INDEX_NAME
    index_path.write_text(index_path.read_text() + "\n")
    with pytest.raises(RuntimeError, match="STOP_STAGE20_RETROSPECTIVE_EVALUATION_FIREWALL"):
        formal_inference_provenance.audit_frozen_formal_inference(
            run_dir=run_dir, test_registry_path=registry_path,
            expected_test_registry_sha256=runtime.file_sha256(registry_path), validation=validation,
        )


def test_formal_wrong_case_sources_scales_and_mask_contract(tmp_path: Path):
    case_ids, seeds, records, validation = _fake_formal_inventory(tmp_path)
    ordered = formal_inference_provenance.validate_formal_output_records(
        records, run_dir=tmp_path, case_ids=case_ids, seeds=seeds, arms=FORMAL_ARMS, **validation
    )
    for row in ordered:
        assert row["mask_source_case_id"] == row["current_case_id"]
        assert row["adapter_scale"] == (0.0 if row["arm"] == "FLOW_ONLY" else 1.0)
        assert (row["evidence_source_case_id"] != row["case_id"]) == (row["arm"] == "ADAPTER_WRONG_CASE")


def test_evaluator_freeze_audit_precedes_any_truth_dereference():
    source = inspect.getsource(evaluate.main)
    assert source.index("audit_frozen_formal_inference(") < source.index('["truth_assets"]["truth"]')


def test_scientific_decision_table_is_total_and_gate_g_has_priority():
    keys = ("A_learned_continuous_evidence_coupling", "B_seismic_increment_beyond_boreholes", "C_case_specificity", "hard_physics", "volume", "global_geology_preservation", "hard_conditions")
    gates = {key: True for key in keys}
    assert evaluate.stage20_scientific_decision(gates) == "CONTINUOUS_IMPEDANCE_ADAPTER_VALIDATED"
    expected = [
        ("hard_conditions", "HARD_CONDITION_INTEGRITY_FAILED"),
        ("A_learned_continuous_evidence_coupling", "CONTINUOUS_EVIDENCE_COUPLING_NOT_VALIDATED"),
        ("B_seismic_increment_beyond_boreholes", "WELL_PRIOR_DOMINATED_NO_SEISMIC_INCREMENT"),
        ("C_case_specificity", "CONTINUOUS_EVIDENCE_NOT_CASE_SPECIFIC"),
        ("hard_physics", "GEOLOGY_IMPROVES_WITHOUT_HARD_SEISMIC_SUPPORT"),
        ("volume", "CONTINUOUS_IMPEDANCE_ADAPTER_NOT_VALIDATED"),
        ("global_geology_preservation", "CONTINUOUS_IMPEDANCE_ADAPTER_NOT_VALIDATED"),
    ]
    for key, decision in expected:
        current = dict(gates); current[key] = False
        assert evaluate.stage20_scientific_decision(current) == decision
    all_bad = {key: False for key in keys}
    assert evaluate.stage20_scientific_decision(all_bad) == "HARD_CONDITION_INTEGRITY_FAILED"
    source = inspect.getsource(evaluate.stage20_scientific_decision)
    assert "ENGINEERING_FAIL" not in source and "STAGE20_PARTIAL_OR_NEGATIVE" not in source
