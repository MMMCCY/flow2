"""CPU regressions for the frozen Stage17 experiment design."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
for root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from scripts.stage17.common import ap_skill, average_precision
from scripts.stage17.build_or_register_cases import _columnwise_contiguous_support
from scripts.stage17.run_multicase_trace_evidence import _validated_case_assets
from scripts.stage17.run_optimizer_calibration import _validate_config


CONFIG_ROOT = (
    PROJECT_DIR / "experiments/stage17_evidence_coupling_attribution/configs"
)


def _read(name: str) -> dict[str, object]:
    return json.loads((CONFIG_ROOT / name).read_text(encoding="utf-8"))


def test_ap_skill_is_prevalence_corrected_and_raw_ap_is_not_substituted() -> None:
    scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
    target = torch.tensor([True, False, True, False])
    ap = average_precision(scores, target)
    assert ap == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)
    assert ap_skill(ap, 0.5) == pytest.approx((ap - 0.5) / 0.5)
    assert ap_skill(ap, 0.5) != pytest.approx(ap)


def test_optimizer_confirmation_is_task_separated_and_smoke_cannot_tune() -> None:
    config = _read("optimizer_calibration_v1.json")
    assert _validate_config(config) == [42, 142, 242]
    assert config["smoke_may_tune_scientific_parameters"] is False
    gate = config["progression_gate"]
    assert gate["task_aggregation"] == "separate_no_pooled_compensation"
    assert set(config["tasks"]) == {"oracle_probability", "property"}


def test_case_registry_contains_all_five_independent_cases() -> None:
    registry = _read("case_registry_v1.json")
    cases = registry["cases"]
    assert len(cases) == 5
    assert len({case["root_seed"] for case in cases}) == 5
    assert registry["case_selection_uses_downstream_performance"] is False


def test_evidence_runner_asset_api_rejects_truth() -> None:
    required = {
        "observation_manifest",
        "observed_seismic",
        "subsurface_mask",
        "binary_fixed_values",
        "binary_fixed_mask",
        "condition_values",
        "condition_mask",
    }
    case = {
        "case_id": "test",
        "assets": {name: {"path": "unused", "sha256": "unused"} for name in required},
    }
    case["assets"]["truth_model"] = {"path": "forbidden", "sha256": "forbidden"}
    with pytest.raises(ValueError, match="invalid evidence asset API"):
        _validated_case_assets(case)


def test_observation_freeze_precedes_absent_evidence_freeze() -> None:
    generation = _read("observation_generation_v2.json")
    assert generation["status"] == "frozen_before_asset_generation"
    # The inversion config is intentionally generated only after observations
    # exist.  Once present, its explicit freeze-order declaration remains the
    # regression assertion.
    evidence_path = CONFIG_ROOT / "evidence_multicase_v1.json"
    if evidence_path.exists():
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        assert evidence["freeze_order"].startswith("observation_config_then_generated_hashed")


def test_columnwise_support_fills_only_cells_below_highest_nonair() -> None:
    truth = torch.tensor([-1, 1, -1, 2, -1]).view(1, 1, 1, 1, 5)
    support = _columnwise_contiguous_support(truth)
    assert support.flatten().tolist() == [True, True, True, True, False]


def test_coupling_config_uses_case_primary_units_and_forbids_dflow() -> None:
    config = _read("coupling_v1.json")
    assert config["status"] == "frozen_before_run"
    assert config["primary_statistical_unit"] == "independent_geology_case"
    assert config["source_seed_role"] == "within_case_stochastic_replicate"
    assert config["pooled_15_pair_role"] == "auxiliary_only"
    assert config["authorized_arms"] == ["FLOW_ONLY", "TRAJECTORY_EVIDENCE"]
    assert config["dflow_arm_authorized"] is False
    assert len(config["cases"]) == 5
    assert all("evidence" in case for case in config["cases"])
