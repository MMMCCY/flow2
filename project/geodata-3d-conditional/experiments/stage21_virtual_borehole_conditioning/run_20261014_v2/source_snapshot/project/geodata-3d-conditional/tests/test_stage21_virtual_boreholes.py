from __future__ import annotations

import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.stage21 import common as c
from scripts.stage21.evaluate import aggregate, common_domain, connected, exposure_table, match_objects, voxel_metrics
from scripts.stage21.prepare import protocol
from scripts.stage21.run import initial_noise, sample_native, independent_n9_reference, verify_completed
import inference_runtime as runtime


def registry_fixture():
    layout = c.plan_layout()
    p = protocol()
    record = lambda n: {"schema": "categorical_borehole_conditions_v1",
                        "origin": "synthetic_truth_sampled_at_frozen_wells_and_original_surface",
                        "oracle": True, "sequence_uncertainty": False,
                        "well_xy": layout["levels"][str(n)],
                        "tensors": {k: {"path": f"conditions/{k}.pt", "file_sha256": "f", "tensor_sha256": "t"}
                                    for k in ("condition_values", "condition_mask", "surface_mask", "borehole_mask")}}
    return {"schema": "stage21_inference_registry_v1", "protocol": p, "protocol_sha256": c.json_hash(p),
            "layout": layout, "layout_sha256": c.json_hash(layout),
            "conditions": {case: {str(n): record(n) for n in c.LEVELS} for case in c.CASES},
            "source_sha256": {}}


def test_layout_nested_replay_and_coverage():
    layout = c.plan_layout()
    assert layout == c.plan_layout()
    sets = [set(map(tuple, layout["levels"][str(n)])) for n in c.LEVELS]
    assert sets[0] < sets[1] < sets[2]
    assert list(map(len, sets)) == [9, 16, 25]
    extra = layout["extra_wells_ordered"]
    assert len({(x//16, y//16) for x, y in extra}) == 16
    assert not (set(map(tuple, extra)) & set(c.W9))
    # The planner's API has no truth/condition argument or file I/O.
    assert list(inspect.signature(c.plan_layout).parameters) == ["seed"]
    assert "load" not in inspect.getsource(c.plan_layout)


@pytest.mark.parametrize("points", [[(0, 0), (0, 0)], [(-1, 2)], [(64, 2)], [(2, 64)], [(2.5, 3)]])
def test_invalid_wells_rejected(points):
    with pytest.raises(ValueError):
        c.well_mask(points)


def test_conditions_mask_firewall_and_observed_air():
    truth = torch.arange(64).reshape(1,1,4,4,4).float() % 15 - 1
    surface = torch.zeros_like(truth, dtype=torch.bool)
    surface[..., -1] = True
    assets = c.condition_assets(truth, surface, [(0, 0)])
    mask = assets["condition_mask"]
    assert torch.equal(assets["condition_values"][mask], truth[mask])
    assert (assets["condition_values"][~mask] == -1).all()
    assert mask[0,0,0,0,0] and truth[0,0,0,0,0] == -1
    changed = truth.clone()
    changed[~mask] = 13
    assert torch.equal(c.condition_assets(changed, surface, [(0,0)])["condition_values"], assets["condition_values"])
    assert c.plan_layout() == c.plan_layout()


def test_historical_n9_and_surface_unchanged():
    truth = torch.zeros(1,1,64,64,64)
    surface = torch.zeros_like(truth, dtype=torch.bool)
    surface[..., 63] = True
    layout = c.plan_layout()
    assets = {n: c.condition_assets(truth, surface, layout["levels"][str(n)]) for n in c.LEVELS}
    for n in c.LEVELS:
        assert torch.equal(assets[n]["surface_mask"], surface)
        old = assets[9]["condition_mask"]
        assert torch.equal(assets[n]["condition_values"][old], assets[9]["condition_values"][old])
        assert int(assets[n]["borehole_mask"].sum()) == n*64


def test_common_domain_and_denominators_na():
    truth = np.array([-1, 1, 9, 9])
    m25 = np.array([True, False, True, False])
    domain = common_domain(truth, m25)
    assert domain.tolist() == [False, True, False, True]
    a = voxel_metrics(np.array([-1,1,9,1]), truth, domain, [1,9])
    assert a["domain_voxels"] == 2 and a["target_truth_volume"] == 1
    assert a["target_recall"] == 0 and a["target_precision"] is None
    empty = voxel_metrics(truth, truth, np.zeros(4, dtype=bool), [])
    assert empty["target_iou"] is None and empty["fixed_class_miou"] is None
    assert empty["target_na_reason"] == "empty_common_domain"
    no_target = voxel_metrics(np.array([9]), np.array([1]), np.array([True]), [1])
    assert no_target["target_iou"] is None and no_target["target_fp"] == 1


def test_merger_cannot_claim_two_matches_and_precision_penalizes_fp():
    truth = np.zeros((9,3,3), dtype=bool)
    truth[0:2,1,1] = True
    truth[7:9,1,1] = True
    pred = np.zeros_like(truth)
    pred[:,1,1] = True
    t, _ = connected(truth)
    p, _ = connected(pred)
    rules = {**protocol()["objects"], "edge_min_intersection": 1}
    rows, preds, stats, matrices = match_objects(t,p,np.ones_like(truth),rules)
    assert sum(r["matched_prediction_id"] is not None for r in rows) == 1
    matched = next(r for r in rows if r["matched_prediction_id"])
    assert matched["matched_precision"] == pytest.approx(2/9)
    assert matched["matched_iou"] == pytest.approx(2/9)
    assert stats["merge_predicted_object_count"] == 1
    assert matrices["intersection_common"].shape == (2,1)
    assert preds[0]["false_positive_common_voxels"] == 5


def test_well_removal_does_not_relabel_predictions():
    truth = np.zeros((9,3,3), dtype=bool)
    truth[:,1,1] = True
    ids,_ = connected(truth)
    domain = np.ones_like(truth)
    domain[4,:,:] = False
    rows,_,stats,_ = match_objects(ids,ids,domain,protocol()["objects"])
    assert stats["predicted_objects_full"] == 1
    assert rows[0]["matched_iou"] == 1
    assert rows[0]["evaluable_voxels"] == 8


def test_original_exposure_and_new_hit_groups_remain_distinct():
    ids = np.zeros((8,8,8),dtype=int)
    ids[0,0,0] = 1
    ids[0,6,7] = 2
    ids[3,3,1] = ids[3,4,1] = 3
    ids[7,7,1] = 4
    truth = torch.from_numpy(np.where(ids > 0,9,1)).reshape(1,1,8,8,8)
    surface = torch.zeros_like(truth,dtype=torch.bool)
    surface[...,7] = True
    layout = {"levels":{"9":[[0,0]],"16":[[0,0],[3,3]],"25":[[0,0],[3,3],[4,4]]}}
    conditions = {n:c.condition_assets(truth,surface,layout["levels"][str(n)]) for n in (9,16,25)}
    domain = common_domain(truth.numpy()[0,0],conditions[25]["condition_mask"].numpy()[0,0])
    rows = exposure_table("fixture",ids,conditions,domain,layout)
    assert [r["exposure_group"] for r in rows] == ["well_hit","well_unhit_surface_visible","strict_unobserved","strict_unobserved"]
    assert rows[2]["n16_status"] == "newly_hit" and rows[2]["common_voxels"] == 1
    assert rows[3]["n25_status"] == "still_unhit"
    assert rows[1]["original_joint_hit_voxels"] == 1


def test_registry_refuses_truth_and_nested_metadata():
    r = registry_fixture()
    c.validate_registry(r)
    r["truth_path"] = "/hidden/truth.pt"
    with pytest.raises(ValueError, match="registry keys"):
        c.validate_registry(r)
    r = registry_fixture()
    r["conditions"][c.CASES[0]]["9"]["hidden_roi"] = []
    with pytest.raises(ValueError, match="condition keys"):
        c.validate_registry(r)
    r = registry_fixture()
    r["conditions"][c.CASES[0]]["9"]["tensors"]["condition_values"]["path"] = "truth/model.pt"
    with pytest.raises(ValueError, match="outside prepared"):
        c.validate_registry(r)


def test_content_hash_and_asset_tampering(tmp_path):
    t = torch.arange(6).reshape(2,3)
    assert c.tensor_hash(t) == c.tensor_hash(t.clone())
    path = tmp_path / "v.pt"
    torch.save(t, path)
    record = {"path": str(path), "file_sha256": c.file_hash(path), "tensor_sha256": c.tensor_hash(t)}
    assert torch.equal(c.verify_tensor(record), t)
    torch.save(t+1, path)
    with pytest.raises(ValueError, match="file hash"):
        c.verify_tensor(record)


class Dummy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_dim = 15
        self.embedding = torch.nn.Embedding.from_pretrained(torch.eye(15), freeze=True)
        class Net(torch.nn.Module):
            def forward(self, x, condition, t):
                return 0.2*x + 0.1*condition + t[:,None,None,None,None]
        self.net = Net()
    def embed(self, raw):
        return self.embedding(raw[:,0].long()+1).permute(0,4,1,2,3).contiguous()
    def decode(self, state):
        return torch.einsum("bexyz,ce->bcxyz", state, self.embedding.weight).argmax(1)


def test_cpu_native_sampler_exact_conditions_noise_and_independent_baseline():
    model = Dummy()
    values = torch.arange(15).reshape(1,1,1,1,15)-1
    mask = torch.ones_like(values, dtype=torch.bool)
    mask[..., 3:6] = False
    before = model.embedding.weight.detach().clone()
    noise = initial_noise(42, 15, [1,1,15])
    repeat = initial_noise(42, 15, [1,1,15])
    assert torch.equal(noise, repeat)
    assert c.tensor_hash(noise) == c.tensor_hash(repeat)
    pred,state,trace = sample_native(model,noise,values,mask,4)
    reference = independent_n9_reference(model,noise,values,mask,4)
    assert torch.equal(state,reference)
    assert torch.equal(pred[mask],values[mask])
    assert torch.equal(before,model.embedding.weight)
    assert [r["time"] for r in trace] == [.125,.375,.625,.875]
    assert all(r["condition_embedding_max_error"] == 0 for r in trace)
    assert torch.equal(noise,repeat)  # sampler did not mutate the full unprojected source


def test_nonfinite_velocity_stops():
    model = Dummy()
    model.net.register_forward_hook(lambda net, args, output: output*float("nan"))
    v = torch.zeros(1,1,1,1,2).long()
    with pytest.raises(FloatingPointError):
        sample_native(model,initial_noise(42,15,[1,1,2]),v,torch.ones_like(v,dtype=torch.bool),2)


def test_runtime_truth_open_firewall(tmp_path):
    path = tmp_path / "truth" / "true_model.pt"
    path.parent.mkdir()
    path.write_bytes(b"test")
    code = (f"import sys;sys.path.insert(0,{str(PROJECT)!r});"
            "from scripts.stage21.run import install_firewall;install_firewall();"
            f"open({str(path)!r},'rb')")
    result = subprocess.run([sys.executable,"-c",code],capture_output=True,text=True)
    assert result.returncode != 0 and "inference truth firewall" in result.stderr


def test_firewall_allows_geometry_library_source(tmp_path):
    path = tmp_path / "site-packages" / "sympy" / "geometry" / "__init__.py"
    path.parent.mkdir(parents=True)
    path.write_text("# library fixture")
    code = (f"import sys;sys.path.insert(0,{str(PROJECT)!r});"
            "from scripts.stage21.run import install_firewall;install_firewall();"
            f"assert open({str(path)!r}).read() == '# library fixture'")
    result = subprocess.run([sys.executable,"-c",code],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr


def test_case_first_medians_and_two_paired_definitions():
    samples = []
    for case, offset in [("a",0), ("b",100)]:
        for seed,x,y in [(42,0,0),(142,10,11),(242,100,12)]:
            for n,value in [(9,x),(16,y),(25,y+1)]:
                samples.append({"case_id":case,"source_seed":seed,"well_count":n,"target_iou":value+offset})
    case_rows, paired, cp, summary = aggregate(samples)
    assert summary["n_cases"] == 2 and summary["n_samples"] == 18
    assert summary["levels"]["9"]["target_iou"]["case_median"] == 60
    a = next(r for r in cp if r["case_id"]=="a" and r["comparison"]=="16-9")
    assert a["target_iou_median_paired_difference"] == 0
    assert a["target_iou_difference_of_seed_medians"] == 1
    assert summary["levels"]["9"]["strict_unobserved_major_matched_iou"]["valid_cases"] == 0


def test_ema_loader_retains_raw_frozen_embedding_and_applies_trainable_shadow(tmp_path):
    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding.from_pretrained(torch.eye(3), freeze=True)
            self.net = torch.nn.Linear(3, 3, bias=False)
        @classmethod
        def load_from_checkpoint(cls, path, map_location):
            return cls()
    path = tmp_path / "fake_checkpoint.ckpt"
    torch.save({"ema_shadow": {"net.weight": torch.full((3,3), 7.0)}, "epoch":1651, "global_step":8376}, path)
    model, report = runtime.load_model_with_weight_policy(Fake,path,"cpu","ema")
    assert torch.equal(model.embedding.weight, torch.eye(3))
    assert torch.equal(model.net.weight, torch.full((3,3),7.0))
    assert report["ema_excluded_frozen_parameters"] == ["embedding.weight"]
    assert report["ema_missing_trainable"] == [] and not model.training


def test_resume_record_verification_rejects_changed_prediction(tmp_path):
    pred_path = tmp_path / "decoded.pt"
    trace_path = tmp_path / "trace.json"
    pred = torch.zeros(1,1,2,2,2).long()
    torch.save(pred,pred_path)
    trace_path.write_text("[]")
    r = {"prediction_path":"decoded.pt", "prediction_file_sha256":c.file_hash(pred_path),
         "prediction_tensor_sha256":c.tensor_hash(pred), "trace_path":"trace.json",
         "trace_file_sha256":c.file_hash(trace_path)}
    assert torch.equal(verify_completed(r,tmp_path),pred)
    torch.save(pred+1,pred_path)
    with pytest.raises(ValueError, match="resume prediction file hash"):
        verify_completed(r,tmp_path)
