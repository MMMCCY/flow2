"""Independent delivery audit of sealed outputs, raw counts, pairing and object assignments."""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from scripts.stage21.common import (
    REPO, file_hash, json_hash, load_conditions, load_tensor, read_json, source_hashes,
    tensor_hash, validate_registry, verify_tensor, write_json,
)


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def same(actual, expected):
    if expected is None:
        assert actual == "", (actual, expected)
    else:
        assert actual != "" and math.isclose(float(actual), float(expected), abs_tol=1e-12), (actual, expected)


def audit(args):
    registry = read_json(args.root / "inference_registry.json")
    p = validate_registry(registry)
    assert source_hashes() == registry["source_sha256"]
    assert file_hash(REPO / p["checkpoint"]) == p["checkpoint_sha256"]
    for mode in ("cuda_smoke", "formal"):
        folder = args.root / mode
        state = read_json(folder / "sampling_state.json")
        sealed = read_json(folder / "completion_receipt.json")
        assert state["status"] == "completed"
        assert state["base_model_unchanged"] and state["base_hash_before"] == state["base_hash_after"]
        assert sealed["state_sha256"] == file_hash(folder / "sampling_state.json")
        assert sealed["records_sha256"] == json_hash(state["records"])
        assert state["model_load_report"]["ema_entry_count"] == 411
        assert state["model_load_report"]["ema_excluded_frozen_parameters"] == ["embedding.weight"]
        assert len(state["records"]) == (3 if mode == "cuda_smoke" else 45)
        for r in state["records"]:
            pred = load_tensor(folder / r["prediction_path"])
            assert file_hash(folder / r["prediction_path"]) == r["prediction_file_sha256"]
            assert tensor_hash(pred) == r["prediction_tensor_sha256"]
            trace = read_json(folder / r["trace_path"])
            assert file_hash(folder / r["trace_path"]) == r["trace_file_sha256"]
            assert len(trace) == 32
            for i, step in enumerate(trace):
                assert step["step"] == i and step["time"] == (i+.5)/32
                assert step["state_finite"] and step["velocity_finite"] and step["condition_embedding_max_error"] == 0
            ts = load_conditions(registry["conditions"][r["case_id"]][str(r["well_count"])] )
            assert np.array_equal(pred[ts["condition_mask"]].numpy(), ts["condition_values"][ts["condition_mask"]].numpy())
            noise_path = folder / "noise" / f"seed{r['source_seed']}.pt"
            assert file_hash(noise_path) == r["initial_noise_file_sha256"]
            assert tensor_hash(load_tensor(noise_path)) == r["initial_noise_tensor_sha256"]
    smoke = read_json(args.root / "cuda_smoke/sampling_state.json")
    assert smoke["independent_n9_equivalence"]
    formal = read_json(args.root / "formal/sampling_state.json")
    for r in smoke["records"]:
        counterpart = next(x for x in formal["records"] if all(x[k] == r[k] for k in ("case_id", "source_seed", "well_count")))
        assert counterpart["prediction_tensor_sha256"] == r["prediction_tensor_sha256"]
    ereg = read_json(args.root / "evaluation_registry.json")
    sr = rows(args.report / "sample_metrics.csv")
    sample_index = {(r["case_id"],int(r["source_seed"]),int(r["well_count"])):r for r in sr}
    assert len(sample_index) == len(sr) == 45
    domain_records = {r["case_id"]:r for r in read_json(args.report / "common_domain_audit.json")}
    for case in p["cases"]:
        truth = verify_tensor(ereg["cases"][case]["truth"]).numpy()[0,0]
        ts = load_conditions(registry["conditions"][case]["25"])
        e = (truth != -1) & ~ts["condition_mask"].numpy()[0,0]
        assert np.array_equal(e, load_tensor(args.report / f"{case}_common_domain.pt").numpy())
        assert tensor_hash(load_tensor(args.report / f"{case}_common_domain.pt")) == domain_records[case]["common_hash"]
        for seed in p["source_seeds"]:
            records = [r for r in formal["records"] if r["case_id"] == case and r["source_seed"] == seed]
            assert len({r["initial_noise_tensor_sha256"] for r in records}) == 1
            for record in records:
                n = record["well_count"]
                pred = load_tensor(args.root / "formal" / record["prediction_path"]).numpy()[0,0]
                row = sample_index[(case,seed,n)]
                tp = np.count_nonzero((pred==9)&(truth==9)&e)
                pn,tn = np.count_nonzero((pred==9)&e),np.count_nonzero((truth==9)&e)
                same(row["target_tp"],tp); same(row["target_fp"],pn-tp);same(row["target_fn"],tn-tp)
                same(row["target_iou"],tp/(pn+tn-tp) if tn else None)
                same(row["target_precision"],tp/pn if pn and tn else None)
                same(row["target_recall"],tp/tn if tn else None)
                assert row["common_domain_hash"] == domain_records[case]["common_hash"]
                ious = []
                for raw_label in np.unique(truth[e]):
                    label = int(raw_label)
                    a,b = (pred==label)&e,(truth==label)&e
                    iou = np.count_nonzero(a&b)/np.count_nonzero(a|b)
                    same(row[f"class_{label}_iou"],iou)
                    ious.append(iou)
                same(row["fixed_class_miou"],np.mean(ious))
    objects = rows(args.report / "object_metrics.csv")
    for case in p["cases"]:
        for seed in p["source_seeds"]:
            for n in p["well_counts"]:
                group = [r for r in objects if r["case_id"]==case and int(r["source_seed"])==seed and int(r["well_count"])==n]
                matched = [r["matched_prediction_id"] for r in group if r["matched_prediction_id"]]
                assert len(set(matched)) == len(matched)
                with np.load(args.report / "overlap_matrices" / f"{case}_seed{seed}_n{n}.npz") as matrix:
                    for r in group:
                        i,j=int(r["object_id"])-1,int(r["matched_prediction_id"])-1 if r["matched_prediction_id"] else None
                        if j is not None:
                            inter,union=matrix["intersection_common"][i,j],matrix["union_common"][i,j]
                            same(r["matched_intersection"],inter);same(r["matched_union"],union)
                            same(r["matched_iou"],inter/union)
    artifacts = read_json(args.report / "artifact_hashes.json")
    for path, expected in artifacts.items():
        assert file_hash(REPO / path) == expected, path
    result = {"status":"passed", "formal_outputs":45,"smoke_outputs":3,
              "formal_velocity_evaluations":1440,"smoke_velocity_evaluations_including_independent_reference":128,
              "zero_condition_violations":True,"common_domain_counts_independently_recomputed":True,
              "model_ema_embedding_unchanged":True,"unique_one_to_one_object_assignments":True,
              "artifact_files_verified":len(artifacts),"source_sha256":file_hash(__file__)}
    write_json(args.report / "DELIVERY_AUDIT.json",result)
    print(result,flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    audit(parser.parse_args())
