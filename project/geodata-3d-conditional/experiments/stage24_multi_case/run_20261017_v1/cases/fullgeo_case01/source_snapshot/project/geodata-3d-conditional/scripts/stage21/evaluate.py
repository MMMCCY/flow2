"""Retrospective common-domain and fixed-identity object evaluation after sealed inference."""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from scripts.stage21.common import (
    REPO, file_hash, json_hash, load_conditions, load_tensor, read_json, tensor_hash,
    validate_registry, verify_tensor, well_mask, write_csv, write_json,
)


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def median(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return statistics.median(values) if values else None


def common_domain(truth, m25):
    return (truth != -1) & ~m25


def voxel_metrics(pred, truth, domain, classes):
    if pred.shape != truth.shape or domain.shape != truth.shape:
        raise ValueError("evaluation shape mismatch")
    p, t = (pred == 9) & domain, (truth == 9) & domain
    tp, pn, tn = int((p & t).sum()), int(p.sum()), int(t.sum())
    total = int(domain.sum())
    result = {"domain_voxels": total, "target_truth_volume": tn, "target_pred_volume": pn,
              "target_tp": tp, "target_fp": pn-tp, "target_fn": tn-tp,
              "target_iou": ratio(tp, pn+tn-tp) if tn else None,
              "target_precision": ratio(tp, pn) if tn else None,
              "target_recall": ratio(tp, tn), "target_signed_relative_volume_error": ratio(pn-tn, tn),
              "target_absolute_relative_volume_error": ratio(abs(pn-tn), tn),
              "voxel_accuracy": ratio(int(((pred == truth) & domain).sum()), total),
              "background_lithology_error": ratio(int(((pred != truth) & domain & (truth != 9)).sum()), total-tn),
              "background_false_target_rate": ratio(pn-tp, total-tn),
              "target_na_reason": "empty_common_domain" if total == 0 else ("no_truth_target" if tn == 0 else ("no_predicted_target_for_precision" if pn == 0 else ""))}
    ious = []
    for label in classes:
        a, b = (pred == label) & domain, (truth == label) & domain
        inter, union = int((a & b).sum()), int((a | b).sum())
        iou = ratio(inter, union)
        result[f"class_{label}_iou"] = iou
        result[f"class_{label}_truth_voxels"] = int(b.sum())
        if iou is not None:
            ious.append(iou)
    result["fixed_class_miou"] = float(np.mean(ious)) if ious else None
    return result


def connected(mask):
    return ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=bool))


def overlap_matrix(truth_ids, pred_ids, domain, nt, npred):
    codes = truth_ids[domain].astype(np.int64) * (npred + 1) + pred_ids[domain]
    return np.bincount(codes, minlength=(nt+1)*(npred+1)).reshape(nt+1, npred+1)


def match_objects(truth_ids, pred_ids, domain, rules):
    """IDs are formed on full volumes; all sizes/intersections/unions here use domain."""
    nt, npred = int(truth_ids.max()), int(pred_ids.max())
    overlap = overlap_matrix(truth_ids, pred_ids, domain, nt, npred)
    ts = np.bincount(truth_ids[domain], minlength=nt+1)[1:]
    ps = np.bincount(pred_ids[domain], minlength=npred+1)[1:]
    intersections = overlap[1:, 1:]
    unions = ts[:, None] + ps[None, :] - intersections
    ious = np.divide(intersections, unions, out=np.zeros_like(unions, dtype=float), where=unions > 0)
    matches = {}
    if nt and npred:
        rows, cols = linear_sum_assignment(ious, maximize=True)
        matches = {int(r+1): int(c+1) for r, c in zip(rows, cols) if intersections[r, c] > 0}
    coverage = np.divide(intersections, ts[:, None], out=np.zeros_like(ious), where=ts[:, None] > 0)
    edges = (intersections >= rules["edge_min_intersection"]) & (coverage >= rules["edge_min_truth_coverage"])
    stats = {"predicted_objects_full": npred, "truth_objects_full": nt,
             "merge_predicted_object_count": int((edges.sum(axis=0) > 1).sum()),
             "split_truth_object_count": int((edges.sum(axis=1) > 1).sum()),
             "unmatched_predicted_object_count": npred - len(matches),
             "unmatched_truth_object_count": nt - len(matches)}
    rows = []
    for i in range(1, nt+1):
        j = matches.get(i)
        size = int(ts[i-1])
        inter = int(intersections[i-1, j-1]) if j else 0
        pred_size = int(ps[j-1]) if j else 0
        union = size + pred_size - inter
        matched_iou = ratio(inter, union) if size else None
        matched_coverage = ratio(inter, size)
        rows.append({"object_id": i, "matched_prediction_id": j,
                     "evaluable_voxels": size, "matched_predicted_volume_common": pred_size,
                     "matched_intersection": inter, "matched_union": union,
                     "matched_iou": matched_iou,
                     "matched_precision": ratio(inter, pred_size), "matched_coverage": matched_coverage,
                     "any_prediction_coverage": ratio(int(intersections[i-1].sum()), size),
                     "detected_diagnostic": bool(j and inter >= rules["edge_min_intersection"]
                                                  and matched_iou >= rules["detection_min_iou"]
                                                  and matched_coverage >= rules["detection_min_coverage"]) if size else None,
                     "significant_prediction_overlaps": int(edges[i-1].sum()),
                     "na_reason": "no_common_domain_object_voxels" if size == 0 else ("unmatched_prediction_precision" if j is None else "")})
    pred_rows = []
    reverse = {v: k for k, v in matches.items()}
    full_sizes = np.bincount(pred_ids.ravel(), minlength=npred+1)[1:]
    for j in range(1, npred+1):
        pred_rows.append({"predicted_object_id": j, "matched_truth_id": reverse.get(j),
                          "full_voxels": int(full_sizes[j-1]), "common_voxels": int(ps[j-1]),
                          "false_positive_common_voxels": int(overlap[0, j]),
                          "significant_truth_overlaps": int(edges[:, j-1].sum())})
    stats["tiny_fragment_full_count_le5"] = int((full_sizes <= 5).sum())
    stats["tiny_fragment_common_volume_le5"] = int(ps[full_sizes <= 5].sum())
    return rows, pred_rows, stats, {"intersection_common": intersections, "union_common": unions,
                                   "truth_sizes_common": ts, "prediction_sizes_common": ps,
                                   "iou_common": ious, "overlap_with_background_common": overlap,
                                   "overlap_full": overlap_matrix(truth_ids, pred_ids, np.ones_like(domain), nt, npred)}


def exposure_table(case, ids, conditions, domain, layout):
    result = []
    masks = {n: conditions[n]["borehole_mask"].numpy()[0, 0] for n in (9, 16, 25)}
    s = conditions[9]["surface_mask"].numpy()[0, 0]
    m9 = conditions[9]["condition_mask"].numpy()[0, 0]
    for i in range(1, int(ids.max())+1):
        o = ids == i
        wh, sh, mh = int((o & masks[9]).sum()), int((o & s).sum()), int((o & m9).sum())
        group = "strict_unobserved" if mh == 0 else ("well_unhit_surface_visible" if wh == 0 else "well_hit")
        row = {"case_id": case, "object_id": i, "full_voxels": int(o.sum()),
               "major": bool(o.sum() >= 20), "exposure_group": group,
               "original_well_hit_count": sum(bool(o[x, y, :].any()) for x, y in layout["levels"]["9"]),
               "original_well_hit_voxels": wh, "original_surface_hit_voxels": sh,
               "original_joint_hit_voxels": mh, "common_voxels": int((o & domain).sum())}
        for n in (16, 25):
            extra_hits = int((o & masks[n] & ~m9).sum())
            row[f"n{n}_extra_hit_voxels"] = extra_hits
            row[f"n{n}_extra_hit_well_count"] = sum(bool((o & ~m9)[x, y, :].any()) for x, y in layout["levels"][str(n)][9:])
            row[f"n{n}_status"] = ("newly_hit" if extra_hits else "still_unhit") if mh == 0 else "originally_observed"
        result.append(row)
    return result


GROUPS = ("strict_unobserved", "well_unhit_surface_visible", "well_hit",
          "strict_newly_hit_by25", "strict_still_unhit_by25")

METRICS = ["target_iou", "target_precision", "target_recall", "target_signed_relative_volume_error",
           "target_absolute_relative_volume_error", "fixed_class_miou", "voxel_accuracy",
           "background_lithology_error", "background_false_target_rate", "target_fp",
           "target_tp", "target_fn", "target_truth_volume", "target_pred_volume",
           "predicted_objects_full", "unmatched_predicted_object_count", "unmatched_truth_object_count",
           "merge_predicted_object_count", "split_truth_object_count", "tiny_fragment_common_volume_le5",
           *[f"{group}_major_{metric}" for group in GROUPS
             for metric in ("matched_iou", "matched_coverage", "any_prediction_coverage")]]


def aggregate(samples):
    cases = sorted({r["case_id"] for r in samples})
    metrics = METRICS + sorted({k for r in samples for k in r if k.startswith("class_") and k.endswith("_iou")})
    case_rows, paired, case_paired = [], [], []
    for case in cases:
        cr = [r for r in samples if r["case_id"] == case]
        for n in (9, 16, 25):
            nr = [r for r in cr if r["well_count"] == n]
            row = {"case_id": case, "well_count": n, "seed_count": len(nr),
                   "common_domain_voxels": nr[0].get("domain_voxels"),
                   "common_domain_hash": nr[0].get("common_domain_hash")}
            for group in GROUPS:
                row[f"{group}_major_object_count"] = nr[0].get(f"{group}_major_object_count", 0)
            for key in metrics:
                values = [r.get(key) for r in nr]
                row[key] = median(values)
                row[key+"_valid_seeds"] = sum(v is not None for v in values)
            case_rows.append(row)
        by_key = {(r["source_seed"], r["well_count"]): r for r in cr}
        for high, low in ((16, 9), (25, 9), (25, 16)):
            for seed in sorted({r["source_seed"] for r in cr}):
                a, b = by_key[(seed, high)], by_key[(seed, low)]
                row = {"case_id": case, "source_seed": seed, "comparison": f"{high}-{low}"}
                for key in metrics:
                    row[key] = a[key]-b[key] if a.get(key) is not None and b.get(key) is not None else None
                paired.append(row)
            selected = [r for r in paired if r["case_id"] == case and r["comparison"] == f"{high}-{low}"]
            row = {"case_id": case, "comparison": f"{high}-{low}"}
            for key in metrics:
                row[key+"_median_paired_difference"] = median([r[key] for r in selected])
                cm = {(r["case_id"], r["well_count"]): r for r in case_rows}
                a, b = cm[(case, high)][key], cm[(case, low)][key]
                row[key+"_difference_of_seed_medians"] = a-b if a is not None and b is not None else None
            case_paired.append(row)
    summary = {"statistical_unit": "five_reused_geological_cases", "n_cases": len(cases),
               "n_samples": len(samples), "seed_uncertainty_calibrated": False, "levels": {}, "paired": {}}
    for n in (9, 16, 25):
        rows = [r for r in case_rows if r["well_count"] == n]
        summary["levels"][str(n)] = {k: {"case_median": median([r[k] for r in rows]),
                                                      "valid_cases": sum(r[k] is not None for r in rows)} for k in metrics}
    for comparison in ("16-9", "25-9", "25-16"):
        rows = [r for r in case_paired if r["comparison"] == comparison]
        summary["paired"][comparison] = {k: {"cross_case_median_of_case_median_paired_differences": median([r[k+"_median_paired_difference"] for r in rows]),
                                                      "valid_cases": sum(r[k+"_median_paired_difference"] is not None for r in rows),
                                                      "positive_cases": sum((r[k+"_median_paired_difference"] or 0) > 0 for r in rows),
                                                      "negative_cases": sum((r[k+"_median_paired_difference"] or 0) < 0 for r in rows)} for k in metrics}
    return case_rows, paired, case_paired, summary


def evaluate(args):
    registry = read_json(args.registry)
    p = validate_registry(registry)
    state = read_json(args.run / "sampling_state.json")
    sealed = read_json(args.run / "completion_receipt.json")
    if (state["status"] != "completed" or state["binding"]["mode"] != "formal"
        or sealed["state_sha256"] != file_hash(args.run / "sampling_state.json")
        or sealed["records_sha256"] != json_hash(state["records"])
        or state["binding"]["registry_file_sha256"] != file_hash(args.registry)
        or not state["base_model_unchanged"]):
        raise ValueError("inference is not sealed and complete")
    expected = {(c, s, n) for c in p["cases"] for s in p["source_seeds"] for n in p["well_counts"]}
    keys = [(r["case_id"], r["source_seed"], r["well_count"]) for r in state["records"]]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("formal output set differs from frozen 45")
    # Verify predictions and noise pairing before opening the evaluation registry/truth.
    predictions = {}
    pairing = []
    for r in state["records"]:
        path = args.run / r["prediction_path"]
        if file_hash(path) != r["prediction_file_sha256"]:
            raise ValueError("prediction file changed")
        pred = load_tensor(path)
        if tensor_hash(pred) != r["prediction_tensor_sha256"]:
            raise ValueError("prediction tensor changed")
        if pred.shape != (1, 1, 64, 64, 64) or not torch.isfinite(pred).all() or pred.min() < -1 or pred.max() > 13:
            raise ValueError("invalid prediction")
        predictions[(r["case_id"], r["source_seed"], r["well_count"])] = pred.numpy()[0, 0]
    for c in p["cases"]:
        for seed in p["source_seeds"]:
            rows = [r for r in state["records"] if r["case_id"] == c and r["source_seed"] == seed]
            hashes = {r["initial_noise_tensor_sha256"] for r in rows}
            noise_path = args.run / "noise" / f"seed{seed}.pt"
            if len(hashes) != 1 or tensor_hash(load_tensor(noise_path)) not in hashes:
                raise ValueError("source pairing mismatch")
            pairing.append({"case_id": c, "source_seed": seed, "well_counts": [9,16,25], "noise_sha256": next(iter(hashes)), "passed": True})
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    write_json(args.out / "pre_evaluation_seal.json", {"inference_receipt_sha256": file_hash(args.run / "completion_receipt.json"),
               "predictions_verified": 45, "all_truth_assets_unopened_until_now": True})
    ereg_path = args.registry.parent / "evaluation_registry.json"
    ereg = read_json(ereg_path)
    samples, exposures, object_rows, pred_rows, domains = [], [], [], [], []
    for case in p["cases"]:
        truth_t = verify_tensor(ereg["cases"][case]["truth"])
        truth = truth_t.numpy()[0, 0]
        conditions = {n: load_conditions(registry["conditions"][case][str(n)]) for n in p["well_counts"]}
        e = common_domain(truth, conditions[25]["condition_mask"].numpy()[0,0])
        classes = sorted(int(x) for x in np.unique(truth[e]))
        ids, count = connected(truth == 9)
        exposure = exposure_table(case, ids, conditions, e, registry["layout"])
        exposures.extend(exposure)
        domain_record = {"case_id": case, "common_voxels": int(e.sum()), "common_hash": tensor_hash(torch.from_numpy(e)),
                         "class_denominators": {str(k): int(((truth == k)&e).sum()) for k in classes},
                         "all_object_count": count, "major_object_count": sum(r["major"] for r in exposure),
                         "evaluable_strict_hidden_major_count": sum(r["major"] and r["exposure_group"] == "strict_unobserved" and r["common_voxels"] > 0 for r in exposure)}
        domains.append(domain_record)
        torch.save(torch.from_numpy(e), args.out / f"{case}_common_domain.pt")
        for seed in p["source_seeds"]:
            for n in p["well_counts"]:
                pred = predictions[(case, seed, n)]
                condition = conditions[n]["condition_mask"].numpy()[0, 0]
                violations = int(((pred != truth) & condition).sum())
                if violations:
                    raise ValueError("retrospective hard-condition mismatch")
                row = {"case_id": case, "source_seed": seed, "well_count": n,
                       "common_domain_hash": domain_record["common_hash"], "condition_violations": violations,
                       **voxel_metrics(pred, truth, e, classes)}
                # Supplemental metrics use explicit prefixes and never enter the common-dose curve.
                for prefix, domain in (("full_nonair_", truth != -1), ("own_free_", (truth != -1)&~condition)):
                    row.update({prefix+k: v for k, v in voxel_metrics(pred, truth, domain, classes).items()})
                row["full_raw_target_pred_volume"] = int((pred == 9).sum())
                row["full_raw_target_truth_volume"] = int((truth == 9).sum())
                pred_ids, _ = connected(pred == 9)
                objects, preds, stats, matrix = match_objects(ids, pred_ids, e, p["objects"])
                row.update(stats)
                prefix = {"case_id": case, "source_seed": seed, "well_count": n}
                for obj in objects:
                    meta = exposure[obj["object_id"]-1]
                    object_rows.append({**prefix, **meta, **obj})
                for obj in preds:
                    pred_rows.append({**prefix, **obj})
                matrix_path = args.out / "overlap_matrices" / f"{case}_seed{seed}_n{n}.npz"
                matrix_path.parent.mkdir(exist_ok=True)
                np.savez_compressed(matrix_path, **matrix)
                for group in GROUPS:
                    def in_group(meta):
                        if group in ("strict_newly_hit_by25", "strict_still_unhit_by25"):
                            status = "newly_hit" if group == "strict_newly_hit_by25" else "still_unhit"
                            return meta["exposure_group"] == "strict_unobserved" and meta["n25_status"] == status
                        return meta["exposure_group"] == group
                    selected = [o for o in objects if in_group(exposure[o["object_id"]-1])
                                and exposure[o["object_id"]-1]["major"] and o["evaluable_voxels"] > 0]
                    row[f"{group}_major_object_count"] = len(selected)
                    for metric in ("matched_iou", "matched_coverage", "any_prediction_coverage"):
                        values = [o[metric] for o in selected]
                        row[f"{group}_major_{metric}"] = float(np.mean(values)) if values else None
                samples.append(row)
    case_rows, paired, case_paired, summary = aggregate(samples)
    summary.update(status="completed_step1_only", scientific_success_gate="not_automatically_assigned",
                   evaluation_registry_sha256=file_hash(ereg_path),
                   protocol_sha256=registry["protocol_sha256"], inference_state_sha256=sealed["state_sha256"],
                   strict_hidden_major_objects=sum(d["evaluable_strict_hidden_major_count"] for d in domains),
                   strict_hidden_major_cases=sum(d["evaluable_strict_hidden_major_count"] > 0 for d in domains),
                   source_sha256={str(Path(__file__).relative_to(REPO)): file_hash(__file__)})
    for name, rows in (("sample_metrics", samples), ("case_level_summary", case_rows), ("paired_deltas", paired),
                       ("case_paired_deltas", case_paired), ("object_exposure", exposures),
                       ("object_metrics", object_rows), ("predicted_object_metrics", pred_rows)):
        write_csv(args.out / f"{name}.csv", rows)
    write_json(args.out / "pairing_audit.json", {"status": "passed", "pairs": pairing})
    write_json(args.out / "common_domain_audit.json", domains)
    write_json(args.out / "summary.json", summary)
    print(f"Retrospective evaluation complete: {args.out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    evaluate(parser.parse_args())
