"""Offline synthetic property construction and declared condition preparation."""
import argparse
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.stage21v2.common import *
from guidance.property_volume import (property_table_from_config, hard_labels_to_properties,
                                      gaussian_blur_property_channels)
from scripts.stage21.common import load_conditions


def protocol():
    return dict(schema="stage21v2_dense_wells_v1", development_case=CASES[0],
                check_cases=list(CASES[1:]), source_seeds=[42, 142, 242], levels=list(LEVELS),
                layout_seed=21601, sigma=1.0, codebook=str(CODEBOOK.relative_to(REPO)),
                codebook_sha256=file_hash(CODEBOOK), checkpoint=CHECKPOINT, checkpoint_sha256=CHECKPOINT_SHA,
                integrator="32 fixed Euler steps; t=(step+0.5)/32; one velocity call/step",
                weights="411 trainable EMA; raw checkpoint frozen embedding",
                precision="FP32 no autocast no TF32", deterministic=True, cublas=":4096:8",
                source="CPU full Gaussian before projection; same hash per case/seed",
                classifier="raw 0..13; weighted squared distance; codebook nonair max-min channel scales; smallest-label exact ties",
                smoothing="full property volume including codebook air; existing separable Gaussian replicate boundaries",
                original_priority="original condition values/mask always preserved; outside mask=-1",
                main_domain="holdout_columns & ~original_condition_mask & truth!=-1",
                auxiliary_domain="~original_condition_mask & truth!=-1",
                class_set="truth-present nonair classes fixed per case/domain before GPU",
                screen="case03 seed42 increasing N; run C only if B holdout mIoU>A; stop higher N at first candidate",
                candidate="C holdout mIoU>A and original_free mIoU>=A and original_free label9 IoU>A",
                confirmation="only candidate density seeds142/242; all 3 holdout paired mIoU>0 and median paired original_free mIoU/label9 IoU>0",
                cross_case="only if confirmed; case01/02 seed42 same N; no density search",
                max_network_outputs=19, max_forward_calls=608, simultaneous_gpu_jobs=1,
                objects=dict(connectivity=26, major_min_voxels=20, **OBJECT_RULES),
                endpoints="4096 oracle/pseudo are direct-write endpoints; no Flow",
                limits="controlled separable-property surrogate; reused historical cases; seeds not independent geology; no training/adapter/physics inversion",
                holdout_caveat="held-out columns contribute to synthetic smoothed properties; not independent observation holdout",
                figure_slices=[32, 32, 32], figure_seed=42,
                prompt_sha256=file_hash(REPO / "STAGE21V2_VSCODE_CODEX_PROMPT_2026-10-16.md"))


def prepare(out):
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(4)
    if file_hash(REPO / CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError("checkpoint hash")
    old = read_json(OLD / "inference_registry.json")
    ev = read_json(OLD / "evaluation_registry.json")
    benchmark = PROJECT / "experiments/full_structuralgeo_benchmark"
    if file_hash(benchmark / "benchmark_manifest.json") != ev["benchmark_manifest_sha256"]:
        raise ValueError("benchmark manifest hash")
    out.mkdir(parents=True)
    plan, p = layout(), protocol()
    for relative in sources():
        destination = out / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, destination)
    shutil.copyfile(REPO / "STAGE21V2_VSCODE_CODEX_PROMPT_2026-10-16.md", out / "source_snapshot" / "TASK_SPEC.md")
    write_json(out / "PROTOCOL.json", p)
    write_json(out / "layout.json", plan)
    shutil.copyfile(CODEBOOK, out / "codebook.json")
    table, weights, metadata = property_table_from_config(read_json(CODEBOOK), 15)
    candidates = table[:, 1:].T
    collisions = [[i, j] for i in range(14) for j in range(i+1, 14)
                  if torch.equal(candidates[i], candidates[j])]
    probe = torch.arange(14).reshape(1, 1, 14, 1, 1)
    roundtrip = classify(hard_labels_to_properties(probe, table), table, weights)
    if collisions or not torch.equal(probe, roundtrip):
        raise ValueError("codebook collision / indexing error")
    write_json(out / "classifier.json", dict(metadata=metadata, collisions=collisions,
               sigma0_nonair_roundtrip_exact=True, scales=(candidates.max(0).values-candidates.min(0).values).tolist(),
               weights=weights.tolist(), rule=p["classifier"], air="not a classification candidate; original hard air preserved"))
    registry = dict(schema="stage21v2_inference_v1", protocol_sha256=json_hash(p),
                    source_sha256=sources(), checkpoint=CHECKPOINT,
                    checkpoint_sha256=CHECKPOINT_SHA, conditions={})
    evaluation = dict(role="offline_preparation_and_retrospective_only", cases={})
    for case in CASES:
        manifest_path = benchmark / "cases" / case / "manifest.json"
        if file_hash(manifest_path) != ev["cases"][case]["historical_manifest_sha256"]:
            raise ValueError("case manifest hash")
        m = read_json(manifest_path)
        truth = verify_tensor(ev["cases"][case]["truth"])
        orig = load_conditions(old["conditions"][case]["9"])
        for name, value in orig.items():
            rel = f"condition/{name}.pt"
            path = benchmark / "cases" / case / rel
            if file_hash(path) != m["saved_file_sha256"][rel] or tensor_hash(value) != m["tensor_content_hashes"][rel]:
                raise ValueError("original condition source mismatch")
        if old["conditions"][case]["9"]["well_xy"] != [list(x) for x in W9]:
            raise ValueError("original XY")
        ov, om = orig["condition_values"], orig["condition_mask"]
        if not torch.equal(ov[om], truth[om]):
            raise ValueError("truth/original condition mismatch")
        properties = hard_labels_to_properties(truth, table)
        sigma0 = classify(gaussian_blur_property_channels(properties, 0), table, weights)
        if not torch.equal(sigma0[truth != -1], truth[truth != -1].long()):
            raise ValueError("sigma0 roundtrip")
        smooth = gaussian_blur_property_channels(properties, 1.0)
        pseudo_raw = classify(smooth, table, weights)
        pseudo = copy_only(pseudo_raw, ov, om)
        evaluation["cases"][case] = dict(truth=ev["cases"][case]["truth"],
            properties=save_tensor(out / "evaluation" / case / "smooth_properties.pt", smooth),
            pseudo_raw=save_tensor(out / "evaluation" / case / "pseudo_raw.pt", pseudo_raw),
            pseudo=save_tensor(out / "evaluation" / case / "pseudo_original_projected.pt", pseudo),
            sigma0_nonair_mismatches=0, domains={})
        for name, domain in evaluation_domains(truth, om, plan["holdout"]).items():
            classes = sorted(int(x) for x in truth[domain].unique())
            evaluation["cases"][case]["domains"][name] = dict(
                tensor=save_tensor(out / "evaluation" / case / f"{name}.pt", domain),
                classes=classes, voxels=int(domain.sum()),
                class_counts={str(k): int(((truth == k) & domain).sum()) for k in classes})
        registry["conditions"][case] = {}
        common_assets = {"original_mask": save_tensor(out / "conditions" / case / "original_mask.pt", om),
                         "original_values": save_tensor(out / "conditions" / case / "original_values.pt", ov)}
        pairs = {"A_9": (ov, om)}
        for n in LEVELS:
            for arm, labels in (("B", truth), ("C", pseudo)):
                pairs[f"{arm}_{n}"] = dense_conditions(ov, om, labels, plan["levels"][str(n)])
            if not torch.equal(pairs[f"B_{n}"][1], pairs[f"C_{n}"][1]):
                raise ValueError("B/C mask mismatch")
        for key, (values, mask) in pairs.items():
            record = {**common_assets,
                      "values": save_tensor(out / "conditions" / case / key / "values.pt", values),
                      "mask": save_tensor(out / "conditions" / case / key / "mask.pt", mask)}
            check_condition(record)
            registry["conditions"][case][key] = record
    validate_inference(registry)
    write_json(out / "inference_registry.json", registry)
    write_json(out / "evaluation_registry.json", evaluation)
    write_json(out / "preparation_receipt.json", dict(git=git_snapshot(), source_sha256=sources(),
               protocol_sha256=json_hash(p), registry_sha256=file_hash(out / "inference_registry.json"),
               evaluation_sha256=file_hash(out / "evaluation_registry.json")))
    print(f"Prepared {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    prepare(parser.parse_args().out.resolve())
