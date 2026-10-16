"""CPU-only retrospective continuous-evidence diagnosis of sealed Stage21v2 assets."""
import argparse
import csv
import json
import os
import platform
import shlex
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
PROJECT = Path(__file__).resolve().parents[2]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from scripts.stage21.common import (
    file_hash, tensor_hash, json_hash, read_json, write_json, write_csv,
    verify_tensor, well_mask, git_snapshot,
)
from scripts.stage21.evaluate import connected
from guidance.property_volume import (
    property_table_from_config, gaussian_blur_property_channels,
)
from metrics import average_precision, fixed_top5, quantiles

CASES = ("fullgeo_case03", "fullgeo_case01", "fullgeo_case02")
EXPECTED = {"fullgeo_case03": (1733, 103), "fullgeo_case01": (118, 11),
            "fullgeo_case02": (16046, 977)}
DOMAINS = ("E_original_free", "E_holdout")
REFERENCE = "fbafbcb5893a1c262a247f53e0fee5dd7072b822"
DEFAULT_SOURCE = PROJECT / "experiments/stage21v2_dense_virtual_boreholes/run_20261016_v1"
DEFAULT_OUT = PROJECT / "experiments/stage21v2_evidence_diagnostics/run_20261016_v1"


def install_readonly_guard(old):
    protected = (old.resolve(), PROJECT / "scripts/stage21v2",
                 PROJECT / "tests/test_stage21v2_dense_virtual_boreholes.py")

    def inside(path):
        return any(path == p or p in path.parents for p in protected)

    def guard(event, args):
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if path.suffix == ".ckpt":
                raise PermissionError("Offline diagnosis must not open a Flow checkpoint")
            mode, flags = args[1:3]
            writing = ((isinstance(mode, str) and any(c in mode for c in "wax+"))
                       or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
            if writing and inside(path):
                raise PermissionError(f"Sealed legacy asset is read-only: {path}")
        elif event in ("os.remove", "os.rmdir", "os.mkdir", "os.rename"):
            for value in args[:2]:
                if isinstance(value, (str, bytes, os.PathLike)) and inside(Path(os.fsdecode(value)).resolve()):
                    raise PermissionError(f"Sealed legacy asset mutation: {event}")
    sys.addaudithook(guard)


def distance_fields(properties, table, weights):
    # Keep the operation order, dtype and candidate order of the frozen classifier.
    candidates = table[:, 1:]
    scales = candidates.max(1).values - candidates.min(1).values
    d = ((properties[:, None] - candidates.T[None, :, :, None, None, None])
         / scales[None, None, :, None, None, None]).square()
    d = (d * (weights / weights.sum())[None, None, :, None, None, None]).sum(2)
    other = [k for k in range(14) if k != 9]
    competitor_distance, competitor_index = d[:, other].min(1, keepdim=True)
    margin = competitor_distance - d[:, 9:10]
    nearest_other = torch.tensor(other)[competitor_index]
    return d.argmin(1, keepdim=True).long(), margin, nearest_other, scales


def prefixed(name, values):
    return {name + "_" + k: v for k, v in quantiles(values).items()}


def run(source, out):
    source, out = source.resolve(), out.resolve()
    if source == out or source in out.parents:
        raise ValueError("New output must be outside the sealed old experiment")
    if out.exists():
        raise FileExistsError(f"Retain existing output; do not overwrite: {out}")
    install_readonly_guard(source)
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    out.mkdir(parents=True)
    start = time.monotonic()
    input_files = {}
    assets = []

    def metadata(name):
        path = source / name
        input_files[str(path.relative_to(REPO))] = file_hash(path)
        return read_json(path)

    def asset(record, role):
        path = Path(record["path"])
        path = path if path.is_absolute() else REPO / path
        tensor = verify_tensor(record)
        if tensor.device.type != "cpu":
            raise ValueError("Asset must be CPU loaded")
        input_files[str(path.relative_to(REPO))] = record["file_sha256"]
        assets.append(dict(role=role, **record, dtype=str(tensor.dtype), shape=list(tensor.shape), reconstructed=False))
        return tensor

    try:
        protocol = metadata("PROTOCOL.json")
        classifier = metadata("classifier.json")
        config = metadata("codebook.json")
        evaluation = metadata("evaluation_registry.json")
        inference = metadata("inference_registry.json")
        layout = metadata("layout.json")
        for name in ("object_metrics.csv", "report/REPORT.md"):
            path = source / name
            input_files[str(path.relative_to(REPO))] = file_hash(path)
        with (source / "object_metrics.csv").open() as stream:
            old_objects = {(r["case_id"], int(r["object_id"])): int(r["full_voxels"]) for r in csv.DictReader(stream)}
        if json_hash(protocol) != inference["protocol_sha256"] or protocol["sigma"] != 1.0:
            raise ValueError("Frozen protocol mismatch")
        if file_hash(source / "codebook.json") != protocol["codebook_sha256"]:
            raise ValueError("Frozen codebook mismatch")
        table, weights, table_metadata = property_table_from_config(config, 15)
        if table_metadata["channel_names"] != ["density_proxy", "susceptibility_contrast_proxy"]:
            raise ValueError("Unexpected channel order")
        if config["channels"][1]["values"]["9"] != .1:
            raise ValueError("Unexpected target susceptibility")
        if weights.tolist() != classifier["weights"]:
            raise ValueError("Classifier weights mismatch")
        # Only check legacy code inputs already sealed by Stage21v2, not historical experiments.
        relevant_sources = ["scripts/stage21/common.py", "scripts/stage21/evaluate.py",
                            "scripts/stage21v2/common.py", "scripts/stage21v2/prepare.py",
                            "guidance/property_volume.py"]
        source_identity = {}
        for relative in relevant_sources:
            path = PROJECT / relative
            key = str(path.relative_to(REPO))
            actual = file_hash(path)
            if actual != inference["source_sha256"][key]:
                raise ValueError(f"Legacy implementation differs from sealed run: {key}")
            input_files[key] = actual
            source_identity[key] = actual
        git = git_snapshot()
        command = " ".join(shlex.quote(s) for s in [".venv/bin/python", str(Path(__file__).relative_to(REPO)),
                                                    "--source", str(source.relative_to(REPO)), "--out", str(out.relative_to(REPO))])
        command = 'CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ' + command
        configuration = dict(schema="stage21v2_offline_evidence_diagnostics_v1", cases=list(CASES),
            source_run=str(source.relative_to(REPO)), primary_domain=DOMAINS[0], auxiliary_domain=DOMAINS[1],
            fixed_scores=["margin9", "susceptibility"], direction="higher is more target-supportive, fixed a priori",
            margin="min_{k in raw0..13,k!=9} d_k - d_9; signed, no clipping/softmax",
            distances="FP32 frozen codebook weights and NONAIR max-min scales; all raw0..13 candidates",
            top_fraction=.05, tie_rule="ceil(0.05*n) rank; include all ties, no truth tie breaking",
            AP="non-interpolated sum of recall increments * precision at complete tied groups; sklearn-equivalent binary unweighted",
            AP_reference="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html",
            AP_backend="local exact grouped implementation (server sklearn unavailable)",
            objects="26-neighbor labels on full truth before masking; original IDs; major full size>=20",
            q9="truth-derived explanatory G_sigma1(1[truth==9]); full volume replicate Gaussian then mask; NOT inference input/probability",
            source_seeds="none: each case analyzed once", scans=False, training_runs=0, Flow_runs=0,
            checkpoint_loaded=False, CPU_only=True, command=command, reference_commit=REFERENCE)
        write_json(out / "CONFIG.json", configuration)
        (out / "COMMAND.txt").write_text(command + "\n", encoding="utf-8")
        write_json(out / "REPO_STATE.json", dict(**git, reference_matches=git["head"] == REFERENCE,
            tracked_diff=subprocess_output(["git", "diff", "--stat"]),
            new_namespace="scripts/stage21v2_diagnostics", legacy_source_identity=source_identity))
        own_sources = {str(p.relative_to(REPO)): file_hash(p) for p in Path(__file__).parent.glob("*.py")}
        for relative in own_sources:
            destination = out / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((REPO / relative).read_bytes())
        write_json(out / "SOURCE_HASHES.json", own_sources)
        summary, distributions, assignments, object_rows, audits = [], [], [], [], []
        plot_data = {}
        for case in CASES:
            records = evaluation["cases"][case]
            truth = asset(records["truth"], f"{case}:retrospective_truth")
            props = asset(records["properties"], f"{case}:saved_smoothed_property")
            pseudo = asset(records["pseudo_raw"], f"{case}:raw_classifier_not_projected")
            original_mask = asset(inference["conditions"][case]["A_9"]["original_mask"], f"{case}:original_hard_mask")
            if truth.shape != (1, 1, 64, 64, 64) or props.shape != (1, 2, 64, 64, 64) or props.dtype != torch.float32:
                raise ValueError("Unexpected asset shapes/dtype")
            decoded, margin, nearest, scales = distance_fields(props, table, weights)
            # One necessary full-volume argmin check against unprojected pseudo_raw.
            mismatch = int((decoded != pseudo).sum())
            if mismatch or scales.tolist() != classifier["scales"]:
                raise ValueError(f"Argmin/scales mismatch: {case}, {mismatch}")
            q9 = gaussian_blur_property_channels((truth == 9).float(), sigma=1.0)
            t = truth.numpy()[0, 0]
            m = margin.numpy()[0, 0]
            sus = props.numpy()[0, 1]
            q = q9.numpy()[0, 0]
            pred = decoded.numpy()[0, 0]
            ids, count = connected(t == 9)
            full_sizes = np.bincount(ids.ravel(), minlength=count+1)
            for object_id in range(1, count+1):
                if old_objects[(case, object_id)] != int(full_sizes[object_id]):
                    raise ValueError("Object ID/size inconsistent with old object table")
            w9 = well_mask(layout["original_wells"]).numpy()[0, 0]
            om = original_mask.numpy()[0, 0]
            fields = out / "fields"
            fields.mkdir(exist_ok=True)
            np.savez_compressed(fields / f"{case}.npz", margin9=m, susceptibility=sus, q9_truth_explanatory=q,
                                nearest_non9=nearest.numpy()[0, 0], truth_object_ids=ids)
            per_case = dict(truth=t, margin9=m, susceptibility=sus, q9=q, ids=ids, domains={})
            for domain_index, name in enumerate(DOMAINS):
                domain_tensor = asset(records["domains"][name]["tensor"], f"{case}:{name}")
                if domain_tensor.dtype != torch.bool:
                    raise ValueError("Domain must be bool")
                domain = domain_tensor.numpy()[0, 0]
                free = (t != -1) & ~om
                held = well_mask(layout["holdout"]).numpy()[0, 0]
                expected_domain = free if name == "E_original_free" else free & held
                if not np.array_equal(domain, expected_domain):
                    raise ValueError("Registered domain not equal to frozen mask definition")
                positive = (t == 9) & domain
                positives = int(positive.sum())
                size = int(domain.sum())
                if positives != EXPECTED[case][domain_index] or size != records["domains"][name]["voxels"]:
                    raise ValueError(f"Known target/domain count mismatch: {case}/{name}")
                pi = positives / size
                audits.append(dict(case_id=case, domain=name, voxels=size, positives=positives,
                                   domain_sha256=tensor_hash(domain_tensor), argmin_mismatches=mismatch,
                                   properties_reconstructed=False))
                per_case["domains"][name] = dict(domain=domain, curves={}, top={})
                for group, mask in (("true9", positive), ("non9", domain & (t != 9))):
                    values = m[mask]
                    distributions.append(dict(case_id=case, domain=name, group=group, count=int(mask.sum()),
                        **quantiles(values), strict_positive_margin_fraction=float((values>0).mean()),
                        zero_margin_fraction=float((values==0).mean()), negative_margin_fraction=float((values<0).mean()),
                        hard_target_count=int((pred[mask]==9).sum()), hard_target_fraction=float((pred[mask]==9).mean())))
                for label in range(14):
                    assignments.append(dict(case_id=case, domain=name, true_label=9, assigned_raw_label=label,
                        count=int(((pred==label) & positive).sum()), target_count=positives,
                        fraction=float(((pred==label) & positive).sum()/positives)))
                for score_name, score in (("margin9", m), ("susceptibility", sus)):
                    labels = t[domain] == 9
                    values = score[domain]
                    ap, curve = average_precision(labels, values)
                    selected, threshold, requested = fixed_top5(values)
                    top = np.zeros_like(domain)
                    top[domain] = selected
                    per_case["domains"][name]["curves"][score_name] = curve
                    per_case["domains"][name]["top"][score_name] = top
                    selected_count = int(selected.sum())
                    tp = int(labels[selected].sum())
                    summary.append(dict(case_id=case, domain=name, score=score_name, domain_voxels=size,
                        positives=positives, prevalence=pi, AP=ap, AP_skill=(ap-pi)/(1-pi),
                        top5_requested_count=requested, top5_threshold=threshold, top5_selected_count=selected_count,
                        top5_actual_fraction=selected_count/size, top5_tp=tp, top5_precision=tp/selected_count,
                        top5_recall=tp/positives, hard_target_count=int(((pred==9)&domain).sum()),
                        hard_target_fraction=float((pred[domain]==9).mean()), hard_target_tp=int(((pred==9)&positive).sum())))
                    pr_dir = out / "pr_curves"
                    pr_dir.mkdir(exist_ok=True)
                    np.savez_compressed(pr_dir / f"{case}_{name}_{score_name}.npz", **curve)
                    for object_id in range(1, count+1):
                        full_object = ids == object_id
                        obj = full_object & domain
                        voxels = int(obj.sum())
                        hits = int((full_object & w9).sum())
                        hard_hits = int((full_object & om).sum())
                        full_volume = int(full_sizes[object_id])
                        hit_wells = sum(bool(full_object[x,y,:].any()) for x,y in layout["original_wells"])
                        object_rows.append(dict(case_id=case, domain=name, object_id=object_id, score=score_name,
                            full_voxels=full_volume, major=full_volume>=20, domain_voxels=voxels,
                            original_well_hit_count=hit_wells, original_well_hit_voxels=hits,
                            original_hard_hit_voxels=hard_hits, original_well_zero_hit=hits==0,
                            original_M9_zero_hit=hard_hits==0, **prefixed("score", score[obj]),
                            **prefixed("q9_domain", q[obj]), **prefixed("q9_full", q[full_object]),
                            top5_covered_voxels=int((obj & top).sum()),
                            top5_coverage=float((obj & top).sum()/voxels) if voxels else None,
                            top5_threshold=threshold, top5_actual_fraction=selected_count/size,
                            hard_correct_target_voxels=int(((pred==9)&obj).sum()),
                            na_reason="empty_object_in_evaluation_domain" if not voxels else ""))
            plot_data[case] = per_case
            print(f"{case}: exact argmin; targets {EXPECTED[case]}; {count} full-volume objects", flush=True)
        must_explain = [r for r in object_rows if r["case_id"] == "fullgeo_case03" and r["object_id"] == 1]
        if not must_explain or any(r["full_voxels"] != 572 or not r["original_well_zero_hit"] for r in must_explain):
            raise ValueError("Missing required case03/572-voxel zero-well-hit object")
        write_csv(out / "score_summary.csv", summary)
        write_csv(out / "margin_distributions.csv", distributions)
        write_csv(out / "true9_assignments.csv", assignments)
        write_csv(out / "object_diagnostics.csv", object_rows)
        write_json(out / "INPUT_ASSETS.json", dict(assets=assets, metadata_and_source_file_sha256=input_files))
        write_json(out / "CHECKS.json", dict(status="passed", cases=audits,
            known_counts_match=True, full_argmin_matches_pseudo_raw=True, original_object_ids_match=True,
            Gaussian_sigma=1.0, Gaussian_boundary="replicate", CPU_only=True, checkpoint_open_guard=True,
            old_run_write_guard=True, new_Flow_runs=0, training_runs=0))
        from report import write_report
        write_report(out, configuration, summary, distributions, assignments, object_rows, plot_data)
        for relative, before in input_files.items():
            if file_hash(REPO / relative) != before:
                raise ValueError(f"Read-only input changed: {relative}")
        for relative, before in own_sources.items():
            if file_hash(REPO / relative) != before:
                raise ValueError("Diagnostic source changed while running")
        forbidden_imports = [name for name in ("scripts.stage21.run", "scripts.stage21v2.run", "scripts.stage21v2.workflow", "model_train_sh_inference_cond") if name in sys.modules]
        if forbidden_imports:
            raise ValueError(f"Inference entry imported: {forbidden_imports}")
        write_json(out / "COMPLETION.json", dict(status="completed", elapsed_seconds=time.monotonic()-start,
            input_hashes_unchanged=True, source_hashes_unchanged=True, Flow_checkpoint_opened=False,
            inference_entry_imported=False, CUDA_VISIBLE_DEVICES=os.environ["CUDA_VISIBLE_DEVICES"],
            torch=torch.__version__, numpy=np.__version__, python=platform.python_version(),
            new_Flow_runs=0, training_runs=0, bootstrap=False, threshold_scan=False))
        write_json(out / "ARTIFACT_HASHES.json", {str(p.relative_to(out)):file_hash(p) for p in sorted(out.rglob("*")) if p.is_file()})
        print(out / "REPORT.md", flush=True)
    except Exception as exc:
        write_json(out / "FAILURE.json", dict(status="failed",error=f"{type(exc).__name__}: {exc}",
                                              new_Flow_runs=0,training_runs=0))
        raise


def subprocess_output(command):
    import subprocess
    return subprocess.check_output(command,cwd=REPO,text=True).strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args=parser.parse_args()
    run(args.source,args.out)
