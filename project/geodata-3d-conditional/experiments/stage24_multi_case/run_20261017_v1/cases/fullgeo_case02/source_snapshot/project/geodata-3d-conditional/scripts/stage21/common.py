"""Stage21 shared contracts. This module has no truth/evaluation dependency."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
REPO = PROJECT.parents[1]
EXPERIMENT = PROJECT / "experiments/stage21_virtual_borehole_conditioning"
CHECKPOINT = "project/geodata-3d-conditional/demo_model/conditional-weights.ckpt"
CHECKPOINT_SHA = "561e94bfda770ec41fc4cbed43436a7e2130eef5dfb7e5d666fcefc0724ff94c"
W9 = [(8, 46), (9, 5), (10, 24), (27, 17), (35, 26), (39, 59), (44, 60), (48, 6), (57, 32)]
CASES = [f"fullgeo_case{i:02d}" for i in range(1, 6)]
SEEDS = [42, 142, 242]
LEVELS = [9, 16, 25]
SOURCE_FILES = [
    "scripts/stage21/common.py", "scripts/stage21/run.py",
    "inference_runtime.py", "model_train_sh_inference_cond.py",
    "guidance/generator_posterior.py", "guidance/dflow.py",
    "guidance/probability_sampling.py", "guidance/probability_volume.py",
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_hash(tensor):
    # Same canonical dtype/shape/content contract as full_structuralgeo_benchmark.
    a = tensor.detach().cpu().contiguous().numpy()
    h = hashlib.sha256()
    h.update(str(a.dtype).encode("ascii"))
    h.update(json.dumps(list(a.shape), separators=(",", ":")).encode("ascii"))
    h.update(a.tobytes(order="C"))
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def source_hashes():
    paths = [PROJECT / p for p in SOURCE_FILES]
    paths += list((REPO / "src/flowtrain").rglob("*.py"))
    paths += [PROJECT / p for p in ("boreholes.py", "callbacks.py", "utils.py")]
    return {str(p.relative_to(REPO)): file_hash(p) for p in sorted(paths)}


def git_snapshot():
    def run(*args):
        return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()
    return {"head": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current"),
            "status_short": run("status", "--short")}


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_tensor(path):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"not a tensor: {path}")
    return value


def check_labels(value):
    if not torch.isfinite(value).all() or not torch.equal(value, value.round()):
        raise ValueError("labels must be finite integers")
    if value.numel() and (value.min() < -1 or value.max() > 13):
        raise ValueError("raw labels must be -1..13")


def check_wells(wells, shape=(64, 64, 64)):
    points = [tuple(p) for p in wells]
    if len(set(points)) != len(points):
        raise ValueError("duplicate well")
    for p in points:
        if len(p) != 2 or any(type(v) is not int for v in p):
            raise ValueError("well coordinates must be integer XY")
        if not (0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]):
            raise ValueError("well out of bounds")
    return points


def plan_layout(seed=21001):
    """PCG64; row-major cells, uniform lexicographic eligible XY, then permutation."""
    rng = np.random.Generator(np.random.PCG64(seed))
    extra = []
    for i in range(4):
        for j in range(4):
            choices = [(x, y) for x in range(16*i, 16*(i+1))
                       for y in range(16*j, 16*(j+1)) if (x, y) not in W9]
            extra.append(choices[int(rng.integers(len(choices)))])
    order = rng.permutation(16).tolist()
    extra = [extra[i] for i in order]
    check_wells(W9 + extra)
    return {"schema": "stage21_layout_v1", "layout_seed": seed,
            "rule": "4x4_row_major_PCG64_uniform_eligible_integer_xy_then_permutation",
            "truth_independent": True, "cell_permutation": order,
            "original_wells": [list(p) for p in W9],
            "extra_wells_ordered": [list(p) for p in extra],
            "levels": {str(n): [list(p) for p in (W9 + extra[:n-9])] for n in LEVELS}}


def well_mask(wells, shape=(64, 64, 64)):
    points = check_wells(wells, shape)
    mask = torch.zeros((1, 1, *shape), dtype=torch.bool)
    for x, y in points:
        mask[0, 0, x, y, :] = True
    return mask


def condition_assets(labels, surface, wells):
    check_labels(labels)
    if labels.ndim != 5 or labels.shape[:2] != (1, 1) or surface.shape != labels.shape:
        raise ValueError("condition shape mismatch")
    if surface.dtype != torch.bool:
        raise ValueError("surface mask must be bool")
    boreholes = well_mask(wells, labels.shape[2:])
    mask = surface | boreholes
    values = torch.full_like(labels, -1)
    values[mask] = labels[mask]
    return {"condition_values": values, "condition_mask": mask,
            "surface_mask": surface.clone(), "borehole_mask": boreholes}


def save_tensor(path, tensor):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)
    return {"path": str(path.relative_to(REPO)), "file_sha256": file_hash(path),
            "tensor_sha256": tensor_hash(tensor)}


def verify_tensor(record):
    path = REPO / record["path"]
    if file_hash(path) != record["file_sha256"]:
        raise ValueError(f"asset file hash mismatch: {path}")
    t = load_tensor(path)
    if tensor_hash(t) != record["tensor_sha256"]:
        raise ValueError(f"asset tensor hash mismatch: {path}")
    return t


def require_keys(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"unexpected {name} keys; allowed: {keys}")


def validate_registry(registry):
    """Strict allowlist: the runner cannot accept truth, ROI, or object metadata."""
    require_keys(registry, ["schema", "protocol", "protocol_sha256", "layout", "layout_sha256",
                            "conditions", "source_sha256"], "registry")
    if registry["schema"] != "stage21_inference_registry_v1":
        raise ValueError("registry schema")
    p = registry["protocol"]
    require_keys(p, ["schema", "scope", "cases", "source_seeds", "well_counts", "shape",
                     "checkpoint", "checkpoint_sha256", "weight_source", "n_steps", "integrator",
                     "precision", "determinism", "layout_seed", "smoke_case", "smoke_seed",
                     "figure_seed", "objects", "evaluation", "budget"], "protocol")
    require_keys(p["objects"], ["raw_label", "connectivity", "major_min_voxels", "matching",
                                "edge_min_intersection", "edge_min_truth_coverage", "detection_min_iou",
                                "detection_min_coverage", "diagnostic_thresholds_are_not_scientific_success_gates"], "object rules")
    require_keys(p["evaluation"], ["domain", "class_set", "case_aggregation", "paired_aggregation",
                                   "undefined", "figure_rule"], "evaluation rules")
    require_keys(p["budget"], ["formal_outputs", "formal_velocity_evaluations", "cuda_smoke_outputs",
                               "independent_n9_regression_velocity_evaluations", "max_concurrent_gpu_jobs"], "budget")
    if json_hash(p) != registry["protocol_sha256"] or json_hash(registry["layout"]) != registry["layout_sha256"]:
        raise ValueError("protocol/layout hash mismatch")
    if (p["cases"] != CASES or p["source_seeds"] != SEEDS or p["well_counts"] != LEVELS
        or p["shape"] != [64, 64, 64] or p["n_steps"] != 32 or p["weight_source"] != "ema"
        or p["checkpoint"] != CHECKPOINT or p["checkpoint_sha256"] != CHECKPOINT_SHA
        or p["precision"] != "float32_no_autocast_no_tf32"
        or p["integrator"] != "fixed_euler_midpoint_v1"
        or p["determinism"] != "torch_deterministic_algorithms_cublas4096_8_cudnn_benchmark_false"
        or p["layout_seed"] != 21001 or p["smoke_case"] != "fullgeo_case03" or p["smoke_seed"] != 142
        or p["figure_seed"] != 142
        or p["scope"] != "step1_accurate_extra_wells_only"):
        raise ValueError("unregistered inference protocol")
    if registry["layout"] != plan_layout(p["layout_seed"]):
        raise ValueError("layout replay mismatch")
    if set(registry["conditions"]) != set(CASES):
        raise ValueError("case registry mismatch")
    for case, levels in registry["conditions"].items():
        require_keys(levels, [str(n) for n in LEVELS], "well levels")
        for n, record in levels.items():
            require_keys(record, ["schema", "origin", "oracle", "sequence_uncertainty", "well_xy", "tensors"], "condition")
            if (record["schema"] != "categorical_borehole_conditions_v1" or not record["oracle"]
                or record["sequence_uncertainty"] is not False
                or record["origin"] != "synthetic_truth_sampled_at_frozen_wells_and_original_surface"
                or record["well_xy"] != registry["layout"]["levels"][n]):
                raise ValueError("condition provenance/layout mismatch")
            require_keys(record["tensors"], ["condition_values", "condition_mask", "surface_mask", "borehole_mask"], "tensor set")
            for asset in record["tensors"].values():
                require_keys(asset, ["path", "file_sha256", "tensor_sha256"], "tensor asset")
                path = Path(asset["path"])
                # Only prepared condition assets, even if a malicious pointer is nested.
                if path.is_absolute() or ".." in path.parts or "conditions" not in path.parts or path.suffix != ".pt":
                    raise ValueError("asset outside prepared conditions")
    return p


def load_conditions(record):
    ts = {k: verify_tensor(v) for k, v in record["tensors"].items()}
    v, m, s, b = [ts[k] for k in ("condition_values", "condition_mask", "surface_mask", "borehole_mask")]
    check_labels(v)
    if v.shape != (1, 1, 64, 64, 64) or any(x.shape != v.shape for x in (m, s, b)):
        raise ValueError("invalid condition shape")
    if any(x.dtype != torch.bool for x in (m, s, b)):
        raise ValueError("masks must be bool")
    if not torch.equal(m, s | b) or not torch.equal(b, well_mask(record["well_xy"])):
        raise ValueError("condition mask contract")
    if not torch.all(v[~m] == -1):
        raise ValueError("hidden labels outside mask")
    return ts
