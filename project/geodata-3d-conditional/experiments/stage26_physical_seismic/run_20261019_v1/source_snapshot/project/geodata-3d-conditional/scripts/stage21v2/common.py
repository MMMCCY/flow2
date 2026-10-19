"""Dense-well protocol primitives; no case truth or evaluation file access."""
from pathlib import Path

import numpy as np
import torch

from scripts.stage21.common import (
    PROJECT, REPO, CHECKPOINT, CHECKPOINT_SHA, W9, file_hash, tensor_hash,
    json_hash, read_json, write_json, load_tensor, save_tensor, verify_tensor,
    well_mask, check_labels, git_snapshot, write_csv, source_hashes,
)

ROOT = PROJECT / "experiments/stage21v2_dense_virtual_boreholes"
OLD = PROJECT / "experiments/stage21_virtual_borehole_conditioning/run_20261014_v2"
CODEBOOK = PROJECT / "experiments/stage2_property/configs/ideal_density_susceptibility_label9_contrast_v1.json"
LEVELS = (512, 2048, 3840)
CASES = ("fullgeo_case03", "fullgeo_case01", "fullgeo_case02")
OBJECT_RULES = dict(edge_min_intersection=5, edge_min_truth_coverage=.05,
                    detection_min_iou=.1, detection_min_coverage=.1)


def layout():
    rng = np.random.Generator(np.random.PCG64(21601))
    cells = [[(x, y) for x in range(4*i, 4*i+4) for y in range(4*j, 4*j+4)
              if (x, y) not in W9] for i in range(16) for j in range(16)]
    holdout = [cell[int(rng.integers(len(cell)))] for cell in cells]
    queues = []
    for cell, held in zip(cells, holdout):
        points = [p for p in cell if p != held]
        queues.append([points[int(i)] for i in rng.permutation(len(points))])
    order = []
    cursor = [0]*256
    while True:
        active = [i for i in range(256) if cursor[i] < len(queues[i])]
        if not active:
            break
        for i in rng.permutation(active):
            order.append(queues[i][cursor[i]])
            cursor[i] += 1
    assert len(order) == 3831
    return {"seed": 21601, "rng": "numpy.Generator(PCG64)",
            "original_wells": W9, "holdout": holdout, "extra_order": order,
            "levels": {str(n): W9 + order[:n-9] for n in LEVELS}}


def dense_conditions(original_values, original_mask, labels, wells):
    check_labels(labels)
    if labels.shape != original_values.shape or original_mask.dtype != torch.bool:
        raise ValueError("invalid shape/mask")
    added = well_mask(wells, labels.shape[2:]) & ~original_mask
    mask = original_mask | added
    values = torch.full_like(original_values, -1)
    values[added] = labels[added].to(values.dtype)
    values[original_mask] = original_values[original_mask]
    return values, mask


def classify(properties, table, weights):
    """All raw labels 0..13; first argmin breaks exact ties toward smaller raw label."""
    candidates = table[:, 1:]
    scales = candidates.max(1).values - candidates.min(1).values
    if (scales <= 0).any() or not torch.isfinite(properties).all():
        raise ValueError("invalid classifier scale/input")
    d = ((properties[:, None] - candidates.T[None, :, :, None, None, None])
         / scales[None, None, :, None, None, None]).square()
    d = (d * (weights / weights.sum())[None, None, :, None, None, None]).sum(2)
    return d.argmin(1, keepdim=True).long()


def evaluation_domains(truth, original_mask, held):
    free = (truth != -1) & ~original_mask
    return {"E_holdout": free & well_mask(held, truth.shape[2:]),
            "E_original_free": free}


def copy_only(baseline, values, mask):
    return torch.where(mask, values.to(baseline.dtype), baseline)


def sources():
    result = source_hashes()
    for p in [*(PROJECT / "scripts/stage21v2").glob("*.py"),
              PROJECT / "scripts/stage21/evaluate.py", PROJECT / "guidance/property_volume.py",
              PROJECT / "tests/test_stage21v2_dense_virtual_boreholes.py"]:
        result[str(p.relative_to(REPO))] = file_hash(p)
    return result


def check_condition(record):
    if set(record) != {"values", "mask", "original_mask", "original_values"}:
        raise ValueError("condition fields")
    ts = {k: verify_tensor(v) for k, v in record.items()}
    v, m, ov, om = (ts[k] for k in ("values", "mask", "original_values", "original_mask"))
    check_labels(v)
    if v.shape != (1, 1, 64, 64, 64) or any(t.shape != v.shape for t in (m, ov, om)):
        raise ValueError("condition shape")
    if m.dtype != torch.bool or om.dtype != torch.bool or (om & ~m).any():
        raise ValueError("condition mask")
    if not torch.equal(v[om], ov[om]) or (v[~m] != -1).any():
        raise ValueError("condition preservation/placeholders")
    return ts


def validate_inference(r):
    if set(r) != {"schema", "protocol_sha256", "source_sha256", "checkpoint", "checkpoint_sha256", "conditions"}:
        raise ValueError("inference fields")
    if r["schema"] != "stage21v2_inference_v1" or r["checkpoint"] != CHECKPOINT or r["checkpoint_sha256"] != CHECKPOINT_SHA:
        raise ValueError("inference model contract")
    for case, arms in r["conditions"].items():
        if case not in CASES:
            raise ValueError("case")
        for key, record in arms.items():
            if key not in {"A_9", *[f"{a}_{n}" for a in "BC" for n in LEVELS]}:
                raise ValueError("arm")
            for asset in record.values():
                if set(asset) != {"path", "file_sha256", "tensor_sha256"}:
                    raise ValueError("asset fields")
                p = Path(asset["path"])
                if p.is_absolute() or ".." in p.parts or "conditions" not in p.parts or p.suffix != ".pt":
                    raise ValueError("inference path")
    return r
