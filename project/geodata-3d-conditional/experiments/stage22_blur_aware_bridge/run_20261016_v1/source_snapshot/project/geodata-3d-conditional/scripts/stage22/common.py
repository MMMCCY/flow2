"""Stage22 fixed contracts, provenance and read-only historical assets."""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
import torch
from scripts.stage21.common import (
    PROJECT, REPO, CHECKPOINT, CHECKPOINT_SHA, read_json, write_json, write_csv,
    file_hash, tensor_hash, json_hash, verify_tensor, save_tensor, well_mask, git_snapshot,
)
from scripts.stage21v2.common import check_condition

OLD = PROJECT / "experiments/stage21v2_dense_virtual_boreholes/run_20261016_v1"
DIAGNOSTIC = PROJECT / "experiments/stage21v2_evidence_diagnostics/run_20261016_v1"
ROOT = PROJECT / "experiments/stage22_blur_aware_bridge"
CASE = "fullgeo_case03"
RULES = dict(edge_min_intersection=5, edge_min_truth_coverage=.05,
             detection_min_iou=.1, detection_min_coverage=.1)


def utc():
    return datetime.now(timezone.utc).isoformat()


def source_hashes():
    paths = list(Path(__file__).parent.glob("*.py")) + [PROJECT / name for name in (
        "scripts/stage21/common.py", "scripts/stage21/run.py", "scripts/stage21/evaluate.py",
        "scripts/stage21v2/common.py", "guidance/property_volume.py", "inference_runtime.py",
        "model_train_sh_inference_cond.py", "guidance/generator_posterior.py", "guidance/dflow.py")]
    return {str(p.relative_to(REPO)): file_hash(p) for p in sorted(paths)}


def guard(mode):
    """Preparation cannot open truth/objects/checkpoints; all phases preserve old work."""
    protected = [OLD, DIAGNOSTIC, PROJECT / "scripts/stage21v2",
                 PROJECT / "tests/test_stage21v2_dense_virtual_boreholes.py"]
    def audit(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        writing = ((isinstance(args[1], str) and any(c in args[1] for c in "wax+"))
                   or (isinstance(args[2], int) and args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        if writing and any(path == p or p in path.parents for p in protected):
            raise PermissionError(f"Historical files are read-only: {path}")
        if mode in ("prepare", "flow") and ("truth" in path.parts or DIAGNOSTIC in path.parents
                or path.name == "object_metrics.csv" or ("geometry" in path.parts and "experiments" in path.parts)):
            raise PermissionError(f"{mode} cannot open truth/object/diagnostic assets: {path}")
        if mode != "flow" and path.suffix == ".ckpt":
            raise PermissionError("Offline solver/evaluator must not load a Flow checkpoint")
        if mode == "flow" and ("evaluation" in path.parts or "preprocessing" in path.parts or path.name == "evaluation_registry.json"
                or path.name in ("restored_properties.pt", "restored_classification.pt", "recovered_z.pt")):
            raise PermissionError("Flow can see prepared conditions only")
    sys.addaudithook(audit)


def protocol():
    return dict(schema="stage22_blur_aware_bridge_v1", case=CASE, sigma=1., TV_weight=1e-4,
        TV="mean over channels and voxels sqrt(sum_xyz(forward_difference**2)+1e-8); exterior differences zero",
        scale="fixed nonair codebook max-min; no centering", bounds="full codebook including air min/max divided by scale",
        original_constraints="unblurred lookup properties fixed in z at original hard mask",
        initialization="clipped y (observed properties/scale), then original hard projection",
        solver="projected accelerated gradient with majorization backtracking and objective-monotonic restart",
        solver_dtype="CPU float64; frozen float32 Gaussian coefficients promoted to float64",
        Gaussian="original radius3 sigma1 separable kernel, replicate padding; autograd exact transpose",
        initial_L=4., backtrack_factor=2., max_iterations=2000,
        stop="50 consecutive accepted-iterate objective relative changes <1e-6; denominator max(abs(previous),1e-30)",
        iterate_choice="final solver iterate only; no truth-visible snapshots",
        classification="FP32 old nonair raw0..13 argmin and fixed normalized weights/ranges",
        acceptance=dict(argmin=9, max_d9=.01, min_margin9=.05),
        layout="old total512 columns; exclude original hard mask; subset depths only; never holdout",
        added_labels=[9], source_seeds=[42,142,242], max_new_Flow_outputs=3,
        offline_gate="added precision>=0.95 AND correct added voxels in object1>=5; no additions => NA/fail",
        seed42_gate="original_free recall>A and precision>=A; object1 one-to-one IoU>=0.10 and coverage>=0.10; holdout mIoU>=A-0.01",
        confirmation="only if seed42 passes; same conditions seeds142/242; all 3 pass every criterion",
        evaluation_domains=["E_original_free", "E_holdout"], object_rules=RULES,
        outside_added_object_evaluation="object1 intersect original_free minus added condition mask; common denominator A/new",
        weights="411 trainable EMA then freeze; raw checkpoint frozen embedding",
        sampler="original sample_native; 32 midpoint-time fixed Euler steps; FP32 no autocast/TF32; deterministic CUBLAS :4096:8",
        noise="load and hash-verify actual paired A source tensor; verify CPU initial_noise generation as well",
        no_training=True, no_adapter=True, no_physics=True, no_scans=True,
        reference_commit="9b46d1af2d08c32ce256ad18afab716b91c92ada",
        specification_sha256=file_hash(REPO / "STAGE22_BLUR_AWARE_BRIDGE_PROMPT_2026-10-16.md"))


def classify_properties(properties,table,weights):
    candidates=table[:,1:]
    scales=candidates.max(1).values-candidates.min(1).values
    d=((properties[:,None]-candidates.T[None,:,:,None,None,None])/scales[None,None,:,None,None,None]).square()
    d=(d*(weights/weights.sum())[None,None,:,None,None,None]).sum(2)
    pred=d.argmin(1,keepdim=True).long()
    d9=d[:,9:10]
    margin=d[:,[i for i in range(14) if i!=9]].min(1,keepdim=True).values-d9
    return pred,d9,margin


def sparse_conditions(values,mask,pred,d9,margin,eligible):
    added=(pred==9)&(d9<=.01)&(margin>=.05)&eligible&~mask
    new_values=torch.full_like(values,-1)
    new_values[mask]=values[mask]
    new_values[added]=9
    return new_values,mask|added,added
