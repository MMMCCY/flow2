"""Stage23 contracts: complete restored pseudo-wells; historical assets read-only."""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
from scripts.stage21.common import (
    PROJECT, REPO, CHECKPOINT, CHECKPOINT_SHA, read_json, write_json, write_csv,
    file_hash, tensor_hash, json_hash, verify_tensor, save_tensor, well_mask, git_snapshot,
)
from scripts.stage21v2.common import check_condition, dense_conditions, classify

OLD = PROJECT / 'experiments/stage21v2_dense_virtual_boreholes/run_20261016_v1'
RESTORED = PROJECT / 'experiments/stage22_blur_aware_bridge/run_20261016_v1'
CASE = 'fullgeo_case03'
ROOT = PROJECT / 'experiments/stage23_full_pseudo_wells/run_20261017_v2'
RULES = dict(edge_min_intersection=5, edge_min_truth_coverage=.05,
             detection_min_iou=.1, detection_min_coverage=.1)


def utc():
    return datetime.now(timezone.utc).isoformat()


def source_hashes():
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += [PROJECT / name for name in (
        'scripts/stage21/common.py', 'scripts/stage21/run.py', 'scripts/stage21/evaluate.py',
        'scripts/stage21v2/common.py', 'guidance/property_volume.py', 'inference_runtime.py',
        'model_train_sh_inference_cond.py', 'guidance/generator_posterior.py', 'guidance/dflow.py')]
    return {str(p.relative_to(REPO)): file_hash(p) for p in sorted(paths)}


def guard(mode):
    protected = [OLD, RESTORED, PROJECT/'scripts/stage21v2', PROJECT/'scripts/stage22',
                 PROJECT/'tests/test_stage21v2_dense_virtual_boreholes.py']
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        writing = ((isinstance(args[1], str) and any(c in args[1] for c in 'wax+')) or
                   (isinstance(args[2], int) and args[2] &
                    (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        if writing and any(path == p or p in path.parents for p in protected):
            raise PermissionError(f'Historical files are read-only: {path}')
        if mode in ('prepare', 'flow') and (
                'truth' in path.parts or ('geometry' in path.parts and 'experiments' in path.parts) or
                'stage21v2_evidence_diagnostics' in path.parts or
                path.name in ('object_metrics.csv', 'evaluation_registry.json')):
            raise PermissionError(f'{mode} cannot read truth/objects/evaluation: {path}')
        if mode != 'flow' and path.suffix == '.ckpt':
            raise PermissionError('Offline process cannot load checkpoint')
        if mode == 'flow' and ('preprocessing' in path.parts or path.name in (
                'restored_properties.pt', 'restored_classification.pt', 'recovered_z.pt')):
            raise PermissionError('Flow receives prepared conditions only')
    sys.addaudithook(audit)


def protocol():
    return dict(schema='stage23_full_pseudo_wells_v1', case=CASE, source_seed=42,
        total_wells=512, layout='unchanged Stage21v2 levels[512]; original condition priority',
        classification='reuse Stage22 saved classification; FP32 raw0..13 first argmin; old codebook weights/ranges',
        added='well_mask(layout512) & ~original_mask; all depths, all predicted raw0..13; no target filter',
        original_priority=True, unknown_value=-1, mask_equal_old_B_C_512=True,
        condition_preparation_truth_visible=False, scientific_gate_before_flow=False,
        weights='411 EMA parameters; raw checkpoint frozen embedding; all frozen',
        sampler='unchanged sample_native, 32 midpoint Euler, FP32 no autocast/TF32, deterministic',
        noise='load actual saved A_seed42 CPU noise and verify its identity',
        domains=['E_holdout', 'E_original_free'], classes='fixed R21 registered classes',
        objects='full-volume 26-connected IDs before fixed domain; old one-to-one IoU matching',
        object_rules=RULES, object1_common_subset='truthID1 & E_original_free & ~Stage23_new_mask',
        figure='one retrospective full XY Z=15, fixed prior display coordinate',
        max_new_Flow_outputs=1, max_velocity_evaluations=32,
        no_training=True, no_deconvolution=True, no_scans=True, no_trimming=True,
        interpretation=dict(conversion='all-label/target/non-target errors versus C on same added_mask',
            modeling='holdout recall and IoU > C; mIoU >= C; object1 common outside-mask TP > A/COPY',
            extent='precision, volume bias and merging versus Stage22, absolute and all reference gaps'),
        reference_commit='8ac499a2b5fc0cacf205066f92b382625c314da4',
        specification_sha256=file_hash(REPO/'STAGE23_FULL_PSEUDO_WELLS_PROMPT_2026-10-17.md'))
