# Phase 3: spatially degraded 3-D property observations

Read `docs/PHASE3_SPEC.md` before running. Phase 3 starts from the distinct
two-channel Phase-2a codebook that passed 12/12 strict pairs and changes only a
fixed spatial observation operator. It is a truth-derived 3-D
inversion-surrogate experiment, not measured or acquisition-domain geophysics.

## Implemented path

- `guidance/spatial_property.py`: validated identity/blur/coarse-grid operators,
  exact hard-property overwrite, depth/missing confidence, deterministic
  observation-only noise, soft and hard observation losses;
- `guidance/property_sampling.py`: default-preserving loss injection into the
  existing projected fixed-Euler sampler;
- `scripts/stage3/run_spatial_property_guidance.py`: immutable observation
  assets, strict pairing, EMA loading, CPU noise sequence, hard geology and
  observation-domain evaluation;
- `scripts/stage3/audit_spatial_screen.py`: source/asset/pairing audit,
  Phase-2a alpha-zero regression and the frozen complete hard gate;
- `configs/gaussian_sweep_manifest_v1.json`: identity then Gaussian sigma
  1/2/4 voxels in predeclared order;
- `tests/test_phase3_spatial_property.py`: the Phase-3 CPU development gate.

Generated tensors belong under `runs/`; derived reports belong under
`reports/`. Both are separate from Phase 1 and Phase 2. Runners refuse to
overwrite a non-empty run directory.

## Reproducing the seed-42 n=1 screen

The frozen screen was run one level at a time from the repository root. To
reproduce its first identity pair:

```bash
PYTHON_BIN=/absolute/path/to/gpu/python \
LEVEL=identity_anchor_v1 \
bash project/geodata-3d-conditional/experiments/stage3_spatial_property/run_phase3_gaussian_screen.sh
```

If the active `python` has CUDA PyTorch, omit `PYTHON_BIN`. The command creates
a new alpha-zero baseline and alpha-0.25 guided directory with one sample,
seed 42 and 32 fixed-Euler steps.

Audit the completed pair:

```bash
PYTHONPATH=src python \
  project/geodata-3d-conditional/scripts/stage3/audit_spatial_screen.py \
  --level identity_anchor_v1 \
  --overwrite
```

The audit requires bytewise equality with the existing Phase-2a alpha-zero
sample. The Phase-3 guided result is not required to be bytewise equal to
Phase 2a because exact known properties are now inserted before the observation
operator, eliminating finite-temperature property leakage around conditions.

Do not use one level's baseline for another: observation hashes are strict
pairing fields.

## First interrupted identity attempt

The first `identity_anchor_v1` invocation using run tag
`seed42_n1_s32_a025_c025` completed the 32-step baseline sample but stopped in
post-sampling hard-observation evaluation. The evaluator passed the saved 3-D
sample directly to a mapping that requires normalized `[B,1,X,Y,Z]` input.
This was a runner output-shape bug, not a model, EMA, solver or GPU failure.

The evaluator now normalizes saved 3-D samples and has a dedicated regression
test. The incomplete directory is retained as invalid evidence with
`run_status=running`; never edit, overwrite or pair against it. Repeat with a
new run tag such as `seed42_n1_s32_a025_c025_fix1`.

## Completed Gaussian n=1 screen

All four frozen levels completed strict seed-42 single-sample runs. Pairing,
EMA loading, 32-step fixed Euler, exact hard conditions and the Phase-2a
alpha-zero regression passed at every valid level. The hard scientific gate
gave:

| Level | Gate | Accuracy delta | Fixed mIoU delta | Label-9 IoU / P / R | Major mean recall |
|---|---:|---:|---:|---|---:|
| identity | pass | 0.0413 | 0.0676 | 0.4881 / 0.9032 / 0.5151 | 0.4984 |
| Gaussian sigma 1 | fail | 0.0312 | 0.0458 | 0.3357 / 0.6758 / 0.4001 | 0.3981 |
| Gaussian sigma 2 | fail | 0.0217 | 0.0272 | 0.2064 / 0.4935 / 0.2619 | 0.2584 |
| Gaussian sigma 4 | fail | 0.0128 | 0.0117 | 0.1026 / 0.2997 / 0.1349 | 0.1305 |

The aggregate decision is: **no nonzero Gaussian blur passed the complete n=1
gate**. Label-9 IoU and major-body mean recall both decline monotonically with
blur even though every observation loss decreases. See
`reports/gaussian_screen_seed42_n1/REPORT.md`.

The frozen promotion rule therefore brackets the identity pass with the
adjacent sigma-1 failure at seed 42, n=4. The identity side has completed and
passed 4/4 complete gates, including diversity:

- mean accuracy delta: `+0.0420`;
- mean truth-present mIoU delta: `+0.0696`;
- mean label-9 IoU/precision/recall: `0.4966/0.9144/0.5211`;
- mean major-body minimum/mean recall: `0.3915/0.5139`;
- exact conditions and Phase-2a alpha-zero regression: passed.

Its report is under
`reports/identity_anchor_v1/seed42_n4_s32_a025_c025/`. This confirms only the
undegraded anchor. The sigma-1 side also completed and failed 0/4. All four
pairs failed the target thresholds; two also failed major-component recovery.
Mean label-9 IoU/precision/recall were `0.3482/0.6938/0.4120`.

Phase 3 is closed. Do not spend n=4 GPU time on sigma 2 or sigma 4 and do not
run seeds 142/242. See `docs/PHASE3_REPORT.md`.

## Required interpretation

A one-sample pass is a screen, not confirmation. A lower observation loss,
smoother volume, larger label-9 volume or appealing soft isosurface cannot
promote a level without the frozen global/class/label-9/component/topology and
endpoint gates. Phase 4 gravity remains a separate experiment.
