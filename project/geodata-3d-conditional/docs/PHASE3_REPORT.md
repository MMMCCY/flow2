# Phase 3 final report: spatially degraded 3-D property guidance

Date closed: 2026-07-31.

## Decision

**CLOSED NEGATIVE RESULT: the undegraded identity anchor is confirmed, but no
nonzero Gaussian spatial degradation passes the frozen geological gate.**

Phase 3 therefore does not establish a degraded 3-D property working point for
multi-seed promotion or Phase-4b joint guidance. This is a valid resolution-
sensitivity result, not a runtime failure and not evidence about measured
geophysics.

## Question and controlled design

Phase 3 started from the distinct two-channel Phase-2a codebook that passed
12/12 strict pairs. It changed only the spatial observation operator: identity
or isotropic Gaussian blur with sigma 1, 2 or 4 voxels. Training, U-Net,
checkpoint, EMA convention, fixed-Euler solver, temperature/controller,
alpha/cap, initial-noise sequence and hard conditions remained fixed.

Known surface, air and borehole properties were inserted before the observation
operator, making their contribution exact and their guidance gradient zero.
Every positive-alpha result was paired with a newly generated alpha-zero run.
Alpha zero was required to reproduce the Phase-2a baseline bytewise.

## Seed-42 n=1 screen

| Level | Complete gate | Accuracy delta | Fixed mIoU delta | Label-9 IoU / P / R | Major mean recall |
|---|---:|---:|---:|---|---:|
| identity | pass | +0.0413 | +0.0676 | 0.4881 / 0.9032 / 0.5151 | 0.4984 |
| Gaussian sigma 1 | fail | +0.0312 | +0.0458 | 0.3357 / 0.6758 / 0.4001 | 0.3981 |
| Gaussian sigma 2 | fail | +0.0217 | +0.0272 | 0.2064 / 0.4935 / 0.2619 | 0.2584 |
| Gaussian sigma 4 | fail | +0.0128 | +0.0117 | 0.1026 / 0.2997 / 0.1349 | 0.1305 |

Label-9 IoU and major-body mean recall decrease monotonically as blur grows.
All four continuous observation losses decrease, so continuous agreement alone
would have produced a false positive conclusion.

## Frozen seed-42 n=4 bracket

The frozen rule selected the identity pass and adjacent sigma-1 failure.

| Level | Classification | Complete pairs | Accuracy delta | Fixed mIoU delta | Label-9 IoU / P / R |
|---|---|---:|---:|---:|---|
| identity | confirmed pass | 4/4 | +0.0420 | +0.0696 | 0.4966 / 0.9144 / 0.5211 |
| Gaussian sigma 1 | confirmed failure | 0/4 | +0.0322 | +0.0485 | 0.3482 / 0.6938 / 0.4120 |

Both levels pass strict pairing, observation hashes, exact hard conditions,
Phase-2a alpha-zero regression and ensemble diversity. Identity passes every
per-pair geology gate. At sigma 1, all four pairs fail the label-9 target gate,
principally because precision remains below the frozen 0.75 threshold. Two of
four also fail major-component recovery. Primary directions, majority-class
improvement, size-stratified topology and endpoint churn pass in all four.

## What Phase 3 proves

- The exact-known-property observation interface and injected-loss sampler are
  correct and reproduce the undegraded ideal-property result robustly.
- The frozen flow2 prior can convert full-resolution distinct property evidence
  into broad hard-label and label-9 improvements without violating conditions.
- A one-voxel Gaussian blur already removes enough spatial/class information to
  make label-9 recovery fail the complete gate at this fixed operating point.
- The decline is repeatable across four paired samples and is not explained by
  loss divergence, condition leakage, solver mismatch or loss of diversity.

## What Phase 3 does not prove

- It does not use measured density, susceptibility, gravity, magnetic or
  seismic data.
- It does not validate geophysics-only reconstruction or removal of surface and
  borehole conditions.
- It does not establish that every possible blurred-property controller must
  fail; alpha and other controls were intentionally not retuned after screening.
- The distinct codebook still gives label 9 special observability and is not a
  calibrated physical property table.

## Consequence for Phase 4

Phase-4a acquisition-domain gravity operator development remains independent
and may proceed. It must start with full-support forward physics, physical units
and derivative validation. Phase-4b joint smooth-3-D-plus-gravity guidance may
not claim a promoted Phase-3 degraded working point: none exists. Any later
joint experiment requires a new frozen design or must explicitly use identity
only as an upper-bound control.

Authoritative generated evidence:

- `experiments/stage3_spatial_property/reports/gaussian_screen_seed42_n1/`;
- `experiments/stage3_spatial_property/reports/identity_anchor_v1/seed42_n4_s32_a025_c025/`;
- `experiments/stage3_spatial_property/reports/gaussian_sigma1_v1/seed42_n4_s32_a025_c025/`.

At Phase-3 closure the complete local lightweight suite was `85 passed`. After
adding the independent Phase-4a CPU gate, the current suite is `96 passed`,
with the same 13 existing Matplotlib/pyparsing deprecation warnings.
