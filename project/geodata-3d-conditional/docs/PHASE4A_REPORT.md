# Phase 4a final report: acquisition-domain gravity-only guidance

Date closed: 2026-07-31.

## Decision

**CLOSED NEGATIVE SCREEN: both pre-registered seed-42 controllers reduce the
hard gravity residual, but neither converts field agreement into recovery of
the three-dimensional hard geology.**

The alpha-0.25 primary screen and the conditionally authorized alpha-0.10 harm
diagnostic both fail the frozen complete geology-plus-gravity gate (`0/1`).
Neither setting is eligible for n=4 or multi-seed promotion. No further alpha
search is authorized in Phase 4a.

## Question and controlled design

Phase 4a asks whether a surface-acquired gravity field can provide the global
information missing from sparse wells while the frozen flow2 prior and exact
surface/borehole constraints reduce gravity non-uniqueness.

The first test deliberately favors gravity: it uses a truth-derived,
full-64-by-64, noiseless synthetic inverse-crime observation and a complete
density table in which label 9 has a distinct `650 kg/m3` contrast. The forward
operator evaluates the downward vertical attraction of every rectangular
prism at every surface station, in SI geometry with mGal output and
full-support zero-padded FFT linear convolution. This is not measured data and
the density table is not site-calibrated petrophysics.

Training, U-Net and checkpoint are unchanged. All 411 trainable parameters use
EMA, while the frozen embedding uses its normal checkpoint value. Baseline and
guided arms use fixed Euler, the same CPU-generated initial noise, 32 steps,
the same time grid and immutable observation tensors. Known surface and
borehole density is inserted exactly before every forward evaluation, giving
condition voxels zero guidance gradient; embeddings are projected before and
after every solver step.

## Frozen seed-42 n=1 results

| Arm | Complete gate | Accuracy | Truth-present mIoU | Hard gravity RMSE (mGal) | Label-9 IoU / P / R | Label-9 volume |
|---|---:|---:|---:|---:|---|---:|
| alpha 0 baseline | reference | 0.58737 | 0.26525 | 0.95848 | 0.02860 / 0.06748 / 0.04728 | 6283 |
| alpha 0.25 | fail | 0.59644 | 0.26536 | 0.89287 | 0.01590 / 0.06383 / 0.02074 | 2914 |
| alpha 0.10 | fail | 0.59347 | 0.26564 | 0.87032 | 0.02111 / 0.06490 / 0.03033 | 4191 |

Both guided arms pass strict pairing, immutable tensor/source hashes,
Phase-2a alpha-zero hard regression, finite traces and exact conditions.
Relative to baseline:

- alpha 0.25 lowers hard gravity RMSE by `0.06561 mGal`, but changes `6562`
  hard voxels; `3392` leave label 9 and only `23` enter it;
- alpha 0.10 lowers hard gravity RMSE by `0.08815 mGal`, but changes `3756`
  hard voxels; `2101` leave label 9 and only `9` enter it;
- each arm improves only four of the eight truth-present classes, below the
  frozen majority requirement;
- major label-9 component mean recall is only `0.0205` at alpha 0.25 and
  `0.0280` at alpha 0.10; minimum recall is `0` for both;
- both fail primary-direction, target-threshold and major-component gates,
  although their endpoint-churn and size-stratified-topology checks pass.

The lower controller changes fewer labels and preserves more of the baseline
label-9 prediction, but it still removes target voxels rather than recovering
the missing truth bodies. Its lower final hard gravity residual also shows
that controller magnitude and decoded field residual are not monotonic along
the nonlinear sampling path.

## Scientific interpretation

The soft-to-hard mechanism is active: gravity guidance changes thousands of
decoded labels while every hard condition remains exact. The failure is
therefore not the earlier problem in which a continuous gradient could not
cross embedding Voronoi boundaries.

Instead, the two-dimensional integral field admits many three-dimensional
density arrangements. The sampler lowers gravity residual by reallocating
density among lithologies and deleting misplaced label-9 mass. In this test,
the frozen learned prior plus sparse conditions are not strong enough to select
the truth-like member of that gravity-equivalent family. A lower continuous
loss, a higher global accuracy dominated by common classes, and thousands of
hard changes do not establish target-body recovery.

## What Phase 4a proves

- The full-support differentiable gravity operator, immutable observation
  path, EMA loader, controller and fixed-Euler strict-pair runner function on
  the real checkpoint.
- Acquisition-domain gravity gradients can materially cross hard-label
  boundaries without violating known surface or borehole conditions.
- Surface gravity residual can improve while label-9 IoU, recall, volume and
  every major-body recovery measure worsen.
- At the two pre-registered strengths, flow2 structure prior and sparse wells
  do not resolve the depth/lithology non-uniqueness of this favorable gravity
  upper bound.

## What Phase 4a does not prove

- It does not prove that every gravity likelihood, posterior algorithm,
  codebook or joint-physics method must fail.
- It does not validate measured gravity or calibrated rock properties.
- It does not support removing surface or borehole constraints.
- It does not provide a multi-sample or multi-seed gravity result; the frozen
  protocol prevents promotion after the n=1 gate failure.
- It does not make the Phase-2 truth-derived 3-D property oracle equivalent to
  a gravity inversion product.

## Decision and next work

Phase 4a gravity-only controller screening is closed. Do not run n=4, the
label-6/9 density-collision control, or additional alpha values: the deliberately
distinct inverse-crime upper bound already failed the geological screen.

The next independent upper bound is convolutional seismic response. Unlike
surface gravity, a 3-D post-stack seismic cube preserves lateral and
time/depth localization. Before GPU sampling, freeze and test the complete
velocity/density/impedance codebook, reflectivity convention, depth-to-time
mapping, wavelet, sampling interval, masks, noise and differentiable adjoint or
finite-difference behavior. Gravity-plus-seismic is considered only after a
seismic-only hard-geology gate; it cannot rescue a failed arm by reporting a
joint continuous loss.

Authoritative generated evidence:

- `experiments/stage4_gravity/reports/seed42_n1_s32_a025_c025/`;
- `experiments/stage4_gravity/reports/seed42_n1_s32_a010_c025/`;
- the corresponding immutable paired run directories under
  `experiments/stage4_gravity/runs/cond_generation_0/phase4a_gravity_v1/`.

At closure, the Phase-4-focused CPU gate is `18 passed` and the complete local
lightweight suite is `103 passed`, with 13 existing Matplotlib/pyparsing
deprecation warnings.
