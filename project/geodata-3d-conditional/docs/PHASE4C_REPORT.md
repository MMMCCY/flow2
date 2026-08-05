# Phase 4c final report: acquisition-domain convolutional-seismic guidance

Date closed: 2026-07-31.

## Decision

**CLOSED NEGATIVE SCREEN: the pre-registered seed-42 alpha/cap 0.25
controller lowers the hard seismic residual but does not recover the
three-dimensional hard geology.**

The single strict pair fails the frozen complete geology-plus-seismic gate
(`0/1`). It is not eligible for n=4 or multi-seed promotion. The conditional
alpha-0.10 diagnostic is not authorized: alpha 0.25 passes both the last-step
churn stability check and the paired endpoint-change limit, so this is not the
pre-registered excessive-change failure that a weaker controller was intended
to diagnose.

## Question and controlled design

Phase 4c asks whether a laterally and two-way-time localized seismic response
can provide the global information missing from sparse wells while the frozen
flow2 prior and exact surface/borehole constraints reduce seismic
non-uniqueness.

The experiment deliberately favors seismic. It uses a truth-derived,
full-64-by-64, noiseless synthetic inverse-crime post-stack response. Every
vertical column is forward modeled independently from a complete synthetic
density/velocity codebook in which label 9 has deliberately distinctive
acoustic impedance. Reflectivity is deposited on a fixed 320-sample, 8 ms
two-way-time grid and convolved with a zero-phase 25 Hz Ricker wavelet. This is
not measured data, calibrated petrophysics, migrated field seismic or a
full-wave simulation.

Training, U-Net and checkpoint are unchanged. All 411 trainable parameters use
EMA, while the frozen embedding uses its normal checkpoint value. Baseline and
guided arms use fixed Euler, the same CPU-generated initial noise, 32 steps,
the same time grid and immutable observation tensors. Known surface and
borehole acoustic properties are inserted exactly before every forward
evaluation, and embeddings are projected before and after every solver step.

## Frozen seed-42 n=1 result

| Arm | Complete gate | Accuracy | Truth-present mIoU | Hard seismic RMSE | Label-9 IoU / P / R | Label-9 volume |
|---|---:|---:|---:|---:|---|---:|
| alpha 0 baseline | reference | 0.58737 | 0.26525 | 0.042262 | 0.02860 / 0.06748 / 0.04728 | 6283 |
| alpha 0.25 | fail | 0.58972 | 0.26565 | 0.039048 | 0.02593 / 0.07027 / 0.03947 | 5038 |

Both arms pass strict pairing, immutable tensor/source hashes, Phase-2a
alpha-zero hard regression, EMA loading, finite traces and exact conditions.
Relative to baseline:

- hard seismic normalized loss falls from `17.86063` to `15.24736`, and RMSE
  falls by `0.003214` amplitude units;
- global accuracy rises by `0.002354` and truth-present mIoU rises by only
  `0.000406`, while the all-code mean IoU falls by `0.041106` because absent
  labels are also introduced;
- label-9 IoU falls by `0.002666` and recall falls from `0.04728` to `0.03947`;
- predicted label-9 volume falls from `6283` to `5038` against `8968` truth
  voxels, increasing absolute volume error from `2685` to `3930`;
- the final guided sample differs from its paired baseline at `3737` voxels
  (`1.4256%`): `1438` leave label 9 and only `193` enter label 9;
- only four of eight truth-present classes improve, below the required five;
- recalls of the four largest truth label-9 bodies are
  `0.0306/0/0.1121/0`, all no better than baseline, with mean `0.0357` and
  minimum `0`;
- the last Euler step changes `702` decoded voxels (`0.2678%`), so the frozen
  endpoint-stability check passes. This last-step churn is distinct from the
  full guided-versus-baseline difference above.

The complete gate fails primary directions, majority-class improvement,
absolute label-9 thresholds and major-component recovery. It passes exact
conditions, size-stratified topology and last-step churn.

## Scientific interpretation

The soft-to-hard mechanism is active: seismic guidance changes thousands of
final decoded labels without violating known conditions. The failure is not a
missing gradient or an inability to cross embedding Voronoi boundaries.

The convolutional response constrains acoustic interfaces and their arrival
times, not absolute lithology identity at every voxel. Even with distinctive
label-9 impedance, a frozen sampler can reduce trace mismatch by shifting
boundaries and exchanging other lithologies while deleting label-9 mass. In
this result the learned prior plus sparse conditions do not select the
truth-like hard model from the response-equivalent alternatives. Slight gains
in accuracy and truth-present mean IoU are dominated by common classes and do
not compensate for worse target-body and component recovery.

## What Phase 4c proves

- The differentiable convolutional operator, immutable seismic observation,
  EMA loader and fixed-Euler strict-pair runner function on the real
  checkpoint.
- Acquisition-domain seismic gradients cross hard-label boundaries while
  surface and borehole conditions remain exact.
- A lower full-cube seismic residual does not necessarily imply recovery of
  label 9 or its major three-dimensional bodies.
- Under this deliberately favorable single-controller upper bound, the frozen
  flow2 prior and sparse wells do not resolve the remaining acoustic/lithology
  non-uniqueness.

## What Phase 4c does not prove

- It does not prove that every seismic likelihood, acquisition geometry,
  elastic property set, posterior-search method or geophysics-aware trained
  model must fail.
- It does not validate measured seismic, calibrated density/velocity or
  field-data inversion.
- It does not support removing surface or borehole conditions.
- It does not provide multi-sample or multi-seed seismic evidence; the frozen
  protocol prevents promotion after the n=1 gate failure.
- It does not authorize gravity-plus-seismic fusion. Combining two failed
  single-physics arms and reporting a weighted continuous loss would not solve
  their geological non-uniqueness.

## Decision and next work

Phase 4c convolutional-seismic-only controller screening is closed. Do not run
alpha 0.10, n=4, more seeds or extra alpha values for this protocol.

The next design must address geological identifiability rather than controller
magnitude. Before another GPU experiment, freeze a separate protocol that
compares at least one posterior-selection or geology-aware observation design
against strict paired inference. Candidate directions include multi-property
or multi-angle/elastic information, structure-aware likelihood terms that do
not use truth labels directly, and ensemble posterior selection. Any new arm
must retain the frozen model boundary, exact hard conditions and complete
hard-geology gates. If sufficient identifiability cannot be obtained at
inference time, the evidence supports a later, explicitly separate training
study in which geophysics is learned jointly; it must not be presented as a
continuation of the frozen-checkpoint experiment.

Authoritative generated evidence:

- `experiments/stage4_seismic/reports/seed42_n1_s32_a025_c025/`;
- `experiments/stage4_seismic/runs/cond_generation_0/phase4c_seismic_v1/seed42_n1_s32_a025_c025/`;
- `experiments/stage4_seismic/observations/cond_generation_0/distinct_upper_bound_v1_fix2/`.

At closure, the Phase-4c-focused CPU gate is `15 passed` and the complete local
lightweight suite is `118 passed`, with 13 existing Matplotlib/pyparsing
deprecation warnings.
