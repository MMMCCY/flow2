# Phase 4c specification: convolutional seismic upper bound

Date frozen: 2026-07-31, after Phase 4a closure and before implementation or
generation of any seismic observation.

## Decision context

Phase 4a lowered a favorable full-grid noiseless gravity residual at both
pre-registered controllers but damaged label-9 hard geology. Surface gravity
does not retain enough depth localization for the frozen flow2 prior and sparse
wells to select the truth-like three-dimensional density arrangement.

Phase 4c therefore tests an independent acquisition-domain response that
retains lateral and two-way-time localization. It does not reinterpret or hide
the Phase-4a negative result, and gravity-plus-seismic is not run until a
seismic-only arm passes the complete hard-geology gate.

## First observation model

The first experiment is a synthetic normal-incidence post-stack convolutional
seismic upper bound. Every `(x,y)` column is an independent vertical trace. For
raw label `k`, the frozen codebook defines positive absolute density
`rho(k)` in `kg m^-3` and P-wave velocity `vp(k)` in `m s^-1`. The differentiable
soft mappings use

```text
Z_pred = sum_k p(k) rho(k) vp(k)
s_pred = sum_k p(k) / vp(k)
```

where `Z` is acoustic impedance and `s` is slowness. At every known surface,
air or borehole voxel, predicted impedance and slowness are overwritten by
their exact hard-codebook values before forward modeling. Their guidance
gradient must therefore be zero.

The observed topographic surface also fixes which voxels are subsurface. In
that known subsurface support, the soft acoustic mapping excludes the air
category and renormalizes the remaining category probabilities. This prevents
transient soft air probability from producing nonphysical underground air
travel time. It does not project or hide the decoded hard geology: any air
label below the known surface remains an error in hard-label evaluation.

The tensor's final spatial axis has larger indices upward. Each trace is
reversed into local surface-to-depth order. The known truth-derived subsurface
mask is allowed only to define the already observed topographic surface and to
exclude the air/rock reflection. It may not reveal internal lithology. For
adjacent subsurface cells,

```text
r_i = (Z_below - Z_above) / (Z_below + Z_above)
t_i = sum_above_and_including_i 2 dz s
```

Reflectivity is linearly deposited into the two nearest samples of a fixed
regular two-way-time grid and convolved with a fixed zero-phase Ricker wavelet.
Convolution uses zero boundary padding, preserves trace length and has no
lateral mixing. The observed data are

```text
d_obs  = F_s(Z_true, s_true; surface_mask) + epsilon
d_pred = F_s(Z_pred_known, s_pred_known; surface_mask)
L_s    = || M (d_pred - d_obs) / sigma ||^2 / sum(M)
```

The first upper bound uses all lateral traces and all time samples, no noise,
fixed scalar uncertainty, and the same operator for construction and guidance.
It is explicitly an inverse crime, not measured seismic.

## Frozen first configuration

- grid: `64 x 64 x 64`, cell size `100 x 100 x 50 m`;
- trace datum: top of the first subsurface cell in each column;
- air/rock interface: excluded;
- time grid: 320 samples at 8 ms (`0..2552 ms`);
- wavelet: zero-phase 25 Hz Ricker, 128 ms span (17 samples), peak-normalized;
- lateral sampling: all `64 x 64` traces;
- sample mask: all time samples;
- amplitude uncertainty: `0.01` in unscaled convolutional amplitude;
- noise: none;
- codebook: complete synthetic density/velocity table with deliberately
  distinctive label-9 impedance; not site-calibrated petrophysics.

The 8 ms grid covers the maximum truth local-datum two-way time under the
frozen codebook and 3.2 km vertical extent. The builder must fail if any valid
truth interface lies outside the recording interval instead of silently
clipping it. During inference, an incorrect slow prediction may move arrivals
beyond the fixed acquisition window; those predicted arrivals are cropped,
exactly as for a finite-duration record, while the underlying decoded labels
remain unchanged and are still penalized by hard-geology metrics.

## Physics scope

This is more localized than gravity but remains an upper-bound convolutional
model. It includes primary normal-incidence acoustic reflectivity and a fixed
wavelet. It does not model offsets, migration, multiples, mode conversion,
attenuation, anisotropy, elastic amplitudes, lateral ray bending or full-wave
propagation. A successful result would justify controlled degradation; it
would not establish field-seismic validation.

## CPU acceptance gate

Before any GPU guidance, tests must establish:

1. complete, finite and positive density/velocity codebook parsing;
2. exact hard-label and one-hot soft impedance/slowness agreement;
3. constant impedance gives a zero trace;
4. reflectivity polarity and magnitude for a single interface;
5. no lateral cross-talk and correct tensor/trace orientation;
6. greater slowness moves an interface to later time;
7. linear time deposition conserves unconvolved reflection amplitude;
8. zero-padded wavelet convolution has no wraparound;
9. finite-difference directional derivative agrees with autograd away from a
   time-bin crossing;
10. mask and uncertainty normalization are exact;
11. exact-condition overwrite gives zero state gradient at known voxels;
12. deterministic immutable observation and noise hashes;
13. alpha zero reproduces the existing projected fixed-Euler baseline.

The builder must also validate that each column contains one contiguous
subsurface interval below its known surface, every valid interface is inside
the time window, all tensors are finite and the output directory is empty.

## Paired experiment protocol

After the CPU gate and immutable observation build, implement a seismic-only
runner through the existing injected-loss fixed-Euler sampler. Preserve EMA,
sequential CPU initial noise, 32 midpoint steps, explicit alpha-zero branch,
condition projection and strict asset/source hashes. Controller levels must be
pre-registered before the first GPU result.

Start with seed 42 and one sample. Promotion to n=4 requires the same complete
hard geology criteria used in Phases 2-4a, with hard seismic residual replacing
hard gravity residual. Continuous trace loss, soft impedance or visually
plausible wiggle matching cannot substitute for global/class/label-9 and
major-component recovery.

## Stop and promotion rules

- If seismic residual decreases but hard geology fails, report field fitting
  and follow only a pre-registered controller bracket.
- Do not add gravity to rescue a failed seismic-only screen.
- If seismic-only passes n=1 and the frozen n=4 confirmation, then compare
  baseline reranking and proceed to multi-seed evidence.
- Only after seismic-only evidence may a separate gravity-plus-seismic joint
  protocol be frozen. The joint result must report each field residual and the
  complete hard geology, not only a weighted total loss.

## Invariants

- No training, U-Net, embedding or checkpoint modification.
- EMA for all trainable parameters; normal checkpoint frozen embedding.
- Strict fixed-Euler paired baseline and guidance; alpha zero must regress.
- Exact surface and borehole conditions at every step.
- Immutable truth-derived observations built before sampling.
- No claim that the synthetic codebook or inverse-crime response is measured
  petrophysics or field seismic.

## Implementation status

The first CPU/operator and strict-runner units are complete:

- `guidance/seismic.py` implements complete acoustic-codebook parsing,
  hard/soft impedance-slowness mapping, known-subsurface air exclusion,
  local-datum two-way time, linear sample deposition, fixed Ricker convolution,
  finite prediction-window cropping, field loss and deterministic observation
  construction;
- `guidance/seismic_sampling.py` injects seismic loss into the existing
  projected fixed-Euler solver without copying or changing the solver;
- `scripts/stage4/build_seismic_observation.py` creates immutable tensors and
  source/config/tensor hashes;
- `scripts/stage4/run_seismic_guidance.py` validates all immutable assets,
  loads EMA, preserves sequential CPU noise and writes complete hard geology,
  acoustic and seismic evidence;
- `scripts/stage4/audit_seismic_screen.py` enforces strict pairing,
  Phase-2a alpha-zero hard regression and the complete geology-plus-seismic
  gate;
- `experiments/stage4_seismic/configs/seismic_controller_manifest_v1.json`
  freezes alpha/cap 0.25 first and a conditional alpha 0.10 harm diagnostic;
- `tests/test_phase4_seismic.py` passes 15 focused tests. The complete local
  lightweight suite passes 118 tests with 13 existing Matplotlib/pyparsing
  deprecation warnings.

The canonical observation is
`experiments/stage4_seismic/observations/cond_generation_0/distinct_upper_bound_v1_fix2/`.
It is finite with shape `1 x 1 x 64 x 64 x 320`, amplitude range
`-0.4716..0.4931`, 193459 valid internal interfaces and maximum truth TWT
`1428.43 ms` inside the `2552 ms` recording window. Real-checkpoint CPU
one-step alpha-zero and positive-alpha strict-pair smokes complete with EMA,
matching initial noise, finite nonzero seismic gradients and zero post-
projection condition violations. These are code gates, not scientific
evidence.

The frozen seed-42 n=1, 32-step alpha/cap 0.25 GPU strict pair is now complete.
Strict pairing, immutable hashes, EMA, alpha-zero Phase-2a regression and exact
conditions pass, but the complete geology-plus-seismic gate is `0/1`. Hard
seismic RMSE falls from `0.042262` to `0.039048`, while label-9 IoU/recall fall
from `0.02860/0.04728` to `0.02593/0.03947`; all four major truth-body recalls
remain below their baselines. The run passes the paired endpoint-change and
last-step churn checks, so the conditional alpha-0.10 excessive-harm diagnostic
is not authorized. Phase 4c is closed without n=4, additional alpha search or
gravity fusion. See `docs/PHASE4C_REPORT.md`.
