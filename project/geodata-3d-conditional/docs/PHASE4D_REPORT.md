# Phase 4d final report: seismic identifiability and posterior selection

Date closed: 2026-07-31.

## Decision

**CLOSED NEGATIVE DIAGNOSTIC: the frozen 12-sample prior pool contains no
candidate meeting the geological support gate, and lower convolutional-seismic
loss does not rank label-9 geology in the desired direction.**

Both the population-support and seismic-ranking gates fail. Phase 4d is not
eligible for a larger ensemble, posterior selection, particle inference or a
reopening of the Phase-4c controller search.

## Controlled design

The candidate population was frozen before its Phase-4c seismic scores were
computed. It contains exactly the 12 existing Phase-2a alpha-zero baseline
samples from seeds 42, 142 and 242, four sequential CPU-noise samples per seed.
All source runs use EMA, 32-step midpoint fixed Euler, the canonical checkpoint
and exact surface/borehole conditions. Their tensor hashes match their source
configs, and seed-42 sample 0 exactly matches the completed Phase-4c alpha-zero
anchor.

No sample was generated, altered or removed. Candidates were ranked only by
ascending hard seismic loss under the immutable Phase-4c full-cube noiseless
inverse-crime observation. Truth labels were revealed only after ranking to
audit geology. A separate truth-derived whole-class substitution matrix tested
operator sensitivity and was not used for selection.

## Fixed-pool result

The seismic-selected top candidate is `seed42_sample1`:

| Metric | Selected top 1 | 12-sample mean | Oracle best |
|---|---:|---:|---:|
| Hard seismic RMSE | 0.041072 | not a geology metric | not selected by truth |
| Global accuracy | 0.6122 | 0.5972 | 0.6133 |
| Truth-present mIoU | 0.2759 | 0.2771 | 0.2979 |
| Label-9 IoU | 0.0245 | 0.0314 | 0.0672 |
| Label-9 recall | 0.0329 | 0.0520 | 0.0959 |
| Four-major-body mean recall | 0.0329 | 0.0534 | 0.0963 |
| Four-major-body minimum recall | 0 | — | below the 0.25 gate |

The oracle-best values use truth and are ceilings, not deployable selections.
Even the best candidate in the entire pool is far below the frozen label-9
IoU/precision/recall and major-component thresholds. The support gate is
therefore `0/12`.

The top-three seismic-ranked candidates are also worse than the ensemble mean:

- label-9 IoU: `0.02359` versus `0.03144`;
- label-9 recall: `0.03702` versus `0.05196`;
- four-major-body mean recall: `0.03905` versus `0.05344`.

Hard seismic loss has the following Spearman relationships with geology:

| Geology metric | Spearman rho | One-sided negative p |
|---|---:|---:|
| Global accuracy | -0.2937 | 0.1697 |
| Truth-present mIoU | -0.0420 | 0.4581 |
| Label-9 IoU | +0.5524 | 0.9676 |
| Label-9 recall | +0.5874 | 0.9758 |
| Four-major-body mean recall | +0.5804 | 0.9748 |

The desired sign is negative because lower loss should accompany higher
geological quality. The positive target coefficients show the opposite trend
inside this fixed pool: candidates with lower trace mismatch tend to have worse
label-9 and major-body recovery. These 12 samples are a diagnostic population,
not a general statistical estimate of all prior draws.

## Whole-class substitution sensitivity

For label 9, `8955` unconditioned truth voxels were replaced while the 13 known
conditioned voxels remained exact. Replacing all those voxels with label 12 is
the least visible complete substitution, but still produces hard seismic RMSE
`0.017692`; alternative replacements span approximately `0.01769..0.02501`.

Thus the forward operator is sensitive to removing the true label-9 body as a
whole. However, that isolated signal is smaller than the approximately
`0.041..0.046` full-model RMSE of the prior candidates. Errors in common-class
interfaces, travel time and other lithologies dominate the global score and
can compensate target changes. Whole-class visibility therefore does not imply
local lithology uniqueness or a useful global ranking.

## What Phase 4d proves

- The validated frozen prior pool does not contain a candidate near the
  required label-9/major-component quality.
- Post-hoc seismic reranking cannot recover geology that the proposal pool did
  not generate.
- In this pool, the current global convolutional-seismic loss is actively
  misaligned with target-body quality despite modest alignment with global
  accuracy.
- The Phase-4c failure is not explained only by gradient optimization: the
  same identifiability problem appears in a gradient-free selector.
- Exact hard conditions and deterministic input/source hashes hold throughout.

## What Phase 4d does not prove

- Twelve samples do not prove that the complete learned prior has zero support
  for better geology.
- The result does not prove that every seismic acquisition, elastic likelihood,
  local/structural score or calibrated property model must fail.
- The substitution matrix is a truth-derived operator diagnostic, not measured
  geophysics or an inversion.
- The result does not authorize deleting sparse hard conditions or combining
  failed gravity and seismic arms.

## Decision and next work

Close Phase 4d without expanding the candidate pool or tuning a truth-informed
ranking score. Together, Phases 4c and 4d show both inadequate proposal support
and likelihood/geology misalignment for the current frozen-checkpoint route.
More alpha values, simple reranking and additional seeds do not address those
two causes.

The next main study should be a separately frozen geophysics-aware training or
fine-tuning phase. Its aim is to make the generative proposal distribution
represent geophysically compatible target bodies while retaining the learned
geological prior and exact sparse conditions. Before implementation it must
define train/validation/test separation, how synthetic forward responses enter
conditioning or the loss, property uncertainty, anti-leakage rules, frozen
baseline checkpoints and hard-geology/geophysics acceptance gates. This changes
the current no-training boundary and therefore requires explicit user
authorization.

Authoritative evidence:

- `experiments/stage4_seismic_identifiability/reports/cond_generation_0/fixed12_v1/`;
- the three immutable Phase-2a alpha-zero source ensembles recorded in its
  `manifest.json`;
- the immutable Phase-4c observation and alpha-zero anchor recorded in the
  same manifest.

At closure, Phase-4d plus Phase-4c focused tests pass `20/20`. The complete
local lightweight suite passes `123` tests with 13 existing
Matplotlib/pyparsing deprecation warnings.
