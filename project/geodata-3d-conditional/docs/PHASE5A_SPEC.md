# Phase 5a specification: no-training acoustic inversion bridge

Date frozen: 2026-07-31, before evaluating the Phase-5a inversion outputs
against the unconstrained truth volume.

## Question and boundary

Phase 4c/4d showed that direct global seismic guidance and post-hoc seismic
ranking can reduce or order trace residuals without recovering the desired
hard geology. Phase 5a asks a narrower question: can the same immutable
synthetic seismic observation be converted into a useful uncertain 3-D
acoustic property volume before it is presented to the frozen flow model?

This is a no-training bridge experiment. It must not modify or load the U-Net,
training loop, EMA state or checkpoint parameters. It does not run guided
sampling and cannot by itself establish geological recovery.

## Immutable inputs

- the Phase-4c `distinct_upper_bound_v1_fix2` full-cube, noiseless,
  inverse-crime convolutional seismic observation and acoustic codebook;
- exactly the 12 Phase-2a alpha-zero EMA/fixed-Euler samples frozen in Phase
  4d: seeds 42, 142 and 242, sample IDs 0--3;
- the Phase-4c alpha-zero anchor, which must equal seed-42 sample 0;
- the existing surface and borehole tensors.

Every input tensor, source config, checkpoint reference and sample hash must be
validated before inversion. No candidate may be generated, removed or selected
using truth. The full truth geology and `truth_acoustic.pt` are forbidden to
the inversion builder. They may only be opened by the separate audit after the
posterior directory has been completed.

## Frozen inversion v1

Each of the 12 hard geological samples supplies a low-frequency impedance
model and a fixed time-depth mapping:

1. map hard labels to the Phase-4c impedance/slowness codebook;
2. replace decoded air below the known surface by the rock category closest to
   the median rock log-impedance (chosen from the codebook alone);
3. overwrite surface and borehole properties exactly before forward modeling;
4. compute the exact nonlinear Phase-4c response and its residual to the
   observed seismic;
5. solve one deterministic linearized post-stack log-impedance correction on
   the regular time grid,

   `argmin ||G dm - residual||^2 + lambda0 ||dm||^2 + lambda1 ||D dm||^2`,

   where `G = W (0.5 D)`, `W` is the exact saved zero-padded same-length Ricker
   operator and `D` is the forward time difference;
6. sample `dm` at prior-derived cell-centre two-way times, add it to prior
   log-impedance, and clamp only to the non-air codebook impedance range;
7. keep prior slowness fixed in v1, then overwrite exact conditions again;
8. evaluate the updated property volume with the exact nonlinear forward
   operator.

The one frozen operating point is `prior_relative_weight=0.001` and
`vertical_smoothness_relative_weight=0.01`, both multiplied by the mean
diagonal of `G^T G`. These dimensionless values were selected from the wavelet
operator scale only, before truth-property evaluation. There is no alpha or
truth-tuned regularization sweep in Phase 5a.

The 12 inverted members define an empirical posterior. Save every member plus
the mean and population standard deviation of log-impedance and slowness. Do
not convert low variance into a claim of confidence; ensemble spread captures
only sensitivity to this small flow-prior pool.

## Exact-condition and anti-leakage rules

- Known air comes from the immutable subsurface mask; known rock comes only
  from non-air borehole entries.
- Exact property values at those voxels come from the acoustic codebook and
  borehole labels, not from unconstrained truth.
- All posterior members and the posterior mean must have zero condition
  violations.
- The builder records no hard-label or truth-property score.
- The audit refuses incomplete builders or mismatched source/tensor hashes.

## Frozen Phase-5a gate

The audit uses unconstrained truth only after construction. Phase 5a is
eligible for a downstream Phase-2-style property-guidance test only if all of
the following hold:

1. all 12 members and the posterior mean preserve exact conditions;
2. at least 9/12 members reduce exact nonlinear seismic RMSE;
3. at least 9/12 members reduce log-impedance RMSE over unconstrained
   subsurface voxels;
4. posterior-mean log-impedance RMSE is lower than the prior-ensemble-mean
   RMSE over the same voxels;
5. posterior-mean absolute log-impedance error inside unconstrained truth
   label 9 is lower than the prior-ensemble-mean error;
6. saved posterior spread is finite, non-negative and nonzero over the
   unconstrained subsurface.

Nearest-codebook hard labels, complete per-class metrics and label-9 metrics
are descriptive diagnostics at this stage. Continuous property improvement is
not geological success. If Phase 5a passes, the later strictly paired
alpha-zero/property-guided flow experiment must still improve hard-label and
three-dimensional component geometry while preserving exact conditions. If it
fails, a sharpened or blurred version of the same inversion is not authorized;
the next route is a conditioning adapter or geophysics-aware fine-tuning.

## Required outputs

- frozen configuration and immutable source/hash manifest;
- 12 prior and inverted acoustic members;
- posterior mean and population standard deviation;
- per-member exact nonlinear residual diagnostics;
- separate truth audit with all frozen gate checks and descriptive hard-label
  projection metrics;
- tests for the convolution matrix boundary, linear solver, time-depth
  sampling, air cleanup, exact conditions, posterior statistics, anti-leakage
  manifest and non-empty-output refusal.

