# Phase 3: spatially degraded 3-D property observations

Date frozen: 2026-07-31, before any Phase-3 GPU run.

## Scientific question

Phase 2a showed that a frozen flow2 prior can use a truth-derived,
full-resolution and noiseless two-channel 3-D property volume to improve hard
geology. Phase 3 asks how much of that result survives when the **spatial
information content** of the same distinct property codebook is reduced.

This remains a synthetic 3-D inversion-surrogate experiment. It is not a
gravity, magnetic or seismic forward experiment, and it is not evidence that
surface or borehole conditions can be removed.

Phase 3 is scientifically independent from Phase 4. It shares an observation
operator interface with Phase 4 so that operator matching, hashing and hard
condition handling have one implementation contract, but Phase-3 and Phase-4
evidence must be reported separately.

## Reciprocal-constraint interpretation

The intended final system is bidirectional:

- broad geophysical observations compensate for sparse borehole coverage;
- the frozen learned geological prior restricts the observation-consistent
  solution set to plausible structures;
- surface, air and borehole values remain exact hard constraints and further
  reduce geophysical non-uniqueness;
- an ensemble represents the remaining conditional ambiguity rather than
  hiding it behind one deterministic reconstruction.

Phase 3 tests the first bridge toward that system: whether an imperfect 3-D
property observation still changes hard geology usefully while the prior and
hard conditions remain active.

## Frozen starting point

All Phase-3 configurations start from the only Phase-2 operating point that
passed the complete multi-seed gate:

- property codebook:
  `experiments/stage2_property/configs/ideal_density_susceptibility_label9_contrast_v1.json`;
- Phase-2a controller: `alpha=0.25`, maximum guidance ratio `0.25`;
- 32-step fixed-Euler midpoint integration;
- seeds 42/142/242, four sequential CPU-generated noise samples per seed for a
  final confirmation;
- the normal frozen categorical embedding and EMA values for all 411 trainable
  entries.

No Phase-2b ambiguous codebook is eligible: `paired_c100` and `paired_c025`
were only 3/4 transition regions and did not pass the frozen promotion rule.

## Observation contract

Let `q(k)` be the complete property codebook, `p_t(k,r)` the existing soft
categorical decoder, and `H` a fixed differentiable spatial observation
operator:

```text
q_t(r)       = sum_k p_t(k,r) q(k)
q_t_known(r) = q_true(r), r in surface/air/borehole conditions
             = q_t(r),    otherwise
y_t          = H(q_t_known)
y_obs        = H(q_true) + epsilon
L_obs        = normalized_weighted_multiscale_mse(y_t, y_obs)
```

`epsilon` is generated once on CPU from the observation configuration and is
added only to the immutable observation. Noise is never re-sampled inside a
guidance step and is never applied to the prediction.

The same deterministic `H` is applied to prediction and truth. Conditioned
properties are overwritten with exact hard values **before** `H`; gradients at
conditioned voxels are therefore zero even when an observation footprint
crosses a borehole or the surface. The state itself is still projected to the
clean embedding before the first step and after every Euler step.

All operator parameters, observation tensors, confidence tensors, noise
tensors, source files and configs receive stable hashes. Generated run
directories are immutable.

## Independent degradation families

Only one family changes relative to the identity anchor in the first screen.
Do not combine failures or tune alpha after observing results.

1. **Identity anchor**: no spatial degradation and no noise. This validates the
   new exact-known-property observation contract. Alpha zero must be bytewise
   identical to the Phase-2a paired baseline with the same seed/sample count.
   The guided trajectory is a new Phase-3 result because the exact property
   overwrite removes finite-temperature leakage around conditions.
2. **Gaussian resolution**: isotropic, replicate-padded blur with sigma 1, 2
   and 4 voxels. The primary Phase-3 working point is selected from this
   family.
3. **Coarse cells**: non-overlapping 3-D average response at factors 2 and 4.
   Comparison occurs on the coarse observation grid; it is not upsampled into
   fake full-resolution data.
4. **Depth confidence**: exponential confidence decay below each column's
   highest non-air voxel, with e-folding depths 16 and 8 voxels. The vertical
   axis is the final tensor axis and larger indices are upward in the current
   data.
5. **Missing support**: fixed, explicit axis-aligned observation blocks remove
   25% and 50% of support. The mask is spatially predeclared and independent of
   truth labels.
6. **Observation noise**: fixed zero-mean Gaussian noise at 2.5% and 5% of each
   channel's confidence-weighted noiseless-observation standard deviation.

The identity and primary Gaussian configs are implemented first. Other
families may use the same tested operator schema after their exact configs are
committed, without changing the identity/Gaussian results.

## Strict pairing and execution gates

Every positive-alpha run requires a newly generated alpha-zero run with the
same:

- checkpoint, EMA policy, frozen embedding, truth and conditioning tensors;
- property and observation configs and all resolved observation hashes;
- CPU initial-noise sequence, time grid, step count and device string;
- temperature, schedule, controller, alpha cap and gradient clipping;
- target-label audit settings and hard-condition projection policy.

`alpha=0` takes an explicit no-gradient path and must reproduce projected
fixed Euler. Any pairing mismatch, non-finite trace, missing sample, condition
violation or observation-hash mismatch invalidates the pair.

Before GPU sampling, CPU tests must pass for:

- identity, blur and coarse-grid shapes and values;
- zero residual for an exact noiseless target;
- deterministic observation/noise hashes;
- noise applied to observations only;
- finite autograd and zero condition-voxel gradients;
- depth and missing-support confidence behavior;
- alpha-zero fixed-Euler equality and every-step condition projection.

## Screening and promotion

Run the identity anchor at seed 42, one sample. If it passes, run Gaussian
levels in increasing sigma order with seed 42, one sample. The most degraded
single-sample pass and its adjacent harder level enter a frozen seed-42
four-sample bracket. Classification follows Phase 2b:

- 4/4 complete pair gates plus ensemble diversity: confirmed seed-42 pass;
- 1/4 to 3/4: transition region, not a pass;
- 0/4: confirmed failure;
- only 4/4 may proceed unchanged to seeds 142 and 242;
- the Phase-3 primary operating point requires 12/12 pair gates and 3/3
  diversity gates.

Secondary degradation families are attributed independently from identity.
They do not delay Phase 4a gravity-operator development, but the combined
Phase-4b smooth-3-D-plus-gravity experiment waits for a frozen primary
Gaussian working point.

## Hard scientific gate

Retain the complete Phase-2a per-pair geology gate without relaxing thresholds:

1. global accuracy and truth-present fixed-set mIoU improve;
2. observation loss and corresponding hard-observation residual decrease;
3. label-9 IoU, precision and recall improve, with guided precision at least
   0.75 and guided recall/IoU at least 0.30;
4. predicted label-9 volume is 35%-120% of truth;
5. at least five of eight truth-present non-air classes improve IoU;
6. all four major truth label-9 components have recall at least 0.25 and mean
   recall at least 0.40;
7. components of at most five voxels contain at most 10% of target mass, the
   eight largest contain at least 75%, and final-step hard churn is at most
   1.5% of the full model;
8. hard conditions are exact and ensemble diversity remains non-zero.

Continuous loss reduction, soft probability surfaces, a visually smoother
volume or a larger label-9 voxel count cannot substitute for these gates.

## Interpretation limits

A Phase-3 pass would show tolerance to a specified 3-D spatial degradation.
It would not validate physical property values, acquisition physics, a real
inversion product, field-data generalization or geophysics-only generation.
The distinct Phase-2 codebook still gives label 9 special observability and all
conclusions must state that limitation.

## Completed seed-42 n=1 Gaussian screen

The frozen identity/sigma-1/sigma-2/sigma-4 sequence completed on 2026-07-31.
All valid pairs passed the runtime invariants, including EMA loading, strict
fixed-Euler pairing, exact hard conditions and Phase-2a alpha-zero bytewise
regression. Identity passed the complete hard scientific gate. Every nonzero
Gaussian level failed it.

The guided label-9 IoU declined from 0.4881 at identity to 0.3357, 0.2064 and
0.1026 as sigma increased. Major-body mean recall likewise declined from
0.4984 to 0.3981, 0.2584 and 0.1305. Continuous observation loss decreased at
all levels, demonstrating why that loss cannot be used as a promotion rule.

The aggregate status is `no_nonzero_blur_passed`. Under the frozen screening
rule, identity and its adjacent harder sigma-1 level enter the seed-42 n=4
bracket. Identity n=4 passed all four complete pair gates and its diversity
gate. Its mean accuracy and truth-present mIoU deltas are +0.0420 and +0.0696;
mean guided label-9 IoU/precision/recall are 0.4966/0.9144/0.5211. Sigma 1 then
failed 0/4: every pair failed the target threshold and two also failed major-
component recovery. Its mean guided label-9 IoU/precision/recall are
0.3482/0.6938/0.4120. Sigma 2 and sigma 4 do not receive n=4 runs.

Phase 3 is closed: the undegraded anchor is confirmed, but no nonzero Gaussian
level is promoted. The authoritative interpretation is in
`docs/PHASE3_REPORT.md`.
