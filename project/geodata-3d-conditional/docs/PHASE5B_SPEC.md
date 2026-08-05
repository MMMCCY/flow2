# Phase 5b specification: inversion-posterior flow bridge

Date frozen: 2026-07-31, after the Phase-5a truth audit and before running a
flow trajectory against the inverted property volume.

## Question and stop boundary

Phase 5a produced a truth-blind fixed-12 three-dimensional impedance posterior
that passes its continuous bridge-eligibility gate, while direct nearest-code
hard projections become worse. Phase 5b asks whether the frozen flow prior and
soft categorical path can translate that uncertain impedance evidence into a
better geologically structured hard model.

Only one strictly paired seed-42/sample-0 screen is authorized initially. A
failure blocks other seeds, n=4, additional alpha values, confidence tuning,
regularization tuning and gravity fusion. No training, U-Net or checkpoint
change is permitted.

## Frozen bridge observation

- source: completed Phase-5a `model_based_fixed12_v1` posterior whose separate
  audit has `promoted_to_property_guidance_bridge_test=true`;
- property channel: log acoustic impedance only;
- category codebook: the log of every Phase-4c acoustic impedance, including
  air, in raw-label order `-1..13`;
- target: Phase-5a posterior mean log impedance;
- active region: unconstrained subsurface only;
- uncertainty: Phase-5a population standard deviation of log impedance;
- confidence: `1 / (1 + (std / median_positive_std)^2)`, where the reference
  median is computed only over positive spread in the active region;
- confidence is exactly zero at surface/borehole conditions and outside the
  subsurface;
- no extra spatial blur: `property_sigma=0`, weight `1`.

The bridge builder must use only completed Phase-5a posterior assets and their
hashes. It may read the audit pass/fail bit but not use truth metrics to alter
the target, confidence or codebook.

## Strict pair

Run the existing property sampler in the dedicated
`phase5b_inversion_property_bridge_v1` mode:

- EMA weights and normal frozen embedding;
- seed 42, one sample, 32 midpoint fixed-Euler steps;
- identical checkpoint, conditions, CPU initial noise and time grid;
- alpha-zero baseline first, then alpha/cap `0.25/0.25`;
- tau `0.5 -> 0.1` cosine, guidance starts at 0.25 with windowed sine;
- reference-norm-relative guidance scaling and gradient clip 1.0;
- conditions projected before the first step and after every step.

Alpha zero must reproduce the historical Phase-2a seed-42 sample-0 baseline
exactly. Baseline and guided runs must contain identical bridge tensor hashes
and initial-noise hashes.

## Frozen gate

The single pair passes only if:

1. immutable assets, EMA, pairing and alpha-zero hard regression pass;
2. all post-projection condition violations are zero;
3. hard log-impedance observation loss decreases;
4. global accuracy, truth-present mIoU and label-9 IoU/precision/recall all
   improve over alpha zero;
5. at least five truth-present class IoUs improve;
6. guided label-9 precision/recall/IoU are at least `0.75/0.30/0.30`, and its
   volume is `0.35..1.20` of truth volume;
7. the four largest truth label-9 components have minimum/mean recall at least
   `0.25/0.40`;
8. tiny-component mass fraction is at most `0.10`, top-eight component mass
   fraction at least `0.75`, and final distinct-step hard churn at most `0.015`.

A lower soft or hard property loss alone is a failed screen. A 1/1 pass only
authorizes a separately frozen n=4 confirmation; it is not multi-seed success.

