# Phase 1: 3D probability-volume guidance

## Objective

Determine whether inference-time differentiable guidance can significantly and
correctly alter the decoded label-9 geometry when supplied with an oracle
three-dimensional target probability volume.

This is an upper-bound mechanism experiment, not a claim of using real
geophysical observations.

## In scope

- Construct a binary label-9 target from the true model.
- Support:
  - all label-9 voxels;
  - largest connected component;
  - explicitly selected connected component.
- Construct one or more smoothed/multiscale 3D target probability volumes.
- Compute differentiable soft class probabilities from the continuous sampling state.
- Add a 3D guidance loss and gradient to the fixed-Euler sampling path.
- Re-project surface and borehole conditions after every guided step.
- Record loss, gradient norms, guidance scaling, condition violations,
  continuous changes and decoded changes.
- Add label-9 and geometric evaluation.
- Preserve strict pairing with the fixed-Euler baseline.

## Out of scope

- Retraining the network.
- Modifying the U-Net architecture.
- Replacing the checkpoint.
- Claiming that the target probability volume is a real geophysical observation.
- Large blind parameter grid searches.
- Combining the new 3D loss with the existing 2D gravity loss before the
  standalone 3D mechanism is verified.

## Required regression checks

- Guidance strength zero reproduces the paired baseline.
- Surface and borehole condition violations remain zero.
- Same seed produces the same initial state.
- Baseline and guided configurations pass strict pairing validation.
- Existing Phase 0 tests remain green.

## Required metrics

- Decoded voxel change count and ratio.
- Global voxel accuracy and mean IoU.
- Label-9 IoU, precision, recall and volume error.
- Label-9 centroid displacement.
- Connected-component count and largest-component size.
- Condition-violation counts.
- Metrics both inside and outside the selected target ROI/component.

## Success gate

Phase 1 succeeds only if the decoded hard-label structure changes materially
and at least one primary label-9/geometric metric improves consistently over
strictly paired baseline samples without breaking conditioning constraints.

Continuous-loss reduction alone is not success.

## Current status

Phase 1 is closed as mechanism validated with topology and endpoint caveats.
Protocol v4 completed 12 strict pairs over seeds 42, 142 and 242. Mean label-9
IoU/precision/recall reached 0.8099/0.8274/0.9747; selected-ROI IoU reached
0.9392; all 12 pairs improved target IoU and centroid distance; all condition
violations were zero. This materially crosses the hard decoder boundary and
satisfies the core success gate.

The pre-registered strong confirmation is not an unqualified full pass. Target
component ratio was 1.2812 versus the 1.25 limit, the analogous ROI diagnostic
was 1.2665, and raw largest-component fraction fell relative to the incorrect
baseline distribution. Size-stratified analysis shows four correct major ROI
bodies in all 12 samples and only 0.66% target mass in fragments of at most
five voxels, but this context does not rewrite the failed raw clauses. A
protocol-v4 fixed-camera render also remains pending. The complete decision and
machine-generated evidence are in `docs/PHASE1_REPORT.md` and
`experiments/stage1_probability/reports/phase1b_v4_12pair/`.
