# Phase-2a 12-pair ideal-property confirmation

## Decision

**PASS: Phase-2a ideal 3-D property upper bound validated with caveats**

This validates a truth-derived, full-resolution two-channel 3-D property upper bound with the frozen EMA model. It does not validate measured geophysics or a 2-D field inversion.

## Strict evidence

- Strict pairs: 12 across seeds [42, 142, 242].
- All per-pair frozen gates pass: True.
- All seed diversity gates pass: True.
- EMA, paired noise/assets, finite traces, exact conditions and confidence locality pass.

## Aggregate hard results

- Global voxel accuracy: `0.5972 -> 0.6381`.
- Truth-present fixed-set mIoU: `0.2771 -> 0.3443`.
- Historical dynamic-union mIoU: `0.1804 -> 0.1980`.
- Hard-property loss: `1.4781 -> 0.5187`.
- Label-9 IoU: `0.0314 -> 0.4808`.
- Label-9 precision: `0.0788 -> 0.9032`.
- Label-9 recall: `0.0520 -> 0.5075`.
- Label-9 centroid distance: `16.9518 -> 3.9825`.

## Class and component findings

- Label 2 is the consistent secondary-class tradeoff: mean IoU delta -0.0129, worse in 12/12 pairs.
- Every pair improves at least five of eight truth-present classes; label 13 remains effectively unrecovered.
- Truth component 1 (4079 voxels): mean guided recall 0.4222, minimum 0.3685.
- Truth component 2 (2192 voxels): mean guided recall 0.6047, minimum 0.5360.
- Truth component 3 (2043 voxels): mean guided recall 0.6138, minimum 0.5091.
- Truth component 4 (627 voxels): mean guided recall 0.3955, minimum 0.3158.
- Tiny-component mass fraction range: 0.0370-0.0806.
- Top-eight target-mass fraction range: 0.8657-0.9436.
- Final hard-churn fraction range: 0.0079-0.0099.

## Caveats and next phase

- The target bodies remain incomplete and more fragmented than truth.
- One repeated guided sample differed by 10 hard voxels across CUDA processes; bitwise determinism is not claimed.
- Phase 2b should first test overlapping/less distinctive property codebooks; Phase 3 can then add resolution, blur, missing regions and noise before any 2-D joint physics experiment.
- Continuous property loss never substitutes for the recorded hard-label and geometry gates.
