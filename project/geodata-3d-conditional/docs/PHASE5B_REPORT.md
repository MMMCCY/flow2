# Phase 5b final report: inversion-posterior flow bridge

Date closed: 2026-07-31.

## Decision

**CLOSED NEGATIVE: the no-training inversion-property flow bridge fails the
single-pair hard-geology gate and must not advance to n=4, additional seeds or
controller/confidence tuning.**

The formal CUDA run is complete and internally valid. It lowers the hard
log-impedance observation loss, but the improvement does not translate into
meaningful hard-label or three-dimensional label-9 recovery. This is exactly
the failure mode the frozen Phase-5b gate was designed to stop.

## Controlled execution and integrity

The seed-42/sample-0 pair used one sample, 32 midpoint fixed-Euler steps, the
same CPU initial-noise hash, the canonical checkpoint, EMA for all 411
trainable parameters and the normal frozen embedding. The baseline used alpha
zero and the guided run used alpha/cap `0.25/0.25`. Both used the same immutable
Phase-5b log-impedance target and spread-derived confidence.

All integrity checks pass:

- strict property assets, hashes and sampler settings match;
- alpha-zero output differs from the historical Phase-2a seed-42 sample 0 by
  exactly `0` hard voxels;
- surface/borehole condition violations are `0` after every step;
- both configs record CUDA execution and valid EMA coverage;
- final distinct-step churn is `0.001575`, below the `0.015` ceiling.

The negative result is therefore not an EMA, solver-pairing, condition
projection, initial-noise or device failure.

## Continuous/property result

The guidance path is active. There are 24 nonzero guidance steps. Soft
property loss is approximately `32.31` at the first active step and `0.5484`
at the endpoint. The final hard inversion-observation loss changes
`0.552753 -> 0.552070` (`-0.000683`), so that gate passes.

However, hard observation MAE slightly increases
`0.0647596 -> 0.0647797`. The truth-codebook hard-property loss also decreases
slightly (`0.939783 -> 0.939071`) while its MAE increases
`0.107931 -> 0.107996`. These mixed, very small changes are not geological
recovery.

## Hard-geology result

| Metric | Alpha zero | Guided | Delta |
|---|---:|---:|---:|
| Global voxel accuracy | 0.587366 | 0.587153 | -0.000213 |
| Truth-present mIoU | 0.265249 | 0.265672 | +0.000423 |
| Label-9 IoU | 0.028596 | 0.028895 | +0.000299 |
| Label-9 precision | 0.067484 | 0.068699 | +0.001215 |
| Label-9 recall | 0.047279 | 0.047502 | +0.000223 |
| Predicted label-9 voxels | 6283 | 6201 | -82 |
| Label-9 components | 37 | 51 | +14 |

The label-9 changes are numerically positive for IoU/precision/recall but far
below the absolute `0.30/0.75/0.30` IoU/precision/recall requirements. The
volume moves farther from the truth count of 8968: absolute volume-error
fraction worsens `0.2994 -> 0.3085`.

There are 1212 paired hard changes (`0.4623%` of the cube), all inside the
active confidence region. Only 112 voxels move from other labels to label 9,
while 194 leave label 9. Large transitions are dominated by common-class
reassignments, including `4->5` (186), `2->5` (126) and `1->4` (107).

Only 4 of 8 truth-present class IoUs improve, below the required five. Global
accuracy declines and global union-class mIoU declines by `0.01579`.

The four largest truth label-9 body recalls are:

- body 1: `0.04192 -> 0.04266`;
- body 2: `0.00091 -> 0.00046`;
- body 3: unchanged at `0.12286`;
- body 4: unchanged at `0`.

Their guided minimum/mean recall are `0/0.04149`, far below `0.25/0.40`.
Thus the small aggregate label-9 increment does not reconstruct missing
three-dimensional bodies.

## Frozen gate verdict

Pass:

- strict pairing and assets;
- Phase-2a alpha-zero hard regression;
- exact conditions;
- lower hard inversion-observation loss;
- size-stratified topology limits;
- endpoint churn limit.

Fail:

- joint primary directions;
- majority of truth-present classes;
- absolute label-9 thresholds;
- four-major-body recovery.

## Interpretation

Phase 5a showed that model-based inversion can create a non-degenerate 3-D
impedance posterior with lower synthetic trace/property error. Phase 5b now
shows that feeding its mean/spread through the existing soft property guidance
does not make the frozen generator recover the correct lithology or target
geometry. The soft-hard mechanism is active and changes 1212 voxels, so the
main failure is not an inability to cross categorical boundaries. The
available impedance posterior remains too non-unique/misaligned with hard
lithology, and the frozen proposal prior does not supply the missing bodies.

This closes the specific no-training inversion-to-property bridge. It does not
prove that every possible acquisition, multi-attribute inversion or learned
conditioning method must fail. It does prove that further alpha/confidence
tuning on this truth-audited target would violate the protocol and would not
address the demonstrated information/alignment limitation.

## Next research decision

Do not run Phase-5b n=4, seeds 142/242, more alpha values, confidence variants,
extra sharpening/blur or gravity fusion. The next scientifically distinct
option consistent with avoiding full retraining is a small geophysics
conditioning adapter with the original flow U-Net and checkpoint frozen. That
requires a new train/validation/test and anti-leakage protocol plus explicit
authorization because it introduces learned parameters and a training path.

Authoritative evidence:

- `experiments/stage5_acoustic_inversion/runs/cond_generation_0/phase5b_inversion_property_bridge_v1/seed42_n1_s32_a025_c025/`;
- its `audit/summary.json`, `audit/REPORT.md`, class and component tables;
- the immutable Phase-5a posterior and Phase-5b bridge manifests referenced by
  the paired configs.

