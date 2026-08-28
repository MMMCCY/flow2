# Stage18 frozen specification: hard-seismic audit and evidence target semantics

Status: **frozen before Stage18 code execution and scientific runs**  
Frozen date: 2026-08-29

## Scope and authorized override

Stage18 answers only two questions:

1. whether the 30 immutable Stage17C hard geologies are more consistent with
   the frozen Stage17 binary seismic observation; and
2. whether Stage17C target-volume inflation is attributable to treating the
   full-trace score as a constant-positive property target rather than a
   continuous normalized binary-property target.

The user explicitly superseded the initial Stage18 instruction's proposed
`confidence = free_subsurface` intervention.  The authorized arm is:

```python
target_properties = score
confidence = score * free_subsurface.float()
```

**Stage18 tests target semantics while holding spatial confidence weighting
fixed.**  The score is a `continuous normalized binary-property target`, not a
calibrated probability of label 9.

## Historical constraints and prior art

The successful Stage15-H/Stage17C lineage uses complete 320-sample traces,
continuous scores, normalized binary endpoints (`label9=1`, all other
classes=0), and no threshold.  Earlier Stage15-H property-bridge attempts v1
and v2 were suppressed because unresolved background dominated the property
loss.  Their historical lesson is binding here: Stage18 does not re-expand
the weight of low-score background voxels.  Keeping
`confidence = score * free_subsurface` preserves the successful spatial
confidence weighting while changing only its target interpretation.

Stage15-G already used a continuous coarse-occupancy BCE target.  Continuous
target semantics are therefore not a wholly new mechanism.  That evidence was
weak, low-resolution and produced severe fragmentation.  Stage18's new value
is strict same-evidence attribution using full-trace evidence that already
passed the Stage17A five-case specificity gate.

The fixed 0.5 Stage15-H core is retrospective/display-only.  Stage15 consensus
thresholds 0.8/0.2 belong to another branch.  Stage18 forbids thresholding,
including 0.6, and forbids score rescaling, case normalization, temperature,
alpha, cap or schedule sweeps.

## Stage18A: immutable Stage17C hard-seismic audit

No Flow sampling is run.  All five cases, seeds 42/142/242 and both immutable
Stage17C arms (`FLOW_ONLY`, `TRAJECTORY_EVIDENCE`) are audited.  Each decoded
geology is mapped to binary occupancy by `decoded_geology == 9`, then evaluated
with `binary_occupancy_to_acoustic` and `seismic_operator_from_config` using
the frozen Stage17 support and observed seismic.

The audit records per sample hard-seismic MSE/RMSE, guided-minus-Flow RMSE,
the existing Stage17C geology metrics, six-connected component statistics,
largest-component fraction and top-4/top-8 component mass.  It also replays
truth binary occupancy through the same operator and records
truth-to-observation hard-seismic closure for every case.

Three seeds are within-case replicates.  Medians are computed inside each case
before the five independent cases are summarized.  Interpretation is limited
to `SURROGATE_OVERSHOOT_SIGNAL`,
`HARD_PHYSICS_COMPATIBLE_BUT_VOLUME_UNCALIBRATED`, or
`MIXED_HARD_PHYSICS_RESPONSE`.

## Stage18B: one new arm

The only new arm is `CONTINUOUS_PROPERTY_TARGET`, producing 15 outputs.  It
uses the exact Stage17A evidence, checkpoint/EMA policy, conditions, source
noise, fixed 32-step Euler solver, controller, alpha/cap, temperature,
schedule, clipping, property sigmas/weights, decoder and binary property table.
Stage17C's 30 outputs are immutable references and are not rerun.

Every new sample saves decoded geology and final continuous state.  At frozen
`tau_end`, `soft_decode_to_probs` supplies label-9 probability.  Both soft mass
and hard label-9 count are computed over the identical free-subsurface mask:

```text
free_subsurface = subsurface & ~condition_mask
M_soft = sum(p9 * free_subsurface)
M_hard = sum((decoded == 9) * free_subsurface)
```

The evaluator reports all required hard-geology metrics, hard seismic,
six-connected components, largest-component fraction, top-4/top-8 component
mass, `M_soft`, `M_hard`, and `M_hard-M_soft`.  Pair validity requires matching
case, evidence tensor, initial noise, checkpoint, condition tensors, solver,
steps and all scientific guidance fields.

The primary comparison is `CONTINUOUS_PROPERTY_TARGET` versus immutable
`TRAJECTORY_EVIDENCE`, while target IoU is also compared with immutable
`FLOW_ONLY`.  The frozen result classes are
`VOLUME_REPAIR_WITH_LOCALIZATION_RETAINED`,
`VOLUME_REPAIR_LOCALIZATION_TRADEOFF`, and `NO_VOLUME_REPAIR`.

## Smoke, formal and stop rule

Focused CPU tests precede CUDA.  CUDA smoke uses only the first registered case
and first seed, but retains all 32 steps and every scientific parameter.  The
formal run is exactly 5 cases x 3 seeds x 1 new arm.

After Stage18A and Stage18B are complete, stop.  No result automatically
authorizes a threshold arm, a threshold sweep, score rescaling, guidance
retuning, hard correction, new decoder, new method family or training.
