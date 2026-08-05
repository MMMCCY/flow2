# Phase 6A report: frozen-flow residual adapter oracle smoke

Date completed: 2026-08-01.

## Decision

**ENGINEERING/ORACLE MECHANISM PASS: a 54,327-parameter external residual
adapter can move the frozen EMA conditional flow across hard-label boundaries
and reconstruct the truth-derived acoustic-oracle geology while preserving
every hard condition and every original model tensor.**

This is not a real-geophysics result.  The adapter was trained and evaluated on
one repeatedly inspected truth using a full-resolution truth-derived acoustic
volume and truth-supervised categorical losses.  It establishes only the
learnability and sampler interface needed for a deterministic held-out Phase-6
study.

## Controlled implementation

The original 53-million-parameter U-Net, frozen embedding, EMA weights and
checkpoint are not modified.  The base velocity is computed under `no_grad`
and detached.  AdamW contains only the 54,327 adapter parameters.  The adapter
receives the current state, detached base velocity, sparse condition tensor,
condition mask, normalized two-channel three-dimensional acoustic oracle and
time embedding.  Four dilated residual blocks predict a condition-zero
velocity correction capped at 25% of the base velocity norm.

Training uses normalized residual flow MSE plus all-class balanced endpoint CE,
macro soft Dice and a small residual norm regularizer.  No label-9-specific
head or training loss exists.  Four deterministic cached states at times
`0.2/0.4/0.6/0.8` are repeated for 80 optimizer steps.  The final check uses a
strict seed-42, 32-step fixed-Euler `adapter_scale=0/1` pair.

## Engineering result

All frozen gates pass:

- adapter parameters: `54,327`, below the two-million limit;
- training time on RTX 4090 D: `2.99 s` after cached base-state construction;
- peak allocated CUDA memory: about `1.97 GB`;
- base-model tensor hash before/after: identical;
- base gradients: absent;
- optimizer: adapter parameters only;
- adapter gradient: finite and nonzero;
- historical seed-42 scale-zero hard sample: exact;
- hard-condition violations: zero;
- first/last ten-step mean loss: `0.205246 -> 0.084735`;
- cached mean loss: `0.194786 -> 0.087870`;
- cached endpoint hard accuracy: `0.941159 -> 0.971063`.

## Same-case oracle sample

| Metric | Scale zero baseline | Oracle adapter | Delta |
|---|---:|---:|---:|
| Global voxel accuracy | 0.587366 | 0.745053 | +0.157688 |
| Truth-present mIoU | 0.265249 | 0.473283 | +0.208033 |
| Label-9 IoU | 0.028596 | 0.511914 | +0.483318 |
| Label-9 precision | 0.067484 | 0.833687 | +0.766203 |
| Label-9 recall | 0.047279 | 0.570138 | +0.522859 |
| Four-major-body mean recall | 0.041423 | 0.507007 | +0.465584 |
| Hard seismic loss | 17.860632 | 19.828848 | +1.968216 |

The adapter changes 59,653 hard voxels (`22.76%`) and preserves all hard
conditions.  It learns a strong same-case categorical correction from the
truth-derived oracle, including the previously missing label-9 bodies.

The hard seismic loss worsens despite the geological gains.  That is not an
implementation failure: the frozen smoke intentionally omitted physical loss
until hard categorical learnability was established.  It is nevertheless a
critical warning.  The adapter has not yet learned the final reciprocal target
of being both observation-consistent and geologically accurate.

## What this proves

- The frozen flow is not intrinsically unable to express much better hard
  geology when an external learned velocity correction is available.
- A very small adapter can cross soft-hard embedding boundaries at useful
  scale without back-propagating into or modifying the original U-Net.
- All-class CE/Dice directly address the alignment problem that continuous
  physical losses did not solve in Phase 4/5.
- Exact sparse conditions and scale-zero baseline equivalence can coexist with
  the learned correction.

## What this does not prove

- It does not prove generalization to another geology, well layout, seed,
  petrophysical realization or observation.
- It does not use truth-blind seismic features and is not an inversion.
- It does not prove that a blurred/inverted/acquisition-domain feature contains
  enough information to recover hard lithology.
- It does not establish calibrated uncertainty, diversity preservation or
  physical consistency.
- The large improvements cannot be reported as final Phase-6 performance
  because both the oracle feature and training labels come from the evaluated
  truth.

## Next authorized work

Phase 6 is started, not complete.  The mechanism pass authorizes:

1. deterministic materialized geology/condition/observation manifests;
2. geology-history-grouped train/validation/test splits excluding all current
   demo cases from formal test;
3. a held-out oracle/degraded-feature pilot to test generalization;
4. only after that, a truth-blind post-stack seismic encoder or deterministic
   backprojection input;
5. matched/zero/shuffled controls and a small pre-registered physical
   consistency auxiliary term, with both hard geology and hard seismic gates.

Do not tune this same-case smoke, promote its checkpoint, or claim label-9
success from it.  The formal test remains sealed until the split, evaluator and
source hashes are frozen.

Authoritative output:

- `experiments/stage6_geo_adapter/runs/cond_generation_0/oracle_tiny_overfit_v1/`;
- its `config.json`, `training_trace.csv`, `sample_metrics.csv`, condition and
  component tables, adapter-only checkpoint and source hashes.

