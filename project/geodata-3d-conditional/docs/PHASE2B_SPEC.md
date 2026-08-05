# Phase 2b: property-codebook ambiguity and label-9 contrast

Date frozen: 2026-07-30, before any Phase-2b GPU run.

## Objective

Determine how much of the Phase-2a hard-geology improvement survives when the
truth-derived, full-resolution property volume no longer assigns easily
distinguishable property vectors to all lithologies. Phase 2b changes only the
categorical property codebook. It does not yet degrade spatial resolution or
introduce field noise.

Phase 2b remains an inversion-surrogate experiment. It is not measured
petrophysics, gravity, magnetics or proof that surface/borehole conditions can
be removed.

## Frozen invariants

- Frozen training, 3-D U-Net, checkpoint and categorical embedding.
- Normal checkpoint value for the frozen embedding and EMA values for all 411
  trainable entries.
- Strictly paired CPU-generated initial noise and 32-step fixed-Euler midpoint
  integration.
- Explicit alpha-zero no-gradient baseline for every property configuration.
- `alpha=0.25`, maximum guidance ratio `0.25`, the Phase-2a temperature,
  controller, schedule, confidence and three-scale loss settings.
- Full-resolution, noiseless truth-derived target properties.
- Surface, air and borehole conditions projected before sampling and after
  every step; violations must remain zero.
- Phase-1 probability guidance and historical 2-D gravity/magnetic guidance
  remain inactive.

Every configuration receives a new strict baseline. A baseline from another
property configuration cannot be reused because the property table, target
property volume and diagnostics have different hashes.

## Predeclared codebooks

The screen contains one implementation anchor and four ambiguity levels.

| Order | Level | Density codebook | Label-9 susceptibility | Purpose |
|---:|---|---|---:|---|
| 0 | `distinct_c100_anchor` | Phase-2a distinct density | 0.100 | confirm no Phase-2b runner regression |
| 1 | `paired_c100` | paired/overlapping | 0.100 | isolate non-target and density ambiguity |
| 2 | `paired_c025` | paired/overlapping | 0.025 | high-to-medium contrast reduction |
| 3 | `paired_c010` | paired/overlapping | 0.010 | low positive contrast |
| 4 | `paired_c004_overlap` | paired/overlapping | 0.004 | exact label-9/label-6 collision |

The overlapping density channel assigns identical values to pairs of raw
labels. In particular, labels 6 and 9 both receive `0.36`. The non-target
susceptibility background also contains shared values; label 6 receives
`0.004`. Therefore the final level gives labels 6 and 9 exactly the same
two-channel property vector. Label 6 is present in the truth, so this is a
genuine tested ambiguity rather than a collision only with absent classes.

These are controlled relative values. No lithological or calibrated physical
meaning is assigned to their magnitude.

## Seed-42 single-sample screen

Run levels in the table order with seed 42, one sample and 32 steps. The anchor
must run under the new Phase-2b stage/source hashes. Its alpha-zero hard sample
must exactly match the saved Phase-2a alpha-zero reference; the independently
guided hard sample may differ by at most 0.1% of the full volume, and its main
hard metrics must remain within the predeclared report tolerances.

For each ambiguity level, the screen records the complete Phase-2a per-pair
gate without changing its thresholds:

1. global accuracy, fixed truth-present mIoU and label-9 IoU/precision/recall
   improve, while hard-property loss decreases;
2. at least five of eight truth-present non-air classes improve IoU;
3. guided label-9 precision is at least 0.75, recall and IoU at least 0.30,
   and predicted target volume is 35%-120% of truth;
4. all four major truth components have recall at least 0.25 and their mean
   recall is at least 0.40;
5. components of at most five voxels contain at most 10% of target mass and
   the eight largest contain at least 75%;
6. final-step hard churn is at most 1.5% of the full volume;
7. pairing, EMA, finite traces, exact conditions and confidence locality pass.

A one-sample pass is only a screening result. Continuous property loss alone
cannot promote a level.

## Promotion and threshold rule

After all four ambiguity levels are screened, choose the most degraded level
that passes the complete single-pair gate. Promote that level and its adjacent
more-degraded level to a seed-42 four-sample bracket. If the exact-collision
level passes, promote it together with the preceding level. If no ambiguity
level passes, Phase 2b records that the usable threshold lies between the
distinct Phase-2a anchor and `paired_c100`; no alpha retuning is allowed.

For the four-sample bracket:

- `4/4` complete per-pair gates is a confirmed seed-42 pass;
- `1/4` to `3/4` is a transition region, not a pass;
- `0/4` is a confirmed seed-42 failure at the frozen operating point;
- all four guided samples must be unique with non-zero outside-ROI
  disagreement.

Only a `4/4` level can be promoted unchanged to seeds 142 and 242. A final
multi-seed claim requires the same 12/12 per-pair and 3/3 diversity gates used
for Phase 2a. The adjacent failing/transitional level remains the lower
observability bracket and is not hidden.

## Interpretation boundaries

- A monotonic decline is scientifically plausible but is not assumed or
  required; all non-monotonic results remain reported.
- Passing the exact-collision control would mean the generative prior and
  spatial context can choose between property-equivalent lithologies. It would
  not mean the property observation itself identifies label 9.
- Failure at reduced contrast locates a limit of this frozen inference path;
  it is not repaired by post-hoc alpha selection in Phase 2b.
- Blur, downsampling, missing regions, depth attenuation and noise remain
  Phase 3. Two-dimensional forward physics remains later work.

## Completed bracket result

The frozen seed-42 n=4 bracket is complete. `paired_c025` passes 3/4 pair gates
and is a transition region; `paired_c010` passes 0/4 and is a confirmed
seed-42 failure. Both diversity gates pass. The bracket therefore promotes no
level to multi-seed testing, and these classifications are final.

The separately frozen post-bracket `paired_c100` robustness check is specified
in `docs/PHASE2B_FOLLOWUP_SPEC.md`. It does not modify this protocol or its
result.
