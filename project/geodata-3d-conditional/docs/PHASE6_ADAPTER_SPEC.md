# Phase 6 specification: frozen-flow geophysics residual adapter

Date frozen for the first engineering/oracle screen: 2026-08-01, before any
Phase-6 CUDA training output is inspected.

## Goal and boundary

Phase 6 tests the first learned route authorized after the inference-only
Phase-4/5 failures.  The original conditional flow remains the geological
prior.  Its U-Net, embedding, EMA values and checkpoint are immutable.  A
small external adapter alone learns a geophysics-conditioned correction to the
flow velocity:

```text
v_total(x_t, t, c, g) = v_EMA(x_t, t, c) + scale * delta_v_adapter(x_t, v_EMA, c, g, t)
```

`v_EMA` is evaluated under `no_grad` and detached.  The optimizer may contain
only adapter parameters.  Adapter checkpoints are stored separately and must
record the original checkpoint hash.

The adapter output is exactly zero at surface/borehole conditions.  Sampling
still uses midpoint fixed Euler and projects clean condition embeddings before
the first step and after every step.  `adapter_scale=0` must exactly reproduce
the existing EMA fixed-Euler baseline without evaluating or adding the
adapter output.

## Why this is different from Phase 4/5

Phase 4 used a global physical gradient to perturb each inference step.  Phase
5 used an inversion-property bridge or direct pCN likelihood without learning
the physical-to-lithology alignment.  Phase 6 learns this alignment from
geological truth while leaving the geological generator frozen.  Its loss
therefore includes categorical endpoint objectives; lower physical or
continuous loss alone cannot pass.

No label receives a privileged output head or loss.  Label 9 remains a legacy
sparse-intrusion audit target only.

## Adapter input and architecture

The first adapter receives:

- current embedded state `x_t`;
- detached EMA velocity `v_EMA`;
- sparse condition tensor `ATb`;
- one-channel hard-condition mask;
- two-channel three-dimensional acoustic feature;
- scalar time through FiLM modulation.

The network is an attention-free residual 3-D CNN with width 12, four residual
blocks and dilations `1/2/4/1`.  Its final 1-by-1 convolution is exactly zero
initialized.  The adapter must remain below two million trainable parameters.
During the first screen, residual velocity is capped at 25% of the detached
EMA velocity norm per sample.

## Loss

Training uses only unconstrained subsurface voxels:

1. normalized residual flow-matching MSE for `v_EMA + delta_v`;
2. all-class balanced cosine-logit cross entropy of
   `x1_hat = x_t + (1-t) * (v_EMA + delta_v)`;
3. all truth-present-class macro soft Dice at the same endpoint estimate;
4. a small residual-velocity norm regularizer.

Frozen first-screen weights are flow `1.0`, CE `0.25`, Dice `0.25`, residual
regularizer `1e-4`.  A physical forward-consistency term is intentionally not
added until the categorical mechanism passes; Phase 4/5 already showed that a
physical loss can dominate while hard geology worsens.

## Phase 6A/B legacy engineering screen

The first run uses `cond_generation_0` and its truth-derived, full-resolution
two-channel acoustic volume as an oracle feature.  This is not deployable
geophysics, validation data or publication evidence.  It only asks whether a
small residual adapter can learn a hard-lithology correction through a frozen
flow.

Frozen settings:

- four cached training states with initial-noise seeds `6100..6103`;
- interpolation times `0.2/0.4/0.6/0.8`;
- 80 optimizer steps, AdamW, learning rate `0.002`, weight decay `1e-4`;
- deterministic seed `6200`;
- one final seed-42, 32-step strict baseline/adapter sampling pair;
- adapter scale `1.0`, residual norm cap `0.25`.

Mandatory engineering gates:

1. model/embedding tensor hash is identical before and after training;
2. every base parameter is frozen, has no gradient, and is absent from the
   optimizer;
3. adapter has nonzero finite gradients and changes from zero initialization;
4. initial zero adapter exactly reproduces the base velocity;
5. adapter output and decoded samples have zero hard-condition violations;
6. `adapter_scale=0` sampling exactly reproduces the historical paired
   baseline;
7. final ten-step mean total loss is lower than the first ten-step mean and
   cached endpoint hard accuracy increases;
8. all outputs, settings and sources have hashes and use a new directory.

The oracle sample is promoted only as an adapter mechanism if truth-present
mIoU improves and label-9 IoU/recall/major-body recall do not simultaneously
regress.  Even a pass authorizes only deterministic data construction and a
held-out oracle pilot; it does not establish real seismic guidance.

## Formal Phase-6 work after the engineering screen

Before any formal training, materialize deterministic geology cases and split
by complete geological realization/history.  Sibling borehole, noise, wavelet,
petrophysical and rotation variants must remain in the same split.  All current
demo cases are excluded from the formal test set.  Train normalization and
petrophysical priors come from train only; the sealed test manifest and
evaluator hash are fixed before training.

The sequence is:

1. held-out oracle/degraded-3-D feature pilot;
2. truth-blind post-stack seismic encoder or deterministic backprojection;
3. matched/zero/shuffled observation controls;
4. property-overlap, wavelet, noise and missing-trace robustness;
5. locked multi-geology, multi-noise fixed-Euler evaluation.

Correct geophysics must improve hard geology beyond sparse-only baseline.
Geology-prior-plus-geophysics must also outperform a geophysics-only control at
comparable physical fit to support the reciprocal non-uniqueness claim.

