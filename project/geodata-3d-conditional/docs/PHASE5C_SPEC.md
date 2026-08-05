# Phase 5c specification: direct conditional-flow posterior search

Date frozen: 2026-08-01, before any Phase-5c CUDA output is inspected.

## Research question

Phase 5c asks whether the frozen conditional flow can be used directly as a
geological generator prior during seismic inversion.  It is a bounded,
no-training test motivated by Mosser et al. (2020) and Fang et al. (2020).

This is scientifically different from the closed Phase-5b bridge.  Phase 5b
first inverted a three-dimensional impedance volume and then supplied its
posterior moments to soft property guidance.  Phase 5c instead searches the
Gaussian initial-noise space of the complete conditional flow and evaluates
every proposal only after it has passed through the frozen generator:

```text
x0 ~ N(0, I)
  -> frozen EMA conditional flow, fixed Euler
  -> exact surface/borehole projection after every step
  -> hard lithology decode
  -> hard acoustic codebook and seismic forward operator
  -> seismic likelihood
```

No training, U-Net change, embedding change, checkpoint update, truth-derived
probability target, or inversion-property bridge is permitted.

## Posterior and proposal

For fixed conditions `c`, let `G_c(x0)` denote the existing 32-step midpoint
fixed-Euler conditional sampler.  The initial state is Gaussian, so Phase 5c
uses the preconditioned Crank-Nicolson proposal

```text
x0' = sqrt(1 - beta^2) * x0 + beta * xi,  xi ~ N(0, I).
```

The proposal preserves the Gaussian prior.  Its Metropolis acceptance ratio
therefore contains only the hard seismic negative log likelihood:

```text
E(x0) = 0.5 * likelihood_weight * hard_normalized_seismic_MSE(G_c(x0))
accept = min(1, exp(-(E(x0') - E(x0))))
```

The hard likelihood is deliberate.  It removes the soft-decoding mismatch
that affected Phases 4 and 5.  Known surface and borehole acoustic properties
are overwritten exactly before every seismic evaluation.

## Frozen legacy mechanism screen

The first CUDA run is an engineering/mechanism screen only:

- case: `samples/jupyter-demo/cond_generation_0`;
- observation: completed Phase-4c `distinct_upper_bound_v1_fix2`;
- checkpoint policy: canonical EMA U-Net and normal frozen embedding;
- initial latent seed: `42`, which must reproduce the historical Phase-4c
  alpha-zero sample exactly;
- proposal seed: `5501`;
- integrator: 32-step midpoint fixed Euler;
- pCN beta: `0.10`;
- likelihood weight: `1.0`;
- primary pilot: 8 proposals after a one-proposal performance smoke;
- no truth metric may influence a proposal, acceptance decision, beta, chain
  length, or retained-sample selection.

`cond_generation_0` has been repeatedly inspected and is not a held-out test.
It may establish implementation integrity and motivate a new-data study, but
it cannot provide publication-level evidence or tune later Phase-6 losses.

## Separation of sampling and truth audit

The chain runner writes immutable samples, hashes, hard seismic likelihoods,
accept/reject decisions, timings, and configuration.  It does not calculate
truth-based geological metrics.  A separate auditor may open the truth only
after the run status is complete and all recorded hashes validate.

The auditor evaluates the initial sample, every retained state, the final
state, and the minimum-seismic-loss retained state.  The latter is selected
using seismic likelihood only, before truth metrics are calculated.  It is
called the minimum-likelihood retained sample, not a calibrated MAP estimate.

## Integrity gates

All of the following are mandatory:

1. EMA covers every trainable U-Net parameter; the frozen embedding keeps its
   checkpoint value.
2. The initial noise hash and decoded sample exactly match the historical
   seed-42 Phase-4c alpha-zero baseline.
3. The checkpoint, truth, boreholes, observation, source files, random seeds,
   pCN settings and output tensors have recorded SHA-256 hashes.
4. Surface and borehole decoded violations are zero for every evaluated
   proposal and retained state.
5. The sampler uses fixed Euler with condition projection before the first
   step and after every step.
6. Acceptance uniforms and Gaussian innovations come from a separate
   deterministic CPU generator.
7. Outputs are written to a new empty directory; no prior result is
   overwritten.

## Mechanism and promotion gates

The one-proposal smoke must finish on CUDA with finite likelihood and record
runtime and peak memory.  The 8-proposal pilot is considered mechanically
active only if it has at least one accepted move, at least two unique retained
hard models, nonzero hard voxel movement, and zero condition violations.

Promotion beyond this legacy screen additionally requires that the
minimum-seismic-loss retained sample has lower hard seismic loss than the
paired baseline and does not trade that reduction for simultaneous regression
of truth-present mean IoU, target IoU, target recall, and major-body recall.
A lower continuous or hard seismic loss alone is not a pass.

Failure to improve one repeatedly inspected case is not a universal rejection
of pCN.  However, zero movement, unusable acceptance, prohibitive proposal
cost, or another physics-better/geology-worse result closes the current
full-dimensional pCN implementation and sends the project to Phase 6.  No
beta or likelihood-weight sweep is authorized on this truth-audited case.

## Possible next step after a pass

A pass authorizes a separately frozen study using new deterministic,
geology-grouped train/validation/test assets.  It must include multiple chains,
matched/zero/shuffled observations, chain diagnostics, per-case uncertainty,
and hard-geology/geometric evaluation.  It does not authorize field-data
claims, full-waveform inversion claims, or removal of sparse hard conditions.

