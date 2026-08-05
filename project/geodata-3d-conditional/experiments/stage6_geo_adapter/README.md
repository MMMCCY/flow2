# Phase 6 frozen-flow geophysics residual adapter

Phase 6 keeps the original EMA conditional U-Net, embedding and checkpoint
frozen and trains only a small external residual-velocity adapter.  Read
`docs/PHASE6_ADAPTER_SPEC.md` before running any code.

The first `cond_generation_0` experiment is a truth-derived acoustic-oracle
engineering screen.  It may validate gradients, memory, exact baseline
regression and hard-label learnability, but it is neither held-out evidence nor
real seismic inversion.

```text
configs/       frozen adapter smoke settings
runs/          adapter-only checkpoints and training/sampling outputs
reports/       later independent audits
splits/        future immutable grouped data manifests
```

## Phase 6A oracle smoke status

The frozen 80-step `oracle_tiny_overfit_v1` run passes all engineering and
same-case oracle mechanism gates.  The original model hash is unchanged and
hard conditions are exact.  Hard geology improves strongly, but hard seismic
loss worsens, so this is only an adapter learnability result.  Read
`docs/PHASE6A_REPORT.md`; the next task is deterministic grouped data and a
held-out oracle pilot, not tuning this legacy case.

## Phase 6P inference-limit audit

Before formal adapter training, `physics_attainment_seismic_endpoint_v1` tests
whether strong truth-blind endpoint physics fitting can make the frozen
network's final hard seismic response approach the Phase-4C observation.  The
optimizer selects by hard seismic loss only; a separate auditor evaluates
geology after the run is frozen.  Read `docs/PHASE6P_INFERENCE_LIMIT_SPEC.md`.

Both frozen Phase-6P diagnostics are complete.  The 0.25-to-4.0 trajectory
ladder reaches at most 9.88% hard-seismic RMSE attainment at ratio 1.0, while
stronger levels damage geology and regress physics.  The 200-step endpoint
optimizer reaches only 1.69%; its soft seismic response improves strongly as
its hard response diverges.  All engineering gates pass.  Read
`docs/PHASE6P_REPORT.md`.  Further same-case inference-strength tuning is
closed; formal adapter training still requires explicit user confirmation.
The authoritative endpoint rerun is tagged
`physics_attainment_seismic_endpoint_v1_specfinal`; it bytewise reproduces the
earlier endpoint tensors/trace while recording the final protocol-spec hash.
