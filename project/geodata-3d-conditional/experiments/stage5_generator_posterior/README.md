# Phase 5c direct generator-posterior search

This directory contains the no-training pCN experiment specified in
`docs/PHASE5C_SPEC.md`.  It searches the Gaussian initial-noise space of the
frozen EMA conditional flow and scores hard decoded models with the immutable
Phase-4c convolutional-seismic observation.

The first `cond_generation_0` run is a legacy mechanism screen, not a formal
held-out result.  Sampling and truth audit are separate commands.  All output
directories are append-only and runners refuse to overwrite non-empty paths.

Planned layout:

```text
configs/       frozen pCN settings
runs/          immutable sampler outputs
reports/       separate truth audits
```

## Closed legacy result

The frozen `primary_pilot_v1` chain completed on an RTX 4090 D and accepted
7/8 moves, but its minimum-seismic-loss retained model worsened label-9 IoU,
recall and four-major-body recovery.  The full-dimensional pCN implementation
is closed without a longer chain or parameter sweep.  Read
`docs/PHASE5C_REPORT.md`; the next main direction is Phase 6.
