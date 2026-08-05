# Phase 4d: seismic identifiability and posterior selection

Read `docs/PHASE4C_REPORT.md` and `docs/PHASE4D_SPEC.md` first. This stage
does not reopen the closed Phase-4c controller screen.

The diagnostic evaluates exactly 12 existing alpha-zero fixed-Euler prior
samples from the validated Phase-2a seed-42/142/242 baselines. It ranks them by
the immutable Phase-4c hard seismic loss, then reveals truth metrics only to
audit proposal support and selection quality. A separate whole-class truth
substitution matrix tests operator sensitivity. No network trajectory or
geological tensor is changed.

The fixed audit is complete and fails both gates. No candidate reaches the
absolute geology-support thresholds. The seismic-selected top-three mean has
lower label-9 IoU, recall and major-body recall than the full ensemble, and the
corresponding loss correlations have the wrong sign. Phase 4d is closed; do not
expand this pool or tune the ranking score. Read `docs/PHASE4D_REPORT.md`.

The reproducibility command from the repository root is:

```bash
OVERWRITE=1 bash \
  project/geodata-3d-conditional/experiments/stage4_seismic_identifiability/run_phase4d_identifiability.sh
```

The default is CPU. Set `DEVICE=cuda` only as a performance option; this stage
does not load the checkpoint into a model or perform sampling. `OVERWRITE=1`
regenerates only the deterministic Phase-4d report directory; source samples
and immutable observations remain read-only.
