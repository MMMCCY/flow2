# Phase 5a: acoustic inversion bridge

This experiment implements the no-training bridge frozen in
`docs/PHASE5A_SPEC.md`. It converts the immutable Phase-4c seismic upper-bound
observation and the exact Phase-4d 12-member alpha-zero prior pool into an
uncertain three-dimensional impedance posterior.

The builder is deliberately truth-blind outside the observed surface and
boreholes. The separate audit opens truth only after a completed posterior has
been written. A Phase-5a pass authorizes a later strictly paired property-
guidance experiment; it is not itself evidence of geological recovery.

Run from `project/geodata-3d-conditional`:

```bash
bash experiments/stage5_acoustic_inversion/run_phase5a_inversion.sh
```

The launcher first builds `outputs/cond_generation_0/model_based_fixed12_v1/`
and then writes the truth audit under its `audit/` subdirectory. Output reuse
is refused unless the explicit builder/auditor overwrite flags are supplied.

## Phase 5b single-pair flow gate

Phase 5a passed only the continuous bridge-eligibility gate; nearest-codebook
hard geology worsened. `docs/PHASE5B_SPEC.md` therefore permits exactly one
seed-42/sample-0 flow screen. Its truth-blind target assets are already built
under
`bridge_observations/cond_generation_0/fixed12_log_impedance_v1/`.

The formal 32-step CUDA strict pair has completed under
`runs/cond_generation_0/phase5b_inversion_property_bridge_v1/seed42_n1_s32_a025_c025/`.
Its frozen audit fails the hard-geology gate despite lower bridge loss. Read
`docs/PHASE5B_REPORT.md`; do not rerun this command or expand to n=4:

```bash
PYTHON_BIN=python DEVICE=cuda \
  bash project/geodata-3d-conditional/experiments/stage5_acoustic_inversion/run_phase5b_bridge_screen.sh
```

The launcher is retained only for reproducibility and will refuse the existing
non-empty run directories. Do not run additional seeds, samples, controller
values, confidence variants or inversion tuning; Phase 5b is closed negative.
