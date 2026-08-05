# Phase 4a: acquisition-domain gravity

Read `docs/PHASE4_SPEC.md` and `docs/PHASE3_REPORT.md` first. This directory is
separate from the historical lightweight gravity proxy.

Implemented CPU-development path:

- `guidance/gravity.py`: analytic rectangular-prism vertical attraction,
  full-support zero-padded FFT evaluation, SI geometry and mGal output;
- `scripts/stage4/build_gravity_observation.py`: immutable truth-derived
  observation assets and hashes;
- `configs/density_distinct_label9_upper_bound_v1.json`: explicitly synthetic
  density upper bound;
- `configs/density_label6_label9_collision_v1.json`: mandatory ambiguity
  control;
- `configs/full_grid_noiseless_inverse_crime_v1.json`: first full-grid
  noiseless survey.
- `scripts/stage4/run_gravity_guidance.py`: EMA/fixed-Euler strict pair runner
  consuming immutable observations read-only;
- `scripts/stage4/audit_gravity_screen.py`: alpha-zero regression, pairing,
  complete hard geology and gravity gate;
- `scripts/stage4/rerank_gravity_ensemble.py`: mandatory n=4 post-hoc baseline
  selection comparator;
- `configs/gravity_controller_manifest_v1.json`: pre-registered alpha/cap
  order included in strict pairing hashes.

The Phase-4-focused CPU gate passes 18 tests and the complete lightweight suite
passes 103 tests. A real-checkpoint CPU alpha-zero smoke run completes with EMA,
immutable hashes and zero condition violations. This is code validation, not a
Phase-4 scientific result.

Build the first observation from the repository root after the CPU gate:

```bash
PYTHONPATH=src .venv/bin/python \
  project/geodata-3d-conditional/scripts/stage4/build_gravity_observation.py \
  --truth-model project/geodata-3d-conditional/samples/jupyter-demo/cond_generation_0/true_model.pt \
  --density-config project/geodata-3d-conditional/experiments/stage4_gravity/configs/density_distinct_label9_upper_bound_v1.json \
  --observation-config project/geodata-3d-conditional/experiments/stage4_gravity/configs/full_grid_noiseless_inverse_crime_v1.json \
  --output-dir project/geodata-3d-conditional/experiments/stage4_gravity/observations/cond_generation_0/YOUR_NEW_IMMUTABLE_TAG \
  --device cpu --dtype float64
```

Generated observation directories are immutable and must not be overwritten.
The canonical completed asset is `observations/cond_generation_0/distinct_upper_bound_v1_fix2/`.
The earlier `distinct_upper_bound_v1/` stores a non-contiguous field view
inefficiently. `fix1` predates stable float64 analytic-kernel construction for
the float32 GPU path. Both remain immutable historical assets; current source-
hash validation intentionally rejects them for new paired runs.

## First GPU strict pair

Run only the pre-registered alpha/cap 0.25 seed-42 single-sample pair from the
repository root:

```bash
PYTHON_BIN=/absolute/path/to/gpu/python \
bash project/geodata-3d-conditional/experiments/stage4_gravity/run_phase4a_gravity_screen.sh
```

If the active `python` is the CUDA environment, omit `PYTHON_BIN`. The launcher
creates a fresh alpha-zero baseline and guided result under
`runs/cond_generation_0/phase4a_gravity_v1/seed42_n1_s32_a025_c025/`.

Audit the completed pair:

```bash
PYTHONPATH=src python \
  project/geodata-3d-conditional/scripts/stage4/audit_gravity_screen.py \
  --run-name seed42_n1_s32_a025_c025 \
  --seed 42 --n-samples 1 --n-steps 32 --guided-name alpha025 --overwrite
```

Do not run n=4 unless this audit passes the complete geology-plus-gravity gate.
Do not run alpha 0.10 merely because alpha 0.25 fails: the pre-registered lower
strength is allowed only when gravity residual improves but excessive hard
change harms geology. It is a diagnostic, not an unconstrained tuning loop.

### Seed-42 alpha-0.25 result

The completed `seed42_n1_s32_a025_c025` audit is a valid strict pair but fails
the complete scientific gate (`0/1`). Pairing and immutable hashes, the
Phase-2a alpha-zero hard regression, and every projected surface/borehole
condition pass. Hard gravity RMSE improves by `0.06561 mGal`, global accuracy
improves by `0.00907`, and truth-present mIoU changes by only `0.00011`.
However, label-9 IoU/precision/recall fall from
`0.0286/0.0675/0.0473` to `0.0159/0.0638/0.0207`, and the minimum recall of
the four major truth components remains `0`. Relative to the paired baseline,
`6562` final hard voxels change: `3392` leave label 9 while only `23` enter it.
The guide therefore fits the acquisition field by reallocating density among
non-unique lithologies and damages the target geology. This is not geological
recovery.

That failure satisfies the controller manifest's precondition for the one
allowed lower-strength diagnostic. Run a fresh strict pair with:

```bash
ALPHA=0.10 \
PYTHON_BIN=/absolute/path/to/gpu/python \
bash project/geodata-3d-conditional/experiments/stage4_gravity/run_phase4a_gravity_screen.sh
```

Its default run tag is `seed42_n1_s32_a010_c025`. Audit it with
`--guided-name alpha010`. Do not start n=4 unless this lower-strength pair
passes the same complete gate. No additional alpha search is pre-registered.

### Phase-4a closure

The alpha-0.10 diagnostic is complete and also fails the full gate (`0/1`). It
lowers hard gravity RMSE from `0.95848` to `0.87032 mGal`, while label-9
IoU/precision/recall fall from `0.02860/0.06748/0.04728` to
`0.02111/0.06490/0.03033`. Of `3756` paired hard changes, `2101` leave label 9
and only `9` enter it; major-component minimum recall remains zero.

Phase 4a is therefore closed as a negative controller screen. Do not run the
n=4 launcher, collision control or unregistered alpha values. The result and
its scope are frozen in `docs/PHASE4A_REPORT.md`.
