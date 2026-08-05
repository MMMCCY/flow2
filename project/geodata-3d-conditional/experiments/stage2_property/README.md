# Phase 2: ideal full-lithology 3-D property volume

Phase 2 is separate from Phase 1 run and figure directories. Its first target
is a full-resolution, noiseless property volume constructed from all truth
labels. It is a controlled oracle/inversion surrogate, not measured
geophysics.

Current implementation state:

- `guidance/property_volume.py` implements explicit full-class property tables,
  hard and soft mappings, matched multi-scale 3-D loss, channel normalization
  and confidence weighting;
- `guidance/property_sampling.py` implements an isolated projected fixed-Euler
  path with an explicit alpha-zero branch and the proven protocol-v4 controller;
- `guidance/property_evaluation.py` records global, per-class, label-9 geometry
  and hard-property metrics, including both the historical dynamic-union mIoU
  and a fixed truth-present-class mIoU;
- `scripts/stage2/run_property_guidance.py` validates strict pairing and writes
  hashed targets/configs, traces, samples, per-class tables and ensemble summaries;
- `tests/test_phase2_property_volume.py` and
  `tests/test_phase2_property_sampling.py` cover the CPU development gate;
- `configs/ideal_distinct_density_proxy_v1.json` is the first controlled scalar
  property codebook;
- `configs/ideal_density_susceptibility_label9_contrast_v1.json` is the second
  controlled observability ablation: it keeps the scalar density proxy and adds
  a complete susceptibility proxy in which label 9 is explicitly distinctive;
- `run_phase2a_ideal_property_smoke.sh` preserves the completed Phase-2a GPU
  path;
- `configs/phase2b_codebook_ambiguity_v1/sweep_manifest.json` freezes the
  Phase-2b anchor and four ambiguity levels;
- `run_phase2b_codebook_screen.sh` is the Phase-2b seed-42 single-sample
  launcher. It creates one new alpha-zero/guided strict pair per level;
- `scripts/stage2/summarize_phase2b_screen.py` audits completed screen levels
  and applies the predeclared promotion rule.
- `run_phase2b_codebook_n4_bracket.sh` and
  `scripts/stage2/summarize_phase2b_n4_bracket.py` preserve the completed frozen
  c025/c010 seed-42 bracket;
- `run_phase2b_codebook_n4_fallback.sh` and
  `scripts/stage2/summarize_phase2b_n4_fallback.py` implement the separately
  frozen post-bracket c100 robustness check.
- `scripts/stage2/visualize_property_guidance.py` validates a completed strict
  Phase-2 pair, renders fixed-camera truth/baseline/guided and paired-change
  figures, renders empirical ensemble label-occurrence surfaces, and exports
  VTK image volumes for PyVista/ParaView.

Large generated artifacts must be stored under `runs/`; derived aggregate
reports belong under `reports/`. Never reuse a Phase-1 baseline for Phase 2.

## First GPU smoke

Run from the repository root in the GPU-enabled environment:

```bash
PYTHON_BIN=/absolute/path/to/python \
  bash project/geodata-3d-conditional/experiments/stage2_property/run_phase2a_ideal_property_smoke.sh
```

If the active `python` already has CUDA PyTorch, omit `PYTHON_BIN`. The default
is one sample, seed 42 and 32 steps. Do not increase `N_SAMPLES` until the
single sample has finite traces, zero condition violations, valid pairing,
useful full-model/per-class hard improvements and acceptable endpoint churn.

The default output is isolated under
`runs/cond_generation_0/ideal_distinct_density_proxy_v1/phase2a_v1/`.
The runner refuses non-empty output directories; use a new `RUN_TAG` after any
failed or intentionally repeated run.

## First scalar smoke result

The strict seed-42, one-sample, 32-step alpha-0.10 pair completed under
`ideal_distinct_density_proxy_v1/phase2a_v1/seed42_n1_s32/`. Pairing passed,
EMA was used, initial-noise and target hashes matched, all 32 trace rows were
finite, and surface/borehole violations were zero.

The scalar property objective was active but did not pass the scientific gate:

- global voxel accuracy changed `0.58737 -> 0.59684` and mean IoU
  `0.19291 -> 0.19527`;
- hard-property loss changed `1.19629 -> 0.98169` (`-17.94%`);
- label-9 IoU changed `0.02860 -> 0.03271`, but recall changed
  `0.04728 -> 0.04472` and predicted volume changed `6283 -> 3691` against
  truth volume `8968`;
- all 4,284 paired hard-label changes were inside the active property
  confidence region, and conditions remained exact.

Therefore lower continuous and hard-property residuals did not establish the
required label-9 geometry improvement. The scalar soft expectation is
under-determined for 15 categories and mostly benefited dominant classes. Do
not scale this configuration to four samples.

## Second observability smoke result

The strict two-channel alpha/cap-0.10 smoke completed under
`ideal_density_susceptibility_label9_contrast_v1/phase2a_v1/seed42_n1_s32/`.
It increased label-9 IoU `0.02860 -> 0.05795`, precision
`0.06748 -> 0.22429`, recall `0.04728 -> 0.07248`, and true positives
`424 -> 650`. Global accuracy increased by `0.01129`, hard-property loss fell
30.06%, and six of eight truth-present classes improved IoU.

This is useful directional evidence, not a pass. Predicted label-9 volume fell
to `2898` against truth `8968`, components increased `37 -> 109`, and the
largest-component fraction decreased. The historical dynamic-union mIoU fell
because four spurious label-8 voxels added a new zero-IoU class to its changing
denominator; fixed truth-present mIoU increased `0.26525 -> 0.27131`. Both
metrics must remain reported.

See `docs/PHASE2_PROGRESS.md` for the full decision record.

## Controller-strength smoke result

The alpha/cap-0.25 pair completed and materially crossed the hard-label
boundary. Label-9 IoU, precision and recall reached 0.4816/0.9005/0.5087;
centroid distance fell from 24.00 to 3.42 voxels; global accuracy and fixed-set
mIoU improved by 0.0410 and 0.0667. All four major truth components were
partially recovered.

Topology remains incomplete: raw components increased to 202, tiny-component
mass is 5.37%, and the four major truth bodies have 37.2%-57.9% recall. The
single sample passes the directional gate, not the final Phase-2 gate.

The evaluator now saves size-stratified topology in `sample_metrics.csv` and
adds `truth_component_recovery.csv` plus paired baseline/delta tables.

## Seed-42 n=4 confirmation result

All eight frozen gates pass. Mean baseline-to-guided changes include global
accuracy `0.59245 -> 0.63412`, fixed-set mIoU `0.26969 -> 0.33847`, and
hard-property loss `1.48448 -> 0.51833`. Mean guided label-9
IoU/precision/recall are 0.4905/0.9116/0.5152. All four samples remain unique;
topology and endpoint gates pass with the documented fragmentation caveat.

## Seed-142 confirmation result

All eight gates pass again. Mean guided label-9 IoU/precision/recall are
0.4735/0.8940/0.5028; fixed-set mIoU improves by 0.0662 and global accuracy by
0.0413. Topology, endpoint, conditions and diversity all pass.

## Final 12-pair confirmation result

Seed 242 also passes. Across seeds 42/142/242, all 12 per-pair gates and all
three seed diversity gates pass. The formal decision is:

**PASS: Phase-2a ideal 3-D property upper bound validated with caveats.**

Rebuild the derived report without changing immutable runs:

```bash
PYTHONPATH=src .venv/bin/python \
  project/geodata-3d-conditional/scripts/stage2/summarize_phase2a.py \
  --overwrite
```

Read `docs/PHASE2A_REPORT.md` and
`reports/phase2a_v1_12pair/REPORT.md`. Phase 2b should test overlapping and
lower-contrast property codebooks before spatial degradation or 2-D physics.

## Phase-2b codebook-ambiguity screen

Read `docs/PHASE2B_SPEC.md` before running. The screen changes only the
property codebook; sampler, alpha/cap, confidence, target resolution and strict
pair protocol remain frozen. Start with the implementation anchor only:

```bash
PYTHON_BIN=/absolute/path/to/gpu/python \
LEVEL=distinct_c100_anchor \
bash project/geodata-3d-conditional/experiments/stage2_property/run_phase2b_codebook_screen.sh
```

If the active `python` is the CUDA environment, omit `PYTHON_BIN`. Do not set
`LEVEL=all` before the anchor report passes. Audit the completed anchor with:

```bash
PYTHONPATH=src python \
  project/geodata-3d-conditional/scripts/stage2/summarize_phase2b_screen.py \
  --level distinct_c100_anchor \
  --overwrite
```

The immutable output is stored under
`runs/cond_generation_0/phase2b_codebook_ambiguity_v1/distinct_c100_anchor/`.
The anchor must reproduce the Phase-2a alpha-zero hard sample exactly and stay
within the frozen guided hard/metric tolerances. Only after that audit should
the four ambiguity levels run in their manifest order.

The anchor has now passed: alpha zero is byte-identical to the Phase-2a
reference, the guided repeat differs at only 8 hard voxels, and the complete
single-pair geology/topology gate passes. `paired_c100` subsequently also
passes while reducing the codebook to nine unique vectors. `paired_c025` also
passes, narrowly at the major-component mean/minimum gates. The next permitted
level `paired_c010` subsequently fails the target/component/topology gates.
Complete the predeclared exact-collision control with:

```bash
PYTHON_BIN=/absolute/path/to/gpu/python \
LEVEL=paired_c004_overlap \
bash project/geodata-3d-conditional/experiments/stage2_property/run_phase2b_codebook_screen.sh
```

The exact-collision control is complete and fails, so the five-level screen is
closed. The frozen promotion result is `paired_c025` plus adjacent
`paired_c010` for a seed-42 four-sample bracket.

Run the passing screen candidate first:

```bash
PYTHON_BIN=/absolute/path/to/gpu/python \
LEVEL=paired_c025 \
bash project/geodata-3d-conditional/experiments/stage2_property/run_phase2b_codebook_n4_bracket.sh
```

Audit completed n=4 levels with:

```bash
PYTHONPATH=src python \
  project/geodata-3d-conditional/scripts/stage2/summarize_phase2b_n4_bracket.py \
  --overwrite
```

The frozen n=4 bracket is complete. `paired_c025` is a 3/4 transition region
and `paired_c010` is a 0/4 confirmed seed-42 failure. Both ensembles remain
diverse, but neither level qualifies for multi-seed testing. Rebuild the final
bracket report with:

```bash
PYTHONPATH=src .venv/bin/python \
  project/geodata-3d-conditional/scripts/stage2/summarize_phase2b_n4_bracket.py \
  --require-both-levels \
  --overwrite
```

The dated post-bracket fallback protocol in `docs/PHASE2B_FOLLOWUP_SPEC.md`
permits one unchanged seed-42 n=4 test of the next higher screened ambiguous
level, `paired_c100`:

```bash
PYTHON_BIN=/absolute/path/to/gpu/python \
bash project/geodata-3d-conditional/experiments/stage2_property/run_phase2b_codebook_n4_fallback.sh
```

The fallback is complete and passes only 3/4 pair gates. Sample 2 misses the
major-component minimum-recall gate (`0.2313 < 0.25`); all other pair gates and
the diversity gate pass. Do not run seeds 142/242. Phase 2b is closed in
`docs/PHASE2B_REPORT.md`; the next experiment must separately pre-register
Phase-3 spatial degradation from the 12/12-validated distinct Phase-2a target.

Phase-2 paper-style visual results are stored under `figures/`. Each figure
directory includes three PNG comparisons, VTK volumes and a manifest recording
the strict pair, selected sample and metrics.
