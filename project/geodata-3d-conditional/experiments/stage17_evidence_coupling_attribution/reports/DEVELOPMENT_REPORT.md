# Stage17 development report

## Protocol and lineage

Stage17 uses all five preregistered Full StructuralGeo cases and source seeds
42, 142 and 242. The checkpoint is unchanged at
`561e94bfda770ec41fc4cbed43436a7e2130eef5dfb7e5d666fcefc0724ff94c`.
Synthetic observation config v2 was frozen before observation generation.
After five observations were generated and hashed, evidence config v1 was
frozen at `2e6cfd2e1ea7999f587454db3ed68564b39599592f375a43fdae6a150b9058b0`.
Only after the Stage17A and Stage17B decisions were complete was trajectory-
only coupling config v1 frozen at
`32c53745f74083b8a1bfecf07f6034a85e38234e6ecdc4daba89f0fd7fc704d1`.

Observation v1 stopped on case04 because its truth-nonair mask contains 23
enclosed air-coded cells and violates the seismic operator's contiguous-
column precondition. That failed directory is retained. Version 2 freezes the
deterministic columnwise support fill below the highest non-air cell; filled
cells use the binary background endpoint and remain outside original-nonair
retrospective scoring.

## Stage17A: evidence

The decision is `EVIDENCE_CASE_SPECIFIC`. All 5/5 prevalence-corrected AP-skill
diagonal advantages are positive; their group median is 0.558335. All five
within-case raw AUPRC values exceed prevalence. Raw AUPRC matrices are
auxiliary; AP skill on the common 124,350-voxel mask is primary. The inversion
runner was truth-blind and did not run Flow.

## Stage17B: optimizer calibration

The decision is `DFLOW_ENGINE_ACTIVE_BUT_WEAK_HARD_CONTROL`. Engineering smoke
is explicitly non-scientific and did not tune parameters. The independent
three-seed confirmation shows:

| Optimizer | Oracle gate | Property gate | Oracle median IoU recovery | Property median IoU recovery |
|---|---|---|---:|---:|
| Adam | fail | fail | 0.154868 | -0.013708 |
| LBFGS | pass | fail | 0.267519 | 0.215156 |

The engine is active, source-only, condition-preserving and model-immutable,
but neither optimizer passes both task gates. No pooled compensation is used.
No D-Flow optimizer is selected, and the geophysical D-Flow arm is not run.
This closes only the two frozen optimizer settings for Stage17C, not D-Flow as
a method family.

## Stage17C: same evidence, trajectory coupling

The decision is `TRAJECTORY_COUPLING_POSITIVE` under the frozen case-level
directional rule. All 5/5 independent cases have positive case-median paired
target-IoU deltas; the cross-case median is +0.125051. Three source seeds are
within-case replicates. The pooled 15-pair result is auxiliary and cannot
override the case-level result.

Trajectory coupling reliably converts case-specific evidence into more target
localization and recall, but it overpredicts volume in every case. Auxiliary
pooled medians show target IoU 0.029990 to 0.237611, precision 0.060150 to
0.261352, recall 0.220419 to 0.857719 and centroid distance 16.5940 to 7.5265
voxels, while absolute volume-error fraction worsens 0.564282 to 2.319874.
This supports evidence and trajectory coupling as active bottleneck-relieving
components, but topology/volume calibration remain final-goal bottlenecks.

## Execution and validation

All scientific commands ran through SSH on `172.27.231.254` with the repository
`.venv` and CUDA device 0. The machine has an RTX 4090 D; no conda environment
named `flowtrain` exists, so the repository `.venv` was the validated runtime.

Engineering-only outputs:

- Stage17A smoke: `stage17a/smoke_case01_v1`, completed one case;
- Stage17B smoke: `stage17b/smoke_seed42_v1`, completed eight arms and is
  explicitly ineligible for scientific evidence;
- Stage17C smoke: `stage17c/smoke_case01_seed42_v1`, completed two arms and is
  explicitly ineligible for scientific evidence.

Formal outputs:

- Stage17A: `stage17a/formal_all5_v1`, five cases;
- Stage17B: `stage17b/formal_all3_v1`, 24 arms;
- Stage17C: `stage17c/formal_all5_all3_v1`, 30 arms.

Focused remote regression command:

```bash
../../.venv/bin/python -m pytest -q \
  tests/test_stage17_frozen_design.py tests/test_dflow.py \
  tests/test_stage15_binary_trace_boundary.py \
  tests/test_stage15_binary_seismic_consensus.py \
  tests/test_stage15_inversion_score_probability.py
```

Result: `45 passed, 13 warnings`.

The complete lightweight suite, run from `project/geodata-3d-conditional`,
returned `341 passed, 2 failed, 13 warnings`. Both failures are pre-existing
paper-evidence generated-artifact drift involving
`paper/manifests/figure03_cfm_structured_cuboid_contrast.json`; Stage17 did not
modify the paper tree. An earlier invocation from repository root also failed
collection because `test_dflow.py` expects the project directory on the import
path; the corrected project-directory invocation produced the counts above.
Static compilation passed for every `scripts/stage17/*.py`, and
`git diff --check` passed.

## Stop

Stage17 is complete. No PGDM, DPS, PnP, SGLD, topology method, multiclass
inversion or training was started. Further method work requires explicit
authorization and a new prospectively frozen protocol.
