# Stage21v2 dense accurate/pseudo virtual wells

Finite frozen-Flow development diagnosis authorized by the user on 2026-10-16.
This is separate from historical Stage21 `run_20261014_v2` (45 completed N9/16/25
predictions). Preserve all historical data and unrelated uncommitted changes.

The full task specification is repository-root
`STAGE21V2_VSCODE_CODEX_PROMPT_2026-10-16.md`; its hash is frozen in each new
run's `PROTOCOL.json`. Fixed development case03, seed42, N512/2048/3840;
PCG64 seed21601 layout has 256 stratified holdout columns and 3831 ordered
eligible additional columns. Codebook is the original Phase2a controlled
density/susceptibility-label9-contrast mapping; sigma=1 replicate Gaussian,
all nonair raw labels0..13, weighted squared nearest-codebook classification.
This is a synthetic separable-property upper-bound diagnostic, not calibrated
petrophysics or a physical inverse result. No training/adapter/physics sweep.

Reused modules: Stage21 common hashing, assets, well masks; its CPU
`initial_noise` and `sample_native`; original EMA runtime; projected fixed-Euler
sampler; full-codebook property mapping and Gaussian; Stage21 voxel metrics,
26-connected IDs and one-to-one object matching. Their global protocols are
unchanged. N9 baselines use original pointers only after a sealed-source,
condition, noise, time-grid, model and environment equivalence audit. Otherwise
only the needed A is recomputed. New CUDA invocations see prepared categorical
conditions only, with strict registry fields and file-open firewalls.

Execute through SSH on `labrs-codex`, repository root, `.venv/bin/python`:

```bash
P=project/geodata-3d-conditional
R=$P/experiments/stage21v2_dense_virtual_boreholes/run_20261016_v1
PYTHONPATH=src:$P .venv/bin/python -m pytest -q "$P/tests/test_stage21v2_dense_virtual_boreholes.py" "$P/tests/test_stage21_virtual_boreholes.py" "$P/tests/test_phase5c_generator_posterior.py"
PYTHONPATH=src .venv/bin/python "$P/scripts/stage21v2/prepare.py" --out "$R"
PYTHONPATH=src .venv/bin/python "$P/scripts/stage21v2/workflow.py" --root "$R"
PYTHONPATH=src .venv/bin/python "$P/scripts/stage21v2/report.py" --root "$R"
PYTHONPATH=src .venv/bin/python "$P/scripts/stage21v2/audit.py" --root "$R"
```

Preparation rejects existing run directories. Workflow can resume interrupted
orchestration by verifying sealed predictions and rebuilding offline evaluation;
failed/incomplete CUDA sample directories are retained and rejected. Diagnose
before choosing a fresh run; never clear a failed sample or overwrite it.
Report also rejects existing report directories. No commit/push is automatic.

The first protocol-valid complete B is the real CUDA check and contributes to
the finite result; no redundant GPU smoke or historical45-output rerun.
Truth-dependent progression occurs only in the external evaluator after each
prediction is sealed; truth never enters a trajectory, sample selection or
the CUDA process. This is explicit development diagnosis, not truth-blind TEST
tuning. The 256 columns contribute to synthetic smoothed attributes, so their
categorical holdout is not an independent observation holdout.

Outputs: frozen protocol/layout/codebook/classifier, split registries and hashed
preparation receipt; immutable per-sample predictions/noise/trace/receipts;
sample/class/object/overlap tables, pseudo-well and whole-volume confusion;
paired differences and explicit execution/skip checklist; three figure types
(fixed-domain dose points, full center slices, confusion/target/object summary);
Chinese report, tests and independent delivery audit. Historical baselines are
not copied or counted as newly sampled outputs.

## Completed delivery (2026-10-16)

[Chinese final report](run_20261016_v1/report/REPORT.md),
[delivery audit](run_20261016_v1/DELIVERY_AUDIT.json),
[protocol](run_20261016_v1/PROTOCOL.json),
[execution checklist](run_20261016_v1/execution_checklist.csv).

34 focused/legacy tests passed. Candidate N512 passed the fixed descriptive
three-seed development confirmation, followed by case01/02 seed42 checks.
10 new Flow outputs / 320 forward calls, 5 reused N9 baseline pointers,
233.98 seconds total new inference time including initialization. Higher
N2048/3840 were not executed under the preregistered first-candidate rule.
Category gains do not establish reliable hidden-object recovery or superiority
over direct whole-property classification. No training, adapter or real physics.

The run additionally retains an offline supplement and its exact source:

```bash
PYTHONPATH=src .venv/bin/python "$R/source_snapshot/postprocess_summary.py"
```

This was run once after report generation and before delivery audit; it appends
whole-pseudo-volume and C-vs-A/D/P comparisons. The final report also includes
reviewed factual conclusions. A retained initial offline path-variable failure
changed no predictions and caused no model calls. Do not rerun this appending
step against the sealed delivery. All frozen inference sources are unchanged.
