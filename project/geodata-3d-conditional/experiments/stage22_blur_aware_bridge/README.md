# Stage22 blur-aware bounded-TV property bridge

Authorized finite case03 experiment from repository-root
`STAGE22_BLUR_AWARE_BRIDGE_PROMPT_2026-10-16.md`. Reference and reviewed HEAD:
`9b46d1af2d08c32ce256ad18afab716b91c92ada` (main). Initial worktree had only
the untracked user prompt. Old Stage21v2/diagnostic runs, old scripts and tests
are preserved. No training/adapter/real-physics/SDF/parameter scan/commit/push.

The preprocessing process has a truth/object/checkpoint open-file guard. It
loads only saved property observations, fixed codebook/classifier, original
categorical conditions and frozen layout; accessing metadata registry truth
pointers does not open truth tensors. METHOD_FREEZE.json, protocol, input hashes,
Git state and source snapshots are saved before optimization. Only final solver
iterate and its resulting target-only conditions are sealed. Evaluation first
opens full truth after PREPROCESSING_RECEIPT.json exists and verifies its hashes.

Objective: mean squared normalized Gaussian discrepancy +1e-4*TVmean, smooth
isotropic per-channel TV with epsilon1e-8 inside square root. Original Gaussian
float32 coefficients/radius3/separable replicate boundaries are preserved;
float64 CPU arithmetic and autograd implement its actual transpose. Both terms
use identical voxel/channel mean conventions. Full-codebook bounds include air;
hard projection fixes original unblurred lookup properties. Projected accelerated
gradient uses majorization backtracking, monotone restart and final-iterate
selection. Maximum2000 iterations; stop after50 consecutive relative objective
changes<1e-6. No truth-driven numerical/regularization settings or snapshots.

Final properties use original FP32 nonair nearest-codebook distances. Accept only
raw9 argmin, d9<=0.01, margin>=0.05 at old total512 columns outside original
conditions. No inferred non-target/air labels; unknowns remain-1. These are
partial target voxels in columns, never512 new complete wells. All FP candidates
remain. Fixedholdout is excluded by layout, without truth filtering.

After sealing, precision>=.95 and>=5 correct added voxels in object1 allow
seed42 Flow. Original-free recall/precision, original-free one-to-one object1
IoU/coverage, and holdout mIoU apply the exact prompt gates. Only a passing42
permits142/242 using identical conditions. At most3 new predictions /96 calls.
Actual oldA CPU noise files/hashes are verified and loaded, rather than merely
assuming equal seeds imply equal sources. Same EMA/embedding/model hash and
native32-step deterministic FP32 Euler sampler. Full oldA/B512/C512 references
are reused with original prediction pointers. Direct recovered classification
and COPY_ONLY add no network calls. Common object1-outside-added subset tests
whether apparent recovery exceeds writing the new conditions themselves.

Actual commands (SSH labrs-codex, repository root, existing `.venv`):

```bash
P=project/geodata-3d-conditional
R=$P/experiments/stage22_blur_aware_bridge/run_20261016_v1
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:$P .venv/bin/python -m pytest -q "$P/scripts/stage22/test_stage22.py" "$P/tests/test_stage21_virtual_boreholes.py::test_cpu_native_sampler_exact_conditions_noise_and_independent_baseline" "$P/tests/test_phase5c_generator_posterior.py::test_projected_fixed_euler_matches_manual_no_guidance_path"
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python "$P/scripts/stage22/prepare.py" --out "$R"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python "$P/scripts/stage22/workflow.py" --root "$R"
```

Check GPU availability before any authorized Flow step; no other jobs are
interrupted and only one Flow process can be active. Preparation rejects existing
run folders. Completed deliveries are immutable; failed numerical/GPU records
remain for diagnosis, never deleted or silently mixed with valid results.

See run PROTOCOL.json, METHOD_FREEZE.json, PREPROCESSING_RECEIPT.json,
solver_history.csv, OFFLINE_GATE.json, inference_registry.json, EXECUTION.json,
sample/object/predicted-object tables, predictions/ receipts and REPORT.md.
Only a convergence plot and fixed full-XY Z15 comparison (same prior diagnostic
retrospective slice and color scale) are produced. The coefficient1e-4 is a
preselected working hypothesis, not an optimality claim or a literature guarantee.

## Completed result

[Chinese report](run_20261016_v1/REPORT.md),
[convergence](run_20261016_v1/figures/solver_convergence.png),
[fixed slice](run_20261016_v1/figures/fixed_object_slice.png).

CPU optimization converged at550 iterations in39.52s, with zero constraint
errors. Frozen candidate conditions:79 true target voxels in36 columns,0 FP;
36 correct conditions in object1. Only seed42 new Flow ran (32 calls,23.77s
including initialization). It increased object1 TP outside added conditions
0 ->392/536, but greatly overpredicted/merged targets. One-to-one object1
matched IoU/coverage stayed0 and holdout mIoU decreased0.148135 ->0.131576,
failing the allowed0.01-drop guard. Seeds142/242 were not run. Direct restored
classification target IoU0.967723 greatly exceeds Flow0.036278. Stop this
fixed target-hard-projection configuration; later interface work requires a
separate protocol. No method or condition changed after truth evaluation.

Six required/new/legacy basic checks passed. Original A noise/model hashes,
hard conditions and held-out columns were checked. Final reviewed conclusions
are retained separately in REVIEWED_CONCLUSIONS.md; no numeric method rerun
or prediction changes accompanied this report review.
