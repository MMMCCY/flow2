# Stage21: accurate wells → geophysical virtual sequences → frozen Flow validation

The full research route has three steps. (1) Sample accurate extra full well
sequences at truth-independent nested positions and test native frozen Flow
completion outside every well. (2) Determine geophysical full multicategory
sequence inference, its errors and joint uncertainty. (3) Compare inferred
virtual wells against original-only, co-located accurate wells and prior-only
virtual wells, then forward-evaluate the final hard geology against observations.

The user explicitly authorized **step 1 including all 45 formal outputs** on
2026-10-14. Steps 2/3 and erroneous-well interventions remain unimplemented;
their physics, sequence construction and uncertainty choices are open.

Five existing full StructuralGeo cases are retained without reselection or
label merging. Their registered truth and original conditions must match both
file and canonical tensor hashes. Original 9 wells and original surface remain
exact. Additional 16 XY positions come from a 4×4 grid with PCG64 seed21001:
row-major cells, uniform integer selection from lexicographic non-original
coordinates, then permutation. Its first seven wells make total N16 and all
sixteen make N25. One layout is shared by every case; no truth drives positions.

Checkpoint `conditional-weights.ckpt` must have SHA-256
`561e94bfda770ec41fc4cbed43436a7e2130eef5dfb7e5d666fcefc0724ff94c`.
The established loader applies EMA before parameters are frozen, retaining
the raw checkpoint embedding. The sampler is the existing pure projected
fixed-Euler sampler: 32 midpoint-time evaluations, one velocity evaluation per
step. FP32, no autocast/TF32, deterministic algorithms and CUBLAS `:4096:8`
are shared across all levels. Each seed42/142/242 draws a complete CPU Gaussian
tensor before any projection. The identical tensor is used by all three arms.

Preparation may open truth to collect declared well columns and retain the
registered surface. Values outside each explicit condition mask are -1
placeholders, never inferred from labels. The runner only opens a strict
allowlisted inference registry and condition tensors, with a runtime file-open
firewall against truth and evaluation assets. It never imports the preparer or
evaluator. Evaluation starts only after the completion receipt seals all45
outputs. A separate evaluation registry carries truth pointers and hashes.

The common evaluation domain is `(truth != -1) & ~M25`. All levels share its
hash and denominators. Truth/prediction 26-connected IDs are created on full
target volumes before domain masking, with one-to-one maximum common-domain
IoU matching. Entire matched predicted components enter union/precision.
Major objects have at least20 full-volume voxels. Merge/split overlap edges
require ≥5 common voxels and ≥5% truth-object coverage. Detection diagnostics
add IoU≥0.1 and matched coverage≥0.1. These predeclared diagnostic thresholds
are not scientific success gates. Raw intersections/unions, unmatched objects,
continuous scores, small fragments and every original exposure group are saved.

Undefined denominators produce null/blank and reasons, never artificial0/1.
Metrics first aggregate the3 seeds within each case, then the5 cases. Paired
deltas first subtract identical-seed samples; the separate difference of seed
medians is also saved and explicitly named. Hidden-object statistics use only
valid cases/objects. Five reused cases are the geological units; seeds are
random realization diagnostics, not calibrated uncertainty or independent cases.

All commands run from the repository root on `labrs-codex` via SSH. The existing
`.venv/bin/python` provides PyTorch2.8.0+cu128 and the RTX4090D CUDA environment.
Choose a fresh run ID. Existing directories are never overwritten; only
hash-verified `run.py --resume` can continue interrupted inference. Failed
engineering runs remain failed and require diagnosis/new output directories.

```bash
P=project/geodata-3d-conditional
R=$P/experiments/stage21_virtual_borehole_conditioning/run_20261014_v2
PYTHONPATH=src .venv/bin/python -m pytest -q "$P/tests/test_stage21_virtual_boreholes.py" "$P/tests/test_phase5c_generator_posterior.py"
PYTHONPATH=src .venv/bin/python "$P/scripts/stage21/prepare.py" --out "$R"
PYTHONPATH=src .venv/bin/python "$P/scripts/stage21/run.py" --registry "$R/inference_registry.json" --mode smoke --out "$R/cuda_smoke"
PYTHONPATH=src .venv/bin/python "$P/scripts/stage21/run.py" --registry "$R/inference_registry.json" --mode formal --smoke "$R/cuda_smoke" --out "$R/formal"
PYTHONPATH=src .venv/bin/python "$P/scripts/stage21/evaluate.py" --registry "$R/inference_registry.json" --run "$R/formal" --out "$R/reports_v3"
PYTHONPATH=src .venv/bin/python "$P/scripts/stage21/report.py" --registry "$R/inference_registry.json" --run "$R/formal" --report "$R/reports_v3"
PYTHONPATH=src .venv/bin/python "$P/scripts/stage21/audit.py" --root "$R" --report "$R/reports_v3"
```

Preparation freezes protocol.json, layout.json, inference/evaluation registries,
asset/provenance audits. CUDA smoke uses case03, seed142, three levels plus an
independent32-step N9 baseline regression; none enter formal statistics. Formal
budget is45 outputs/1440 velocity evaluations, with tests/smoke separate.
Reports include all sample/case/object tables, three paired comparisons,
common-domain and pairing audits, and all5 cases at fixed seed142 in two full
domain fixed camera views. No best-seed/case selection, truth ROI, adapter,
guidance, source optimization, training, physics inversion or seed sweep occurs.

Scientific interpretation remains bounded: accurate-well off-well benefit can
motivate studying the virtual-sequence bridge, but neither establishes that
geophysics supplies those sequences nor proves final physical consistency.
No-benefit/nonmonotonic outcomes are retained; neither case count, source seeds,
layout nor thresholds are changed to pursue a positive result. Pretraining
overlap cannot be excluded and this reused cohort is not untouched TEST data.

`run_20261014_v1` retains a pre-sampling initialization failure: an overbroad
data-file guard rejected the SymPy geometry library during Lightning import.
No model calls or scientific results came from it. A focused regression fixes
the distinction between library sources and retrospective experiment geometry.
`run_20261014_v2` uses the unchanged scientific configuration and corrected guard.

The completed45-output delivery is `run_20261014_v2/formal/`; the authoritative
report is [REPORT.md](run_20261014_v2/reports_v3/REPORT.md), with
[independent audit](run_20261014_v2/reports_v3/DELIVERY_AUDIT.json).
`reports_v1` retains a final-manifest path-packaging error. `reports_v2` passed
the audit but had a well-map legend overlap; `reports_v3` moves that legend
outside the data. All three contain identical scientific metrics from the same
45 outputs; no model rerun, layout change or metric threshold change occurred.

Case-first median target IoU is0.0284/0.0494/0.0421 for N9/N16/N25; fixed-class
mIoU is0.1875/0.2183/0.2499. Median within-case same-seed25-9 IoU differences
are positive in5/5 cases (cross-case median +0.0124), but original hidden-major
recovery remains weak:7 objects in3 cases, only1 newly hit, and none satisfying
the predeclared diagnostic detection rule. Off-well categorical benefit does
not establish hidden-body recovery or a successful geophysical bridge.

To verify the existing delivery without rerunning inference:

```bash
R=project/geodata-3d-conditional/experiments/stage21_virtual_borehole_conditioning/run_20261014_v2
PYTHONPATH=src .venv/bin/python project/geodata-3d-conditional/scripts/stage21/audit.py --root "$R" --report "$R/reports_v3"
```
