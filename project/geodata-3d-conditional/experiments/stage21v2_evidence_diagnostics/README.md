# Stage21v2 continuous evidence: offline diagnostic

Independent CPU-only analysis of sealed Stage21v2 `run_20261016_v1` assets.
No old run file, old `scripts/stage21v2` file or original test is modified.
No Flow checkpoint is opened; no model/adapter/physics run or parameter scan.

Task: determine whether continuous margin9 and raw susceptibility still rank
label9 locations when the frozen nearest-codebook classifier predicts no9.
Each of case03/01/02 is analyzed once, without repeating source seeds. Primary
domain is the registered E_original_free; E_holdout is auxiliary. The two
score directions and fixed5% coverage budget are specified before analysis.
Full-volume26-connected IDs precede domain masking; major full size>=20.
Gaussian sigma1/replicate q9 is truth-derived explanation only, never inference.

[Final Chinese report](run_20261016_v1/REPORT.md) ·
[score table](run_20261016_v1/score_summary.csv) ·
[object table](run_20261016_v1/object_diagnostics.csv).

Run on labrs-codex, repository root, existing `.venv`:

```bash
P=project/geodata-3d-conditional
R=$P/experiments/stage21v2_evidence_diagnostics/run_20261016_v1
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 .venv/bin/python "$P/scripts/stage21v2_diagnostics/test_metrics.py"
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python "$P/scripts/stage21v2_diagnostics/analyze.py" --source "$P/experiments/stage21v2_dense_virtual_boreholes/run_20261016_v1" --out "$R"
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 .venv/bin/python "$P/scripts/stage21v2_diagnostics/finalize.py" --root "$R"
```

Commands above describe the completed immutable delivery; use a fresh output
directory for a separate replay. Analysis rejects existing outputs; finalization
rejects an already finalized report. Saved continuous properties were available
and hash verified, so no asset reconstruction was necessary.

Server sklearn is absent. `metrics.py` uses the strictly equivalent binary
unweighted non-interpolated AP: group equal scores, accumulate recall/precision,
sum recall increments times group precision. Five focused fixtures cover a
constant score (AP=prevalence), tied groups, the documented sklearn example,
all15 nonempty binary labelings of a four-point tied example, and complete
boundary ties in fixedtop5%. No threshold is optimized.

Results support retained continuous localization in case03 and case01, with
loss at the pure-class nearest-codebook interface. Case03's572-voxel zero-well
major object is fully covered by both fixedtop5% sets despite allnegative margin.
Case01 has no zero-well major object; its zero-well components have1/6/3 voxels.
Case02's strong global AP does not rescue its32/41-voxel hidden components at
the fixed5% budget. These are controlled synthetic-property diagnostics, not
evidence of field validity or guaranteed success of any new soft interface.
