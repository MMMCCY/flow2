# Stage 8A-v3 — hard-loss trust-region birth continuation re-gate

## Machine decision

`FAIL_STAGE8A_STOP_BEFORE_STAGE8B`

The unchanged original Stage8A gates were applied first. Stage8B was not run,
and no second algorithmic modification was implemented.

## Frozen protocol and execution

- Ladder: `(0.25, 0.5, 0.75, 1.0)` times the frozen full target size.
- Center, full-size draw, shape, label, orientation, seeds, ranker, beam,
  nonbirth moves, bounds, controls, hard projection/mapping/forward, and hard
  RMSE selection remained frozen from v2.
- Each of 16 arms used exactly 961 hard forwards; total hard forwards were
  15,376.
- Execution: NVIDIA GeForce RTX 4090 D, PyTorch 2.8.0+cu128, CUDA, host LabRS.
- Relevant tests: 43 passed. Full project suite: 212 passed, 13 existing
  warnings.
- Gate hash: `2781042756561e3307f21d3760f8b3993ef59f5a6fd9d0be0f9f4e722e5eabc7`.
- Frozen v2 ranker hash:
  `640c50b1718661293bfcbc862a3ee72561f7ea28da6e84dfac42a555b12a662f`.
- v3 config hash:
  `9633b2704ba2c87881040f2b831069ecf4bdb6bf367b97142d366ab2fb03857a`.
- v3 preflight-manifest hash:
  `24cd63b0fbe3dd160bc40387a0e586395e9dc7a7f55426a59b0476c49b3cf3b2`.
- Frozen v1/R1/v2/R2 artifact-tree hashes were verified unchanged before the
  run and recorded in the preflight validation embedded in the summary.

## Original gate result

Passed: required cases, correct-observation specificity, exact conditions,
truth-blind selection, identical budgets, and unknown-count tolerance.

Failed: analytic hidden IoU, analytic hidden recall, native mean IoU, native
mean recall, and geometry improvement from empty.

The correct arm ranked strictly first against the correct observation in all
four cases, but recovery was far below the unchanged geological thresholds:

| Correct arm | Hard attainment | Hidden IoU | Hidden recall | Smallest-scale improving births | Growth probes | Improving growth probes |
|---|---:|---:|---:|---:|---:|---:|
| analytic_five_body | 0.012281 | 0.040625 | 0.040625 | 1 | 0 | 0 |
| native_seed20260807 | 0.000186 | 0.002123 | 0.002123 | 1 | 0 | 0 |
| native_seed20260808 | 0.000372 | 0.000000 | 0.000000 | 0 | 0 | 0 |
| native_seed20260809 | 0.000158 | 0.000000 | 0.000000 | 1 | 0 | 0 |

Native correct-arm mean hidden IoU and recall were both `0.000707714`.

## Trust-region mechanism diagnostics

Across all case/control arms, 8,270 new-center probes were evaluated at scale
0.25. Ten had `delta_rmse < 0` versus their own parent: three correct-arm
probes, six zero-control probes, and one wrong-case probe. Shape-wise, every
improving correct-arm smallest probe was an ellipsoid; no correct-arm
dike-hemisphere probe improved its parent.

No continuation-growth probe was realized in any arm. Therefore the
hard-RMSE-improving growth-step count is zero and its requested distribution is
empty (`count=0`, all quantiles null). Every branch's maximum evaluated scale
was 0.25.

The trace explains this outcome without a new forward: none of the ten
parent-improving smallest-scale children survived the frozen beam as a parent
of a later proposal. Consequently no active branch encountered a later legal
birth slot from which the controller could allocate scale 0.5. This is an
observed v3 mechanism limitation under the frozen beam/generation structure;
it is not retrospectively repaired or rerun here.

The post-selection v2-to-v3 geometry tables matched 96–144 frozen v2 full
targets per arm. Zero matched target had both a failed v2 full-size step and an
improving v3 scale-0.25 step. Thus the authoritative run provides no observed
full-size-failure/small-size-rescue example for the same matched target.

## Audit artifacts

Each arm contains:

- `trust_region_probes.csv`: center rank/score, parent, scale, full/probe
  geometry, hard RMSE, deltas, condition violations, authorization and
  termination;
- `trust_region_branches.json`: complete branch and per-scale probe records;
- `v2_full_vs_v3_nested_alignment.csv`: post-selection same-full-geometry
  comparison to the frozen v2 run;
- `sensitivity_rankings.json` and `sensitivity_maps/`: unchanged v2 ranker
  state/residual/sensitivity hashes and deterministic rankings;
- `proposal_trace.json`, `selection.json`, selected hard labels/response, and
  retrospective metrics.

All condition violations are zero. Truth is absent from the ranker,
continuation API, proposal selection, continuation decision, and gate inputs;
retrospective truth metrics were opened only after selection files were frozen.

## Stop status

Stage8A-v3 is a frozen negative gate. Stage8B remains blocked. No training,
fine-tuning, LoRA, broader search, added cuboid, larger hard-forward budget,
ladder sweep, rerun, or second algorithmic change was performed.
