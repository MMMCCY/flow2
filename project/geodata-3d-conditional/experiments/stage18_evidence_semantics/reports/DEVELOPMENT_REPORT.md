# Stage18 development report

## Protocol and repository state

Stage18 was developed on remote `main` at
`551e77bcc4e504a5af956f89b0c8477600887e03`.  The pre-existing untracked
`project/FLOW2_STAGE18_CODEX_INSTRUCTIONS.md` was preserved.  No Stage15 or
Stage17 file/output was modified, moved, rerun or overwritten.

The single Stage18B intervention is:

```python
target_properties = binary_impedance_score
confidence = binary_impedance_score * free_subsurface.float()
```

Thus Stage18 tests target semantics while holding Stage17C spatial confidence
weighting fixed.  It does not threshold or rescale the score.  The score is a
continuous normalized binary-property target, not calibrated `P(label9)`.

The Stage15-H v1/v2 unresolved-background suppression is a binding historical
constraint: Stage18 does not increase low-score background weight.  Stage15-G
continuous coarse-occupancy BCE is acknowledged prior art, but its evidence
was weak, low-resolution and fragmented; Stage18 uses the full-trace evidence
that passed the Stage17A five-case specificity gate.

Key hashes:

- checkpoint: `561e94bfda770ec41fc4cbed43436a7e2130eef5dfb7e5d666fcefc0724ff94c`;
- Stage17 observation/evidence/coupling configs:
  `a44497fe9937bdaa9d8ecbb471308a4a63f8427b85c693653c966518bf8c3112`,
  `2e6cfd2e1ea7999f587454db3ed68564b39599592f375a43fdae6a150b9058b0`,
  `32c53745f74083b8a1bfecf07f6034a85e38234e6ecdc4daba89f0fd7fc704d1`;
- Stage18 config:
  `3676193093f89f5bf87ec8dd72a0a7885c5c62f90d44efd3bff787c4ab05b04b`;
- Stage18 formal run manifest/sample manifest:
  `f98d34401b412913d6804b579e47d912bf66a5c044a5724897aab5f606cf8d54`,
  `d8e3df130c5e2b7403cd490c58cfc8c959b11f41ba04d56419d5e0b7482360d4`.

## Stage18A hard-seismic audit

The audit read all 30 immutable Stage17C decoded geologies and ran no Flow.
It maps only hard label 9 to binary occupancy and reuses the frozen Stage17
binary acoustic endpoints, seismic operator, support and observations.  This
is not a full-lithology or field-seismic claim.

Decision: **SURROGATE_OVERSHOOT_SIGNAL**.

All 5/5 independent cases have positive case-median target-IoU change and
worse case-median hard-seismic RMSE under `TRAJECTORY_EVIDENCE`.  All 5/5 also
retain positive target-volume error.

| Case | Delta target IoU | Delta hard-seismic RMSE | Guided volume-error fraction |
|---|---:|---:|---:|
| case01 | +0.007168 | +0.001550 | 84.8917 |
| case02 | +0.133220 | +0.008958 | 1.5353 |
| case03 | +0.125051 | +0.010432 | 5.7604 |
| case04 | +0.104195 | +0.003890 | 1.4130 |
| case05 | +0.219219 | +0.009669 | 2.3199 |

Truth-to-observation hard-seismic replay closes at maximum RMSE `4.448e-7`
across the five CUDA replays; case01 is exactly zero.  This validates that the
audit is comparing against the intended frozen observation model.

## Stage18B formal result

CUDA smoke completed case01/seed42 with all 32 steps and unchanged scientific
parameters.  Formal then generated exactly 15 new outputs: five cases, seeds
42/142/242, and only `CONTINUOUS_PROPERTY_TARGET`.  All 15 model-state hashes
are unchanged, conditions are exact, final states/soft probabilities are
saved, decoded outputs are unique, and all 15 provenance pairs pass.

Decision: **VOLUME_REPAIR_WITH_LOCALIZATION_RETAINED**.

- case-median absolute volume error improves against Stage17 positive-only in
  5/5 cases;
- case-median target IoU remains above Flow-only in 4/5 cases;
- hard-seismic RMSE is lower than positive-only and Flow-only in 5/5 cases;
- case04 is the explicit localization tradeoff: new IoU is below Flow-only.

| Case | IoU Flow / positive / new | Abs. volume error positive / new | Hard RMSE Flow / positive / new | New components | New largest / top-8 mass |
|---|---|---|---|---:|---|
| case01 | .0043 / .0116 / .0059 | 84.8917 / 60.4833 | .03961 / .04046 / .03797 | 72 | .8488 / .9827 |
| case02 | .1044 / .2445 / .1731 | 1.5353 / .4051 | .08528 / .09372 / .08319 | 853 | .1554 / .4453 |
| case03 | .0057 / .1308 / .0928 | 5.7604 / .9291 | .03232 / .04275 / .02920 | 328 | .7112 / .8531 |
| case04 | .1950 / .2886 / .0536 | 1.4130 / .4502 | .04833 / .05243 / .04118 | 309 | .3770 / .8558 |
| case05 | .0300 / .2492 / .1251 | 2.3199 / .3634 | .03800 / .04758 / .03287 | 406 | .2551 / .6171 |

Across the five case medians, Flow / positive-only / new are:

- target IoU: `0.02999 / 0.24451 / 0.09284`;
- precision: `0.06015 / 0.26135 / 0.14612`;
- recall: `0.22042 / 0.86013 / 0.23198`;
- absolute volume-error fraction: `0.59467 / 2.31987 / 0.45016`;
- centroid distance: `16.594 / 7.232 / 10.868`;
- hard-seismic RMSE: `0.03961 / 0.04758 / 0.03797`;
- truth-present fixed mIoU: `0.20356 / 0.20371 / 0.19664`;
- global voxel accuracy: `0.52462 / 0.51906 / 0.52522`;
- connected components: `49 / 46 / 328`;
- largest-component fraction: `0.8584 / 0.8811 / 0.3770`;
- top-8 component mass: `0.9730 / 0.9926 / 0.8531`.

The volume and physical repair is therefore real under the registered rule,
but topology is substantially more fragmented and fixed truth-present mIoU is
slightly worse.  These metrics do not compensate for one another.

## Soft-hard mass diagnostic

`M_soft` and `M_hard` use the identical `subsurface & ~condition_mask` mask.
Case-median `M_hard-M_soft` is `-12.46`, with all five case medians negative
(`-12.46`, `-148.41`, `-5.63`, `-26.30`, `-6.00`).  Across individual samples
the range is `-195.29..+3.00` voxels.

The hard decoder does not systematically amplify label-9 volume.  Remaining
volume behavior is already present in the final soft state; nearest-embedding
decode is a small, usually volume-reducing adjustment under this experiment.

## Interpretation and boundaries

Holding evidence, confidence weighting, source noise and every guidance/Flow
parameter fixed while changing only the target semantics repairs volume in
5/5 cases and retains localization in 4/5.  This supports the Stage18 causal
interpretation that positive-only target semantics were a principal cause of
Stage17C overprediction under the frozen protocol.  It does not prove they
were the only cause, because fragmentation remains severe and case04 loses the
Flow-only localization advantage.

Stage18A shows that positive-only guidance overshoots the binary observation;
Stage18B's target-only correction then improves hard-seismic RMSE below both
references in every case.  Current evidence therefore does not support
automatically prioritizing a new hard-physics correction.  It also does not
establish full-lithology acoustic consistency, measured-field generalization,
calibrated probabilities, topology fidelity or final-goal success.

## Exact remote validation commands

```bash
../../.venv/bin/python -m py_compile \
  scripts/stage18/*.py tests/test_stage18_evidence_semantics.py

../../.venv/bin/python -m pytest -q \
  tests/test_stage18_evidence_semantics.py \
  tests/test_stage17_frozen_design.py \
  tests/test_dflow.py \
  tests/test_stage15_binary_trace_boundary.py \
  tests/test_stage15_binary_seismic_consensus.py \
  tests/test_stage15_inversion_score_probability.py \
  tests/test_phase2_property_sampling.py
# 61 passed, 13 warnings

../../.venv/bin/python scripts/stage18/audit_stage17c_hard_seismic.py \
  --device cuda \
  --output-dir experiments/stage18_evidence_semantics/hard_seismic_audit/formal_v1

../../.venv/bin/python scripts/stage18/run_continuous_property_target.py \
  --device cuda --smoke \
  --output-dir experiments/stage18_evidence_semantics/smoke/case01_seed42_v1

../../.venv/bin/python scripts/stage18/run_continuous_property_target.py \
  --device cuda \
  --output-dir experiments/stage18_evidence_semantics/formal/continuous_property_target_all5_all3_v1

../../.venv/bin/python scripts/stage18/evaluate_continuous_property_target.py \
  --device cuda \
  --run-dir experiments/stage18_evidence_semantics/formal/continuous_property_target_all5_all3_v1 \
  --stage18a-dir experiments/stage18_evidence_semantics/hard_seismic_audit/formal_v1 \
  --output-dir experiments/stage18_evidence_semantics/reports/continuous_property_target_all5_all3_v1
```

## Stop

Stage18A and Stage18B are complete.  No threshold arm, 0.6 test, parameter
sweep, hard correction, decoder experiment or later method was started.
