# Stage 8A-v4 final standalone structured-search report

## Authoritative machine decision

`FAIL_STAGE8A_STOP_BEFORE_STAGE8B`

The unchanged original Stage8A gates were applied. Standalone Stage8A is now
closed regardless of this result. Stage8A-v5 is forbidden and was not
implemented. Stage8B and training were not run.

## Frozen execution

- Device: NVIDIA GeForce RTX 4090 D, CUDA, PyTorch 2.8.0+cu128.
- Arms: four cases times correct/zero/shuffled/wrong-case = 16.
- Hard forwards: exactly 961 per arm and 15,376 total.
- Conditions: zero violations.
- Selection: hard observed seismic RMSE only; truth absent from ranking,
  continuation and selection.
- Retrospective geometry was computed only after each selection file froze.
- Full suite before source freeze: 215 passed, 13 existing warnings.
- Original gate hash:
  `2781042756561e3307f21d3760f8b3993ef59f5a6fd9d0be0f9f4e722e5eabc7`.
- Frozen v2 ranker hash:
  `640c50b1718661293bfcbc862a3ee72561f7ea28da6e84dfac42a555b12a662f`.
- v4 config hash:
  `b47197967dc77c841c2c1288d67f759de67427d0c62399d25a323853f822a8dc`.
- v4 preflight-manifest hash:
  `2172d651107f9a5766374f43de9495ec56286fd7f9a97f17ccf0cf188d57f6d2`.

## Original gate result

Passed:

- all required cases present;
- exact hard conditions;
- identical 961-call budgets;
- truth-blind selection.

Failed:

- correct/control specificity (`2/4` cases strictly correct-first, below the
  unchanged required fraction);
- analytic hidden IoU and recall;
- native mean hidden IoU and recall;
- geometry improvement from empty across every native case;
- unknown-count recovery.

## Correct-arm result

| Case | Hard attainment | Hidden IoU | Hidden recall | Improving scale-0.25 seeds | Growth attempts | Best body count |
|---|---:|---:|---:|---:|---:|---:|
| analytic_five_body | 0 | 0 | 0 | 0 | 0 | 0 |
| native_seed20260807 | 0.000697 | 0.008493 | 0.008493 | 0 | 0 | 1 |
| native_seed20260808 | 0.000729 | 0.010000 | 0.010000 | 0 | 0 | 1 |
| native_seed20260809 | 0 | 0 | 0 | 0 | 0 | 0 |

Native correct-arm mean hidden IoU and recall were both `0.00616419`.

## Lineage continuation and slot accounting

Across all arms:

- initial-empty slots: 16;
- new-center probes: 7,742;
- nonbirth probes: 7,616;
- reallocated growth slots: 2;
- locally improving scale-0.25 seeds: 2;
- continuation attempts: 2;
- successful `0.25 -> 0.50`: 0;
- attempted/successful `0.50 -> 0.75`: 0/0;
- attempted/successful `0.75 -> 1.00`: 0/0.

Both active lineages occurred in zero-observation controls, not correct arms:

- native seed 20260807 zero arm: RMSE
  `0.001533970 -> 0.001532116 -> 0.002647921`; scale 0.25 improved by
  `-1.8543e-06`, scale 0.50 failed by `+0.001115805`, maximum attained scale
  0.25, and the branch did not survive the global beam;
- native seed 20260809 zero arm: RMSE
  `0.001555558 -> 0.001553730 -> 0.002666387`; scale 0.25 improved by
  `-1.8281e-06`, scale 0.50 failed by `+0.001112657`, maximum attained scale
  0.25, and the branch did not survive the global beam.

Thus no correct-arm lineage exercised the authorized continuation mechanism,
and no branch successfully reached scale 0.50.

## Post-run conformance caveat

The v4 seed state used proposal-move metadata
`birth_lineage_new_center`, whereas frozen v3 used
`birth_trust_region_new_center`. The proposal kernel's downstream deterministic
RNG seed hashes the complete parent state record, including this metadata.
Consequently v4 paths can diverge from v3 even before a local lineage is
activated. This explains why the three R3 correct-arm improving seeds were not
reproduced in the v4 trace and is a protocol-conformance limitation relative
to the requirement that continuation be the sole path-changing mechanism.

No rerun or repair was performed: another 16-arm run would exceed the final
authorized standalone iteration and its hard-forward authorization. The
machine gate result above is preserved, but v4 does not provide a clean test of
whether R3's three correct-arm seeds would benefit from lineage-preserving
continuation.

## Terminal project state

Standalone Stage8A development is closed. No Stage8A-v5, broader structured
search, beam change, new shape, ranker change, ladder change or sweep is
authorized. Because the original gate failed, Stage8B was not run.

The next project-level action is a Flow-prior integration decision under a new
explicit authorization: decide whether to design a new integration protocol
around the frozen flow prior and the now-closed standalone evidence, or stop
the post-Stage7 route. It is not another standalone structured-search
iteration.
