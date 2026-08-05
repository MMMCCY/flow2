# Phase-2b codebook-ambiguity seed-42 screen

## Status

**SCREEN RESULT: n=4 bracket identified; not yet confirmed**

This is a full-resolution truth-derived property-codebook ablation. It is not measured geophysics and a one-sample pass is not confirmation.

## Completed levels

| Level | Role | Target exact group | Nearest distance | Pair gate | Label-9 IoU / P / R |
|---|---|---|---:|---|---|
| distinct_c100_anchor | implementation_anchor | [9] | 0.676429 | True | 0.4818 / 0.9004 / 0.5089 |
| paired_c100 | ambiguity_sweep | [9] | 0.676429 | True | 0.4260 / 0.8699 / 0.4550 |
| paired_c025 | ambiguity_sweep | [9] | 0.581157 | True | 0.3951 / 0.8632 / 0.4215 |
| paired_c010 | ambiguity_sweep | [9] | 0.386900 | False | 0.2370 / 0.7569 / 0.2566 |
| paired_c004_overlap | ambiguity_sweep | [6, 9] | 0.000000 | False | 0.0355 / 0.1391 / 0.0455 |

## Promotion rule result

- Status: `candidate_identified_not_confirmed`.
- Selected level: `paired_c025`.
- Four-sample bracket: `['paired_c025', 'paired_c010']`.
- Continuous property loss is never sufficient for promotion.
