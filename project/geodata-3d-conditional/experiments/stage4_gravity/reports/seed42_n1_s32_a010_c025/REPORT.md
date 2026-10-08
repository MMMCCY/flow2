# Phase-4a gravity screen

## Decision

**FAIL: single-sample gravity screen**

This is a truth-derived full-grid synthetic inverse-crime gravity field, not measured geophysics.

## Strict pair

- Pairing and immutable gravity hashes: `True`.
- Phase-2a alpha-zero hard regression: `True`.
- Hard conditions exact: `True`.
- Complete geology-plus-gravity gates: `0/1`.
- Post-hoc baseline reranking: `not_applicable_single_sample`.

## Hard result

- Global accuracy delta: `0.006100`.
- Truth-present mIoU delta: `0.000389`.
- Hard gravity loss delta: `-1612.180664`.
- Hard gravity RMSE delta: `-0.088155` mGal.
- Label-9 IoU / precision / recall: `0.0211` / `0.0649` / `0.0303`.
- Improved truth-present classes: `4.0`.
- Major-component minimum / mean recall: `0.0000` / `0.0280`.
- Final hard churn fraction: `0.001568`.

A lower gravity residual alone is field fitting, not geological recovery.
