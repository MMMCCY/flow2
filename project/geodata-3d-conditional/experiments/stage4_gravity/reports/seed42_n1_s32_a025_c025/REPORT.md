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

- Global accuracy delta: `0.009071`.
- Truth-present mIoU delta: `0.000109`.
- Hard gravity loss delta: `-1214.598633`.
- Hard gravity RMSE delta: `-0.065606` mGal.
- Label-9 IoU / precision / recall: `0.0159` / `0.0638` / `0.0207`.
- Improved truth-present classes: `4.0`.
- Major-component minimum / mean recall: `0.0000` / `0.0205`.
- Final hard churn fraction: `0.001881`.

A lower gravity residual alone is field fitting, not geological recovery.
