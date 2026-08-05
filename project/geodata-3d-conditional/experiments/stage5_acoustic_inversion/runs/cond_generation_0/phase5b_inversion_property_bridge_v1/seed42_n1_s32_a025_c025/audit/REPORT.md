# Phase 5b inversion-property flow bridge

Decision: **FAIL: close the no-training inversion-property flow bridge**

- Alpha-zero mismatch voxels: `0`.
- Hard inversion-observation loss delta: `-0.00068318844`.
- Global accuracy delta: `-0.00021260977`.
- Truth-present mIoU delta: `0.00042271614`.
- Guided label-9 IoU / precision / recall: `0.02890` / `0.06870` / `0.04750`.

A lower continuous or hard property loss alone is not a pass.

## Checks

- PASS — `strict_pairing_and_assets`
- PASS — `alpha_zero_phase2a_hard_regression`
- PASS — `conditions_exact`
- PASS — `hard_inversion_observation_loss_improved`
- FAIL — `primary_directions`
- FAIL — `majority_classes`
- FAIL — `target_thresholds`
- FAIL — `major_component_recovery`
- PASS — `size_stratified_topology`
- PASS — `endpoint_churn`
