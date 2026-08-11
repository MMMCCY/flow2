# Stage 14 — GANSim-style geophysical probability guidance pilot

This is a new end-to-end exploratory experiment testing whether the previous bridge-only stop rule was overly conservative. It does not reopen or alter Stage10, Stage12, or Stage13 decisions.

## Decision

`GANSIM_STYLE_GEO_GUIDANCE_NOT_SUPPORTED`

Primary endpoint: overall paired median hidden-label9 IoU change = **-0.031508**; positive case medians = **1/5**.

| case | baseline hidden IoU | guided hidden IoU | median paired delta | hidden recall delta | largest hidden component recall delta | mIoU delta | accuracy delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| fullgeo_case01 | 0.003270 | 0.000000 | -0.003025 | -0.309322 | -0.318627 | -0.002671 | 0.008715 |
| fullgeo_case02 | 0.108540 | 0.015213 | -0.093144 | -0.220927 | -0.218697 | -0.017396 | -0.000656 |
| fullgeo_case03 | 0.007585 | 0.008982 | 0.001386 | -0.010964 | 0.005831 | -0.031209 | -0.002241 |
| fullgeo_case04 | 0.176004 | 0.007106 | -0.166759 | -0.400108 | -0.400553 | -0.020698 | 0.005130 |
| fullgeo_case05 | 0.031951 | 0.000443 | -0.031508 | -0.077767 | -0.008470 | -0.011436 | -0.000488 |

## Decision clauses

- at_least_4_of_5_cases_positive_median_hidden_label9_iou_delta: FAIL
- overall_paired_median_hidden_label9_iou_delta_positive: FAIL
- zero_hard_condition_violations: PASS
- no_catastrophic_global_geology_degradation: PASS

Hard-condition violations across both arms: **0**.

The Stage12B post-seismic P(label9) volumes were consumed as continuous volumes through the frozen Phase1 interface, with no bridge redevelopment, pre-Flow AUPRC gate, probability preprocessing, parameter sweep, training, truth-based tuning, or best-sample selection. The experiment stops at this decision.
