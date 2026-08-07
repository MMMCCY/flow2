# Stage 7 Report — Observation Specificity and Structured Hard-Geophysics

Decision: **STRUCTURED_HARD_INFERENCE_VALIDATED**

## D7 mechanism closure

| Rank | Mechanism | Support score |
|---:|---|---:|
| 1 | S1_residual_similarity | 0.9208 |
| 2 | S4_categorical_hard_transition_collapse | 0.0340 |
| 3 | S2_jacobian_vjp_projection_collapse | 0.0242 |
| 4 | S3_controller_normalization_cap_collapse | 0.0000 |

## Analytic cuboid controls

| Optimized by | Correct-field attainment | Hidden IoU | Hidden recall | Forward calls |
|---|---:|---:|---:|---:|
| correct | 1.0000 | 1.0000 | 1.0000 | 889 |
| zero | 0.0000 | 0.0000 | 0.0000 | 889 |
| shuffled_xy | 0.0000 | 0.0000 | 0.0000 | 889 |
| wrong_case_observation | -0.4142 | 0.0000 | 0.0000 | 889 |

## StructuralGeo deterministic replicas

| Case | Correct ranks first | Correct-field attainment | Hidden IoU | Hidden recall |
|---|---:|---:|---:|---:|
| native_seed20260807 | True | 0.4500 | 0.9690 | 0.9958 |
| native_seed20260808 | True | 0.2385 | 0.9142 | 0.9160 |
| native_seed20260809 | True | 0.5923 | 0.9874 | 0.9933 |

## Paired comparison

| Method | Correct hard RMSE | Attainment vs BASE flow endpoint | Hidden IoU | Hidden recall |
|---|---:|---:|---:|---:|
| BASE_frozen_flow_sample | 0.0441471 | 0.0000 | 0.0000 | 0.0000 |
| continuous_BASE_PLUS_PHYSICS_best_hard_state | 0.0415255 | 0.0594 | 0.0236 | 0.0352 |
| continuous_BASE_PLUS_PHYSICS_final_state | 0.0415255 | 0.0594 | 0.0236 | 0.0352 |
| structured_hard_geophysics | 0 | 1.0000 | 1.0000 | 1.0000 |
| Q1_oracle_like_structured_reference | 0 | 1.0000 | 1.0000 | 1.0000 |

## Required questions

1. Correct/zero/shuffled similarity is attributed in D7 in this order: S1_residual_similarity, S4_categorical_hard_transition_collapse, S2_jacobian_vjp_projection_collapse, S3_controller_normalization_cap_collapse; identical-state residual/gradient/controller/hard-transition evidence prevents trajectory-state confounding.
2. The dominant specificity-loss location is S1_residual_similarity, with the remaining supported mechanisms explicitly ranked rather than conflated.
3. Structured hard inference restored correct-observation specificity on the analytic benchmark when every arm was scored against the same correct field.
4. Correct-arm hard attainment was 100.000%, compared with the existing full-flow maximum/final 30.643%/11.524%.
5. Its hidden-body IoU/recall were 1.000/1.000; truth metrics were retrospective and never selected proposals.
6. Across 3 deterministic StructuralGeo replicas, the correct arm ranked first against the correct field in 100.0%; mean correct hard attainment was 42.695%.
7. Training is still unnecessary for this bounded structured family.

## Gates

- Cuboid correct-control specificity: `True`
- Cuboid hard-attainment improvement: `True`
- Native replication: `True`
- Training performed: `False`

All proposal acceptance and beam selection used hard observed seismic RMSE only. Geological truth metrics were computed retrospectively.
