# Stage 7 Report — Observation Specificity and Structured Hard-Geophysics

Decision: **STRUCTURED_HARD_INFERENCE_PARTIAL_OR_FAILED**

## Required questions

1. Correct/zero/shuffled similarity is attributed in D7 in this order: S1_residual_similarity, S4_categorical_hard_transition_collapse, S2_jacobian_vjp_projection_collapse, S3_controller_normalization_cap_collapse; identical-state residual/gradient/controller/hard-transition evidence prevents trajectory-state confounding.
2. The dominant specificity-loss location is S1_residual_similarity, with the remaining supported mechanisms explicitly ranked rather than conflated.
3. Structured hard inference restored correct-observation specificity on the analytic benchmark when every arm was scored against the same correct field.
4. Correct-arm hard attainment was 100.000%, compared with the existing full-flow maximum/final 30.643%/11.524%.
5. Its hidden-body IoU/recall were 1.000/1.000; truth metrics were retrospective and never selected proposals.
6. Across 3 deterministic StructuralGeo replicas, the correct arm ranked first against the correct field in 100.0%; mean correct hard attainment was 0.000%.
7. Training may now be discussed because structured native-family inference did not clear every frozen gate, but no training was started.

## Gates

- Cuboid correct-control specificity: `True`
- Cuboid hard-attainment improvement: `True`
- Native replication: `True`
- Training performed: `False`

All proposal acceptance and beam selection used hard observed seismic RMSE only. Geological truth metrics were computed retrospectively.
