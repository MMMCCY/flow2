# Stage 9: frozen Flow-prior support and posterior evidence

Stage9A is frozen in `docs/STAGE9A_FLOW_PRIOR_SUPPORT_SPEC.md`. It creates
three deterministic StructuralGeo-native benchmark cases, draws exactly 1024
independent samples per case from the unchanged conditional EMA Flow, ranks
the fixed pools by hard seismic RMSE under four observations, and opens truth
only in a separate retrospective auditor.

The primary scientific gate is `stage9a_prior_support_v1`. Engineering smoke
outputs and incomplete staging directories cannot count as evidence. All
prediction caches are float32, losslessly gzip-compressed, and hash-validated
after decompression.

Stage9A stops after writing its machine decision. This directory does not
authorize Stage9B, Stage9C, posterior chains, structured-search changes, or
training.
