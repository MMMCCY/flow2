# Stage 10: truth-blind geophysical probability bridge

Stage 10 tests one frozen-inference route from observed seismic to a soft
categorical probability field and then, only after an information gate, to the
existing Phase-1 probability-guidance interface.  The Flow checkpoint,
EMA/raw policy, embedding, 32-step midpoint Euler solver, condition projection
and hard categorical decode are frozen.

The three cases are pre-registered StructuralGeo-native cases
`native_seed20260901`, `native_seed20260902` and `native_seed20260903`.  For
each case, property inversion reuses the Phase-5a model-based Tikhonov method
with the unranked Stage-9A candidates 100--111 as its fixed 12-member prior.
Only inverted log acoustic impedance enters the bridge.  Geological truth and
truth physical properties are not accepted by the builder or Flow runner.

The class likelihood model is frozen in
`configs/petrophysical_class_model.json` before retrospective evaluation.  It
uses registered codebook log-impedance means, one codebook-resolution scale
derived from the median adjacent rock-code separation, and a uniform prior.
It is a synthetic inverse-crime reference model, not site-calibrated
petrophysics.

Run from `project/geodata-3d-conditional` with the repository `.venv`:

```bash
../../.venv/bin/python scripts/stage10/build_probability_bridge.py --device cuda
../../.venv/bin/python scripts/stage10/evaluate_bridge_information.py
../../.venv/bin/python scripts/stage10/finalize_stage10.py
```

The evaluator validates all bridge manifests and hashes before it opens the
retrospective truth assets.  Later scripts may run only when the preceding
machine gate passes.  Existing non-empty outputs are never overwritten.

## Frozen outcome

Stage10-A passed in only one of three cases, so the registered machine action
is `STOP_BRIDGE_NO_GEOPHYSICAL_INFORMATION`.  The correct bridge beat constant
and XY-shuffled controls in all three cases, but it did not beat the cyclic
wrong-case bridge in `native_seed20260901` or `native_seed20260903`.  Per the
frozen rule, no Flow pilot, formal experiment, spatial Flow control, guidance
sweep, or new main-paper figure was run.  See `reports/STAGE10_REPORT.md`.
