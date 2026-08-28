# Stage17 evidence generalization and coupling attribution

The frozen protocol is `docs/STAGE17_SPEC.md`. This directory contains only
new immutable Stage17 assets and outputs. Existing Stage15/Stage16 results are
read-only regression references.

Run classes are explicit: `engineering_smoke` is engineering evidence only;
`formal_evidence_confirmation`, `optimizer_calibration_confirmation`, and
conditional `formal_coupling_confirmation` are the scientific run classes.

Synthetic observations must be built from the already-frozen observation
generation config before an evidence inversion config containing their exact
hashes is frozen. Every command must target a new empty output directory.

Stage17 is complete. The authoritative decisions are:

- Stage17A: `EVIDENCE_CASE_SPECIFIC`;
- Stage17B: `DFLOW_ENGINE_ACTIVE_BUT_WEAK_HARD_CONTROL`;
- Stage17C: `TRAJECTORY_COUPLING_POSITIVE`, with systematic volume
  overprediction and unresolved topology.

The D-Flow geophysical arm was not run because the property calibration gate
failed independently. See `reports/DEVELOPMENT_REPORT.md` and the evaluation
reports beneath each formal run. Do not continue automatically to another
method family.
