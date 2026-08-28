# Stage18 evidence semantics

Stage18 performs an immutable Stage17C hard-seismic audit and one strictly
paired new Flow arm.  The arm changes the property target from constant one to
the existing full-trace score while retaining Stage17C's score-weighted spatial
confidence.  See `docs/STAGE18_SPEC.md`.

Scientific outputs are written only to new empty directories:

- `hard_seismic_audit/formal_v1` for Stage18A;
- `smoke/case01_seed42_v1` for engineering smoke;
- `formal/continuous_property_target_all5_all3_v1` for the 15 new outputs;
- `reports/continuous_property_target_all5_all3_v1` for retrospective results.

Stage17A/Stage17C assets are read-only references.  No threshold or parameter
sweep is authorized.
