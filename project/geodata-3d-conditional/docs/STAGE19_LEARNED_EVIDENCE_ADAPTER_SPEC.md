# Stage19 frozen specification: learned evidence-to-velocity adapter

Status: **frozen before cohort generation and CUDA training**  
Baseline: `main@3134684cb9715a8cc58e3758d8770fde075cdd5d`

Stage19 trains only the external `ResidualVelocityAdapter` while the existing
EMA geological CFM, embedding and checkpoint remain immutable.  Its sole
geophysical input is the unthresholded Stage17A continuous binary impedance
score multiplied by the subsurface mask.  The complete protocol, stop rules,
scientific gates and reporting boundaries are frozen in
`../FLOW2_STAGE19_LEARNED_EVIDENCE_ADAPTER_CODEX_SPEC.md`; the executable
parameters are frozen in `experiments/stage19_learned_evidence_adapter/configs`.

The independent StructuralGeo cohort is generated once from disjoint root-seed
ranges: 64 train cases from 220260001, 8 validation cases from 220270001 and
12 test cases from 220280001.  Eligibility and the nine fixed wells are copied
from `full_complexity_targeted_v1`.  The historical five cases are excluded.

Execution must stop before training unless the validation evidence reuse gate
passes.  Formal training is exactly four epochs and 1024 updates; the epoch-4
checkpoint is used without model selection.  Formal inference contains only
FLOW_ONLY, STAGE18_CONTINUOUS_TARGET, ADAPTER_CORRECT, ADAPTER_ZERO and
ADAPTER_WRONG_CASE, using seeds 42/142/242 and case-first evaluation.

This is a binary, noiseless, inverse-crime mechanism upper bound.  It is not
field seismic inversion, realistic multiclass petrophysics, or a calibrated
probability experiment.
