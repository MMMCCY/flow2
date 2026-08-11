# Stage 11 — Independent Diverse-Geometry Probability-Bridge Validation

## Machine decision

`STOP_BENCHMARK_NOT_DIVERSE`

The registered N=5 benchmark could not be constructed. StructuralGeo produced an
empty registered target event for `stage11_case03`. Cases 01 and 02 were built;
cases 04 and 05 were not attempted after the stop. The case registry was not
modified, N was not reduced, and no replacement case was selected.

## Geometry gate

The full pairwise diversity criterion was **not evaluable** because only 2/5
registered cases were successfully built. The single partial pair is retained in
`audit/geometry_diversity.csv` for forensic provenance and is not treated as a
benchmark result.

## Stage11-A status

Stage11-A was **NOT EXECUTED**. Consequently there is no N×N AUPRC, Brier, or
ROC-AUC transfer matrix; no prior-versus-post result; and no shuffled/constant
control result. Producing those artifacts after the failed geometry gate would
violate the pre-registered stop rule.

## Execution counts

- Synthetic seismic forwards: 0
- Flow prior forwards: 0
- Flow guidance forwards: 0
- Property inversions: 0
- Probability bridges: 0

## Frozen prior conclusions

The original Stage10 decision remains `STOP_BRIDGE_NO_GEOPHYSICAL_INFORMATION`.
The Stage10R interpretation remains `CASE_GEOMETRY_CONFUNDED`, with complementary
finding `SEISMIC_ADDS_INCREMENTAL_INFORMATION`. Neither is modified by this
benchmark-construction failure.

## Authorization

Stage11-B is **not authorized**. No next-stage computation was implemented.
