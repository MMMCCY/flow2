# Stage15 binary inversion pilot diagnostics

Status: truth-blind pilot complete; retrospective truth evaluation and Flow execution not run.

## Observation validation

- Case: `cond_generation_0`, parsed from the authoritative Phase1 protocol-v4 baseline config.
- Phase1 truth, borehole, and EMA checkpoint file hashes matched the frozen records.
- Tensor shape: `[1,1,64,64,64]`.
- Synthetic scenario: deliberately binary, high-contrast, full-cube, noiseless inverse-crime upper bound; all non-label9 subsurface voxels use the raw-label0 acoustic reference.
- Direct `ConvolutionalSeismic` closure maximum absolute error: `0.0`.
- Flow categorical condition voxels: 69,107; binary geophysical well voxels: 4,518.

## N=4 smoke

Authoritative completed run: `inversion/smoke_n4_v2`.

- Hard seismic RMSE: `[0.06864680, 0.06861626, 0.06836443, 0.06805022]`; median `0.06849034`, range `0.06805022–0.06864680`.
- Target voxel count: `[115160, 114101, 114743, 117232]`; range `114101–117232`.
- Target fraction: range `0.577566–0.593414`.
- Unique hard models: `4/4`.
- Condition violations: `[0,0,0,0]`.
- Hard/STE forward maximum absolute error: `[0,0,0,0]`.
- Total runtime: `6.7925 s`.

The earlier `inversion/smoke_n4_v1` is an explicitly failed implementation run. It stopped at the hard/STE closure gate because the algebraic STE expression accumulated a float32 cancellation residual (`1.48e-5`). The implementation was replaced by an exact-hard custom autograd STE; the failed directory was retained and not overwritten.

## N=16 pilot

Completed run: `inversion/pilot_n16_v1`.

- Hard seismic RMSE: median `0.06840608`, range `0.06805022–0.06884073`.
- Target voxel count: median `115111.5`, range `113663–117232`.
- Target fraction: median `0.582681`, range `0.575349–0.593414`.
- Unique hard models: `16/16`.
- Condition violations: all zero.
- Hard/STE forward maximum absolute error: all zero.
- Per-member runtime: median `1.4034 s`, range `1.3319–2.4623 s`.
- Total runtime: `24.1123 s`.

## Fixed 0.8/0.2 consensus

Completed output: `consensus/pilot_n16_t080_t020_v1`.

- Positive voxels: 38,831.
- Negative voxels: 10,292.
- Unknown voxels: 148,432.
- Confidence coverage within subsurface: `0.248655`.
- Unconditioned Flow guidance ROI: 44,605 voxels.

Truth-blind geometry diagnostics show a non-empty positive consensus, but not a compact one: it has 789 six-connected components, only two components contain at least 20 voxels, the largest contains 24,897 voxels (`64.12%` of positive mass), and the positive bounding box spans the full XY grid (`[0,0,0]` to `[63,63,48]`). Therefore the pilot does not support describing the positive core as spatially concentrated.

## Stop decision

Do not start N=100, threshold tuning, inversion parameter sweeps, retrospective truth evaluation, or Flow guidance automatically. The positive consensus is stable enough to be non-empty but is broad and fragmented under truth-blind diagnostics. Manual review is required before authorizing retrospective truth evaluation; Flow execution is not recommended at this point.
