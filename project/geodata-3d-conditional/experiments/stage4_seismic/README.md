# Phase 4c: convolutional seismic upper bound

Read `docs/PHASE4A_REPORT.md` and `docs/PHASE4C_SPEC.md` first. This experiment
is independent of the closed gravity-only controller screen.

The first implementation unit contains a differentiable normal-incidence
post-stack convolutional forward model, a complete synthetic acoustic
codebook, immutable observation configuration and CPU acceptance tests. It is
an inverse-crime upper bound, not measured seismic or a full-wave simulator.

The CPU gate passes 15 focused tests and the complete suite passes 118 tests.
The canonical immutable observation is
`observations/cond_generation_0/distinct_upper_bound_v1_fix2/`: shape
`1 x 1 x 64 x 64 x 320`, amplitude range `-0.4716..0.4931`, 193459 valid
subsurface interfaces, maximum valid TWT `1428.43 ms`, and no noise. Every
column has one contiguous known subsurface interval.

The earlier `distinct_upper_bound_v1/` predates the explicit soft-acoustic
known-subsurface policy. A real-checkpoint smoke test showed that transient air
probability below the observed surface could move predicted TWT outside the
recording window. `fix1` added that policy, after which hard decoded underground
air exposed the distinct finite-recording-window case. `fix2` additionally
requires all truth arrivals to fit when building the observation and crops only
out-of-window predicted arrivals during inference/evaluation. Both earlier
directories remain immutable historical evidence. `fix2` uses the same
observation tensors and records both policies; current source-hash validation
intentionally rejects the earlier assets for new runs.

The EMA/fixed-Euler runner, complete hard evaluator, alpha-zero regression
audit and controller manifest are implemented. The pre-registered seed-42 n=1
alpha/cap 0.25 pair is complete and fails the full gate `0/1`. Hard seismic
RMSE decreases from `0.042262` to `0.039048`, but label-9 IoU/recall decrease
from `0.02860/0.04728` to `0.02593/0.03947`; major-body recovery also worsens.
The final guided sample changes `1.4256%` of baseline voxels and the last Euler
step changes `0.2678%`, both inside their frozen limits. The conditional
alpha-0.10 excessive-harm diagnostic is therefore not authorized, and neither
n=4 nor gravity fusion may run under this protocol. Read
`docs/PHASE4C_REPORT.md` before designing the next experiment.
