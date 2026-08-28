# Stage17B optimizer calibration confirmation

Decision: **DFLOW_ENGINE_ACTIVE_BUT_WEAK_HARD_CONTROL**

Selected optimizer: none

Oracle and property gates were evaluated independently; pooled compensation was not used.

| Task | Optimizer | Material seeds | Median IoU recovery | Passed |
|---|---|---:|---:|---|
| oracle_probability | DFLOW_LEGACY_ADAM | 2/3 | 0.154868 | False |
| oracle_probability | DFLOW_PAPER_ALIGNED_LBFGS | 3/3 | 0.267519 | True |
| property | DFLOW_LEGACY_ADAM | 1/3 | -0.0137077 | False |
| property | DFLOW_PAPER_ALIGNED_LBFGS | 2/3 | 0.215156 | False |

Both optimizers reduced their endpoint objectives and changed only source
state for every seed. All model state hashes were unchanged and all hard
condition violations were zero. This establishes an active mechanism, not
adequate hard control.

LBFGS passes the oracle gate, but its property median target-IoU reference
recovery is 0.215156, below the frozen 0.25 requirement. Adam fails both
tasks. Oracle and property were evaluated independently; oracle success cannot
compensate for property failure. No optimizer is selected and the Stage17C
D-Flow geophysical arm is forbidden under this protocol.
