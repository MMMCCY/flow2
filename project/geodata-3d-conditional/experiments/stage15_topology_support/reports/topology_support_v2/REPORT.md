# Stage15 topology support audit

Decision: **PRIOR_SUPPORTS_RING_BUT_SEISMIC_BRIDGE_INSUFFICIENT**

This is a frozen-checkpoint topology stress test, not a certified historical held-out split.

## A_SOLID

| Arm | median IoU | median P/R | median seismic RMSE | median beta1 |
|---|---:|---:|---:|---:|
| FLOW_ONLY | 0.0000 | 0.0000/0.0000 | 0.027529 | 0.0 |
| SEISMIC_GUIDED | 0.3145 | 0.3238/0.9241 | 0.032911 | 2.0 |
| ORACLE_GUIDED | 0.9937 | 0.9937/1.0000 | 0.003785 | 0.0 |

## B_RING

| Arm | median IoU | median P/R | median seismic RMSE | median beta1 |
|---|---:|---:|---:|---:|
| FLOW_ONLY | 0.0002 | 0.0004/0.0004 | 0.028314 | 0.0 |
| SEISMIC_GUIDED | 0.3092 | 0.3110/0.9818 | 0.034300 | 1.0 |
| ORACLE_GUIDED | 0.9937 | 0.9937/1.0000 | 0.003795 | 1.0 |

