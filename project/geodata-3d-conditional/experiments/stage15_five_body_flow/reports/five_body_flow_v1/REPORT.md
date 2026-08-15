# Stage15 guidance on the Stage7-style five-body case

All five bodies are raw label9; bodies 0--2 are drilled and bodies 3--4 are hidden.

| Arm | full IoU | hidden IoU | hidden P/R | seismic RMSE | merged truth-body pairs |
|---|---:|---:|---:|---:|---:|
| FLOW_ONLY | 0.1146 | 0.0106 | 0.0114/0.1375 | 0.044608 | 4.0 |
| SEISMIC_GUIDED | 0.1608 | 0.0679 | 0.0684/0.9016 | 0.048009 | 10.0 |
| ORACLE_GUIDED | 0.9459 | 0.8749 | 0.8749/1.0000 | 0.012146 | 0.0 |

The case-level risk gate passed: all five equal-volume, disjoint bodies use raw label9; three are intersected by wells and two have no hard-condition overlap. The current checkpoint, hard conditions, and frozen Stage15 binary seismic physics are reused.

Flow-only occasionally intersects one hidden body in individual seeds, but does not recover both hidden bodies consistently. Seismic guidance raises the aggregate hidden-body recall while overproducing label9 and merging the five truth bodies into one connected system. Oracle guidance recovers the separated bodies, so the result diagnoses an indirect-evidence/topology limitation rather than an absolute inability of the frozen generator to represent the target.
