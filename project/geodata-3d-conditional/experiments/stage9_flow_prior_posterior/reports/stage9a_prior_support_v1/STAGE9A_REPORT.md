# Stage 9A report: frozen Flow-prior support and geophysical enrichment

Decision: **SUPPORT=FAIL; DISCRIMINATION=FAIL**

Machine next action: `STOP_REASSESS_FROZEN_INFERENCE_ROUTE`.

## Frozen execution

- Primary cases: 3 deterministic StructuralGeo-native cases; 1024 independent frozen-Flow samples per case.
- Hard seismic forwards: 3072 candidate forwards.
- Flow velocity forwards: 98304 (32 per candidate).
- Candidate-generation runtime: 10040.0 seconds total.
- Frozen EMA policy, normal embedding, 32-step midpoint fixed Euler, exact condition projection, hard decode, and hard seismic forward were used throughout.
- No training, structured correction, pCN, D-Flow, SMC, gravity, or truth-visible ranking was run.

## Inference-visible evidence

| Case | Correct seismic top | Hard RMSE | Unique models |
|---|---|---:|---:|
| `native_seed20260901` | `candidate_000066` | 0.0318388 | 1024/1024 |
| `native_seed20260902` | `candidate_000862` | 0.03101524 | 1024/1024 |
| `native_seed20260903` | `candidate_000567` | 0.03161827 | 1024/1024 |

All four rankings used the same cached float32 hard seismic prediction for every candidate. Rankings were frozen by ascending RMSE with candidate-ID ties before truth was loaded.

## Retrospective truth evidence and oracle support ceiling

| Case | Support | Passing candidates | Oracle best label-9 IoU | Discrimination |
|---|---:|---:|---:|---:|
| `native_seed20260901` | False | 0 | 0.1170 | False |
| `native_seed20260902` | False | 0 | 0.1348 | False |
| `native_seed20260903` | False | 0 | 0.1289 | False |

Overall support passes in 0/3 cases; discrimination passes in 0/3 cases.

The best truth candidate is an oracle support ceiling and is not a deployable selector. The correct-observation seismic ranking is deployable in the synthetic protocol because it never receives truth.

## Scope and stop

A lower seismic loss alone is not project success. These three synthetic inverse-crime cases do not establish field generalization. Stage9A determines only frozen-prior support and hard-likelihood enrichment under this registered setup.

Stage9A stops here. The machine next action is a recommendation only; Stage9B, Stage9C, posterior weighting, adaptive proposals, SMC, D-Flow, new likelihoods, and training were not implemented.
