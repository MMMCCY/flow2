# Stage27 endpoint proximal inference

Run from the repository root with `PYTHONPATH=project/geodata-3d-conditional:src`
and `PYTHONDONTWRITEBYTECODE=1`. The authoritative scientific specification is
`STAGE27_FLOW_PROXIMAL_EXECUTION_PROMPT_2026-10-30.md`.

- `prepare`: generate the single prescribed G2, check original conditions,
  reuse old y1, generate y2 once, freeze role-separated inputs and source snapshot.
- `checks`: small CPU/CUDA FP64 helper checks, without checkpoint/network calls.
- `run`: load the frozen EMA model once and produce the six fixed 32-NFE outputs.
  No truth or geometry is accepted by the sampler. A durable ledger reserves each
  NFE before its call; existing ledgers are never automatically replayed.
- `evaluate`: original hard decode metrics, all components and boundary centers,
  hard/continuous physical differences, and both observation/truth cross matrices.
- `plot`: existing approved_v1 renderer and fixed Y26, including controls and failures.
- `report`: produce the evidence-based report and final checks from saved results.

All commands take `--out` except `checks` (which takes `--device cpu|cuda`).
`run` also takes `--prox-device cuda|cpu`; the formal network remains CUDA FP32.
Use a new output directory for preparation. Preserve numerical progress and the
ledger on failure; inspect the remaining fixed budget before any recovery.
Completed sampling is never rerun to fix evaluation, figures or reports.

The inference registry is deliberately stripped of truth/geometry/evaluation
inputs. Runtime constants save the immutable embedding for postprocessing, which
does not reload the checkpoint. The final FP64 proximal endpoint and the FP32
endpoint passed to the original hard decoder are both retained. Soft physics
conditions on non-air probabilities underground; final hard labels retain air.
