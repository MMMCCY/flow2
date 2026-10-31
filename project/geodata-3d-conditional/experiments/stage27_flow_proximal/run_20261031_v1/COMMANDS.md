# Actual commands

Run from /home/xmj/mcy/flowtrain_stochastic_interpolation-main.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=project/geodata-3d-conditional:src CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m scripts.stage27_flow_proximal.checks --device cpu
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=project/geodata-3d-conditional:src CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m scripts.stage27_flow_proximal.checks --device cuda
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=project/geodata-3d-conditional:src CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m scripts.stage27_flow_proximal.prepare --out project/geodata-3d-conditional/experiments/stage27_flow_proximal/run_20261031_v1
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=project/geodata-3d-conditional:src CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m scripts.stage27_flow_proximal.run --out project/geodata-3d-conditional/experiments/stage27_flow_proximal/run_20261031_v1 --prox-device cuda
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=project/geodata-3d-conditional:src CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m scripts.stage27_flow_proximal.evaluate --out project/geodata-3d-conditional/experiments/stage27_flow_proximal/run_20261031_v1
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=project/geodata-3d-conditional:src CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m scripts.stage27_flow_proximal.plot --out project/geodata-3d-conditional/experiments/stage27_flow_proximal/run_20261031_v1
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=project/geodata-3d-conditional:src CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m scripts.stage27_flow_proximal.report --out project/geodata-3d-conditional/experiments/stage27_flow_proximal/run_20261031_v1
```

run launched with subprocess.Popen(start_new_session=True), output to sampling.log; PID in LAUNCH.json. Each later phase runs separately; no automatic resampling.

## Postprocessing after the first completed evaluation

The `plot` and `report` commands above were each executed twice: the second plot
reused DONE plates and added condition-overlay sections; the second report added
cap/TV interpretation and the disclosed metadata issue. Evaluate ran once; run
ran once. No extra checkpoint or velocity load was performed. Metadata review and
finalization used read-only numerical checks, CPU noise reconstruction, and JSON/
Markdown receipt writes; no geophysical forward calls. Current numerical sampler
and runner bytes match the executed snapshot. Corrected config contracts are for
new preparations; the existing run is bound to the preserved original snapshot.
