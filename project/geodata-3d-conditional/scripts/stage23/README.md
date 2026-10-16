# Stage23 complete restored pseudo-wells

Independent entry points reuse the saved Stage22 classification and old native
sampler/evaluator. Preparation never loads full truth; Flow receives conditions
and paired actual noise only. No precision gate or new restoration is run.
Historical sources/results stay read-only. Existing run/output paths refuse overwrite.

Run from the repository root on SSH `labrs-codex`, using the repository `.venv`:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python project/geodata-3d-conditional/scripts/stage23/test_stage23.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python project/geodata-3d-conditional/scripts/stage23/prepare.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python project/geodata-3d-conditional/scripts/stage23/run.py --registry project/geodata-3d-conditional/experiments/stage23_full_pseudo_wells/run_20261017_v2/inference_registry.json --seed 42 --out project/geodata-3d-conditional/experiments/stage23_full_pseudo_wells/run_20261017_v2/predictions/seed42
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python project/geodata-3d-conditional/scripts/stage23/evaluate.py
```

Exactly one case, one seed, one new prediction, 32 velocity calls. Only after
prediction completion does evaluation load truth. Old A/B/C/Stage22 and direct
restoration predictions are reused. COPY_ONLY is the sole new non-network control.
Object IDs are full-volume 26-connected; object1 off-condition comparison uses
one shared Stage23-mask exclusion for every arm. The figure fixes full XY Z=15.
No additional experiments follow the outcome. Artifact roles/hashes, numerical
comparisons and limitations are saved in the run directory's Chinese REPORT.

The retained v1 attempt failed before any network forward because its file guard
blocked the dependency `sympy.geometry`. v2 narrows that guard to experimental
geometry assets; scientific method/condition tensors are unchanged. Both attempts
and zero-versus-32 call counts are explicitly retained in the final delivery.
