# Actual Stage24 commands

Host: SSH labrs-codex; working directory /home/xmj/mcy/flowtrain_stochastic_interpolation-main.
All Python uses the repository .venv; PYTHONDONTWRITEBYTECODE=1.

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python project/geodata-3d-conditional/scripts/stage24/test_stage24.py
CUDA_VISIBLE_DEVICES="" .venv/bin/python project/geodata-3d-conditional/scripts/reports/build_stage24_approved_figures.py --case fullgeo_case03
.venv/bin/python project/geodata-3d-conditional/scripts/stage24/prepare.py --case fullgeo_case01
.venv/bin/python project/geodata-3d-conditional/scripts/stage24/prepare.py --case fullgeo_case02
.venv/bin/python project/geodata-3d-conditional/scripts/stage24/run.py --case fullgeo_case01
.venv/bin/python project/geodata-3d-conditional/scripts/stage24/run.py --case fullgeo_case02
.venv/bin/python project/geodata-3d-conditional/scripts/stage24/evaluate.py --case fullgeo_case01
.venv/bin/python project/geodata-3d-conditional/scripts/stage24/evaluate.py --case fullgeo_case02
CUDA_VISIBLE_DEVICES="" .venv/bin/python project/geodata-3d-conditional/scripts/reports/build_stage24_approved_figures.py --case fullgeo_case01
CUDA_VISIBLE_DEVICES="" .venv/bin/python project/geodata-3d-conditional/scripts/reports/build_stage24_approved_figures.py --case fullgeo_case02
.venv/bin/python project/geodata-3d-conditional/scripts/stage24/report.py
```

Commands are executed sequentially; existing immutable case outputs refuse overwrite. case03 scientific compute=0. Initial style-only failed attempt is retained in engineering/style_attempt1; fixed namespace, no scientific retry.

Case02 plotting was retried after relaxing assertion safety margins only (no camera/style/science changes). Failure log: engineering/case02_margin_attempt1/failure.log. Final style regression and final CPU audit commands:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" .venv/bin/python /tmp/stage24_final_style_regression.py
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" .venv/bin/python /tmp/stage24_final_check.py
```

Sources of both delivery checks are retained in delivery_checks/. No new scientific forward/solve in these checks.
