# 命令（在远程仓库根目录）

环境：SSH xmj@172.27.231.254:22，主机LabRS，仓库main@aad385771d59e8486d62e20d1df1c4a9caf9180a，.venv/bin/python。密码不记录到文件。

## CPU检查（已执行，16 passed）
```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q project/geodata-3d-conditional/scripts/stage25/test_stage25.py project/geodata-3d-conditional/scripts/stage22/test_stage22.py project/geodata-3d-conditional/scripts/stage23/test_stage23.py project/geodata-3d-conditional/scripts/stage24/test_stage24.py
```

## 已执行的数据准备
```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python project/geodata-3d-conditional/scripts/stage25/generate.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python project/geodata-3d-conditional/scripts/stage25/restore.py
```
生成命令实际在旧ROOT=run_20261019_v1时执行过一次；源码快照在v1/source_snapshot内。v1恢复预检查失败，未调用solver，见ENGINEERING.md。v2只重封存同一组输入，不运行generate.py。当前generate.py仍有不可覆盖保护，不应对已存在ROOT重跑。

## 正式推理（每条只运行一次）
```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python project/geodata-3d-conditional/scripts/stage25/run.py --registry project/geodata-3d-conditional/experiments/stage25_laccolith_shape/run_20261019_v3/arms/A/inference_registry.json --seed 42 --out project/geodata-3d-conditional/experiments/stage25_laccolith_shape/run_20261019_v3/arms/A/predictions/seed42
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python project/geodata-3d-conditional/scripts/stage25/run.py --registry project/geodata-3d-conditional/experiments/stage25_laccolith_shape/run_20261019_v3/arms/B/inference_registry.json --seed 42 --out project/geodata-3d-conditional/experiments/stage25_laccolith_shape/run_20261019_v3/arms/B/predictions/seed42
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python project/geodata-3d-conditional/scripts/stage25/run.py --registry project/geodata-3d-conditional/experiments/stage25_laccolith_shape/run_20261019_v3/arms/C/inference_registry.json --seed 42 --out project/geodata-3d-conditional/experiments/stage25_laccolith_shape/run_20261019_v3/arms/C/predictions/seed42
```

## 评价、绘图、报告
```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python project/geodata-3d-conditional/scripts/stage25/evaluate.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python project/geodata-3d-conditional/scripts/stage25/plot.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python project/geodata-3d-conditional/scripts/stage25/report.py
```
报告命令要求先在RENDER_CONFIG.json记录实际人工成图核验，不能未看图就标记。

## 无网络重绘（请选择新目录）
```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python project/geodata-3d-conditional/scripts/stage25/plot.py --out project/geodata-3d-conditional/experiments/stage25_laccolith_shape/run_20261019_v3/redraw_v1
```
保留已完成结果，任何重绘都不调用Flow。实际执行状态以各receipt/log为准。

实际正式结果位于v3；v2源通道错误在零前向处停止，v3仅重封存15维源及原条件，恢复场直接复用v2。

## 可选地震实际命令与停止点
```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q project/geodata-3d-conditional/tests/test_phase5a_seismic_inversion.py project/geodata-3d-conditional/tests/test_phase4_seismic.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python project/geodata-3d-conditional/scripts/stage25_physics/observe.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python project/geodata-3d-conditional/scripts/stage25_physics/invert.py
```
23项测试通过；observe完成；invert在唯一数值调用返回后的首次写出失败。后续D命令和evaluate_plot没有执行。禁止在已有physics目录重跑：脚本有不可覆盖/单次调用保护。本轮不再执行数值反演。
