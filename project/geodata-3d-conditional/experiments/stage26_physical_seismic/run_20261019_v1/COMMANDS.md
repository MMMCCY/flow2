# Stage26 命令

在SSH LabRS的仓库根目录运行，环境.venv/bin/python；基线main@e97a92677463f31162f5bec4e9a7bc78a5e0ac47。

## 已执行：针对性CPU检查
```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q project/geodata-3d-conditional/scripts/stage26/test_stage26.py project/geodata-3d-conditional/tests/test_phase5a_seismic_inversion.py
```
10 passed；包含实际保存→rb哈希→加载和禁止输入访问检查，不调用模型。

## 已执行：唯一物理反演与条件封存
```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python project/geodata-3d-conditional/scripts/stage26/prepare.py
```
只读取旧Stage25地震观测/地表支持、A预测/原条件和固定配置；没有再次生成观测、没有读取真值/B/C/Gaussian恢复。完成1次invert_acoustic_member，先原子保存数值bundle，再整理hash/metadata；后续故障可复用bundle，不再求解。完成后目录有不可覆盖保护。

## 已执行：D0/D各一次正式Flow
```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python project/geodata-3d-conditional/scripts/stage26/run.py --registry project/geodata-3d-conditional/experiments/stage26_physical_seismic/run_20261019_v1/arms/D0/inference_registry.json --seed 42 --out project/geodata-3d-conditional/experiments/stage26_physical_seismic/run_20261019_v1/arms/D0/predictions/seed42
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python project/geodata-3d-conditional/scripts/stage26/run.py --registry project/geodata-3d-conditional/experiments/stage26_physical_seismic/run_20261019_v1/arms/D/inference_registry.json --seed 42 --out project/geodata-3d-conditional/experiments/stage26_physical_seismic/run_20261019_v1/arms/D/predictions/seed42
```
D0/D条件不同，所以2份共64前向；A/B/C原样复用、没有重跑。

## 已执行：事后评价及作图/报告
```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python project/geodata-3d-conditional/scripts/stage26/evaluate.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python project/geodata-3d-conditional/scripts/stage26/plot.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python project/geodata-3d-conditional/scripts/stage26/report.py
```
另对已保存硬标签调用原interface_response做记录窗/地下空气诊断，输出physical_window_diagnostics.csv；没有网络或反演调用。

## 只重绘（选择新输出目录）
```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python project/geodata-3d-conditional/scripts/stage26/plot.py --out project/geodata-3d-conditional/experiments/stage26_physical_seismic/run_20261019_v1/redraw_v1
```
只读已有标签/物理场，原approved_v1渲染，无模型调用。原图不覆盖。报告依赖已实际完成的图件核验，不能未看图就标记。
