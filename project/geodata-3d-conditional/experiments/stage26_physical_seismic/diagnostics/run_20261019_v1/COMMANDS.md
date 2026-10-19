# 离线诊断命令

在SSH LabRS的仓库根目录执行，基线main@a251087d4e958bdc0c30fe98d29d24252f53b8fc。独立结果目录：experiments/stage26_physical_seismic/diagnostics/run_20261019_v1/。

## 已执行的小检查（4 passed）
```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q project/geodata-3d-conditional/scripts/stage26_diagnostics/test_diagnostics.py
```
包括真实保存→hash(rb)→加载、候选只能来自原条件、分类并列/空气/原条件优先与单阻抗规则、TWT top-down到原XYZ的轴向转换。未运行历史测试或模型。

## 已执行：候选来源及旧分类复现
```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python project/geodata-3d-conditional/scripts/stage26_diagnostics/prepare.py
```
独立truth-blind进程先冻结FULL/OBSERVED集合，再重现保存prior/updated的旧FULL分类。没有从真值或B/C决定集合。

## 已执行：唯一新增ORACLE_DIAGNOSTIC_ONLY反演
```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python project/geodata-3d-conditional/scripts/stage26_diagnostics/oracle.py
```
只换固定慢度模块，A阻抗起点逐体素相同，不使用真值阻抗；原观测、码本、原条件、正则及单次函数不变。数值优先原子保存。已完成的oracle拒绝覆盖，若数值保存但后续元数据失败则恢复bundle；不得重复求解凑结果。该入口不是可用推理方法，不能将结果送入Flow。

## 已执行：六个离线分类、评价及最多两张图
```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python project/geodata-3d-conditional/scripts/stage26_diagnostics/evaluate.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python project/geodata-3d-conditional/scripts/stage26_diagnostics/plot.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python project/geodata-3d-conditional/scripts/stage26_diagnostics/report.py
```
六种分类不是六次反演；只有oracle一行新增一次求解。硬标签物理误差均用各自类别重新查表速度，不借oracle慢度。

## 仅重绘（新目录，0反演/Flow）
```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python project/geodata-3d-conditional/scripts/stage26_diagnostics/plot.py --out project/geodata-3d-conditional/experiments/stage26_physical_seismic/diagnostics/run_20261019_v1/redraw_v1
```
保留旧正式数值及本轮结果，不修改训练/采样/物理算子，不新建正式推理注册表，不推送GitHub。
