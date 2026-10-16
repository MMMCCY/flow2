# Stage15 trace v4 与 Stage19R：同案例集合采样

当前授权：两条方法均使用 `stage19_test_case004`。该案例是按已有
`reports/formal_v2/sample_metrics.csv` 中 `ADAPTER_CORRECT` 的三个种子
目标 IoU 中位数事后选择的最佳 TEST case（0.6710347944）。这是选择案例的
展示实验，不作为无偏跨案例比较或泛化证明。选择规则与完整排名进入 protocol。

本目录是新增实验，不覆盖 Stage1–20 既有结果。`run_v1` 是首个日志格式
读取失败的工程目录，没有 Flow 样本；`run_v2` 为修复后的采样目录。

资产根：`experiments/stage19_learned_evidence_adapter/` 的
`{cohort_v2,observations_v2,evidence_v2,checkpoints/formal_v2}`。
基础 checkpoint：`demo_model/conditional-weights.ckpt`，SHA256
`561e94bfda770ec41fc4cbed43436a7e2130eef5dfb7e5d666fcefc0724ff94c`。
adapter：`checkpoints/formal_v2/adapter_checkpoint.pt`，SHA256
`cdb740bfeccc08f7010048c69deb8e44a13917fe53e3b2f4b46910bf536f20bb`。
论文位于仓库根 `Synthetic_Geology_Structural_Geology_Meets_Deep_Le.pdf`。

Stage15 trace v4 的已有样本入口实际是 `scripts/stage2/run_property_guidance.py`，
不是一个臆造的 Stage15 sampler。本入口直接复用同一 `fixed_euler_property_sample`，
绕开原 runner 的真值读取，用 TEST registry 的已观测条件构造 conditioning。
固定的 positive endpoint target=`subsurface_mask.float()`、
confidence=`binary_impedance_score * free_subsurface` 与历史 v4 导出的
property tensors 逐体素一致（在历史 cond_generation_0 已核对）。
同案例转移不称为历史 cond_generation_0 结果。

Stage19R 复用正式 epoch4 adapter 与 `fixed_euler_adapter_sample`。
不训练，不改变 Flow、EMA、decoder、32-step solver 或科学参数。
生成时 CPU 单独创建每个 source，种子 `42 + 100*i`，i=0..199，
同源 baseline/两种 guided 共享相同初始张量。baseline 计算一次并用于
两条方法各自的200对；不把其共享引用计为同一集合内多个模型。
前三个固定 seed 与历史 Stage19R 完整硬模型逐体素核对。

在仓库根用远程 CUDA 环境执行：

```bash
RUN=project/geodata-3d-conditional/experiments/ensemble_200_selected_case/run_v2
PYTHONPATH=src .venv/bin/python project/geodata-3d-conditional/scripts/ensemble_workflow.py --action sample --case stage19_test_case004 --out "$RUN" --resume
PYTHONPATH=src .venv/bin/python project/geodata-3d-conditional/scripts/ensemble_workflow.py --action postprocess --out "$RUN"
```

新建另一个目录可以用 `--action all --out NEW_DIR` 完成整条流程。
接入检查用 `--limit 3`，仍冻结200 IDs；只生成前三个独立draw，必须
加 `--diagnostic` 才可后处理真实N=3。不把它称200实验。
`--action audit --out NEW_AUDIT_DIR` 只核对资产并输出哈希。

只后处理已有本入口结果：

```bash
PYTHONPATH=src .venv/bin/python project/geodata-3d-conditional/scripts/ensemble_workflow.py --action postprocess --out "$RUN" --report-dir NEW_REPORT_DIR
```

只绘图（复用已冻结统计而不采样）：

```bash
PYTHONPATH=src .venv/bin/python project/geodata-3d-conditional/scripts/ensemble_workflow.py --action plot-only --out "$RUN" --report-dir EXISTING_REPORT_DIR --plot-dir NEW_FIGURE_DIR
```

`--skip-plots` 可先快速完成全部统计。plot-only 输出新的 PNG/SVG，
不覆盖已有文件。通用 `load_hard(path, offset)` 支持`.pt/.pt.gz` chunk，
批次必须显式 offset；不会对已解码标签再次减1。

全量保存600个硬模型：200 baseline、200 Stage15、200 adapter，各有
独立 hard seismic prediction、hard RMSE 和sampling trace。筛选采用两种
guided 和共享 baseline 各自的固定池，all200/random100/obs_top100/oracle_top100。
obs为未归一化 hard RMSE（振幅单位，低者优先），ties按冻结次序；oracle为
完整raw-label9真实IoU，仅事后诊断。先写obs IDs，再读取评估truth。

统计重新累加所选硬模型的整数count。严格超过40/60/80%与全部命中，
N200要求81/121/161/200，N100要求41/61/81/100。频率未经校准；
阈值mask不是完整地质模型，不赋予继承的seismic RMSE。
全空间与地下自由区的图使用同坐标、相机、100/100/50m物理aspect与真实
体素外露面；不平滑、不填洞、不删小体。红色条件体素是独立overlay；
只有aggregate mask时不臆造surface与well分类。包含固定最多12个成员、
四阈值、等距/正视、论文/汇报两套PNG/SVG及baseline/guided比较。

指标含全部候选的目标IoU/P/R、TP/FP/FN、体积、6邻域组件、最大组件比、
global accuracy、truth-present mIoU和逐类别IoU，hard seismic RMSE分布；
阈值在full/free/condition三个域分别评价，前50/100/200频率稳定性。
没有开展五体任务，不作五体对象恢复结论。

原参考 `ensemble_occurrence.py` 在工作区和服务器均缺失，本次按README
及随附6个测试重新实现。原文件并未被“读取”或假称已存在；缺失在资产
审计说明中保留。重建模块独立支持README里的numpy/Pillow CLI。

出版绘图建议使用独立的渲染入口（不变更采样程序、统计或mask）：

```bash
PYTHONPATH=src .venv/bin/python project/geodata-3d-conditional/scripts/render_ensemble_publication.py --action postprocess --out "$RUN" --report-dir NEW_PUBLICATION_REPORT_DIR
PYTHONPATH=src .venv/bin/python project/geodata-3d-conditional/scripts/verify_ensemble_run.py --run "$RUN" --report NEW_PUBLICATION_REPORT_DIR
```

该入口提供真实轴端点标注、正交俯视等距相机、各方向面明暗、字体，
论文版与1920×1080汇报版；PNG与SVG来自同一真实体素外露面。
`renderer_manifest.json`单独记录其代码hash。采样进程使用原先已经冻结
的推理代码，出版渲染器不介入推理。

最终完成产物：`run_v2/publication_v1/DEVELOPMENT_REPORT.md` 与
`FIGURE_INDEX.md`。真实N200、600完整硬模型、203正式PNG/对应SVG、零失败、
零硬条件违背；`VALIDATION_REPORT.json`为PASS。
`MANUAL_VISUAL_CHECK.md`保存已打开图件的检查记录，`ARTIFACT_MANIFEST.json`
保存结果hash。`run_v2/statistics_v1`是无图数值版，与出版版12份统计数组
及summary完全一致；N3与人工测试目录另列，不作200实验结果。
