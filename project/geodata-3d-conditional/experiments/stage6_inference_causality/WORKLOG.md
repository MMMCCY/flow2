# Phase 6Q work log

## 2026-08-04：阶段启动

- 用户授权在远端 `/home/xmj/mcy/flowtrain_stochastic_interpolation-main`
  撰写并运行代码；正式训练仍不启动。
- 远端：`172.27.231.254:22`，branch `main`，commit `5b717ec`。
- GPU：NVIDIA GeForce RTX 4090 D，24564 MiB；`.venv` Python 3.10.12。
- 工作树已有大量用户/既有实验未提交文件；本阶段只新增
  `stage6_inference_causality`、Phase 6Q 文档、对应 guidance/script/test。
- 首轮顺序：Q0 硬候选枚举，然后 Q1 soft 与 STE 候选系数优化。

## 2026-08-04：Q0-Q2c

- Q0/Q1 权威目录：`runs/five_body_cuboid_v1/q0_q1_full_v2`；v1 因 STE
  hard-best 初值语义错误保留但废止。Q0 的66个二体组合中，五种观测的 truth pair
  均为唯一数值零、rank 1；Q1 correct 的 soft/STE 均精确恢复 `[4,6]`。
- Q2 权威目录：`q2_full_v2`；v1 因 entropy trace 浮点 NaN 保留但废止。correct
  seismic：soft hard attainment 3.30%、STE 10.26%；gravity soft/STE hard attainment
  93.85%/73.78%，但 hidden IoU 仅9.53%/6.54%。
- Q2b 权威目录：`q2b_full_v1`。hard-only 单调坐标求解得到 correct seismic
  attainment 15.64%、hidden recall 70%；gravity attainment 85.78% 但 hidden IoU=0。
- Q2c 事后细步长目录：`q2c_seismic_fine_v1`。将 proposal 改为1/2/4/8/16/32
  后 correct seismic 只达到4.00%、hidden recall 12.5%；因此 Q2b 的最小32体素并非
  停滞主因，较大的协调更新反而必要。

## 2026-08-04：Q3/Q3b checkpoint embedding endpoint

- 冻结配置：`configs/embedding_endpoint_v1.json`；检查点 SHA256
  `561e94bf...94c`，只读取 `embedding.weight [15,15]`，不加载 U-Net。
- property 冒烟：`q3_property_smoke_v1`，两方法均在 step30 得到 hard attainment=1、
  hidden IoU=1。
- 首次完整目录 `q3_full_v1` 在地表以上精确 one-hot air 进入旧声学 helper 时触发
  “non-air support empty”；它是未完成目录，保留不覆盖。修正为只在 subsurface
  检查/归一化 rock probability，新增回归测试。
- Q3 权威目录：`q3_full_v2`。correct seismic 的 `soft_embedding` hard-best 仍为
  I0（attainment=0），最终 soft attainment=12.59% 而 hard=-87.64%；
  `ste_embedding_rock` hard attainment=37.80%、hidden IoU/recall=20%。shuffled STE
  attainment=16.12% 但 IoU=0.22%。gravity STE attainment=70.67%，全部1,304个变更
  体素为 raw label6、没有 label9。
- 查看 Q3 后冻结 Q3b 全类别竞争诊断：`configs/embedding_binary_control_v1.json`，
  权威目录 `q3b_binary_full_v1`。限制为 raw label0/9 后，correct seismic 结果精确
  重现 Q2：soft attainment=3.30%、recall=10%，STE attainment=10.26%、recall=40%。
  因此真实 simplex embedding 连续几何本身不是新增主因；Q3 全类别物性替代提高了
  地震拟合但降低了目标类别可辨识性。
- 远端测试：Phase6Q 专项 `16 passed`；完整项目测试 `168 passed, 13 warnings`；
  warnings 均为既有 matplotlib/pyparsing deprecation。GPU peak 约221 MB。
- 已执行中文 Notebook：`flow2_phase6q_simple_causality_summary_zh.ipynb`，25 cells、
  9/9 code cells 已执行、5 个嵌入 PNG、0 error；SHA256 `ef91707e...7b89`。
- 人工结论：`docs/PHASE6Q_CAUSALITY_REPORT.md`。正式训练未启动。
