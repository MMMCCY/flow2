# 新对话继续开发入口

更新日期：2026-08-02

本文档是下一次对话的导航入口，不替代各阶段规范、报告或
`DEVELOPMENT_HANDOFF.md`。新对话应先核对仓库的实际状态，再以本文记录的
科学结论和边界继续 Phase 6。不要把本文中的阶段编号理解成新的独立模型版本。

## 可直接粘贴到新对话的指令

```text
你将接手 flow2 的 Phase 6 开发。当前不是从头设计，也不要重复已经关闭的
Phase 1/2/4/5 实验。

项目目录：
/home/xmj/mcy/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional

在修改代码前，请完整读取并遵守：
1. docs/AGENTS.md
2. docs/NEXT_CONVERSATION_PROMPT.md
3. docs/PROJECT_BASELINE.md
4. docs/EXPERIMENT_PROTOCOL.md
5. docs/RESEARCH_GOAL.md
6. docs/DEVELOPMENT_HANDOFF.md
7. docs/PHASE5C_SPEC.md
8. docs/PHASE5C_REPORT.md
9. docs/PHASE6_ADAPTER_SPEC.md
10. docs/PHASE6A_REPORT.md
11. docs/PHASE6P_INFERENCE_LIMIT_SPEC.md
12. docs/PHASE6P_REPORT.md
13. experiments/stage5_generator_posterior/README.md
14. experiments/stage6_geo_adapter/README.md

同时检查 git status、当前分支、最新提交和所有重叠差异。工作树有大量属于
用户和既有实验的未提交文件；不得 reset、checkout 覆盖、删除、静默整理、
commit、push 或上传覆盖。只修改本次任务必要的文件。

最终研究目标是：利用地震等地球物理信息弥补稀疏钻井/地表条件在深部和全局
范围的约束不足，提高解码后的三维岩性与几何精度；同时利用冻结生成网络学到的
地质结构先验和精确钻井/地表条件，缩小地球物理反演的非唯一解集合。最终输出
应是条件严格一致、地球物理一致、地质合理且能表达不确定性的三维模型集合。

当前最新结论：Phase 5c 的全维初始噪声 pCN 后验搜索已经关闭，不能继续在
cond_generation_0 上延长链或调 beta/likelihood；Phase 6A 的 54,327 参数外置
残差速度适配器已通过同一样本、真值派生声学体的机制上限实验，且冻结基模保持
不变。它显著改善 hard geology，但 hard seismic loss 变差，所以它只证明适配器
具有跨越 soft-hard 映射的类别修正能力，没有证明泛化或真实地球物理有效性。

Phase 6P 已完成训练前的推理极限审计。与 Phase 4C 完全同轨迹的
alpha=cap `0.25/0.5/1/2/4` 阶梯最多只达到 9.88% 的 hard-seismic RMSE
消除率（ratio 1），ratio 2/4 物理回退且三维地质显著恶化。绕开轨迹的 200 步
连续终点优化也只达到 1.69%；其 soft seismic 持续改善而最终 hard seismic
严重变差，确认 soft-hard relaxation gap。所有基模 hash、条件、历史回归和独立
真值审计门均通过。禁止继续在 cond_generation_0 上扩大推理权重或调整阶梯。

下一项工作是 Phase 6B：先建立确定性、按完整地质 realization/history 分组、
可审计且无泄漏的数据基础，再做 held-out oracle/degraded-feature pilot。不要直接
开展大规模训练，也不要打开正式 test 集。第一步只实现小规模 manifest smoke
及其审计/测试，提出冻结协议并等待我确认后再运行较大 GPU 实验。

建议文件边界：
- guidance/adapter_dataset.py
- scripts/stage6/build_adapter_dataset.py
- scripts/stage6/audit_adapter_dataset.py
- tests/test_phase6_adapter_data.py
- experiments/stage6_geo_adapter/splits/

数据要求：不能直接把现有 stateful GeoData3DStreamingDataset.__getitem__ 当作
固定样本索引使用。必须显式固化完整 geology truth、生成器版本/种子、历史或族组
标识、钻井布局与种子、岩性物性映射/扰动、正演参数、子波、噪声、旋转、split、
训练集统计量及内容 hash。同一地质 realization 派生的所有井、噪声和物性版本
必须落在同一个 split。当前四个 demo 只能作为 legacy/debug，不得进入正式 test。

Phase 6B 首先用很小的 train/val/test 计数做 CPU manifest smoke，验证确定性、
分组隔离、hash 和重复构建；正式规模要依据构建/显存基准再预注册。随后才做多
realization 的 held-out oracle 上限实验。oracle 只用于研究上限，不能作为最终
输入。只有 held-out 上限成立，才进入 Phase 6C 的 truth-blind 地震编码/回投影。

不可突破的边界：冻结原 U-Net、checkpoint、EMA 和 embedding 约定；优化器只
包含外置适配器；fixed-Euler 严格配对；adapter_scale=0 必须逐元素回归 paired
baseline；每步投影地表和钻井条件且最终条件违规为零；不使用 label9 特化 loss；
不把连续 loss 下降当作成功；暂不融合二维重力；不得使用 test 选择超参数。

验收必须同时覆盖：基模 hash 不变、无基模梯度、scale=0 等价、条件严格保持、
分组无泄漏、可复现；以及 held-out hard-label 全局精度、truth-present mIoU、各类
IoU/precision/recall、稀疏目标 cohort、连通体/几何、hard seismic loss 和集合
多样性。必须有 correct observation、zero observation、shuffled observation
控制；正确观测应优于零/打乱观测。具体数值门槛必须在查看正式 test 前冻结。

请先向我报告：实际仓库状态、你理解的执行路径、已证明/未证明内容、Phase 6B
拟修改文件、数据泄漏防护、分步骤计划和验收门。未经我确认不要进行正式训练。
轻量只读检查和测试可以运行。
```

## 当前权威状态摘要

### 总目标

项目不是单纯追求某个连续地球物理 loss 变小，也不是只恢复 label 9。目标是让
地球物理的全局信息与生成模型的地质先验、稀疏硬条件相互约束：前者改善深部与
全局三维地质，后者减少地球物理非唯一性。评价对象必须回到 hard-label 三维体、
地质几何、物理一致性和后验多样性。

### 已经关闭的路线

- Phase 1：真值派生 label-9 二值概率体证明强三维 oracle 引导可以驱动 hard
  label 改变，但它依赖真值、只针对单类，不能视作真实地球物理成果。
- Phase 2：全岩性三维属性 oracle 在码本区分充分时可工作；属性模糊、平滑或
  近同码会迅速丧失 hard-label 可辨识性。
- Phase 3：非零空间模糊均未形成可推广工作点，作为 Phase 4 的降质桥梁关闭。
- Phase 4：重力、卷积地震与固定候选选择可以降低相应物理残差，但没有稳定改善
  hard label 9 或主要岩体几何，暴露 proposal support 与 likelihood alignment
  两类限制。
- Phase 5a/5b：无训练反演属性桥能改善连续属性或 bridge loss，但 hard geology
  失败。
- Phase 5c：在冻结条件流初始高斯噪声上做 pCN 后验搜索，8 个 proposal 接受
  7 个且地震 loss `17.860632 -> 17.653818`，但 label-9 IoU
  `0.028596 -> 0.025298`、recall `0.047279 -> 0.040700`、四大岩体召回
  `0.041423 -> 0.034587`，因此禁止在旧案例上继续链长或参数搜索。

### 当前有效起点：Phase 6A

外置残差速度适配器共 54,327 个参数，原冻结基模约 5,300 万参数。适配器接收
当前状态、条件体、条件 mask、两通道三维声学特征和时间；输出在硬条件处为零。
同一样本 truth-derived full-resolution acoustic oracle 的 80 步小过拟合结果为：

- global accuracy：`0.587366 -> 0.745053`
- truth-present mIoU：`0.265249 -> 0.473283`
- label-9 IoU：`0.028596 -> 0.511914`
- label-9 recall：`0.047279 -> 0.570138`
- 四大岩体平均召回：`0.041423 -> 0.507007`
- hard seismic loss：`17.860632 -> 19.828848`（变差）

该结果说明冻结生成器之外的小型学习模块确实可以把连续修正转化为明显的类别和
几何变化，缓解此前的 soft-hard 不敏感问题。它没有证明跨地质模型泛化，也没有
证明从真实/模拟地震观测可以获得这种三维特征，更没有完成地质与物理的双重改善。
不得继续调优或宣传这个 same-case oracle checkpoint。

### 训练前机制结论：Phase 6P

Phase 6P 对“是否只需把推理物理权重继续加大”给出否定结果：0.25 到 4 倍的
物理最优点是 1 倍，但 hard-seismic 达到率仍只有 9.88%；4 倍已改变 29.90%
体素，全局 mIoU `0.19291 -> 0.10201`，目标 IoU
`0.02860 -> 0.01334`。直接终点拟合中，soft seismic RMSE 约降低 47%，
hard seismic 最终却恶化，证明现有连续代理可在类别混合空间投机拟合。

这不证明冻结网络数学上绝对不可达，但证明当前生成器、soft decoder 和推理
controller 的主要瓶颈不是 0.25 cap。当前制导地震仍远未到达真值观测，所以
非唯一性不能单独解释 notebook 中“制导更像基线”的现象。学习路线必须同时
解决 hard-aware 类别对齐和 physical consistency，而不是仅加大 inference alpha。

## Phase 6B/6C 的开发顺序

1. 建立确定性 materialized 数据构建器和 manifest schema。
2. 用小样本构建 smoke split；证明相同配置重建 hash 一致。
3. 审计 geology-group split，确保所有派生 sibling 永不跨 split。
4. 固化训练集专属归一化和 evaluator/config hashes。
5. 在查看正式 test 前冻结 held-out oracle 上限实验门槛与稀疏目标 cohort 规则。
6. 训练多 realization 的小适配器，仅用 train 选参、val 决策；最后一次性评估
   test，并同时检查 hard geology 与 hard seismic。
7. 若 oracle 不能在 held-out geology 上泛化，停止，不进入更难的地震编码器。
8. 若 oracle 通过，再把输入替换为 truth-blind post-stack seismic
   encoder/backprojection，并加入 matched/zero/shuffled 控制。
9. 只有正确观测在 hard geology 和 hard seismic 两方面均优于 paired baseline
   且优于零/打乱观测，才考虑多 seed、集合不确定性和后续物理辅助 loss。

Phase 6B 的 oracle pilot 与 Phase 6C 的 truth-blind 地震输入应分开记录。低权重
物理辅助 loss 只能在上限泛化成立后按预注册配置加入，不能用 test 结果反复调节。

## 远端 GPU 运行环境

远端服务器曾实际识别到 NVIDIA RTX 4090 D（24564 MiB）以及
PyTorch `2.8.0+cu128`。本地执行环境看不到 GPU 时，不代表远端不可用。不要在
仓库文档、脚本或命令历史文件中保存密码。

```bash
ssh xmj@172.27.231.254
cd /home/xmj/mcy/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional
nvidia-smi
/home/xmj/mcy/flowtrain_stochastic_interpolation-main/.venv/bin/python --version
```

远端应使用仓库根目录的 `.venv` Python；系统 Python 缺少 PyTorch。正式命令应在
相应阶段脚本完成并通过 CPU/lightweight 测试后，再由新对话给出。不要覆盖已有
run tag，尤其不要复用已知不完整的旧运行目录。

## 交接时的验证基线

- 最新完整本地轻量测试：`152 passed, 13 warnings`。
- Phase 6A 基模参数 hash 前后相同、基模无梯度、硬条件违规为零。
- Phase 6P 两个运行的工程门全部通过；0.25 级逐体素回归 Phase 4C 历史样本，
  基模 hash 不变、无基模梯度、条件违规为零。
- `adapter_scale=0` 具有显式基线分支。
- 最新已知分支为 `main`，交接时起点提交为 `5b717ec`；新对话必须以实际
  `git status`/`git log` 重新核对，不得假定工作树干净。
- 当前工作树含大量未提交用户资产和各阶段实验文件。任何整理、提交或上传都必须
  获得用户单独授权。

详细运行目录、审计数据与阶段结论以 `DEVELOPMENT_HANDOFF.md`、Phase 报告和
各实验目录 README 为准。若本文摘要与机器生成结果冲突，以实际 result artifact
和审计脚本输出为准，并在修改前向用户指出差异。
