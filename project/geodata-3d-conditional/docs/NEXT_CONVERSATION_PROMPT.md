# 新对话继续开发入口

更新日期：2026-08-10

本文档是下一次对话的导航入口，不替代冻结规范、机器结果或
`DEVELOPMENT_HANDOFF.md`。当前 Stage 9A 已按停止规则结束；下一次对话不得自动
实施 Stage 9B/9C、训练或新的推理算法。

## 可直接粘贴到新对话的指令

```text
你将接手 flow2 在正式 Stage 9A 终止后的研究决策。

项目目录：
/home/xmj/mcy/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional

修改任何文件前完整读取：
1. docs/AGENTS.md
2. docs/RESEARCH_GOAL.md
3. docs/EXPERIMENT_PROTOCOL.md
4. docs/DEVELOPMENT_HANDOFF.md
5. experiments/stage8_structured_posterior/runs/stage8a_v4/STAGE8A_V4_FINAL_REPORT.md
6. docs/STAGE9A_FLOW_PRIOR_SUPPORT_SPEC.md
7. experiments/stage9_flow_prior_posterior/README.md
8. experiments/stage9_flow_prior_posterior/reports/stage9a_prior_support_v1/STAGE9A_REPORT.md
9. experiments/stage9_flow_prior_posterior/reports/stage9a_prior_support_v1/summary.json

先报告 git status、分支、HEAD 和重叠差异。不得 reset、checkout 覆盖、删除、
重组、commit、push 或上传覆盖 Stage 9A 代码和不可变正式结果。

Stage 9A 使用三个新确定性 StructuralGeo 案例，每例 1024 个独立冻结 Flow
先验样本。冻结设置为正常 embedding、EMA、32 步 midpoint fixed Euler、逐步精确
条件投影、hard decode 和 hard seismic。正确、零、XY 打乱、独立错误案例四种
观测对同一组缓存预测排序。候选生成和排序均不可读取真值；pool/ranking 清单及
哈希冻结后，独立 retrospective auditor 才能读取真值。

正式结果：3072 个候选全部为唯一 hard model；hard seismic forwards=3072，
Flow velocity forwards=98304。三个案例均为 SUPPORT_PASS=false，且完整地质支持门
通过候选数均为 0/1024。三个案例也均为 DISCRIMINATION_PASS=false：正确观测排序
没有同时富集 label-9 IoU、label-9 recall 和主要连通体 mean recall。总体支持和
判别力均为 0/3，低于 2/3 门槛。

机器终止动作是 STOP_REASSESS_FROZEN_INFERENCE_ROUTE。不得把较低地震 RMSE
解释为项目成功，也不得把 truth-oracle 最优候选当成可部署选择器。三个合成
inverse-crime 案例不能证明场地泛化。

当前协议明确没有实施训练、structured-search 修改、posterior chain、Stage 9B、
Stage 9C、posterior weighting、adaptive proposal、SMC、D-Flow、新 likelihood 或
重力融合。不要自动继续这些路线。下一步必须先由用户明确选择新的科学路线和
授权边界，再冻结新协议；在获得该授权前只允许只读审计和复现验证。
```

## 权威结果

- 规范：`docs/STAGE9A_FLOW_PRIOR_SUPPORT_SPEC.md`
- 人类报告：
  `experiments/stage9_flow_prior_posterior/reports/stage9a_prior_support_v1/STAGE9A_REPORT.md`
- 机器摘要：同目录 `summary.json`
- 正式证据：
  `experiments/stage9_flow_prior_posterior/runs/stage9a_prior_support_v1/formal/`
- 详细历史与边界：`docs/DEVELOPMENT_HANDOFF.md`

若本文与机器摘要冲突，以通过哈希校验的 `summary.json` 和冻结审计结果为准。
远程执行使用仓库根目录 `.venv`；不得在仓库文档、脚本或命令文件中保存密码。
