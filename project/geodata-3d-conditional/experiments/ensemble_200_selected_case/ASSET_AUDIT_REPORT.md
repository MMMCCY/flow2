# 开工资产核对（2026-10-09）

已通过 SSH 登录用户指定 `xmj@172.27.231.254:22`，主机 `LabRS`，RTX4090D。
远程开发根与当前工作区相同：
`/home/xmj/mcy/flowtrain_stochastic_interpolation-main`。
远程 `.venv/bin/python` 为 PyTorch 2.8.0+cu128、CUDA 可用；系统
`python` 无 numpy，使用项目虚拟环境。两个终端看到同一工作区改动与资产。
项目本地 SSH alias 配置为根目录 `labrs-codex.ssh.conf`，可用
`ssh -F labrs-codex.ssh.conf labrs-codex`。密码没有写入配置或文件。

Git：`main`，HEAD `b780b9ae3650b426400d3258b8e33f10a349bf6c`。
开工时已有未提交 Stage19R report 及 visualization 改动，全部保留。
没有 checkout/reset/clean/push，没有更新旧实验输出。

已先读取03提示词、Stage19原始开发规范、补充README和随附测试。
Stage19与Stage19R开发已存在；本次不重复cohort生成或适配器训练。
后续规范中的 Stage19R 正式 epoch4 adapter 为 54,003 参数，正式训练
1024 updates；采用 formal_v2 checkpoint，不使用 smoke checkpoint。

## 路线定位

- Stage15 trace v4：`experiments/stage15_binary_seismic_consensus/trace_boundary/flow_property_seed{42,142,242}_v4/{baseline,guided}`。
  `property_manifest.json` 实际位于各 arm 子目录，采样入口为
  `scripts/stage2/run_property_guidance.py`，复用
  `guidance/property_sampling.py:fixed_euler_property_sample`。
  property资产为 `trace_boundary/property_assets_cond_generation_0_v4`。
- Stage19R：`experiments/stage19_learned_evidence_adapter/formal/inference_v2`，
  12 cases × 3 seeds × 5 arms，正式样本180个，单案例/arm仅3个，不能冒充200。
- Stage15五体配置与资产存在。用户明确本次不做五体，故只作为审计参考。
- Stage9是未制导先验池，不替代此次路线。

## checkpoint、数据与论文

基础 checkpoint：
`project/geodata-3d-conditional/demo_model/conditional-weights.ckpt`，
849415038 bytes，SHA256：
`561e94bfda770ec41fc4cbed43436a7e2130eef5dfb7e5d666fcefc0724ff94c`。
embedding采用raw checkpoint，其余411个trainable tensor采用EMA。
Stage19R adapter：
`experiments/stage19_learned_evidence_adapter/checkpoints/formal_v2/adapter_checkpoint.pt`，
SHA256：`cdb740bfeccc08f7010048c69deb8e44a13917fe53e3b2f4b46910bf536f20bb`。
已在SSH终端实际计算两个文件SHA，与正式记录一致。

实验数据根为 `project/geodata-3d-conditional/experiments/`。
本次共享案例根为
`experiments/stage19_learned_evidence_adapter/{cohort_v2,observations_v2,evidence_v2}/stage19_test_case004`。
推理仅使用 stripped `evidence_v2/test_inference_registry.json` 的case记录，
condition_values/mask、support、q、observed seismic均逐文件验证hash。
真值由后处理从独立 `evidence_v2/evidence_registry.json` 读取。
详细实际路径、hash、config、代码hash在 `run_v2/protocol.json`。

论文：仓库根 `Synthetic_Geology_Structural_Geology_Meets_Deep_Le.pdf`。
已用pdftotext读取Figure8/9及3.3节：12个实现、100个集合，原阈值
5/33/62/90%。本次严格>40/60/80%及全部命中采用独立图注。

## 具体缺口与处理

`ensemble_occurrence.py` 在工作区及服务器搜索均未找到；补充README与
6个测试存在。不是缺少checkpoint或地质数据；缺少的是参考程序本体。
本次按给定接口和测试重建该程序，记录这一差异，不冒称已读到原文件。
原始6个测试全部通过。新增chunk/offset、raw标签、外露面边界/物理间距、
200无放回选择、稳定ties、阈值嵌套、非法模型拒绝测试；与Stage19回归合计
36 passed（13个既有Matplotlib弃用警告）。

## 案例选择与迁移声明

用户授权以历史adapter三种子中位目标IoU最高案例作为两条方法共享案例。
选择 `stage19_test_case004`：IoU `[0.6798255341,0.6710347944,0.6421464352]`，
median `0.6710347944`。全部12案例排序保存到新protocol。
这是真值辅助事后case选择，不能支持无偏跨案例性能结论。

Stage15固定方法在此新TEST案例上迁移；不是历史cond_generation_0实验。
两条方法均复用原连续full-trace evidence，不重反演或调参数。
历史cond_generation_0的v4 target/support与confidence/score构造已实际
逐体素核对一致。新case前三个seed的Stage19R baseline与adapter输出
逐体素等于已有正式输出，独立property alpha0/adapter scale0也一致。
