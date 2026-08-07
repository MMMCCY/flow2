# 推理阶段地球物理制导：完整因果诊断

## 最终结论

D1/D2 四个 implementation gate 全部通过：正演与观测完全闭合，解码/物性映射等价，float64 有限差分与 autograd 一致，实际 Euler 物理更新方向正确且局部下降。因此，本次失败不是正演、梯度符号、控制器符号或条件投影实现错误。

首个可重复的行为分叉出现在 **L3 reflectivity/TWT**：L0 probability、L1 expected property、L2 blurred property 的软/硬 attainment 均约为 100%，但 L3 最大软/硬仅为 10.80%/0%，L4 seismic 最大/最终软为 19.09%/19.09%，最大/最终硬仅为 2.44%/-0.80%。完整冻结 Flow 虽把最大硬 attainment 提高到 30.64%，最终又退化到 11.52%，并且 zero/shuffled 控制得到几乎相同或更强的改善。因此主要因果结论是：**高维冻结流状态上的物理 attainment 不足且缺乏正确观测特异性，随后受到软—硬边界、类别替代和轨迹终点破坏的共同限制。**

## 强制报告的 18 项结果

1. **Forward/observation closure**：PASS。property、reflectivity、seismic、gravity 的 truth loss、绝对差和相对差均为 0；baseline 与 truth 明确分离。
2. **Decoder/mapping equivalence**：PASS。多个连续状态与四个温度下 top-1 mismatch=0；所有类别 one-hot property/acoustic 误差=0；subsurface air renormalization 单独通过。
3. **Gradient correctness**：PASS。五条链的最佳相对误差为 `2.71e-9`–`1.83e-8`，方向符号全部匹配，负梯度小步全部下降。
4. **Raw gradient vs actual controller**：PASS。`cos(-grad, actual update)=0.9999999999999998`；实际 Euler 物理步使 soft loss 降低 `1.1335e-5`。控制器 unit-time 大步可破坏 hard loss，但这属于跨类别边界的尺度效应，不是符号实现错误。
5. **最大/最终 soft attainment**：generator-free seismic 为 `19.09% / 19.09%`；cuboid full-flow BASE_PLUS 为 `81.05% / 9.19%`；native 为 `80.79% / 12.56%`。
6. **最大/最终 hard attainment**：generator-free seismic 为 `2.44% / -0.80%`；cuboid full-flow 为 `30.64% / 11.52%`；native 为 `31.48% / 14.44%`。
7. **响应更接近 baseline 还是 truth**：D3 seismic 终点 hard response 更接近 baseline（距 baseline `0.7166`，距 truth `10.2136`）；D4 full-flow 终点相对 paired BASE 更接近 truth（`47.5412 < 62.7718`），但仍只取得 11.52% hard attainment，不能称为成功拟合。
8. **首次软/硬分叉层**：L3 reflectivity/TWT。L2 仍有约 100% hard attainment，L3 hard 始终为 0。
9. **条件投影抹除**：D3 projection erasure fraction=0；D4 条件状态修正范数最大 `7.1976`，但 projection 前后 soft/hard loss 差均为 0，故不是因果故障。
10. **决策边界穿越**：D3 seismic 最佳 hard step 126 只有 64 个 crossing，全部进入 raw label 9；step 200 crossing=0，尽管 soft loss 继续改善。full-flow 终点相对 BASE 有 14,055 个标签变化。
11. **错误岩性替代**：D3 step126 错类 crossing=0；full-flow 终点 14,055 个变化中 12,005 个落在非目标类，说明它是完整 flow 的次级机制，而不是 generator-free 首个故障。
12. **梯度空间分配**：D3 step126 的 hidden ROI 梯度能量占 `99.9996%`，施加更新占 `97.5365%`，condition 占 0；初始步较分散，但后期已正确局部化，仍未得到充分 hard attainment。
13. **地震 recording-window/dead-zone**：cropped interface fraction=0；最佳步 TWT RMSE `29.617 ms`、reflectivity RMSE `0.01807`、spike RMSE `0.00645`、wavelet RMSE `0.00807`，各层对更新均有响应，不支持窗口裁切或死区。
14. **最佳中间态 vs 最终态**：D3 的 observation-only hard-best 为 step126，`2.44% → -0.80% @200`，是明确的终点破坏。D4 的 paired hard advantage 为 `30.64% @23 → 11.52% @32`，但绝对 observed hard loss 仍在下降，所以 observation-only hard-best 正确地选中最终 step32；这里被破坏的是相对 BASE 的因果优势，而不是绝对观测拟合。末四分之一路径的相对优势已平台/恶化，故没有执行 2× budget。
15. **PHYSICS_ONLY vs BASE_PLUS_PHYSICS**：cuboid 最大 hard 为 `9.30% vs 30.64%`，不满足“PHYSICS_ONLY 成功、BASE_PLUS 失败”的 frozen-prior cancellation 模式；冻结先验抵消不是主要解释。
16. **correct vs zero vs shuffled**：cuboid BASE_PLUS 最大/最终 hard 分别为 correct `30.64/11.52%`、zero `30.25/29.94%`、shuffled `30.35/17.52%`；endpoint 为 `58.78%/64.47%/58.52%`。改善缺乏正确观测特异性。
17. **cuboid vs StructuralGeo-native**：原生事件历史为五个 `IntrusionSpec(kind=hemisphere)`，事件标签 9–13 的独立 mask 被保存后才合并为 audit label9。8/8 先验样本在条件外有目标标签，7/8 有原生尺度相容连通分量；但 correct native BASE_PLUS 仍只有 `31.48/14.44%` 最大/最终 hard，zero/shuffled 为 `29.58/27.31%`、`31.56/18.83%`。故 cuboid OOD 不是主要原因。
18. **机制排序**：① 高维冻结流状态上不足且非观测特异的 physics attainment（H6/H11）；② L3 起的 soft-hard gap 与类别边界饱和（H7/H8）；③ 最佳中间态被后续轨迹破坏（H13）；④ full-flow 错误岩性替代（H9）；⑤ 原生先验拓扑支持不完美，但仅为次级因素（H15）。

## 假说裁决摘要

| 假说 | 状态 | 核心判据 |
|---|---|---|
| H1 forward/observation mismatch | not_supported | 四条 truth closure 精确为零 |
| H2 decoder/mapping mismatch | not_supported | top-1 与 one-hot 全部等价 |
| H3 gradient error | not_supported | FD/autograd 与局部下降通过 |
| H4 controller sign/scaling implementation | not_supported | 实际更新方向和 Euler soft descent 通过 |
| H5 projection erasure | not_supported | loss erasure=0 |
| H6 insufficient attainment | supported | full-flow correct 最终 hard 仅11.52% |
| H7 soft-hard gap | supported | seismic 19.09% soft、-0.80% hard |
| H8 boundary saturation | supported | 最终零 crossing、梯度趋小 |
| H9 wrong-lithology substitution | partially_supported | full-flow 12,005 wrong-class flips，D3 最佳步为0 |
| H10 spatial misallocation | not_supported | 最佳步梯度99.9996%在 hidden ROI |
| H11 high-D/local landscape | partially_supported | Q1=100%，高维控制缺乏观测特异性 |
| H12 recording dead zone | not_supported | crop=0，各中间响应可变 |
| H13 stopping destruction | partially_supported | D3 绝对 hard-best 退化；D4 仅 paired advantage 退化 |
| H14 frozen-prior cancellation | not_supported | BASE_PLUS 明显优于 PHYSICS_ONLY |
| H15 cuboid OOD/support mismatch | not_supported | native 粗支持存在且复现失败 |
| H16 true nonuniqueness | not_supported | 尚未拟合 truth physics，不满足使用前提 |

每个假说的 strongest evidence、counter evidence、run 与 metric 均见 [diagnostic_verdict.json](diagnostic_verdict.json)。

## 可复现性与停止条件

- 权威代码基线：`main @ 11caa498b15e6b89891604e9537830b30df504fa`。
- 正式 CUDA：NVIDIA GeForce RTX 4090 D，PyTorch 2.8.0+cu128，冻结 EMA checkpoint SHA256 `561e94bf...94c`。
- 所有正式目录不可覆盖；最初 D2 float32 不可表示探针结果保留在 `d2_gradient_audit_v1`，权威修正版为 `d2_gradient_audit_v1_fix1`。
- 完整项目测试：`181 passed, 13 warnings`；warnings 均为 matplotlib/pyparsing deprecation。
- 未训练、未微调、未新增推理算法、未做 alpha/mu/tau/cap sweep。D6 到此停止。
