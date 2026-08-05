# Research goal

最终目标：在地表和稀疏钻井不足以约束深部侵入体时，利用推理阶段
全局三维/地球物理观测改善生成模型的hard-label几何；同时用flow2
学习到的地质结构先验以及地表/钻井硬条件限制地球物理的非唯一解。
最终结果应是观测一致、条件严格满足、地质结构合理且保留可解释
不确定性的三维模型集合，而不是只降低连续损失的单一结果。

当前证据（2026-07-31）：
- cond_generation_0中label 9共有8968体素；
- 钻井只约束13个；
- 当前2D重力引导平均只改变31.25个hard体素；
- oracle三维概率体在3个seed的12/12严格配对样本上改善label 9的
  IoU、Precision、Recall和质心距离，且条件违背为0；
- Phase 1b protocol-v4在3个seed共12/12严格配对样本上将label 9
  IoU由0.0314提高到0.8099、Precision由0.0788提高到0.8274、
  Recall由0.0520提高到0.9747，ROI IoU达到0.9392；
- 平均5.18%的全模型hard体素改变，其中98.76%位于预先定义的ROI，
  地表/钻井条件违背为0，3组ensemble均保留4/4个独立样本；
- 原始连通域比例1.281略高于预注册1.25阈值，不能宣称强门槛全部
  通过；但12个样本均恢复4个大于等于20体素的ROI主体，微小碎片
  仅占ROI目标体素约0.66%；
- 这证明推理时三维可微引导和soft-hard跨越机制有效，但仍未证明
  真实地球物理能够单独重建三维地质。

当前阶段：
- Phase 0已完成；
- Phase 1以“机制验证成功、拓扑和末端稳定性保留局限”结束，正式
  报告见`docs/PHASE1_REPORT.md`；
- Phase 2a已经完成：全类别soft probability到多通道期望物性的映射、
  匹配算子的多尺度三维属性loss、独立fixed-Euler sampler、全局/逐类别
  及几何评价、严格配对runner和alpha=0回归均已实现；
- 理想全分辨率density+susceptibility属性上限在3个seed的12/12严格配对
  样本上通过冻结门槛；Phase 2b物性重叠/对比度敏感性的协议、配置、
  runner阶段标记、单样本launcher和汇总门槛已实现；distinct codebook
  锚点GPU严格配对及回归门槛通过；首个高对比度paired codebook也通过
  单样本完整门槛；label 9 susceptibility降至0.025后仍窄幅通过，
  降至0.010后首次在绝对label 9、主体恢复和碎片质量门槛失败，表明当前
  单样本转折位于0.025与0.010之间；0.004完全同码负控制同样失败且不再
  恢复label 9主体；冻结规则选择0.025/0.010进入seed42 n=4确认；0.025
  的n=4结果为3/4通过，属于过渡区；0.010为0/4确认失败。两者均保持
  ensemble多样性，但都不能进入多seed。另行预注册的高对比度paired_c100
  seed42 n=4同样为3/4过渡态；失败样本仅在四个主要真值体的最小召回率
  `0.2313 < 0.25`。Phase 2b因此关闭，没有歧义codebook满足冻结的4/4
  多seed晋级条件；
- 第一组物性仍是truth-derived、全分辨率、无噪声的理想属性oracle；
- 理想属性体通过后才进行降分辨率、模糊、缺失、深度衰减和噪声；
- 暂不合并2D重力、磁法或其他正演场引导。
- Phase 3和Phase 4的协议已分别冻结：Phase 3独立研究三维属性的空间
  退化，Phase 4研究采集域物理；二者共享观测算子接口但不能混合证据；
- Phase 3的identity/Gaussian核心、严格配对runner、审计及汇总脚本、
  n=4边界launcher和CPU门已实现；seed42单样本
  四档筛选已完成：identity通过，sigma 1/2/4均未通过完整hard门槛，尽管
  连续观测loss全部下降；label 9 IoU和主要真值体平均召回率随模糊程度
  单调下降，因此当前没有非零高斯退化层级被晋级；下一证据是按冻结规则
  复核identity与sigma 1的seed42 n=4边界；identity已4/4完整通过并保持
  ensemble多样性，确认未退化锚点稳健；sigma 1则0/4确认失败，四对均未
  通过label 9目标阈值、其中两对还未通过主体恢复。Phase 3因此以“无非零
  高斯退化工作点”的负结果关闭，不再运行sigma 2/4多样本或seed142/242；
- Phase 4a的全支撑矩形棱柱重力正演、SI到mGal单位、零填充FFT线性卷积、
  全类别密度映射、mask/不确定性loss、硬条件零梯度覆盖、固定观测资产
  builder以及既有fixed-Euler注入已实现；严格配对GPU runner、alpha=0
  回归审计、n=4基线重排比较器和controller预注册manifest也已实现；解析
  几何核固定以CPU float64构建后再转推理dtype，避免深层远场float32相消；
  Phase4专项18项CPU门和完整103项测试均通过；
  cond_generation_0的64立方无噪声全网格观测及首个seed42、n=1、32步、
  alpha/cap=0.25 GPU严格配对已经完成。配对、alpha=0回归及逐步硬条件均
  通过，hard重力RMSE下降0.06561 mGal，但完整地质门槛为0/1：label 9的
  IoU/precision/recall由0.0286/0.0675/0.0473降至
  0.0159/0.0638/0.0207，预测体积由6283降至2914，四个主要真值体的
  最低召回仍为0。6562个hard体素发生变化，其中3392个由label 9转出，
  仅23个转入。这证明当前二维重力可通过岩性/密度非唯一重排来降低场
  残差，却没有恢复三维目标体。alpha=0.10低强度伤害诊断也已完成并为
  0/1失败：hard重力RMSE进一步降至0.87032 mGal，但label 9的
  IoU/precision/recall仍降至0.02111/0.06490/0.03033，2101个体素由
  label 9转出而仅9个转入，主要真值体最低召回仍为0。两个预注册控制器
  均已耗尽，Phase 4a以筛选级负结果关闭，不运行n=4或额外alpha。下一
  独立上限转向具有时间/深度定位的卷积地震响应，重力只保留为后续联合
  物理的一个非唯一观测。历史9×9、G=1的局部proxy只保留为负面对照。
- Phase 4c卷积地震上限的第一实现单元已经完成：完整密度/速度/阻抗/慢度
  码表、局部地表基准双程时、线性时间采样、25 Hz Ricker卷积、地下soft
  空气排除、有限记录窗预测裁剪、条件零梯度、不可变观测builder、EMA/
  fixed-Euler严格配对runner和完整hard审计均已实现。canonical fix2观测
  为1×1×64×64×320、振幅范围-0.4716至0.4931、最大真值双程时
  1428.43 ms；Phase4c专项15项和完整118项测试通过，真实checkpoint的
  alpha=0及正alpha一步CPU配对smoke均通过且条件违背为0。seed42、n=1、
  32步、alpha/cap=0.25 GPU严格配对现已完成：hard地震RMSE由0.042262
  降至0.039048，但完整地质门槛为0/1；label 9的IoU/recall由
  0.02860/0.04728降至0.02593/0.03947，四个主要真值体的恢复均未优于
  baseline。最终guided相对baseline改变3737个体素，其中1438个由label 9
  转出、仅193个转入；总变化率1.4256%，末Euler步churn为0.2678%，均未
  超过冻结上限。因此失败不是预注册的“过强变化”情形，不运行alpha=0.10；
  Phase 4c以负筛选关闭，不运行n=4、不额外搜索alpha，也不合并重力。正式
  结论见`docs/PHASE4C_REPORT.md`。下一方案必须针对声学响应到hard岩性的
  可辨识性，而不能只调整控制器强度。
- Phase 4d已经完成固定先验候选的可辨识性与后验选择诊断。它只读取
  Phase 2a中seed42/142/242共12个EMA、fixed-Euler、alpha=0样本，不生成或
  修改地质体，并用Phase 4c hard地震loss独立排序。12个候选无一达到
  label 9和主要真值体支持门槛；oracle最优IoU/recall/主体平均召回也仅为
  0.0672/0.0959/0.0963。地震top-3的对应均值反而低于全体均值，loss与
  label 9 IoU、recall、主体平均召回的Spearman系数为
  +0.552/+0.587/+0.580，与“低loss对应好地质”的预期方向相反。把真值中
  8955个未约束label 9全部替换仍产生至少0.017692地震RMSE，说明算子能看见
  整体变化，但该信号被候选约0.041至0.046的其他界面/时深误差淹没。
  Phase 4d因此以“候选支持不足且似然排序失配”的双重负结果关闭，不扩大
  pool、不调truth-informed排序。下一主线应单独冻结并授权地球物理感知
  训练或微调，使生成分布本身学习物理兼容性，而不是继续调推理alpha。
- 用户随后授权先尝试不进行大规模训练的有界桥接方案。Phase 5a已完成：
  使用同一Phase 4c地震观测和固定12个flow先验进行模型驱动log-impedance
  反演，构建器不读取未约束真值，独立审计后地震RMSE、未约束阻抗RMSE和
  label 9区域阻抗MAE均在12/12成员改善，条件违背为0，因此允许进入一次
  严格配对的flow属性桥接测试。但最近声学码表hard投影的平均全局accuracy
  从0.5972降至0.5163、label 9 IoU/recall从0.03144/0.05196降至
  0.01912/0.02496，说明连续三维反演体仍未解决岩性非唯一性。Phase 5a
  不是最终成功；下一步只能做单样本Phase 5b gate，若hard-label和主体几何
  不改善则停止无训练桥，不得以连续loss下降替代目标。
- Phase 5b的代码、truth-blind log-impedance目标、spread置信度、严格配对
  launcher和hard审计已完成；CPU真实checkpoint单步pair确认EMA、同噪声、
  非零梯度和条件违背0。正式seed42、n=1、32步GPU证据已经完成并失败。
  配对、EMA、alpha=0历史回归、条件和hard bridge loss下降均通过，但全局
  accuracy下降0.000213，只有4/8个真值类别IoU改善；guided label 9的
  IoU/precision/recall仅0.02890/0.06870/0.04750，四个主要真值体召回为
  0.04266/0.00046/0.12286/0。1212个hard体素发生改变，说明soft-hard机制
  活跃，但属性后验仍不能确定正确岩性和结构。Phase 5b按冻结规则关闭，
  不运行n=4、多seed或alpha/confidence搜索。下一方案若继续，应是冻结原
  U-Net/checkpoint、只训练小型地球物理条件adapter的新协议。

Phase 2成功标准：
- 严格配对的多个seed上，全局mIoU和体素准确率一致改善；
- 多数真值中出现的非空气类别IoU改善，且不能由单一类别吞并体积；
- label 9仍保持实质hard-label与几何改善；
- 三维属性loss的下降必须转化为hard类别和hard属性残差改善；
- 地表和钻井违背保持0，结果保持合理多样性；
- 原始拓扑、尺寸分层拓扑和末端hard churn必须同时报告。

边界结论：
- label 9仍是稀疏条件压力测试，不需要因13个钻井命中而换label；
- 稀疏钻井使全局三维信息更有研究价值，不等于可以删除条件；
- 概率体是truth-derived oracle，不是实测地球物理；
- Phase 2理想属性体同样是truth-derived oracle/inversion surrogate，
  不是实测或真实反演属性；
- 当前受控density proxy不是校准物性，只用于验证全类别属性路径；
- 第一组scalar density proxy严格GPU配对已完成：全局accuracy、mIoU和
  hard-property residual改善，但label 9召回率与体积误差恶化，因此没有
  达到Phase 2地质验收门槛，不能只凭连续loss下降宣称成功；
- 第二组density+susceptibility contrast严格GPU配对已完成：label 9的
  IoU、precision、recall和真阳性数均提高，说明“物性可辨识性”确实影响
  hard-label方向；但体积仍严重不足且组件数由37增至109，尚未实现完整
  三维几何恢复。它仍不是真实磁法数据；
- alpha=cap=0.10时，24个有效引导步中18步触及cap；下一步保持物性目标
  不变完成了alpha=cap=0.25单样本控制变量上限实验；label 9的
  IoU/precision/recall达到0.4816/0.9005/0.5087，质心距离降至3.42，四个
  主要真值体均得到部分恢复；
- 该上限实验仍有明显碎片化，但top-8组件包含87.2%的预测质量，≤5体素
  碎片占5.37%；下一步按`docs/PHASE2_PROGRESS.md`中预注册门槛运行
  seed42 n=4；该批次全部8项门槛通过，四对的全局accuracy、固定类别
  mIoU及label 9 IoU/precision/recall均改善；随后完成的seed142/242也按
  相同冻结门槛评价，仍不直接宣称完整Phase 2或真实地球物理成功；
- seed142和seed242同样通过，最终12/12严格配对通过Phase 2a门槛；平均
  label 9 IoU/precision/recall由0.0314/0.0788/0.0520提高到
  0.4808/0.9032/0.5075，全局accuracy与固定类别mIoU也一致改善；
- Phase 2a因此以“理想三维物性上限验证成功，几何/类别/确定性仍有限”
  结束；正式结论见`docs/PHASE2A_REPORT.md`。下一步先做Phase 2b物性
  重叠、对比度降低和codebook敏感性，再做空间分辨率与噪声退化；
- Phase 2b已经完成：c100/c025均为3/4过渡态，c010为0/4失败，完全同码
  负控制在单样本筛选中失败。该阶段证明物性歧义会降低hard几何恢复的
  稳健性，但不等于真实地球物理必然失败；Phase 3应从唯一12/12验证的
  distinct Phase-2a属性上限出发，单独研究空间退化，避免混淆两种失效；
- 连续loss和概率等值面不能替代hard-label及连通几何评价。
- “地球物理补全稀疏条件”和“生成先验减少地球物理非唯一性”是同一
  最终系统的两个方向；当前阶段不删除地表/钻井条件，也不宣称仅凭
  地球物理可唯一还原三维岩性。
