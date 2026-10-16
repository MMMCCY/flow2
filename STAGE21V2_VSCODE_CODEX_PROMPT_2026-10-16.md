# Stage21v2：密集准确井与同位置伪井的冻结Flow实验

你将直接在服务器上的代码库完成开发、GPU推理、结果分析和图件交付。这个任务已经获得用户明确授权。不要只返回计划、开发建议或一份新的提示词；先核查实际环境，然后完成本任务定义的最小实验。遇到真正改变科学含义、无法按既定协议解决的问题时才询问用户。

## 一、工作位置和授权范围

- 预期VSCode已通过Remote SSH打开服务器 `labrs-codex`。
- HostName：`172.27.231.254`；User：`xmj`；Port：`22`。
- 仓库：`/home/xmj/mcy/flowtrain_stochastic_interpolation-main`。
- 主要项目：`project/geodata-3d-conditional`，下文记作P。
- GitHub：`https://github.com/MMMCCY/flow2.git`。
- 使用服务器仓库的 `.venv/bin/python`。已确认系统默认 `/a/anaconda3/bin/python` 没有torch，不要用它运行实验。
- 2026-10-16只读检查时GPU为RTX4090D，约24GB显存，处于空闲；执行前重新检查占用，不中断其他任务。
- 本次明确授权代码开发及下面定义的有限实验；不训练、不微调、不新增adapter、不修改基础U-Net、embedding、checkpoint或既有采样器。
- 不自动运行真实地球物理正反演、不开展参数扫描、不自动commit/push、不覆盖历史结果。
- 如果当前窗口不在预期服务器上，先用已有Remote SSH配置和已有凭据连接；不要在源码、配置、日志或报告里保存SSH密码。

本任务是用户对既有MCYR/MCYC工作分工的本次具体开发授权。仓库中的历史Stage7/8路线和停止条款是背景，不能把它们误读为禁止这次已经明确授权的新实验；保留数据隔离、条件精确和历史结果不可覆盖等通用要求。

## 二、接手时已核实的仓库状态

2026-10-16服务器HEAD和GitHub main均为：

`b780b9ae3650b426400d3258b8e33f10a349bf6c`

origin为 `https://github.com/MMMCCY/flow2.git`，当时分支为main。远程服务器有尚未推送的开发和结果，因此GitHub内容并不代表服务器完整状态。

当时已有两份tracked修改：

1. `P/docs/DEVELOPMENT_HANDOFF.md`：新增Stage21及200集合实验交接内容。
2. `P/experiments/stage19_learned_evidence_adapter/reports/STAGE19R_DEVELOPMENT_REPORT.md`：新增可视化说明。

还有untracked的ensemble脚本、结果、Stage19图件，以及完整Stage21脚本/测试/结果。它们都是已有用户工作，不能reset、clean、删除、覆盖或重新整理。

原Stage21已经存在：

- `P/scripts/stage21/{common,prepare,run,evaluate,report,audit}.py`
- `P/tests/test_stage21_virtual_boreholes.py`
- `P/experiments/stage21_virtual_borehole_conditioning/`
- 已完成：`.../run_20261014_v2/formal/`
- 当前报告：`.../run_20261014_v2/reports_v3/REPORT.md`
- `.../run_20261014_v1` 是保留的初始化失败，不当成科学结果。
- `run_20261014_v2/reports_v1/v2/v3` 是相同45份预测的报告修订，不是三次独立实验。

原Stage21已做5个fullgeo案例×3个source seeds（42/142/242）×总井数9/16/25，共45份正式预测。准确新增井带来了部分井外类别收益，但7个原零井命中主要对象只有1个被新增井命中，未证明可靠隐藏对象恢复。不要重新运行这45份实验，不要将9/16/25当成本次密集井实验。

本窗口之前的SSH会话只读取了仓库和环境，没有创建Stage21v2，没有修改服务器文件，也没有启动新GPU任务。

## 三、新工作命名和先做的检查

原Stage21已经存在，因此使用：

- 新代码命名空间：`P/scripts/stage21v2/`
- 新实验根：`P/experiments/stage21v2_dense_virtual_boreholes/`
- 新测试：`P/tests/test_stage21v2_dense_virtual_boreholes.py`

不要把原 `run_20261014_v2` 误认为新阶段stage21v2。若接手时上述新目录已经被其他工作创建，先检查能否安全续接，不覆盖，不擅自开始第三个新阶段。

先完成并保存简洁的 `REPO_DIFFERENCES.md`：当前branch/HEAD、GitHub main、tracked diff摘要、与本任务有关的untracked内容、旧Stage21的实际完成状态。优先只读查询GitHub引用，不为对比而pull/merge/stash。记录当前代码和结果的来源，不收集无关大文件。

阅读适用AGENTS.md、最新handoff、原Stage21 README/报告及相关实现。旧文档不能代替实际源码和结果。把不影响本实验的旧阶段规范当历史背景，避免重复执行全部历史验证。

## 四、科学问题和结论边界

唯一核心问题：

> 在保持原9口真实井、地表和Flow权重不变时，增加准确完整井能否改善未直接作为条件提供的区域？将同位置井标签替换为平滑属性重新分类得到的伪井后，这种收益是否仍存在？

三个环节必须区分：

1. 准确井条件的信息效应；
2. 退化属性转换为伪岩性的错误；
3. 冻结Flow对带误差伪井的使用效果。

这不是证明真实地球物理已成功，也不是精确后验采样或不确定性校准研究。当前单例用于开发诊断；同一地质模型的多个source seeds不是独立地质案例。

Flow使用固定类别嵌入、条件flow matching和硬解码。原训练 `make_boreholes_mask` 实际使用 `torch.randint(8,32)`，即8–31口完整竖直井；512及以上属于条件密度外推。井位置为整数XY，每口覆盖全部64个Z体素。raw标签-1…13，embedding索引0…14，不能弄错加1/减1。

当前采样会硬写回条件，因此4096口完整真值井给出真值、4096口伪井给出伪岩性体，是写入端点。不要为这两个端点运行Flow，也不要用其高准确率宣称生成器恢复能力强。

## 五、固定数据与可复用实现

开发案例固定为 `fullgeo_case03`，不按本轮指标选择案例。选择理由是旧Stage21和已有smoke已有完整来源与基线，便于复用；不是因为它表现最好。

来源以服务器实际清单为准：

- `P/experiments/full_structuralgeo_benchmark/benchmark_manifest.json`
- `P/experiments/full_structuralgeo_benchmark/cases/fullgeo_case03/`
- 该案例 `truth/true_model.pt` 与 `condition/` 下的condition_values、condition_mask、surface_mask、borehole_mask、well_xy.json。
- 可以从旧Stage21的inference/evaluation registry定位经过核验的相同资产，不能靠猜文件名替换。

保持原9井、原surface/air条件、原地下范围与标签语义。原井XY为：

`[(8,46),(9,5),(10,24),(27,17),(35,26),(39,59),(44,60),(48,6),(57,32)]`

checkpoint：`P/demo_model/conditional-weights.ckpt`

预期SHA-256：

`561e94bfda770ec41fc4cbed43436a7e2130eef5dfb7e5d666fcefc0724ff94c`

复用以下已成功实现，先检查实际函数契约：

- `inference_runtime.load_model_with_weight_policy`：EMA加载，冻结embedding用checkpoint原值。
- `guidance/generator_posterior.py` 的 `projected_fixed_euler_prior_sample`。
- `scripts/stage21/run.py` 的 `initial_noise`、`sample_native`：已有条件精确/有限值检查。
- `scripts/stage21/common.py` 中的hash、well_mask、条件资产等通用功能。
- `scripts/stage21/evaluate.py` 中的 `voxel_metrics`、26连通ID、`match_objects` 等。
- `guidance/property_volume.py` 的完整类别物性映射与Gaussian平滑实现。

不要修改旧模块的全局常量，或为让旧registry通过而篡改原Stage21的CASE/LEVELS约束。新协议在stage21v2适配；尽量调用已有纯函数，不复制整套框架。

旧Stage21对应案例/seed的N9预测可作为A基线复用，前提是核对checkpoint/EMA、原条件、投影前初噪声、32步时间网格、精度/确定性、采样实现来源相符，并校验预测文件hash。保存复用指针和审计依据，不复制为新独立实验。无法确认等价时仅重算所需A，不重跑旧45份。

## 六、固定的伪岩性构建方式：不训练，不扫描

先只构建一次完整伪岩性体，后续各井数只对它采样。

使用既有Phase2a双属性码本：

`P/experiments/stage2_property/configs/ideal_density_susceptibility_label9_contrast_v1.json`

该码本为人为设定的density_proxy及susceptibility_contrast_proxy，label9具有额外对比。它不是校准岩石物理，不是重力/磁场或地震反演。必须在报告中说明这是可分属性上限下的受控证据，不能悄悄换为更有利的码本。

冻结以下规则：

1. 用完整raw类别表将真值映射为两个属性通道。
2. 非零平滑只用 `sigma=1.0 voxel`，三个轴相同，复用已有separable Gaussian及其replicate边界约定；不试sigma2/4，不加噪声、不搜索最佳sigma。
3. 对各非空气候选类别0…13计算属性距离。每通道用码本非空气候选值的max-min作固定尺度归一化；使用配置中通道权重归一化后的加权平方欧氏距离。尺度不得使用当前真值类别统计或TEST数据拟合。
4. 取距离最小的类别；并列按raw标签从小到大打破。所有非空气类别都参与，不根据本例缺哪些岩性删候选。
5. 空气/地表和原9井仍由原condition_values/mask保证准确。伪井只能改变新增位置中不属于原硬条件的体素；原条件永远优先。
6. 平滑时使用完整属性体，不先按拟新增井取样再平滑。原已知空气的属性也是码本定义的值；按旧Gaussian实现处理，不另引入mask-normalized平滑或隐藏真值ROI。
7. sigma=0只做CPU往返诊断，不运行额外Flow。检查码本类别是否可分，报告碰撞；不把碰撞伪装成分类器错误。这个固定双属性码本预期可分，若实际不符先诊断配置/索引。

保存码本与hash、完整平滑属性、完整伪岩性体、分类规则、sigma0往返记录。对伪岩性体一次性计算全地下/原未知区域的混淆矩阵、各类指标、目标precision/recall/IoU和对象诊断；不因该结果不漂亮而调整平滑尺度或映射规则。

完整伪岩性体P也是不调用Flow的廉价比较对象。推理时只能看到所抽取的伪井，不能把完整P传给Flow。

Stage19/19R adapter不是物性→岩性的独立分类器：它依赖状态、基速度、条件、时间和q输出速度修正。本轮不加载它，不重新训练分类器。

## 七、井位和固定评价域

布局seed固定 `21601`，numpy `Generator(PCG64)`。布局与真值内容无关，开发例与后续核对例共用。

确定性构造如下：

1. 将64×64 XY网格划为16×16个4×4小格，按小格行优先顺序处理。
2. 每格从按XY字典序排列、排除原9井的坐标中均匀选1点，得到256个holdout柱。所有格都有可选点。
3. 从4096个XY中排除W9和holdout，得到3831个新增可用坐标。
4. 同一RNG继续：每个4×4格的剩余坐标先独立随机排列；再逐轮遍历仍有坐标的小格，每轮随机排列这些小格，每格取其队列下一个坐标。得到完整新增井顺序，兼顾嵌套与空间覆盖。
5. 总井数N的井集为W9加顺序前N-9个点，N取512、2048、3840。保存全顺序和各档XY，不重复抽样、无坐标碰撞。

所有臂共同的主评价域：

`E_holdout = holdout_columns & ~original_condition_mask & (truth != -1)`

辅助评价域：

`E_original_free = ~original_condition_mask & (truth != -1)`

主域永不加入新硬条件，三档同域同分母。辅助域允许包含新井位置，反映伪井错误和最终整体代价。按档自身free域可作为额外描述，但不能替换共同主域作主曲线。

这些域中的truth!=−1只用于离线评价，不可作为推理时新增掩膜信息。holdout没有作为类别硬条件输入，不代表它没有贡献给合成平滑属性；本任务不是严格独立的观测留出验证。

评价某类在固定域无真值时记N/A，不填0/1，不根据真值重选holdout或井位。主mIoU的类别集合按该案例固定评价域的真值出现类别预先定义，所有臂保持相同；保存逐类交并和分母。

## 八、最小实验臂和推进规则

每一井数的主要比较：

- A / `FLOW_ORIGINAL_9`：原9口真实井＋原surface，全部N共用。
- B / `FLOW_ORACLE_N`：同一新增井坐标取真值标签的准确井上限。
- C / `FLOW_PSEUDO_N`：同一新增井坐标取固定伪岩性体的标签。
- D / `COPY_ONLY_N`：只在A的硬结果上写入C的新增硬条件，不重新运行Flow。D不产生额外网络前向；其holdout结果必须与A相同。
- P / `PROPERTY_CLASSIFICATION_ONLY`：完整伪岩性体，加回原硬条件，不运行Flow。P不受新增井数量影响。

先用开发案例、source seed42筛查，A仅需一份：

1. 按512→2048→3840顺序运行B。
2. 只有当B在E_holdout的固定类别mIoU严格高于A时，才运行该档C。否则保留负结果并进入下一档，不宣告所有伪井不可能有用。
3. C同时满足以下条件时，只标为“待确认候选”，暂停更高密度筛查：
   - E_holdout固定类别mIoU > A；
   - E_original_free固定类别mIoU >= A；
   - E_original_free目标label9 IoU > A。
4. 若该档C不满足，保留结果并进入下一档；不改sigma、分类器、井位或seed。
5. 最多筛查三档。3840仍无候选就完成负/混合结果报告，不运行4096口Flow。

有首个候选时，只在该一个密度补source seeds142/242，复用/生成各自A、运行对应B/C，D照常离线生成。不做三井数×三seed完整矩阵。

候选确认的描述性判据：三个seed的C−A主域mIoU均>0，且三个seed配对差中位数在E_original_free的mIoU与目标IoU均>0。此规则只决定是否值得下一步核对，不是通用科学成功阈值；不能把极小正差写成实质突破。

同时保留precision/recall、体积、合并/分裂和碎片：若类别分数虽改善但对象结构明显恶化，应明确报告“类别收益、对象恢复未成立”，不能宣称隐藏体已恢复。不要临时添加一个挑结果的结构阈值。

确认不一致时，本轮记为混合证据并收束；不要靠增加更多seed或密度重新寻找通过点。没有测试到的更高密度记为未测试，不能据此否定它。

开发例确认通过后，固定该密度、相同布局/分类/sigma，在既有 `fullgeo_case01`、`fullgeo_case02` 各用seed42做A/B/C核对，D/P离线计算；不在这些案例重新搜索数量。它们也是历史用过的案例，不称全新TEST。若未通过开发例确认，不运行这一步。

网络预算：全部基线需重算时，开发例最多7次筛查输出（A一次，B/C最多各三次）＋6次候选确认输出；补两案例最多6次。正确复用旧A会减少相应次数。未出现候选时没有确认/跨例运行。CPU转换、指标和图不算新增Flow推理。

注意：判定在完成每个固定臂并封存预测后由外部离线评价程序作出，只决定是否开展下一份开发实验。模型内部、单条生成轨迹、样本选择不读取真值；这是一套明确使用开发真值的顺序诊断，不伪装成truth-blind方法调参或正式TEST。

## 九、推理协议与必要的工程检查

- 原Flow与embedding冻结；正确加载411项trainable EMA再冻结，embedding用原checkpoint值。
- 32步fixed Euler；时间为(step+0.5)/32，每步一次速度评估，不是二阶RK2。
- FP32，无autocast/TF32；沿用旧Stage21的确定性设置与CUBLAS约定。
- CPU用source seed生成完整Gaussian初噪声；先生成再按各臂mask投影。同case/seed的A/B/C必须使用同一投影前噪声hash。
- 相同N的B/C条件mask完全相同，只允许新增井的类别值不同。原9井和surface/air逐元素不变，mask外condition_values保持未知占位值-1。
- 复用现成sample_native和硬投影检查，原真实条件违例与新增条件违例分别记录。C必须符合其声明的伪条件，不能暗中用真值纠正新增井。
- 准备阶段允许读取truth构造准确井与合成属性；推理registry只列出准备好的条件资产和必要模型配置，不含truth/full伪体/对象几何路径。评价阶段单独读取truth和完整P。
- 每份预测保存condition/noise/decoded hash、case/seed/N/arm、source/config/基础checkpoint来源、耗时和实际前向数；记录基模型不变。复用已有hash工具，不为每一输出重复建设复杂审计框架。
- 不覆盖已有run目录；中断可按保存的hash和完成状态续接。不要把失败运行目录清空再跑同名。

只做与新增行为有关的轻量测试：

1. 嵌套井数量、无重复、holdout不相交及确定性布局；
2. B/C同mask，原条件优先，mask外无隐藏标签；
3. sigma0及固定分类距离的类别往返/并列策略；
4. 共同评价域/类别集合不随N或arm改变；
5. COPY_ONLY在holdout与A一致。

再运行相关已有采样器测试即可，不重做D1–D7、全库历史测试、梯度差分、全部旧smoke。新增runner需要一次真实CUDA检查，可将第一个有效、协议一致且完整保存的B输出计入本轮开发结果，不必机械额外生成同配置smoke；如果CPU/GPU检查暴露代码错误，修复并保留失败记录，不能混入有效结果。

不要将旧Stage21 run.py自身的“必须45份formal＋独立smoke”命令行门槛原样搬到新阶段。复用其已测试采样函数，在stage21v2定义本次最小驱动。

## 十、评价与最少图件

输出一张完整样本表，包含case/seed/N/arm、域、准确率、固定类别mIoU、逐类IoU/分母、label9 IoU/P/R、目标体积/绝对相对体积误差。主/辅助域列名明确区分。

复用旧对象评价：在完整目标体上建立26连通ID，再在评价域作一对一匹配；保存原M9零命中主要对象（全体积>=20）的恢复及新增命中体素数。连通分量是对象代理，不声称等同地质事件。不要因为密集井命中目标就称其仍是“完全隐藏对象”。

复用原Stage21对象诊断约定：overlap边至少5体素且覆盖真值对象>=5%；检测辅助阈值IoU>=0.1及coverage>=0.1，只作诊断；连续匹配IoU/coverage必须保留。报告合并、分裂、小碎片和未匹配对象，不重新搜索阈值。

图件控制为三类，不做几十张重复3D散点图：

1. 已实际执行档位的主域mIoU与目标IoU曲线/点图，区分B/C，A水平线；未执行点留空，不插值冒充数据。P在相同域可作辅助水平参照。
2. 固定seed42、固定坐标轴的地质剖面对照：truth、P、A及已执行B/C，配差异图；剖面位置按域中心预定，不按结果挑选。
3. 伪井混淆矩阵及目标P/R/对象恢复的简洁比较；可将少量全域目标投影整合进同一图，不裁truth ROI。

选中的密度完成三seed后，报告每个seed的配对差及其中位数，不把组中位之差冒充配对差中位，不做n=3的夸大显著性或泛化推断。

## 十一、交付内容与停止位置

在新实验目录交付：

- `README.md`：任务、范围、复用模块、准确命令。
- `PROTOCOL.json`：在首次新GPU结果前冻结上述数据/码本/sigma/分类/布局/数量/seed/推进规则。
- `REPO_DIFFERENCES.md`：实际远程/GitHub差异及旧Stage21情况。
- 独立的新run目录：条件、source/noise/预测、原始指标CSV、对象表、图件和`REPORT.md`。
- 简洁执行清单：实际执行与跳过的case/seed/N/arm及原因、Flow前向次数、耗时、测试结果。
- 对未执行内容写明未执行；训练、真实正反演、Stage19复用等没有做就不能写完成。

只在拿到实际结果后给 `P/docs/DEVELOPMENT_HANDOFF.md` 追加本轮事实，保留其原有未提交内容，不重写历史报告。记录新源码hash及基础采样器未修改的证据。不要自动push。

最终用简体中文先报告：

1. 远程与GitHub差异、本轮新增文件及复用了什么；
2. 准确井在井外是否有效、需要测试到多少口；
3. 同位置伪井能否保留收益，是否主要来自直接写入，是否优于直接属性分类P；
4. 对象/体积是否与类别收益一致；
5. 是否值得下一步用真实正反演替换平滑证据，或应先解决伪井/硬条件问题。

达到上述有限任务的实际结果与报告后停止，不自动进入新训练、噪声实验、物性重设或真实正反演阶段。不要为了获得正结果改变已冻结的case、seed、密度序列、sigma、井位、分类规则或评价域。
