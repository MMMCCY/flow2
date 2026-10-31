# Stage27：冻结Flow的干净终点近端修正与路径重投影
## 可直接交给Remote SSH中的VSCode Codex执行

请在实际服务器仓库中完成实现、必要检查、下述全部正式运行、评价与交付。不要只提交计划、补丁或smoke后等待再次授权。该任务书包含独立背景，不需要读取本地MCYR/FM或此前聊天。

本任务的科学目标和运行范围已在本文固定。允许处理路径、设备、序列化、内存等常规工程问题；不得擅自改变核心算法、追加扫描、挑seed、换评价口径或重训模型。指标不改善不是停止理由，负结果完整保存。

## 1. 背景、当前证据与本次唯一目标

仓库：MMMCCY/flow2。
本地讨论端2026-10-30核实默认分支main与以下commit一致：
de06168fb57f2462499a99885c584fc3e60fbeff。

实际远程工作目录历史上为 /home/xmj/mcy/flowtrain_stochastic_interpolation-main，先定位真实路径，读取适用AGENTS并保留现有改动。记录实际HEAD与相关未提交差异；若HEAD改变，只检查本任务相关差异，不重新审计全库。

项目以冻结地质Flow为先验，真实井/地表给出精确类别条件。已有Stage25采用人工Gaussian双属性恢复，新增伪井近乎完全正确，原生岩盖目标IoU约0.118→0.787；它不是真实地球物理成功。Stage26换为合成逐柱卷积地震，一次固定A速度的log阻抗更新虽降低连续地震误差1.849→1.140，但分类伪井更差，最终D目标IoU0.0865，低于A约0.118和D0约0.112。后续真速度oracle及候选缩减仍未建立可靠前端分类。

核心问题：不要求反演先得到唯一、准确的完整岩性体，希望观测能给井外结构提供增量，同时不把一次错误分类永久固定。

旧代码已有：
- 对当前生成中间状态的外部物理/属性梯度；
- DFlow穿过全生成链优化源噪声；
- 一次物理反演→分类→硬伪井。
本任务不重复它们。

本次唯一新算法：原网络只接收真实井/地表条件；每一步估计干净终点，在该终点变量上做少量带信赖项的物理/局部结构修正，再重新投影至线性训练路径。不是对当前noisy状态直接叠加旧梯度，不是优化完整源噪声，也不生成新的硬伪井。

本轮是单源、同原条件两结构的机制实验，不声称泛化、校准后验或唯一重建。研究闭环为：
新算法→物理/结构最小消融→同条件观测反事实→最终硬地质评价。
不开展后验频率校准或增加种子。

## 2. 明确删除的历史与验证性工作

不要运行：
- Stage25原A/B/C、Gaussian恢复，或Stage26旧D0/D和诊断oracle；
- 128/512/2048/3840井密度实验、topK/阈值扫描；
- 新的单类软约束、允许集合S1/S2或旧Stage17/18重跑；
- DFlow、pCN、候选大池筛选或旧adapter训练；
- 符号/梯度/全算子/全拓扑的历史回归审计；
- 噪声、波子、物性、seed或多几何扫描；
- FWI、联合反演、新网络、模型架构或权重更新。

本轮只保留实际发现本次新增错误所需的小检查，列于第10节。历史结果按身份一致复用，不能重复计算来制造“本轮完成量”。

## 3. 仓库入口及资产

以下均为已核实路径；项目路径前缀P=project/geodata-3d-conditional。

可复用实现：
- P/model_train_sh_inference_cond.py：Geo3DStochInterp、embed/decode、训练终点重建；
- src/flowtrain/interpolation/interpolation.py：LinearInterpolant，alpha=1-t、beta=t，one_sided时额外噪声项为0；
- P/inference_runtime.py：现有EMA/embedding正确加载；
- P/scripts/stage21/run.py：initial_noise、sample_native，注意native内部不适合本新采样器直接改；
- P/guidance/generator_posterior.py：project_conditions及旧native sampler；
- P/guidance/dflow.py：模型冻结/hash工具，不能调用其源优化替代本算法；
- P/guided_geophysical_sampling.py：soft_decode_to_probs；
- P/guidance/seismic.py：概率→物性、原条件物性覆盖、逐柱地震正演；
- P/scripts/stage26/common.py：setup、relative_l2及旧码本接口；
- P/scripts/stage26/prepare.py：旧观测、prior/updated等资产来源；
- P/scripts/stage25/generate.py：原生岩盖生成；
- P/scripts/stage25/evaluate.py、P/scripts/stage26/evaluate.py：既有评价；
- P/visualization_standards/geology_approved_v1：认可渲染器及配置。

数据入口：
R25=P/experiments/stage25_laccolith_shape/run_20261019_v3/
R26=P/experiments/stage26_physical_seismic/run_20261019_v1/

通过这些目录的OBSERVATION_FREEZE、INPUT_FREEZE、条件及评价registry定位：
- G1真值，仅生成/评价过程可读；
- 原真实条件values和mask M0；
- 已知地下支持subsurface；
- 旧seed42真实15维源；
- 声学码本、地震观测配置及旧y1；
- 已完成A、D0、D及必要DIRECT结果；
- 旧512井总mask M512与256holdout。

不要把旧main入口导入执行而写回旧ROOT。优先复用纯函数，新脚本和新输出目录独立。建议新代码P/scripts/stage27_flow_proximal/，新产物P/experiments/stage27_flow_proximal/run_<日期>_v1/；若已存在则另起后缀，禁止覆盖。

实际代码以当前HEAD及运行快照为准；函数不存在时按此功能定位，不虚构或强行套用不兼容接口。结果中写清路径。

## 4. 必须保持的模型和变量语义

- 原架构、checkpoint、全部参数及embedding冻结，无训练、fine-tune、adapter或新可训练网络。
- 采用现有411项EMA、冻结embedding策略，实际embedding维度15；以已审阅加载入口及checkpoint元数据核实，不能把默认构造参数20当实际维度。
- 模型eval，模型参数requires_grad=False，不积累模型梯度。
- 网络FP32，不开autocast或TF32；物理和近端更新优先FP64，可CPU或GPU，保持端到端对u的可微关系。
- 张量几何[B,channel,X,Y,Z]，64³，Z向上；spacing=3840/63米。
- 原raw类别-1为空气，0…13为非空气；embedding/物性category索引=raw+1，raw9对应category10。
- 真实条件M0贯穿网络condition和硬投影。伪井、真值、反演完整分类不作为新网络condition。
- conditioning=embed(original_values)*M0，真实条件嵌入B=embed(original_values)。未观测值沿用现有代码语义，不能把地下未知-1解释成真实空气。
- 自由变量域U=subsurface AND NOT M0；近端只能改U，已知区和地上区始终还原B。不要在地上引入无约束噪声。
- 原始地表信息可决定subsurface；不得借完整真值决定地下支持、候选类、时深关系或优化权重。
- 最终hard decode使用原model.decode，raw=category-1，保持原条件。地下空气不清理，计数和物理后果如实报告。

## 5. 本次算法的完整定义

这是受PnP-Flow/Flower启发的项目迁移原型。不是原文完整后验采样器，不使用其在线性高斯情形的协方差保证。

### 5.1 网络评估、时间与重投影

每份输出N=32次velocity网络前向，j=0…31。
名义时间a_j=j/N，下一重投影时间b_j=(j+1)/N。
网络评估时间t_j=clip(a_j, checkpoint训练time_range的下上界)；默认已核实源码time_range=[0.0001,0.9999]。记录实际metadata。此clip仅处理训练端点，不能根据真值或输出修改时间表。首步名义0在1e-4评估属于明确数值近似，报告它；若实际训练路径不是当前线性独立Gaussian定义，应停止新采样器并回报版本冲突，不偷偷替换理论。

初始化：
x_0=原seed42源；在M0及非U区投影回B。
每步在no_grad中：
v_j=model.net(x_j, conditioning, t_j)
e_hat_j=x_j+(1-t_j)*v_j
对e_hat_j在非U位置投影回B，并detach。

在干净候选变量u上执行下节修正，得到u_star。
之后：
x_{j+1}=b_j*u_star+(1-b_j)*epsilon_j，
再对非U区投影回B。

epsilon_j必须是新独立标准Gaussian CPU-FP32源转到网络设备，不能复用同一张原始噪声当全部步骤噪声。为所有方法臂及两观测提供相同配对重噪：
epsilon_j=CPU torch.randn(shape=[1,15,64,64,64], generator.manual_seed(1042+j))。
j=31时系数为0，不必生成该噪声。
初始源复用原保存seed42张量，逐元素身份以tensor hash确认。
保存PRNG实现、种子规则和每步noise tensor hash即可，不要求保存整套约0.5GB堆叠噪声；记录足以CPU精确重建。

不从完整生成路径反传；近端梯度只针对当前detach后的u。不得让模型参数或前面步骤参与backward。

最终x_32=u_star(last)，按原model.decode转成硬类别并保存。主评价使用这份原始硬预测。

### 5.2 固定soft decode与声学映射

q(u)=soft_decode_to_probs(u, embedding.weight, tau=0.1)，tau全程固定，无温度搜索。0.1是本次封存的原型选择，不声称它已由旧实验验证为最优。

物理近端若采用FP64，则将detached embedding常量和声学表显式转换到u的dtype/device再计算soft decode，避免einsum的FP32/FP64冲突；不改变模型里原embedding参数，也不对该转换常量求参数梯度。

调用现有probabilities_to_subsurface_acoustic(q,table,subsurface)及overwrite_exact_condition_acoustic，在地下采用非空气条件化概率形成期望阻抗和慢度，已知条件处覆盖真实物性。
这与现有物理helper保持一致；它会在优化用软物理中排除地下空气，而最终hard decode仍可能有空气，必须分开报告。
时间—深度与反射到时由当前候选自身慢度计算，不固定旧A速度、不用真速度、不按真值对齐。

forward=现有ConvolutionalSeismic，无新物理算子/波子实现。
y1沿用Stage26旧观测：全64×64道、400样点、8ms、25Hz Ricker、128ms、无新增噪声，具体以旧freeze身份为准。
观测有效窗和mask沿用旧配置。

数据能量：
E_y(u)=sum_valid[(F(acoustic(u))-y)^2] / max(sum_valid[y^2],1e-12)。
用平方相对L2能量，不混成relative L2本身。
不把它称已校准似然；没有新增噪声方差或人为似然温度。
保存优化前后的E_y，以及最终硬岩性用自身物性查表的relative L2。

### 5.3 固定局部结构项

只针对研究目标raw9的连续占据，不对岩性编号做TV，不强设单连通、已知体积或真值轮廓。

用现有地下非空气归一化概率取得q_target：
global category10；若先去掉air，则rock index9。
真实已知地下位置的q_target覆盖为真实类别one-hot指示。
结构边仅取两端均在subsurface的X/Y/Z正向6邻接，每条无向边计一次。边权常数；spacing不等时乘spacing_ref/spacing_axis，本次立方spacing权重均为1。

R(u)=mean_edges[sqrt((q_target[i]-q_target[j])^2+1e-6)-1e-3]。
边的平均覆盖整个已知地下域，允许原条件对邻居产生影响，但原条件本身不更新。
R不使用真值体积、组件数、边界、DT、skeleton或Betti。保留所有对象，禁止删小块或只保最大组件。
固定lambda_R=0.05；这是无测试调参的初始方法设置，与均值归一化的数据能量/信赖项一起记录，不宣称物理最优。不得偷偷换成edge-aware权重或加更多损失。

### 5.4 近端目标与固定求解器

定义D_free=15*U体素数，所有mean只对U的15维自由元素计算。
nu_j²=(1-t_j)^2 / (t_j²+(1-t_j)^2)。

实际优化的尺度等价目标：
Q_j(u)=0.5*mean_free[(u-e_hat_j)^2] + nu_j²*[a*E_y(u)+r*0.05*R(u)]，
a/r由方法臂控制；e_hat_j固定，不反传网络。
等价于把信赖项写为1/(2nu²)，但采用上述缩放形式避免接近t=1时数值爆炸。

每个outer step：
- u从e_hat_j初始化；
- 最多3次近端迭代，所有臂相同固定次数，不追求完全近端收敛；
- 求g=D_free*grad_u Q，非U位置置0；
- s=RMS_free(g)。若s=0则本次不更新；
- d=g/max(1,s/0.1)，限制一次尝试方向的自由区RMS≤0.1；
- alpha按[1,1/2,1/4,1/8,1/16,1/32]依次尝试，候选u'=project_nonfree(u-alpha*d)；
- 用同Q做Armijo：Q(u')≤Q(u)-1e-4*alpha*mean_free(g*d)；
- 首次满足则接受；六次均不满足则本次保留原u并记录line_search_exhausted，继续剩余预定流程，不增加迭代或调参数。

每次新梯度重新detach u后requires_grad=True，不保留跨inner/outer图。候选能量可no_grad评估；不额外运行model.net。
数据a=0且r=0时无需inner求解，u_star=e_hat，不能白跑物理。
结构-only臂不调用地震算子，记录其数据能量仅在最终评价计算。

line search是算法内部稳定求解，不是实验超参扫描；它不读取真值。线搜索耗尽或正收益不足不是科学失败门槛，保存次数并按封存预算完成。出现NaN、错误坐标、条件破坏才按工程错误处理。

数值配置固定eps=1e-12；平滑TV中的eps为1e-6。CPU/GPU、FP64物理张量传输需保持u梯度，不能因detach设备转换而让优化变空操作。

## 6. 两个结构、六份输出：最小正式运行闭环

### 6.1 G1沿用，G2预定

G1就是Stage25/26原生laccolith；旧完整真值不进入新采样器。
G2用原生生成历史的同一参数，只把cap_diam从22*spacing改为18*spacing，其余origin、height=8*spacing、stem_diam=4.4*spacing、背景历史、raw9、地表和seed均不变。

仅生成/评价角色可读取两个真值。生成G2后验证：
- labels shape、轴和单位相同；
- 从原M0直接逐体素比较G2[M0]==G1[M0]；
- 地表/地下支持不变；
- 两个几何的帽部确实不同。
这是配对输入定义检查，不是要求网络或指标达到收益。
Laccolith事件会改周围地层，不允许假定条件自动相同。

只构造这一个预定G2，不搜索多个直径/几何；不通过修改井标签强行满足同条件。若固定G2与M0不一致，保存冲突位置与原因，暂停仅依赖G2的两份输出，继续G1四臂完整实验并回报配对未完成。不要擅自挑另一个更利于方法的几何。输入定义确实无效时暂停该部分，不属于收益gate。

y2由G2真实标签→同一码本→同正演生成一次，只保存观测和配置给推理端；这是标准合成观测生成，不是推理泄漏真值。y1原资产直接复用，不重算旧反演。
本任务不需要任何传统反演，新增传统反演调用=0。

### 6.2 固定运行表

同初始源和所有重噪序列，仅改变开关/观测：
1. F0：a=0,r=0，无物理无结构；
2. Fr：a=0,r=1，仅结构；
3. Fy_G1：a=1,r=0，y1；
4. Fyr_G1：a=1,r=1，y1；
5. Fy_G2：a=1,r=0，y2；
6. Fyr_G2：a=1,r=1，y2。

F0/Fr条件和噪声完全相同，无观测依赖，两个世界共享，不能重复生成。
不按G1结果在Fy/Fyr中选一个再做G2，两者均按表完成，避免选择性验证。
已有A/D0/D仅用于基准参考，不是本轮新实验，不重跑。
没有额外true-label条件oracle、错图随机打乱、零图或三seed。
y2输出对G1的交叉评价就是反事实观测对照，不额外生成wrong臂。

### 6.3 固定数量上限

完整运行：
- 新模型加载可复用一次，避免每臂重新加载；
- 六份输出，每份32次velocity前向，共192次；
- 活跃物理输出4份，最多4*32*3=384次物理梯度计算；
- 每次inner至多1次当前目标正演+6次candidate正演，因此内部物理正演上限4*32*3*7=2688次；可缓存未变的当前目标减少实际次数；
- 另加y2生成和紧凑最终评价正演，分别计数；
- 传统反演0，训练0，其他科学输出0。

192不是总物理成本；报告分列网络NFE、物理forward/backward、近端接受/跳过次数及时间。
常规工程恢复可以在预算内继续，完整数值已保存时只补评价/图/receipt，不重采样。需要重跑一份已消耗NFE的科学采样会超出192时，应先回报原因与剩余可执行部分，不暗中增加预算。
科学负结果不触发更多seed、参数修改或后续方案。

## 7. 评价：输出是否利用了额外结构信息

### 7.1 固定评价域

复用原M0、M512、holdout和G1帽部box。
U_original=subsurface AND NOT M0。
U_common=subsurface AND NOT M512，供全部方法包括无伪井的新方法共同比较。
FULL保留，不能只挑有利域。

G2另外保存两个真值帽部并集box和几何差异区，仅评价角色读取；其定义在任何生成结果之前冻结，绝不进入优化。
共同自由真目标的分母随G1/G2各自定义，不能把G1的1933无条件用于G2。
holdout表示无硬条件，不是物理观测没有覆盖。

### 7.2 每份硬输出的最小指标

- U_common和U_original目标TP/FP/FN、P/R/IoU、预测/真目标体积比；
- 既有帽部指标、双向边界体素中心距离，空目标给NA/明确失败状态而非0完美；
- 原条件违例数，地下空气数；
- 目标组件数量与体积序列、与真值对象的空间对应；不要引入新的PH/Betti开发；
- 最终硬标签按自身声学码本同时查阻抗和慢度，计算对y1/y2的同窗relative L2；
- 末步优化用连续场的物理误差另列，不能代替硬模型误差；
- 无关背景逐类混淆或固定类mIoU，用原口径即可。

原地表以下空气要保留，不能清理后才正演。声学helper软排空气的约定与硬输出的gap在报告说明，不把差异隐藏。

### 7.3 配对信息检验

同一Fy/Fyr的G1/G2输出，检查：
- 观测由y1换y2，帽部范围/目标占据变化方向是否与预定G1/G2差异一致；
- 在未变区域是否发生不必要破坏；
- 对G1/G2及y1/y2交叉评价。对角是否优于反事实，在物理和结构上分别判断，不强制单一汇总评分；
- Fyr相对Fy的改善，是否超出Fr相对F0的平滑效果；
- 新采样器F0与旧A的差异单列，不能把新采样器自身变化归给地球物理。

可报告结构IoU的交叉偏好：
D_struct=0.5*[(I(pred_y1,G1)-I(pred_y1,G2))+(I(pred_y2,G2)-I(pred_y2,G1))]，
分别对Fy/Fyr及相同共同域计算。它只是本配对描述值，不用作调参、选图或pass/fail。仍保存原始矩阵，避免聚合掩盖某方向失败。
物理能量也保存2×2矩阵，不将不同观测norm的数值直接混成未经解释的概率。

### 7.4 结论尺度

- 有方向的结构变化+观测匹配改善，且原条件保持：支持单对结构的额外信息利用，不要求精确还原；
- 只有Fr/F0差异：结构先验作用，不是物理信息作用；
- 只有连续物理变好，硬场仍差：松弛/分类/结构问题仍在；
- 组件变少但帽部消失或假阳性膨胀：不称结构恢复；
- F0已经大幅退化：新路径与checkpoint适配是重要限制，但仍按有效实现完成科学结果；
- 负结果不能据此宣布所有推理制导无效；
- 单源配对不能证明真实地球物理泛化或校准多解。本轮不为补齐这些声明追加实验。

## 8. 代码组织和资产隔离

建议逻辑模块：
- config/contracts：所有固定数值、开关、资产记录与预算；
- sampler：单步终点估计、prox求解、重噪/投影；
- prepare_pair：仅合成生成角色读取真值并构造y2；
- run：只读取真实条件、地下支持、观测、模型、seed与配置；
- evaluate/plot/report：读取保存预测和真值。

具体文件名执行端可按仓库惯例决定；不做无关重构、不改变旧shared sampler行为。优先新增独立sampler，旧函数需要的纯逻辑可复用。

限制：
- 新推理入口不读取R25 truth、几何参数、准确井B、旧C完整标签、evaluation registry或任何oracle；
- 观测y2配置可以在prepare角色由真值生成，推理registry只包含观测与公开支持信息，剥离真值/几何；
- 代码源快照与实际运行配置一起保存。不要为本任务重新建立大型防火墙框架；
- 沿用可用防火墙理念，保留最小输入白名单或相应进程角色隔离，不关闭旧正式防火墙；
- 模型before/after身份与无参数梯度检查复用现有工具，不写全库哈希清单；
- 数据优先原子保存；图片/receipt失败后只恢复后处理。

设备：
网络CUDA FP32，物理GPU FP64可用则优先；若确定性scatter或算子不支持该设备，允许近端移到CPU FP64，保存设备差异，网络仍CUDA。e_hat detach后CPU近端再转回网络设备符合本算法，因为不要求网络梯度；u到物理算子的梯度必须保留。不要混淆这两种detach。
可调整CPU线程、张量分块和设备以完成同一计算，不能因此更换算子或优化目标。

## 9. 最少配置清单

运行前将下列固定值写入机器可读配置并冻结：
N=32
initial_source=old saved seed42
reprojection_noise_seed(j)=1042+j
time_nominal(j)=j/32
time_eval=clamp(nominal, recorded_checkpoint_time_range)
time_next(j)=(j+1)/32
tau=0.1
nu2=(1-t_eval)^2/(t_eval^2+(1-t_eval)^2)
prox_steps=3
prox_gradient_scale=D_free
direction_rms_cap=0.1
line_search_steps=[1,0.5,0.25,0.125,0.0625,0.03125]
armijo_c=1e-4
lambda_structure=0.05
tv_smoothing_eps=1e-6
energy_eps=1e-12
real_condition_projection=every stage
target_raw=9 / target_category=10
G2_cap_diameter_voxels=18
same_model_weights=True
parameter_sweep=False
new_traditional_inversions=0
max_network_evaluations=192
precision_network=FP32_noTF32_noautocast
precision_prox_physics=FP64
new_observation_noise=none

不要以“参数尚待优化”为由停在实现而不运行；这是一套完整封存的原型设置，不保证正收益。没有科学调参权限。

## 10. 必要检查：合并做一次，不建立多轮验证项目

在正式长流程前只完成：
1. 模型/shape/类别映射、原资产读取和一次已知条件embed/decode对应检查，复用现有能力；
2. 一个小张量上核对Q的自由域梯度有限、非U为0、无数据无结构分支不调用物理；实际物理helper上确认u确实产生非零梯度。优先复用已有测试，不做全物理有限差分；
3. G1原资产身份和G2同条件检查；
4. 第一个正式输出本身作为端到端检查。不要另起完整32步smoke再重跑同一结果；
5. 结束时参数未变、真实条件无违例、NFE计数和原始预测可加载。

F0有结构错误、物理误差高或结果不理想不属于输入工程失败，不阻止其他臂运行。
若发现梯度断开、坐标错误、条件被覆盖等明确实现错误，修复受影响的新代码，记录旧失败产物；不要修改科学方法来追收益。
线搜索耗尽只是该近端步未更新，按算法继续，不当工程失败重新采样。

## 11. 正式运行与异常处理

完成定位和必要检查后直接按六份表执行，不等待第二次授权。
建议先完成F0、Fr和G1两个物理臂保存；G2观测可在独立生成角色准备完成后运行两臂。顺序不是效果筛选，四个G1结果都要保留。
每份结果立即保存，不让后面的配对或渲染失败抹掉已完成结果。

不估计凭空GPU小时；根据实际前两个outer step的耗时和内存记录量级，按固定操作预算运行，不能因为预计更久自行缩为smoke。若资源不足以完成固定范围，保存进度、报告真实资源/剩余量及独立已完成部分，不无限等待、不擅自开更多GPU或扩大费用。

若G2同条件不成立，只停配对两臂，G1四臂照常完成；回传明确主方法实验已完成/配对未完成，不能把整个闭环写成完成。
若模型版本/真实训练路径与本任务前提冲突，保存核实事实，暂停受影响新sampler；不要通过猜测替代。
不因任何IoU/物理误差未超过阈值而自动修改方法、放宽标准、换seed或追加实验。

未经用户另行要求，不commit、push、创建PR或发布。按仓库规范保留修改与源码快照，用户后续同步。

## 12. 必要图件及交付

沿用approved_v1真实硬标签渲染与固定Y26，统一坐标/视角/色表，保留所有碎片和误报。主图可以：
- G1：Truth/A参考/F0/Fy/Fyr，Fr放必要附图；
- G2：Truth/同F0/Fy/Fyr；
- 一张固定剖面解释配对变化和原条件。
面板过多按同风格分两张，不堆大表或结论进图，不删坏结果。无需新绘图系统或大批动画。

必须交付：
1. REPORT.md：完成范围、方法实际落实、每臂结果、配对因果解释、失败/跳过和限制；
2. 固定配置及单条/批次实际运行命令，实际路径、HEAD/工作树差异、代码快照、必要环境；
3. registry：模型/条件/观测/布局/初始源身份，重噪PRNG规则与hash，推理不读真值的范围；
4. 每份原始连续终点与硬预测，简洁每步trace：NFE、t、nu2、数据/结构/信赖项、梯度RMS、接受步长、耗尽次数、物理调用数和耗时；
5. metrics.csv、cross_observation_matrix.csv、condition_checks.json、resource_usage.json；必要组件/边界摘要，避免大量无关审计文件；
6. G2观测生成记录与同条件逐元素检查（评价/生成角色），旧y1复用身份；
7. 认可风格PNG/PDF图件，全部失败或跳过项；
8. 结尾分别回答：新推理是否正确实施；物理是否带来额外井外结构；TV是帮助组织还是仅平滑；这次单源配对不能回答什么。

失败结果按原口径报告。不得只给优化loss、漂亮剖面或程序成功退出代替结论。

## 13. 参考依据：正文已经给出核心算法，不依赖外链才能执行

PnP-Flow，ICLR2025：
https://proceedings.iclr.cc/paper_files/paper/2025/hash/708e58b0b99e3e62d42022b4564bad7a-Abstract-Conference.html

Flower，ICLR2026，方法原文：
https://arxiv.org/html/2509.26287v2
https://proceedings.iclr.cc/paper_files/paper/2026/hash/939af25e6dc31d3d020fda2f64016917-Abstract-Conference.html

本任务采用这些工作的终点/近端/路径重投影思想，具体非线性物理、条件投影、3步信赖求解及TV为MCYR原型设计，不声称逐字复现论文或继承其线性高斯后验保证。不得因想“更像原文”擅自改掉本任务固定范围。

