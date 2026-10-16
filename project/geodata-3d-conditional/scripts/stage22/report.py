"""Concise Stage22 results with fixed retrospective slice and convergence diagnostics."""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from scripts.stage22.common import (
    REPO,CASE,source_hashes,read_json,write_json,file_hash,verify_tensor,
)


def fmt(v):
    return "NA" if v is None else f"{v:.6f}"


def report(root,result):
    if (root/"REPORT.md").exists():
        raise FileExistsError("Retain sealed report")
    pp=read_json(root/"PREPROCESSING_RECEIPT.json")
    freeze=read_json(root/"METHOD_FREEZE.json")
    inputs=read_json(root/"EVALUATION_INPUTS.json")
    truth=verify_tensor(inputs["truth"]).numpy()[0,0]
    from scripts.stage21.evaluate import connected
    ids,_=connected(truth==9)
    added=verify_tensor(read_json(root/"preprocessing_assets.json")["added"]).numpy()[0,0]
    history=list(csv.DictReader((root/"solver_history.csv").open()))
    figures=root/"figures"
    figures.mkdir()
    fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout="constrained")
    iterations=[int(r["iteration"]) for r in history]
    for key in ("objective","data_mse"):
        axes[0].semilogy(iterations,[float(r[key]) for r in history],label=key)
    axes[0].set(xlabel="Accepted iteration",title="Fixed normalized objective and data MSE")
    axes[0].legend(fontsize=8)
    axes[1].plot(iterations,[float(r["TVmean"]) for r in history],label="TVmean")
    axes[1].set(xlabel="Accepted iteration",title="Isotropic per-channel TV mean")
    axes[2].semilogy(iterations[1:],[max(float(r["relative_change"]),1e-18) for r in history[1:]])
    axes[2].axhline(1e-6,c="black",ls="--")
    axes[2].set(xlabel="Accepted iteration",title="Objective relative change (50 consecutive)")
    fig.savefig(figures/"solver_convergence.png",dpi=160)
    plt.close(fig)
    samples=result["samples"]
    selected=[("truth",truth)]
    for arm,label in (("A_9","A original 9"),("C_512","old C full pseudo"),("FLOW_STAGE22","Stage22 sparse target"),("RESTORED_CLASSIFICATION","deconvolution direct")):
        rows=[s for s in samples if s["arm"]==arm and s["seed"] in (42,None)]
        selected.append((label,verify_tensor(rows[0]["prediction"]).numpy()[0,0] if rows else None))
    cmap=ListedColormap(["white",*plt.cm.tab20(np.linspace(0,1,14))])
    norm=BoundaryNorm(np.arange(-1.5,14.5),15)
    fig,axes=plt.subplots(2,5,figsize=(15,6.7),layout="constrained")
    for j,(name,volume) in enumerate(selected):
        if volume is None:
            for ax in axes[:,j]:
                ax.text(.5,.5,"Not executed by fixed stopping rule",ha="center",va="center",wrap=True)
                ax.set_axis_off()
            continue
        im=axes[0,j].imshow(volume[:,:,15].T,origin="lower",interpolation="nearest",cmap=cmap,norm=norm)
        axes[1,j].imshow((volume[:,:,15]==9).T,origin="lower",interpolation="nearest",cmap="Reds",vmin=0,vmax=1)
        axes[0,j].set_title(name,fontsize=9)
        axes[1,j].set_title("label9 only",fontsize=9)
        for ax in axes[:,j]:
            ax.set(xticks=[0,32,63],yticks=[0,32,63],xlabel="X",ylabel="Y")
    fig.colorbar(im,ax=list(axes[0]),ticks=range(-1,14),label="Raw lithology label",shrink=.9)
    fig.suptitle("case03 seed42 | full XY, fixed Z=15 from prior retrospective object display")
    fig.savefig(figures/"fixed_object_slice.png",dpi=160)
    plt.close(fig)
    gate=result["offline_gate"]
    direct=next(s for s in samples if s["arm"]=="RESTORED_CLASSIFICATION")
    lines=["# Stage22：平滑算子感知的属性恢复与稀疏目标条件", "",
        f"本轮停止位置：`{result['stop_reason']}`。新增Flow输出 **{result['new_Flow_outputs']}份**，网络前向{result['velocity_evaluations']}次；训练、adapter、真实物理正反演及参数扫描均未运行。只使用已见case03，source seeds是同案例随机实现，不是独立地质案例。", "",
        "## 目标条件是否可用", "",
        f"最终恢复场经固定码本与三个固定几何筛选条件，在旧512列布局内新增 **{gate['added_voxels']}个label9体素**，分布于 **{gate['columns']}列**。TP={gate['TP']}，FP={gate['FP']}，precision={fmt(gate['precision'])}；原M9零命中572体素对象1内正确新增{gate['object1_correct_added_voxels']}个。所有FP保留，无真值剔除或补足井数。这是部分目标位置，不是512口完整新增井。", "",
        f"离线开发停止门槛（precision≥0.95且对象1正确条件≥5）：**{gate['passed']}**；求解器收敛标记：{pp['converged']}。空新增集precision=NA、不通过。该真值门槛在条件封存后仅决定是否停止整个实验，不是部署筛选器或独立TEST证明。", "",
        "## 方法冻结与数值求解", "",
        f"方法冻结时间：{freeze['frozen_at_utc']}；最终场及条件封存：{pp['sealed_at_utc']}；首次完整真值评价：{gate['evaluation_started_utc']}。各阶段hash见METHOD_FREEZE与PREPROCESSING_RECEIPT；未根据真值选择迭代快照。", "",
        "目标为 mean((G(z)−y)²)+1e−4·TVmean(z)。TV采用每体素/通道三维前向差分L2范数的平均，体积外差分0，以sqrt(sum(diff²)+1e−8)平滑。G保留旧sigma1/radius3的float32核系数，CPU float64提升算术，separable replicate边界，用自动微分获得正确伴随。与原FP32算子和边界方向导数的小检查通过。", "",
        "采用投影加速梯度：majorization回溯确定数值步长，目标上升时从上次接受迭代重启；不搜索科研正则参数。原条件的未平滑查表属性固定在z中，完整含空气码本界限投影。初始化为有界观测并投影原条件。最终仅使用最后接受迭代。", "",
        f"迭代{pp['iterations']}次；objective {pp['initial_objective']:.9g} → {pp['final']['objective']:.9g}；final data MSE={pp['final']['data_mse']:.9g}，TVmean={pp['final']['TVmean']:.9g}，约束误差={pp['hard_constraint_error']}，边界误差={pp['bounds_error']}。耗时{pp['elapsed_seconds']:.2f}s。", "",
        "![求解收敛](figures/solver_convergence.png)", "",
        "## 直接前处理、生成与原有参考", "",
        "恢复属性乘回固定尺度，再用原raw0…13候选、权重、非空气范围和argmin。新增点还须d9≤0.01及margin≥0.05；它们是几何筛选，不是校准置信度。新增只标9、其余未知，原井/地表优先，固定holdout不写入。", "",
        "RESTORED_CLASSIFICATION为恢复场全域最近码本结果加回原条件；COPY_ONLY只在原A写入本轮条件，没有网络前向，holdout逐元素等于A。B512是完整准确密集条件参考，新方案与旧C信息量/掩膜/标签类型不同，不能单独分离去卷积或条件数量的因果贡献。", "",
        "|臂|seed|域|TP|FP|FN|precision|recall|IoU|固定类mIoU|对象1新增条件外TP/真值数|", "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for s in samples:
        for name,m in s["metrics"].items():
            lines.append(f"|{s['arm']}|{s['seed'] if s['seed'] is not None else '—'}|{name}|{m['target_tp']}|{m['target_fp']}|{m['target_fn']}|{fmt(m['target_precision'])}|{fmt(m['target_recall'])}|{fmt(m['target_iou'])}|{fmt(m['fixed_class_miou'])}|{m['outside_added_object1_TP']}/{m['outside_added_object1_voxels']}|")
    lines.extend(["", "## 隐藏对象及直接写入之外的响应", "",
        "所有对象在完整真值和预测上先建立26连通ID，再在固定域一对一最大IoU匹配；原ID、连续coverage和原重叠诊断阈值保留。以下主域包括全部对象；完整两域表与预测对象表另存。对象1新增条件之外的TP在A和新结果的同一个子集上计算，不按各自mask改分母。", "",
        "|臂|seed|对象ID|完整体素|匹配IoU|coverage|正确新增条件数|", "|---|---:|---:|---:|---:|---:|---:|"])
    for s in samples:
        for o in s["objects"]["E_original_free"]:
            volume=int((ids==o["object_id"]).sum())
            hits=int(((ids==o["object_id"])&added).sum())
            lines.append(f"|{s['arm']}|{s['seed'] if s['seed'] is not None else '—'}|{o['object_id']}|{volume}|{fmt(o['matched_iou'])}|{fmt(o['matched_coverage'])}|{hits}|")
    for g in result["flow_gates"]:
        lines.extend(["",f"seed{g['seed']}预定判据：{g}。"])
    if not result["new_Flow_outputs"]:
        lines.extend(["", "未执行新Flow，因此不能判断本轮条件能否被Flow利用，也不能把前处理门槛失败解释为Flow失败。" if pp["converged"] else "求解未满足预定收敛规则：只能记数值未收敛，不能解释为科学方案失败；本轮不以未收敛结果进入Flow或自动更改正则参数。"])
    for s in [s for s in samples if s["arm"]=="FLOW_STAGE22"]:
        n=s["metrics"]["E_original_free"]
        d=direct["metrics"]["E_original_free"]
        a=next(a for a in samples if a["arm"]=="A_9" and a["seed"]==s["seed"])["metrics"]["E_original_free"]
        delta=n["outside_added_object1_TP"]-a["outside_added_object1_TP"]
        lines.extend(["",f"seed{s['seed']}：Flow−前处理直接分类的主域目标IoU差={n['target_iou']-d['target_iou']:+.6f}、固定类mIoU差={n['fixed_class_miou']-d['fixed_class_miou']:+.6f}。对象1在新增条件之外相对A的TP变化={delta:+d}；该共同子集覆盖率变化={n['outside_added_object1_coverage']-a['outside_added_object1_coverage']:+.6f}。"])
    lines.extend(["",f"该开发案例三seed配对重复支持：{result['paired_repeat_supported']}；全部已运行seed的对象1新增条件之外TP都增加：{result['outside_added_TP_increases_for_all_new_seeds']}。研究判据通过不替代直接写入以外的生成收益检查，也不以中位数掩盖失败seed。", "",
        "![固定回顾性截面](figures/fixed_object_slice.png)", "",
        "## 下一研究决策与限制", "",
        ("该开发案例支持继续研究此桥接，但应保留前处理直接对照和条件外收益核对；本轮停止，不扩展案例、seed或物理设置。" if result["paired_repeat_supported"] and result["outside_added_TP_increases_for_all_new_seeds"] else
         "本轮未建立完整的配对重复生成补全支持。按停止规则收束，依据前处理精度、直接分类与生成之间的差距，另行决定是否修改条件接口或停止这一固定路线；不自动降低阈值、换TV系数或增加伪井。"), "",
        "使用已知Gaussian核、无额外噪声、人工强目标对比及已见case03；成功只是理想代理下可行，失败也不否定所有去卷积或空间反演。TV系数1e−4是事先工作假设，没有最优性或文献成功保证。", "",
        "方法背景：[TV去卷积](https://www.ipol.im/pub/art/2012/g-tvdc/)、[投影/加速近端梯度的FISTA基础](https://doi.org/10.1137/080716542)、[物性与地质联合约束背景](https://academic.oup.com/gji/article/219/3/1989/5556947)。本轮并非Split Bregman或PGI复现，不声称方法创新。", "",
        "执行命令、测试结果、源代码和输入身份见README、TEST_RESULTS、METHOD_FREEZE、inference_registry及source_snapshot。PREPROCESSING_RECEIPT先于完整真值评价；预测receipt记录实际A噪声指针/hash、EMA/embedding、模型不变、条件违例和前向数。", ""])
    (root/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    if source_hashes()!=freeze["source_sha256"]:
        raise ValueError("Frozen Stage22 source changed")
    write_json(root/"DELIVERY.json",dict(status="completed",source_unchanged=True,
        preprocessing_converged=pp["converged"],method_frozen_before_truth=True,
        new_Flow_outputs=result["new_Flow_outputs"],velocity_evaluations=result["velocity_evaluations"],
        stop_reason=result["stop_reason"],condition_outside_holdout=True,only_added_label9=True))
    write_json(root/"ARTIFACT_HASHES.json",{str(p.relative_to(root)):file_hash(p) for p in sorted(root.rglob("*")) if p.is_file()})
