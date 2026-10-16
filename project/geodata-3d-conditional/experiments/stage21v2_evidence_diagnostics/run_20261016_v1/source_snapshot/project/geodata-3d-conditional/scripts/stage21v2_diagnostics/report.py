"""Three fixed retrospective figures and Chinese diagnostic report."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CASES = ("fullgeo_case03", "fullgeo_case01", "fullgeo_case02")
DOMAINS = ("E_original_free", "E_holdout")
SCORES = ("margin9", "susceptibility")
COLORS = dict(margin9="tab:blue", susceptibility="tab:orange")


def number(value, precision=6):
    return "NA" if value is None else f"{value:.{precision}g}"


def write_report(out, config, summary, distributions, assignments, objects, data):
    figures = out / "figures"
    figures.mkdir()
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for i, domain in enumerate(DOMAINS):
        for j, case in enumerate(CASES):
            ax = axes[i,j]
            rows = [r for r in summary if r["case_id"]==case and r["domain"]==domain]
            for row in rows:
                score=row["score"]
                curve=data[case]["domains"][domain]["curves"][score]
                ax.step(np.r_[0.,curve["recall"]], np.r_[1.,curve["precision"]], where="pre",
                        label=f"{score}: AP={row['AP']:.4f}", color=COLORS[score],lw=1.5)
            pi=rows[0]["prevalence"]
            ax.axhline(pi,color="gray",ls="--",label=f"pi={pi:.5f}")
            ax.set(xlim=(0,1),ylim=(0,1.02),xlabel="Recall",ylabel="Precision",
                   title=f"{case} | {domain}\npositives={rows[0]['positives']} / {rows[0]['domain_voxels']}")
            ax.grid(alpha=.15)
            ax.legend(fontsize=8)
    fig.savefig(figures / "precision_recall.png",dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2,3,figsize=(16,8),constrained_layout=True)
    for i,domain in enumerate(DOMAINS):
        for j,case in enumerate(CASES):
            ax=axes[i,j]
            d=data[case]
            mask=d["domains"][domain]["domain"]
            for group,color,selected in (("true9","tab:red",mask & (d["truth"]==9)),
                                         ("non9","tab:gray",mask & (d["truth"]!=9))):
                values=np.sort(d["margin9"][selected])
                ax.step(values,np.arange(1,len(values)+1)/len(values),where="post",color=color,label=f"{group} n={len(values)}",lw=1.4)
            ax.axvline(0,color="black",ls="--",lw=.8,label="margin = 0 (ties use argmin)")
            ax.set(xlabel="Signed margin9 (not probability)",ylabel="Empirical CDF",title=f"{case} | {domain}")
            ax.grid(alpha=.15)
            ax.legend(fontsize=8)
    fig.savefig(figures / "margin_distributions.png",dpi=160)
    plt.close(fig)
    fig=plt.figure(figsize=(16,10),layout="constrained")
    outer=fig.add_gridspec(2,1,height_ratios=[1,1.1])
    top=outer[0].subgridspec(1,3)
    bottom=outer[1].subgridspec(1,4)
    for j,case in enumerate(CASES):
        ax=fig.add_subplot(top[0,j])
        d=data[case]
        mask=d["domains"]["E_original_free"]["domain"]
        h=ax.hexbin(d["q9"][mask],d["margin9"][mask],gridsize=55,bins="log",mincnt=1,cmap="viridis")
        fig.colorbar(h,ax=ax,label="Voxel count (log color)")
        if case=="fullgeo_case03":
            selected=(d["ids"]==1)&mask
            ax.scatter(d["q9"][selected],d["margin9"][selected],s=6,c="tab:red",alpha=.35,label="572-voxel object 1")
            ax.legend(fontsize=8)
        ax.axhline(0,c="black",ls="--",lw=.8)
        ax.set(xlim=(0,1),xlabel="q9: truth-derived local Gaussian fraction",ylabel="Signed margin9",
               title=f"{case} | E_original_free")
    d=data["fullgeo_case03"]
    target=d["ids"]==1
    coords=np.argwhere(target)
    z=int((coords[:,2].min()+coords[:,2].max())//2)
    for j,(name,values,cmap) in enumerate((("truth label9",d["truth"]==9,"gray_r"),
                   ("q9 (truth-derived)",d["q9"],"viridis"),
                   ("susceptibility (raw)",d["susceptibility"],"magma"),
                   ("signed margin9",d["margin9"],"coolwarm"))):
        ax=fig.add_subplot(bottom[0,j])
        im=ax.imshow(values[:,:,z].T,origin="lower",interpolation="nearest",cmap=cmap)
        if target[:,:,z].any():
            ax.contour(target[:,:,z].T,levels=[.5],colors="cyan",linewidths=1)
        ax.set(title=f"{name}\ncase03 object 1 retrospective Z={z}",xlabel="X",ylabel="Y",xticks=[0,32,63],yticks=[0,32,63])
        fig.colorbar(im,ax=ax,shrink=.75)
    fig.savefig(figures / "dilution_mechanism.png",dpi=160)
    plt.close(fig)

    lines=["# Stage21v2：连续属性线索与最近码本硬分类的离线诊断", "",
           "本轮仅回顾性分析保存的连续属性、真值、原始硬分类与两个既定评价域。三个案例各分析一次，不按Flow source seeds重复；使用CPU，零新增Flow、训练、adapter或物理正反演。旧实验与旧代码只读。", "",
           "## 资产与一致性", "",
           "三个案例完整FP32 argmin逐元素等于注册的 pseudo_raw（不是加回条件后的pseudo），固定码本权重/非空气范围和候选0…13保持不变。", "",
           "|案例|原未知域正体素|固定holdout正体素|原始硬分类原未知域目标数|",
           "|---|---:|---:|---:|"]
    for case in CASES:
        primary=next(r for r in summary if r["case_id"]==case and r["domain"]==DOMAINS[0])
        auxiliary=next(r for r in summary if r["case_id"]==case and r["domain"]==DOMAINS[1])
        lines.append(f"|{case}|{primary['positives']}|{auxiliary['positives']}|{primary['hard_target_count']}|")
    lines.extend(["", "margin9=min(其他类别距离)−目标距离，正值才表示目标严格更近；零值仍按原raw标签顺序argmin。susceptibility保持原值，两分数均事先指定为越大越支持目标；无温度softmax，不将分数称为概率。", "",
                  "## 连续排序与固定top5%", "",
                  "AP按完整同分组的recall增量加权precision求和，与二元无权重sklearn定义等价，未使用梯形PR-AUC。[官方AP定义](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)。服务器无sklearn，使用本地严格等价实现；常数分数AP=pi、已知例及同分组核对通过。AP_skill=(AP−pi)/(1−pi)。", "",
                  "|案例|域|分数|pi|AP|AP_skill|top5%实际选中比例|precision|recall|",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|"])
    for r in summary:
        lines.append("|"+"|".join(str(r[k]) if k in ("case_id","domain","score") else number(r[k]) for k in
                     ("case_id","domain","score","prevalence","AP","AP_skill","top5_actual_fraction","top5_precision","top5_recall"))+"|")
    lines.extend(["", "固定5%选集合：取ceil(0.05*n)名的分数作为界限，所有同分全部纳入，因此实际比例可大于5%。它是统一回顾性覆盖摘要，不是优化分类阈值或推荐伪井规则；未扫描预算、选择最佳阈值或按真值拆ties。", "",
                  "![三个案例、两固定域的PR](figures/precision_recall.png)", "",
                  "## 硬分类目标为何缺失", "",
                  "|案例|域|真9margin最小|中位数|p90|最大|严格正margin比例|零margin比例|真9分配到的raw类别:计数|",
                  "|---|---|---:|---:|---:|---:|---:|---:|---|"])
    for r in distributions:
        if r["group"]!="true9":
            continue
        assigned=[f"{a['assigned_raw_label']}:{a['count']}" for a in assignments if a["case_id"]==r["case_id"] and a["domain"]==r["domain"] and a["count"]]
        lines.append(f"|{r['case_id']}|{r['domain']}|"+"|".join(number(r[k]) for k in ("min","median","p90","max","strict_positive_margin_fraction","zero_margin_fraction"))+"|"+", ".join(assigned)+"|")
    lines.extend(["", "完整真9/非9分布分位数及实际硬分类比例见 margin_distributions.csv；全部逐类目标分配含零项见 true9_assignments.csv。负余量的绝对数值与分布均保留，不能仅因目标排名上升就称已跨越硬分类边界。", "",
                  "![有符号余量经验分布](figures/margin_distributions.png)", "",
                  "## 原9井零命中对象与空间稀释", "",
                  "对象首先在完整真值label9上按26邻接建立ID，然后截取既定域；ID和完整体积与旧object_metrics.csv核对一致。major为完整体积≥20。原井几何零命中与原硬条件M9零命中分别记录：前者可能仍被地表观测。几何穿过对象不等于错误伪井正确报告label9。", "",
                  "下面列出所有原9井零命中的主要对象，两域及两种分数均保留；域内无体素时为NA。完整对象（含小对象）见 object_diagnostics.csv。q9为真值辅助解释量，不是可用推断输入或分类概率。", "",
                  "|案例/对象ID|域|分数|完整/域内体素|原M9命中体素|score中位/p90/最大|top5%覆盖|q9域内中位/p90/最大|",
                  "|---|---|---|---|---:|---|---:|---|"])
    for r in objects:
        if not r["major"] or not r["original_well_zero_hit"]:
            continue
        lines.append(f"|{r['case_id']}/{r['object_id']}|{r['domain']}|{r['score']}|{r['full_voxels']}/{r['domain_voxels']}|{r['original_hard_hit_voxels']}|"+
                     "/".join(number(r["score_"+k]) for k in ("median","p90","max"))+f"|{number(r['top5_coverage'])}|"+
                     "/".join(number(r["q9_domain_"+k]) for k in ("median","p90","max"))+"|")
    lines.extend(["", "q9=G_sigma1(1[truth==9]) 在完整体积上计算，使用原separable Gaussian及replicate边界，然后才截域。对象完整q9和域内q9的min/median/p90/max等均保留在CSV。不能假设q9=0.5是通用分类界线，也不能把susceptibility简单归一化当作精确q9：背景物性不恒定，且密度通道与不同竞争类别影响margin。", "",
                  f"下图上排为主域q9/margin的密度图；下排是case03原井零命中572体素对象的回顾性展示，以完整对象Z包围盒中点确定Z={z}，不按分数或恢复效果挑截面；保留全64×64 XY范围，青色轮廓标出对象1。此处使用真值定位只为解释失败，不产生新输入或候选。", "",
                  "![局部平均稀释机制](figures/dilution_mechanism.png)", "",
                  "## 解释边界与研究决策", "",
                  "应结合AP相对pi的提升、固定top5%的precision/recall和原井零命中对象覆盖共同解释；全局较好可能被较大对象主导。case01辅助holdout只有11个正体素，结果仅为描述，不能当作强泛化证据。没有做体素独立bootstrap或显著性推断。", "",
                  "若连续分数可定位但argmin不选9，支持当前硬标签接口压缩线索；若susceptibility与margin差异较大，应重点检查既定码本距离几何的读取代价。若两种简单读出均弱且q9稀释明显，只能说当前设置缺少容易读取的体素级证据，不能证明全部信息不可逆消失、所有空间解码器失败或Flow必须重训。", "",
                  "本轮不实施任何后续表示方法，不证明真实反演有效、软约束必然成功或更高伪井密度能补救目标漏失。下一研究决策应依据全局与具体隐藏对象的差别另行冻结，不现场调sigma、权重、阈值、码本或seed。", "",
                  "## 运行与交付", "", "```bash", config["command"], "```", "",
                  "- CONFIG.json、REPO_STATE.json、INPUT_ASSETS.json、SOURCE_HASHES.json：固定诊断口径、提交/工作区、实际输入身份及新代码快照。",
                  "- score_summary.csv、margin_distributions.csv、true9_assignments.csv、object_diagnostics.csv：主表与完整对象数据。",
                  "- pr_curves/：按相同分数整组累计的完整PR阈值点；fields/：margin、原始susceptibility、真值解释量q9和完整对象ID。",
                  "- CHECKS.json、COMPLETION.json、ARTIFACT_HASHES.json：必要一致性核对、旧输入前后哈希不变及交付哈希。",
                  "- 仅CPU张量读取与计算；零新增Flow推理、训练、adapter、checkpoint加载、参数/阈值扫描。", ""])
    (out / "REPORT.md").write_text("\n".join(lines),encoding="utf-8")
