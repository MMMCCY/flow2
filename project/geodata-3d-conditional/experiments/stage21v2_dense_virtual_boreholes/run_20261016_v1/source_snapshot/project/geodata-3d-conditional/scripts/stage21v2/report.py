"""Compact fixed-domain plots and factual Chinese report from sealed predictions."""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.stage21v2.common import *


def generate(root):
    state=read_json(root / "execution_state.json")
    if state["status"] != "completed":
        raise ValueError("workflow incomplete")
    ev=read_json(root / "evaluation_registry.json")
    objects=list(csv.DictReader((root / "object_metrics.csv").open()))
    pred_objects=list(csv.DictReader((root / "predicted_object_metrics.csv").open()))
    out=root / "report"
    if out.exists():
        raise FileExistsError("report directories are immutable")
    out.mkdir()
    figures=out / "figures"
    figures.mkdir()
    samples=state["samples"]
    cases=[c for c in CASES if any(r["case_id"]==c for r in samples)]
    for case in cases:
        selected=[r for r in samples if r["case_id"]==case and r["source_seed"] in (None,42)]
        a=next(r for r in selected if r["arm"]=="A")
        p=next(r for r in selected if r["arm"]=="P")
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
        for ax,metric,title in zip(axes,("fixed_class_miou","target_iou"),("Holdout fixed-class mIoU","Holdout label9 IoU")):
            ax.axhline(a["metrics"]["E_holdout"][metric],color="black",label="A: original 9")
            ax.axhline(p["metrics"]["E_holdout"][metric],color="gray",ls="--",label="P: property classification")
            for arm,color in (("B","tab:blue"),("C","tab:orange")):
                ss=[r for r in selected if r["arm"]==arm]
                ax.scatter([r["well_count"] for r in ss],[r["metrics"]["E_holdout"][metric] for r in ss],label=arm,c=color)
            ax.set(xticks=LEVELS,xlabel="Total full wells",ylabel=title,title=case)
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.savefig(figures / f"{case}_holdout_dose.png",dpi=160)
        plt.close(fig)
        truth=verify_tensor(ev["cases"][case]["truth"]).numpy()[0,0]
        ordered=[p,a]+sorted([r for r in selected if r["arm"] in "BC"],key=lambda r:(r["well_count"],r["arm"]))
        volumes=[("truth",truth)]+[(f"{r['arm']}{'' if r['well_count'] is None else r['well_count']}",verify_tensor(r["prediction"]).numpy()[0,0]) for r in ordered]
        colors=["#ffffff",*plt.cm.tab20(np.linspace(0,1,14))]
        cmap=ListedColormap(colors)
        norm=BoundaryNorm(np.arange(-1.5,14.5,1),15)
        fig,axes=plt.subplots(6,len(volumes),figsize=(2.4*len(volumes),12),squeeze=False,layout="constrained")
        for j,(name,volume) in enumerate(volumes):
            for axis in range(3):
                sl=np.take(volume,32,axis=axis).T
                tr=np.take(truth,32,axis=axis).T
                axes[2*axis,j].imshow(sl,origin="lower",cmap=cmap,norm=norm,interpolation="nearest")
                axes[2*axis+1,j].imshow(sl != tr,origin="lower",cmap="Reds",vmin=0,vmax=1,interpolation="nearest")
                axes[2*axis,j].set_title(f"{name} | {'XYZ'[axis]}=32",fontsize=8)
                axes[2*axis+1,j].set_title("Difference from truth (full slice)",fontsize=7)
                for ax in axes[2*axis:2*axis+2,j]:
                    ax.set_xticks([0,32,63]);ax.set_yticks([0,32,63])
        fig.savefig(figures / f"{case}_fixed_center_slices.png",dpi=140)
        plt.close(fig)
        confusion=read_json(root / "confusion" / f"{case}_E_original_free.json")
        matrix=np.array(confusion["matrix"])[1:,1:]
        normalized=np.divide(matrix,matrix.sum(1,keepdims=True),out=np.zeros_like(matrix,dtype=float),where=matrix.sum(1,keepdims=True)>0)
        fig,axes=plt.subplots(1,3,figsize=(16,4.5),layout="constrained")
        im=axes[0].imshow(normalized,vmin=0,vmax=1,cmap="Blues")
        axes[0].set(xticks=range(14),yticks=range(14),xlabel="Pseudo raw label",ylabel="True raw label",title="Raw pseudo / original-free confusion")
        fig.colorbar(im,ax=axes[0])
        labels=[n for n,v in volumes[1:]]
        for metric,mark in (("target_precision","o"),("target_recall","s"),("target_iou","^")):
            vals=[r["metrics"]["E_original_free"][metric] for r in ordered]
            axes[1].plot(range(len(ordered)),[np.nan if v is None else v for v in vals],marker=mark,label=metric)
        axes[1].set(xticks=range(len(labels)),xticklabels=labels,ylim=(0,1),title="Original-free target scores")
        axes[1].tick_params(axis="x",rotation=60)
        axes[1].legend(fontsize=7)
        for j,r in enumerate(ordered):
            rr=[o for o in objects if o["case_id"]==case and o["arm"]==r["arm"]
                and o["source_seed"]==("" if r["source_seed"] is None else str(r["source_seed"]))
                and o["well_count"]==("" if r["well_count"] is None else str(r["well_count"]))
                and o["domain"]=="E_original_free" and o["original_zero_M9_major"]=="True" and o["matched_iou"]]
            for o in rr:
                axes[2].scatter(j,float(o["matched_iou"]),color="tab:purple",marker="o")
        axes[2].set(xticks=range(len(labels)),xticklabels=labels,ylim=(0,1),title="Originally M9-hidden major object matched IoU")
        axes[2].tick_params(axis="x",rotation=60)
        fig.savefig(figures / f"{case}_pseudo_and_objects.png",dpi=150)
        plt.close(fig)
    text=["# Stage21v2：密集准确井与同位置伪井", "",
          "本轮按冻结协议完成有限顺序诊断。准确井、伪井和直接写入对照共用固定留出柱；评价在每份预测封存后由独立离线程序进行。", "",
          f"新 Flow 输出 {state['flow_outputs']} 份，网络前向 {state['velocity_evaluations']} 次；复用旧 N9 基线 {state['reused_baselines']} 份；新推理总耗时（含加载）{state['elapsed_seconds']:.2f} 秒。",
          f"首个候选密度：{state['candidate_density'] or '无'}；三 seed 描述性确认通过：{state['confirmation_passed']}。未执行配置详见 execution_checklist.csv，不能据此判断更高密度或其他案例。", "",
          "## 固定域类别结果", "",
          "P 为完整属性分类（加回原条件），不随 N 变化。A 是原9井；B 是准确井；C 是同位伪井；D 仅将 C 条件写入 A。D 的留出结果逐元素等于 A。", "",
          "|案例|seed|臂|总井数|holdout mIoU|holdout label9 IoU|original-free mIoU|original-free label9 IoU|original-free P/R|目标预测/真值体积|",
          "|---|---:|---|---:|---:|---:|---:|---:|---|---| "]
    fmt=lambda x:"N/A" if x is None else f"{x:.6f}"
    for r in samples:
        h,f=r["metrics"]["E_holdout"],r["metrics"]["E_original_free"]
        text.append(f"|{r['case_id']}|{r['source_seed'] if r['source_seed'] is not None else '—'}|{r['arm']}|{r['well_count'] or '—'}|{fmt(h['fixed_class_miou'])}|{fmt(h['target_iou'])}|{fmt(f['fixed_class_miou'])}|{fmt(f['target_iou'])}|{fmt(f['target_precision'])}/{fmt(f['target_recall'])}|{f['target_pred_volume']}/{f['target_truth_volume']}|")
    text.extend(["", "## 准确井、伪井与写入效应", ""])
    for case in cases:
        ss=[r for r in samples if r["case_id"]==case and r["source_seed"] in (None,42)]
        a=next(r for r in ss if r["arm"]=="A")
        p=next(r for r in ss if r["arm"]=="P")
        for r in ss:
            if r["arm"] in "BC":
                h,f=r["metrics"]["E_holdout"],r["metrics"]["E_original_free"]
                text.append(f"- {case}/seed42/{r['arm']}/N{r['well_count']}：相对 A 的 holdout mIoU 差 {h['fixed_class_miou']-a['metrics']['E_holdout']['fixed_class_miou']:+.6f}；original-free mIoU 差 {f['fixed_class_miou']-a['metrics']['E_original_free']['fixed_class_miou']:+.6f}；相对 P 的 holdout mIoU 差 {h['fixed_class_miou']-p['metrics']['E_holdout']['fixed_class_miou']:+.6f}。")
    text.extend(["", "D 在 holdout 与 A 完全相同，因此 B/C 在该域的差异属于重新运行冻结 Flow 后的补全响应；辅助域包括新井，必须结合 D 比较直接写入贡献。井数512及以上超过训练时8–31口井的条件密度。", "",
                 "## 对象与体积", "",
                 "26连通对象 ID 建立于完整体积，再在固定域一对一匹配。major=全体积≥20；重叠边≥5体素且覆盖真值对象≥5%；IoU/coverage≥0.1只作诊断。密集井已命中的对象不能称仍完全隐藏。连续匹配分数、新增命中体素、分裂、合并、未匹配和小碎片见对象表。", "",
                 "|案例|seed|臂|N|原M9零命中主要对象ID|新增井命中体素|original-free匹配IoU|coverage|诊断检测|", "|---|---:|---|---:|---:|---:|---:|---:|---|"])
    for o in objects:
        if o["domain"]=="E_original_free" and o["original_zero_M9_major"]=="True":
            text.append("|"+"|".join(o[k] or "N/A" for k in ("case_id","source_seed","arm","well_count","object_id","new_well_hit_voxels","matched_iou","matched_coverage","detected_diagnostic"))+"|")
    if state["paired_differences"]:
        text.extend(["", "## 同 seed 配对差 C−A", "", "|seed|holdout mIoU|original-free mIoU|original-free label9 IoU|", "|---:|---:|---:|---:|"])
        for r in state["paired_differences"]:
            text.append(f"|{r['source_seed']}|{r['holdout_miou']:+.6f}|{r['original_free_miou']:+.6f}|{r['original_free_target_iou']:+.6f}|")
        text.append("|配对差中位数|"+"|".join(f"{np.median([r[k] for r in state['paired_differences']]):+.6f}" for k in ("holdout_miou","original_free_miou","original_free_target_iou"))+"|")
    text.extend(["", "## 结论边界与下一步", "",
                 "本轮区分了准确井信息效应、平滑属性转岩性的错误和冻结 Flow 对错误硬井的响应。类别收益不自动证明对象恢复；应结合上述对象匹配、目标体积和碎片数据解读。", "",
                 ("开发例确认通过，可考虑另行冻结真实正反演替代方案，但跨例、对象/体积和直接分类 P 仍须共同支持；当前结果不等于真实物理桥成功。" if state["confirmation_passed"] else "本轮尚未获得通过预定确认规则的伪井工作点；优先诊断伪井错误及硬写入的代价，不直接转入真实正反演或新增训练。"), "",
                 "码本为人为设定的 density_proxy 与 label9 对比 susceptibility_contrast_proxy，是可分属性上限的受控合成证据，不是校准岩石物理或真实反演。sigma=1、无噪声，无参数扫描；sigma0仅CPU往返且非空气分类无碰撞。所有0…13类别参与分类，固定非空气码本尺度；原硬空气及9井优先。", "",
                 "holdout未作为类别条件输入，但参与合成完整属性平滑；不构成严格独立观测留出。案例历史已使用，预训练重叠未知；三个source seeds不是独立地质样本，不做n=3显著性或泛化声明。4096口端点为直接写入，未运行Flow。未运行训练、微调、adapter、真实正反演、sigma/井位/seed搜索。", "",
                 "## 资产与核验", "",
                 "- PROTOCOL.json / layout.json / classifier.json：冻结规则、全井序及sigma0审计。",
                 "- inference_registry.json 与 evaluation_registry.json：准备好的条件与离线评价资产分离；CUDA进程文件防火墙拒绝真值和完整P。",
                 "- predictions/*/receipt.json：源代码、条件、CPU噪声、预测和checkpoint哈希、EMA、耗时、前向数及原/新增条件违例。A复用保留原预测指针和复用依据。",
                 "- sample_metrics.csv：两固定域、逐类交并/分母/IoU/P/R、准确率及目标体积误差。",
                 "- object_metrics.csv / predicted_object_metrics.csv / overlap_matrices：全对象诊断。",
                 "- pseudo_well_confusion.csv / confusion：逐档伪井混淆、完整伪体各固定域混淆。",
                 "- execution_checklist.csv / execution_state.json：实际执行、跳过原因、配对差与网络预算。",
                 "- TEST_RESULTS.json / DELIVERY_AUDIT.json：测试及独立资产核验。", ""])
    for case in cases:
        text.extend([f"![{case} dose](figures/{case}_holdout_dose.png)",f"![{case} slices](figures/{case}_fixed_center_slices.png)",f"![{case} confusion and objects](figures/{case}_pseudo_and_objects.png)",""])
    (out / "REPORT.md").write_text("\n".join(text),encoding="utf-8")
    write_json(out / "artifact_hashes.json",{str(p.relative_to(root)):file_hash(p) for p in out.rglob("*") if p.is_file()})
    print(out / "REPORT.md",flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True,type=Path)
    generate(parser.parse_args().root.resolve())
