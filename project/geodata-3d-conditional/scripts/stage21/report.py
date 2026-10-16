"""Fixed-selection retrospective figures and report for the completed Stage21 first step."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from scripts.stage21.common import (
    REPO, file_hash, load_tensor, read_json, verify_tensor, write_json,
)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def num(row, key):
    value = row.get(key, "")
    return float(value) if value not in (None, "", "None") else None


def fmt(value):
    return "N/A" if value is None else f"{value:.4f}"


def save(fig, path, svg=False):
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    if svg:
        fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def boundary(mask):
    return np.argwhere(mask & ~ndimage.binary_erosion(mask))


def draw_target(ax, target, truth=None):
    groups = [(target, "#d89423", "Truth label 9")] if truth is None else [
        (target & truth, "#167ea6", "True positive"), (target & ~truth, "#d94942", "False positive")]
    for mask, color, label in groups:
        xyz = boundary(mask)
        if len(xyz):
            ax.scatter(*xyz.T, s=2.5, marker="s", c=color, alpha=.72, linewidths=0, label=label, rasterized=True)
    ax.set(xlim=(0,63), ylim=(0,63), zlim=(0,63), xlabel="X index", ylabel="Y index", zlabel="Z index (up)")
    ax.set_box_aspect((1,1,1))
    ax.set_xticks([0,32,63]); ax.set_yticks([0,32,63]); ax.set_zticks([0,32,63])
    ax.tick_params(labelsize=7)


def build_report(args):
    args.registry = args.registry.resolve()
    args.run = args.run.resolve()
    args.report = args.report.resolve()
    p = read_json(args.registry)["protocol"]
    registry = read_json(args.registry)
    summary = read_json(args.report / "summary.json")
    state = read_json(args.run / "sampling_state.json")
    ereg = read_json(args.registry.parent / "evaluation_registry.json")
    cases = read_csv(args.report / "case_level_summary.csv")
    delta = read_csv(args.report / "case_paired_deltas.csv")
    exposure = read_csv(args.report / "object_exposure.csv")
    samples = read_csv(args.report / "sample_metrics.csv")
    object_metrics = read_csv(args.report / "object_metrics.csv")
    if (args.report / "REPORT.md").exists() or (args.report / "figures").exists():
        raise FileExistsError("report/figures already exist; use a new evaluation directory")
    figures = args.report / "figures"
    figures.mkdir()
    colors = plt.cm.tab10(np.linspace(0, .8, 5))
    keys = ["target_iou", "target_precision", "target_recall", "fixed_class_miou",
            "strict_unobserved_major_matched_iou", "target_absolute_relative_volume_error"]
    labels = ["Label-9 IoU", "Label-9 precision", "Label-9 recall", "Fixed-class mIoU",
              "Strict hidden major: matched IoU", "Absolute relative target-volume error"]
    fig, axes = plt.subplots(2,3,figsize=(14,8))
    for ax,key,label in zip(axes.flat,keys,labels):
        for case,color in zip(p["cases"],colors):
            cr = sorted((r for r in cases if r["case_id"] == case), key=lambda r:int(r["well_count"]))
            y = [num(r,key) for r in cr]
            if any(v is not None for v in y):
                ax.plot(p["well_counts"], [np.nan if v is None else v for v in y], "o-", color=color, lw=1.2, label=case)
        overall = [summary["levels"][str(n)][key]["case_median"] for n in p["well_counts"]]
        ax.plot(p["well_counts"], overall, "s--", color="black", lw=2, label="Median over case medians")
        ax.set(title=label, xlabel="Total complete wells", xticks=[9,16,25])
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    fig.suptitle("Stage21 common domain: (truth != air) AND NOT (surface OR 25 well columns)\n"
                 "Three-seed median per case; 5 reused diagnostic cases; hidden-major metric valid in 3 cases")
    fig.tight_layout(rect=(0,0,1,.92))
    save(fig, figures / "common_domain_dose_curves.png", svg=True)
    fig,axes = plt.subplots(1,3,figsize=(14,4.8))
    for ax,key in zip(axes,["target_iou", "fixed_class_miou", "strict_unobserved_major_matched_iou"]):
        for offset,comparison in zip([-.22,0,.22],["16-9","25-9","25-16"]):
            rows = [next(r for r in delta if r["case_id"] == c and r["comparison"] == comparison) for c in p["cases"]]
            values = [num(r,key+"_median_paired_difference") for r in rows]
            ax.bar(np.arange(5)+offset, [np.nan if v is None else v for v in values], width=.21, label=comparison)
        ax.axhline(0,color="black",lw=.8)
        ax.set(title=key.replace("_"," "),xticks=range(5),xticklabels=[f"case{i:02d}" for i in range(1,6)])
        ax.tick_params(axis="x",rotation=30)
        ax.grid(axis="y",alpha=.2)
    axes[0].legend()
    fig.suptitle("Within-case median of same-seed paired differences (N/A: no strict hidden major object)")
    fig.tight_layout()
    save(fig, figures / "paired_case_differences.png", svg=True)
    record_map = {(r["case_id"], r["source_seed"], r["well_count"]):r for r in state["records"]}
    for case in p["cases"]:
        truth = verify_tensor(ereg["cases"][case]["truth"]).numpy()[0,0]
        preds = [load_tensor(args.run / record_map[(case,p["figure_seed"],n)]["prediction_path"]).numpy()[0,0] for n in p["well_counts"]]
        for view,azim in enumerate([-60,145],1):
            fig = plt.figure(figsize=(16,4.7))
            for i,(name,volume) in enumerate(zip(["Truth","N9","N16","N25"],[truth,*preds])):
                ax = fig.add_subplot(1,4,i+1,projection="3d")
                draw_target(ax, volume==9, None if i==0 else truth==9)
                ax.view_init(elev=22,azim=azim)
                title = f"{name}: full target voxels={int((volume==9).sum())}"
                if i:
                    r = next(r for r in samples if r["case_id"] == case and int(r["source_seed"]) == p["figure_seed"] and int(r["well_count"]) == p["well_counts"][i-1])
                    title += f"\ncommon IoU={fmt(num(r,'target_iou'))}"
                ax.set_title(title,fontsize=10)
                if i in (0,1) and ax.get_legend_handles_labels()[0]:
                    ax.legend(loc="lower left",fontsize=7)
            fig.suptitle(f"{case}, fixed source seed 142, view {view}; full 64-cube domain\n"
                         "Boundary-voxel centers, equal axes; blue=true positive, red=false positive. Overlap can occlude interiors.")
            fig.tight_layout(rect=(0,0,1,.89))
            save(fig,figures / f"{case}_seed142_targets_view{view}.png")
        fig,ax = plt.subplots(figsize=(7.2,6))
        ax.imshow((truth==9).any(axis=2).T,origin="lower",extent=(-.5,63.5,-.5,63.5),cmap="Greys",alpha=.35)
        xy = registry["layout"]["levels"]["25"]
        for start,end,color,name in [(0,9,"#222222","Original 9"),(9,16,"#e18b1e","Added to N16"),(16,25,"#087fa2","Added to N25")]:
            points = np.array(xy[start:end])
            ax.scatter(points[:,0],points[:,1],c=color,s=65,marker="o",label=name,edgecolor="white",zorder=3)
        for i,(x,y) in enumerate(xy):
            hits = int((truth[x,y,:]==9).sum())
            ax.text(x+.8,y+.8,f"{i+1}"+(f" ({hits})" if hits else ""),fontsize=6)
            if hits:
                ax.scatter([x],[y],s=110,facecolors="none",edgecolors="red",linewidths=1,zorder=4)
        for v in (15.5,31.5,47.5):
            ax.axvline(v,color="gray",alpha=.25);ax.axhline(v,color="gray",alpha=.25)
        ax.set(xlim=(-1,65),ylim=(-1,65),xlabel="X index",ylabel="Y index",aspect="equal",
               title=f"{case}: frozen nested wells, layout seed 21001\nGray=target XY footprint; red ring=target hit; parentheses=hit voxels")
        ax.legend(loc="upper center",bbox_to_anchor=(.5,-.12),ncol=3,fontsize=8)
        fig.tight_layout()
        save(fig,figures / f"{case}_well_target_hits.png")
    base_iou = summary["levels"]["9"]["target_iou"]["case_median"]
    end_iou = summary["levels"]["25"]["target_iou"]["case_median"]
    base_miou = summary["levels"]["9"]["fixed_class_miou"]["case_median"]
    end_miou = summary["levels"]["25"]["fixed_class_miou"]["case_median"]
    paired_iou = summary["paired"]["25-9"]["target_iou"]
    hidden_major = [r for r in exposure if r["major"] == "True" and r["exposure_group"] == "strict_unobserved"]
    newly_hit = sum(r["n25_status"] == "newly_hit" for r in hidden_major)
    lines = ["# Stage21 第一步：准确新增完整井条件实验", "",
             "主要发现：准确新增井带来了部分共同井外域类别/目标指标改善，但当前证据不足以支持原严格隐藏主要对象已恢复。", "",
             f"N9→N25的跨案例、案例内seed中位IoU为{fmt(base_iou)}→{fmt(end_iou)}，固定类别mIoU为{fmt(base_miou)}→{fmt(end_miou)}。"
             f"单独计算同case/seed配对差后，25−9的跨案例中位为{fmt(paired_iou['cross_case_median_of_case_median_paired_differences'])}，"
             f"{paired_iou['positive_cases']}/{paired_iou['valid_cases']}案例的配对差中位为正；绝对目标IoU仍低。", "",
             f"7个原严格隐藏主要对象中只有{newly_hit}个被新增井命中。N25仍未命中组的跨案例匹配IoU中位为0；"
             "不能将总体类别收益解释为发现并恢复了隐藏主体。", "",
             "本轮完成准确新增井实验。第二步（地球物理转完整虚拟井及错误/不确定性）和第三步（虚拟井输入后的地质与物理验证）尚未实施。", "",
             "5个复用诊断案例 × 3个CPU source seeds（42/142/242）× 总井数9/16/25，共45份正式输出；smoke另计。"
             "Flow主干、embedding和checkpoint冻结，411项网络参数采用EMA，embedding保持checkpoint原值。32步midpoint时间网格Euler，每步一次速度评估。", "",
             "新增井来自同一真值的oracle采集；原9井及原surface完整保留。该实验检验原生类别条件接口的信息利用能力，不能声称地球物理虚拟井已成功。", "",
             "## 工程验证", "",
             f"- 正式样本：{len(state['records'])}/45；条件违例：{sum(r['condition_violations'] for r in state['records'])}。",
             "- CUDA smoke：case03、seed142、三档；独立N9连续状态逐元素完全等价；smoke不参与统计。",
             "- SSH focused tests及已有采样器/冻结轨迹回归：29项通过；准确命令及测试源码hash见上级 TEST_RESULTS_final.json。",
             "- 模型、EMA、embedding内容hash逐样本一致；每步输入条件精确、state和velocity有限；三档使用投影前相同CPU初噪声。",
             "- 推理进程仅使用条件清单，运行时文件访问guard禁止真值、原benchmark和评价清单；完成receipt先于回顾评价保存。",
             f"- Checkpoint SHA-256：`{p['checkpoint_sha256']}`。",
             f"- 推理代码、registry、模型张量、预测及初噪声hash见 `{args.run.name}/sampling_state.json` 与 `inference_registry.json`。", "",
             "## 共同井外域结果", "",
             "所有三档共享 E=(truth!=-1) AND NOT(surface OR B25)。以下为先取每案例3个seed中位、再跨5案例取中位；隐藏主要对象仅3个有效案例。", "",
             "|共同域指标|N9|N16|N25|", "|---|---:|---:|---:|"]
    for key,label in [("target_iou","label9 IoU"),("target_precision","precision"),("target_recall","recall"),
                      ("fixed_class_miou","固定真值类别 mIoU"),("target_absolute_relative_volume_error","绝对相对体积误差"),
                      ("target_fp","假阳性目标体素"),("merge_predicted_object_count","合并预测对象数（诊断规则）"),
                      ("split_truth_object_count","被分裂真值对象数（诊断规则）"),
                      ("strict_unobserved_major_matched_iou","原严格未观测主要对象匹配IoU"),
                      ("strict_unobserved_major_any_prediction_coverage","原严格未观测主要对象任意预测覆盖"),
                      ("strict_newly_hit_by25_major_matched_iou","N25新增命中原隐藏主要对象匹配IoU"),
                      ("strict_still_unhit_by25_major_matched_iou","N25仍零命中原隐藏主要对象匹配IoU")]:
        lines.append("|"+label+"|"+"|".join(fmt(summary["levels"][str(n)][key]["case_median"]) for n in (9,16,25))+"|")
    lines += ["", "## 每案例配对差", "", "同一case/seed先计算差，再对3个seed取中位；这不同于两个seed中位数之差。两种均保存在 case_paired_deltas.csv，下面只用前者。", "",
              "|案例|IoU 16−9|IoU 25−9|IoU 25−16|隐藏主要对象匹配IoU 25−9|", "|---|---:|---:|---:|---:|"]
    for case in p["cases"]:
        rows = {r["comparison"]: r for r in delta if r["case_id"]==case}
        lines.append("|"+case+"|"+"|".join(fmt(num(rows[k],"target_iou_median_paired_difference")) for k in ("16-9","25-9","25-16"))+"|"+fmt(num(rows["25-9"],"strict_unobserved_major_matched_iou_median_paired_difference"))+"|")
    lines += ["", "配对方向仅是描述，不作为自动成功判断：", ""]
    for comparison in ("16-9", "25-9", "25-16"):
        t = summary["paired"][comparison]["target_iou"]
        h = summary["paired"][comparison]["strict_unobserved_major_matched_iou"]
        lines.append(f"- {comparison}：目标IoU案例内配对差中位在{t['positive_cases']}/{t['valid_cases']}案例为正、{t['negative_cases']}案例为负；"
                     f"跨案例中位差={fmt(t['cross_case_median_of_case_median_paired_differences'])}。"
                     f"原隐藏主要对象匹配IoU在{h['positive_cases']}/{h['valid_cases']}有效案例为正、{h['negative_cases']}案例为负。")
    lines += ["", "## 对象曝光与匹配解释", "",
              f"原严格未观测、至少20体素且在共同域可评价的目标分量共{summary['strict_hidden_major_objects']}个，分布于{summary['strict_hidden_major_cases']}个案例。case01/04对该组记N/A，未填0参与聚合。", "",
              "|案例/对象ID|完整体素|共同域体素|N16新增命中体素|N25新增命中体素|", "|---|---:|---:|---:|---:|"]
    for r in exposure:
        if r["major"] == "True" and r["exposure_group"] == "strict_unobserved":
            lines.append(f"|{r['case_id']}/{r['object_id']}|{r['full_voxels']}|{r['common_voxels']}|{r['n16_extra_hit_voxels']}|{r['n25_extra_hit_voxels']}|")
    lines += ["", "预登记检测诊断（不是通用成功标准）：", ""]
    for n in (9,16,25):
        selected = [r for r in object_metrics if r["well_count"] == str(n) and r["major"] == "True" and r["exposure_group"] == "strict_unobserved"]
        lines.append(f"- N{n}：{sum(r['detected_diagnostic'] == 'True' for r in selected)}/{len(selected)}个对象实现满足诊断阈值。"
                     "这里是7个对象各3个随机实现，不能当作21个独立地质对象。")
    lines += ["", "真值和预测的26连通ID都在完整体上建立，再截取共同域计交并比。一对一最大IoU匹配防止一个吞并体重复领取恢复；匹配precision/IoU包含整个匹配预测分量在共同域中的假阳性。"
              "完整和共同域重叠矩阵、并集、未匹配真值/预测分量分别保存在 overlap_matrices、object_metrics.csv、predicted_object_metrics.csv。对象是连通分量代理，不必等同地质事件。", "",
              "合并/分裂边规则在推理前冻结：共同域交集至少5体素且覆盖该真值对象至少5%；检测诊断还要求一对一匹配IoU及覆盖均至少0.1。"
              "这些是可复核的诊断约定，不是公认检测标准或方法成功门槛；连续IoU/覆盖全部保留。主结构分层阈值为完整分量至少20体素，其余分量也全部列出。", "",
              "## 解释边界与下一研究决策", "",
              "共同域数值变化反映井列以外的补全响应。新增井命中后改善的对象证明Flow利用准确目标条件的能力；仍未命中对象的变化需单独检查。"
              "全类改善、召回改善、假阳性和合并/分裂须联合解读；没有以微小正差自动宣布整个方法成功。", "",
              "第一步结论限定于当前单一布局、5个已在历史中使用的探索案例及3次随机实现；不能称独立TEST、现场验证或校准后验。原Flow预训练重叠无法排除。"
              "无地球物理转换、物理筛选、adapter、新学习或物理正演结果，本轮不推断物理一致性。", "",
              "下一决策由研究者根据全部案例和原隐藏对象表确定：先判断当前井位信息量（7个主要隐藏对象仅1个新命中）与低绝对恢复水平，是否足以支持进入虚拟井序列桥。"
              "若进一步研究新布局或信息量，应另行预注册而不替换本轮结果；若继续序列桥，再冻结一两个界面偏移/漏薄段错误诊断，并确定物理类型、物性歧义、时深关系来源和联合相容完整序列情景。"
              "后续第三步需保留原井、同位准确井、观测虚拟井、仅先验虚拟井四臂，并检验最终硬地质的原观测一致性。", "",
              "## 文件与图件", "",
              "- sample_metrics.csv / case_level_summary.csv：共同域主指标、明确前缀的full_nonair与own_free辅助指标。",
              "- paired_deltas.csv / case_paired_deltas.csv：样本配对差及两种案例聚合。",
              "- object_exposure.csv / object_metrics.csv / predicted_object_metrics.csv：全对象曝光、匹配、假阳性和未匹配对象。",
              "- common_domain_audit.json：各案例共同域hash、类别分母、有效对象数；pairing_audit.json：完整初噪声配对。",
              "- figures/common_domain_dose_curves.png：同一指标、同一域的三档曲线；paired_case_differences.png：每案例配对差。",
              "- 全5案例固定seed142、两固定相机的truth/N9/N16/N25完整域目标图及井位命中图；未按IoU挑选。",
              "- 三维图使用边界体素中心散点而非实体渲染，重叠可能遮挡内部；两视角及完整体素输出供复核，未裁真值ROI。", ""]
    (args.report / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(args.report / "report_source_hashes.json", {str(Path(__file__).relative_to(REPO)):file_hash(__file__),
               "evaluation_summary_sha256":file_hash(args.report / "summary.json")})
    # Covers figures, tables, masks, matrices, reports, and original condition/noise/output ledgers.
    roots = [args.registry.parent / "conditions", args.run, args.report]
    files = [f for root in roots for f in root.rglob("*") if f.is_file() and f.name != "artifact_hashes.json"]
    files += [f for f in args.registry.parent.iterdir() if f.is_file()]
    write_json(args.report / "artifact_hashes.json", {str(f.relative_to(REPO)):file_hash(f) for f in sorted(set(files))})
    print(f"Report and fixed-selection figures written to {args.report}",flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry",type=Path,required=True)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    build_report(parser.parse_args())
