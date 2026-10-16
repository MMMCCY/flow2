"""Offline-only supplement: whole-volume classifier diagnostics and paired interpretation.

Added after the protocol-valid CUDA screening/confirmation; never imported by
inference, and does not change conditions, progression, thresholds or sources.
"""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[6]))
# Resolve from the experiment's immutable run root, independently of cwd.
RUN_ROOT=Path(__file__).resolve().parents[1]
PROJECT=RUN_ROOT.parents[2]
sys.path.insert(0,str(PROJECT))
from scripts.stage21v2.common import *
from scripts.stage21v2.workflow import metrics
from scripts.stage21.evaluate import connected,match_objects


def summarize(root):
    torch.set_num_threads(4)
    state=read_json(root / "execution_state.json")
    if state["status"] != "completed":
        raise ValueError("wait for sealed workflow completion")
    ev=read_json(root / "evaluation_registry.json")
    reg=read_json(root / "inference_registry.json")
    rows,objects,predicted=[],[],[]
    supplements=[]
    for case in CASES:
        if not any(s["case_id"]==case for s in state["samples"]):
            continue
        truth=verify_tensor(ev["cases"][case]["truth"])
        orig=check_condition(reg["conditions"][case]["A_9"])
        domains={"full_nonair":truth!=-1,
                 "E_original_free":verify_tensor(ev["cases"][case]["domains"]["E_original_free"]["tensor"])}
        ti,_=connected(truth.numpy()[0,0]==9)
        for kind,key in (("RAW_CLASSIFICATION","pseudo_raw"),("P_ORIGINAL_PROJECTED","pseudo")):
            pred=verify_tensor(ev["cases"][case][key])
            pi,_=connected(pred.numpy()[0,0]==9)
            for name,domain in domains.items():
                classes=sorted(int(x) for x in truth[domain].unique())
                mm=metrics(pred.numpy()[0,0],truth.numpy()[0,0],domain.numpy()[0,0],classes)
                oo,pp,ss,_=match_objects(ti,pi,domain.numpy()[0,0],OBJECT_RULES)
                identity=dict(case_id=case,kind=kind,domain=name)
                rows.append({**identity,**mm,**ss})
                objects.extend({**identity,**o} for o in oo)
                predicted.extend({**identity,**p} for p in pp)
        raw=next(r for r in rows if (r["case_id"],r["kind"],r["domain"])==(case,"RAW_CLASSIFICATION","E_original_free"))
        supplements.append(f"- {case}：原未知域原始伪岩性 label9 预测体素 {raw['target_pred_volume']}，真值 {raw['target_truth_volume']}；precision={raw['target_precision'] if raw['target_precision'] is not None else 'N/A（无预测）'}，recall={raw['target_recall']:.6f}，IoU={raw['target_iou']:.6f}。这是固定 sigma1 转换后的损失，不因结果调整码本或尺度。")
    write_csv(root / "pseudo_volume_metrics.csv",rows)
    write_csv(root / "pseudo_volume_objects.csv",objects)
    write_csv(root / "pseudo_volume_predicted_objects.csv",predicted or [dict(status="no_predicted_target_objects")])
    paired=[]
    for c in [s for s in state["samples"] if s["arm"]=="C"]:
        case,seed,n=c["case_id"],c["source_seed"],c["well_count"]
        others={s["arm"]:s for s in state["samples"] if s["case_id"]==case
                and (s["source_seed"]==seed or s["arm"]=="P")
                and (s["well_count"] in (None,9,n))}
        for arm in "ADP":
            for domain in ("E_holdout","E_original_free"):
                cc,base=c["metrics"][domain],others[arm]["metrics"][domain]
                row=dict(case_id=case,source_seed=seed,well_count=n,comparison=f"C-{arm}",domain=domain)
                for key in ("fixed_class_miou","target_iou","target_precision","target_recall","target_absolute_relative_volume_error"):
                    row[key]=cc[key]-base[key] if cc[key] is not None and base[key] is not None else None
                paired.append(row)
    write_csv(root / "C_vs_A_D_P_paired_differences.csv",paired)
    appendix=["", "## 补充：完整伪体与直接写入的定量解释", "",*supplements,"",
              "下表先按同case/seed相减，再对该案例已执行seed取中位；case03为3个seed，case01/02各1个seed。C−D描述超出直接写入的生成响应，C−P比较完整直接属性分类。mIoU与目标IoU的优劣分别解释。", "",
              "|案例|比较|域|seed数|配对ΔmIoU中位|配对Δlabel9 IoU中位|配对Δ目标recall中位|", "|---|---|---|---:|---:|---:|---:|"]
    for case in CASES:
        for comparison in ("C-A","C-D","C-P"):
            for domain in ("E_holdout","E_original_free"):
                rr=[r for r in paired if (r["case_id"],r["comparison"],r["domain"])==(case,comparison,domain)]
                if rr:
                    vals=[np.median([r[k] for r in rr]) for k in ("fixed_class_miou","target_iou","target_recall")]
                    appendix.append(f"|{case}|{comparison}|{domain}|{len(rr)}|"+"|".join(f"{v:+.6f}" for v in vals)+"|")
    appendix.extend(["", "结构与体积仍需独立核对：", "",
              "|案例|seed|臂|N|original-free目标体积|绝对相对体积误差|完整目标连通分量|合并数|分裂数|≤5体素碎片数|", "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"])
    import csv
    with (root / "sample_metrics.csv").open() as stream:
        for r in csv.DictReader(stream):
            if r["domain"]=="E_original_free":
                appendix.append("|"+"|".join(r[k] or "N/A" for k in ("case_id","source_seed","arm","well_count","target_pred_volume","target_absolute_relative_volume_error","predicted_objects_full","merge_predicted_object_count","split_truth_object_count","tiny_fragment_full_count_le5"))+"|")
    appendix.extend(["", "完整伪体的全地下/原未知域逐类指标及对象诊断见 pseudo_volume_metrics.csv、pseudo_volume_objects.csv。混淆矩阵使用原始分类体；P另加回原硬条件。precision分母为0记N/A，recall/IoU在存在真值且无预测时为0，未伪装为分类碰撞。", ""])
    path=root / "report/REPORT.md"
    with path.open("a",encoding="utf-8") as stream:
        stream.write("\n".join(appendix))
    snapshot_verified=all(file_hash(root / "source_snapshot" / rel)==h for rel,h in reg["source_sha256"].items())
    if not snapshot_verified:
        raise ValueError("source snapshot mismatch")
    write_json(root / "OFFLINE_SUPPLEMENT_AUDIT.json",dict(status="passed",source_snapshot_exact=True,
               supplementary_source_sha256=file_hash(Path(__file__)),
               pseudo_volume_metrics_sha256=file_hash(root / "pseudo_volume_metrics.csv"),
               paired_differences_sha256=file_hash(root / "C_vs_A_D_P_paired_differences.csv"),
               no_new_model_calls=True,original_sampling_sources_unchanged=sources()==reg["source_sha256"]))
    write_json(root / "report/artifact_hashes.json",{str(p.relative_to(root)):file_hash(p) for p in (root / "report").rglob("*") if p.is_file() and p.name!="artifact_hashes.json"})


if __name__=="__main__":
    summarize(RUN_ROOT)
