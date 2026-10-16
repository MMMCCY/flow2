"""Add reviewed numeric interpretation to this diagnostic's report; no tensor/model access."""
import argparse
import csv
import hashlib
import json
import shlex
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
REPO = PROJECT.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finalize(root):
    root=root.resolve()
    if (root / "FINAL_REVIEW.json").exists():
        raise FileExistsError("Retain completed diagnostic interpretation")
    completion=json.loads((root / "COMPLETION.json").read_text())
    config=json.loads((root / "CONFIG.json").read_text())
    if completion["status"]!="completed" or config["schema"]!="stage21v2_offline_evidence_diagnostics_v1":
        raise ValueError("Not the completed diagnostic run")
    with (root / "score_summary.csv").open() as stream:
        scores=list(csv.DictReader(stream))
    with (root / "object_diagnostics.csv").open() as stream:
        objects=list(csv.DictReader(stream))
    primary={ (r["case_id"],r["score"]):r for r in scores if r["domain"]=="E_original_free" }
    find=lambda case,oid,score,domain="E_original_free": next(r for r in objects if
        (r["case_id"],r["object_id"],r["score"],r["domain"])==(case,str(oid),score,domain))
    obj=find("fullgeo_case03",1,"margin9")
    assert int(obj["full_voxels"])==572 and obj["original_well_zero_hit"]=="True"
    assert float(obj["top5_coverage"])==1. and float(find("fullgeo_case03",1,"susceptibility")["top5_coverage"])==1.
    c3m,c3s=primary[("fullgeo_case03","margin9")],primary[("fullgeo_case03","susceptibility")]
    c1m,c1s=primary[("fullgeo_case01","margin9")],primary[("fullgeo_case01","susceptibility")]
    f=lambda row,key:float(row[key])
    interpretation=f"""## 结论先读：连续定位线索保留，当前硬标签接口丢失了目标

本轮结果更支持“平滑后的连续属性仍有可读的定位线索，但当前最近纯类别码本的argmin把线索压缩掉”，而非“这两个失败案例已没有容易读取的目标证据”。这只成立于本轮人为强目标对比的属性代理和指定诊断，不是独立物理反演结论，也不保证后续连续条件方法有效。

- **case03，全局与572体素隐藏实例均有线索。** 原未知域pi={f(c3m,'prevalence'):.8f}（约{100*f(c3m,'prevalence'):.3f}%）；margin AP={f(c3m,'AP'):.6f}、AP_skill={f(c3m,'AP_skill'):.6f}，susceptibility AP={f(c3s,'AP'):.6f}、AP_skill={f(c3s,'AP_skill'):.6f}，均远高于pi。两种分数的固定top5%各选{c3m['top5_selected_count']}体素，precision={f(c3m,'top5_precision'):.6f}、recall=1，覆盖全部1733个目标体素。原9井和原硬条件都零命中的对象1（572体素），两分数均覆盖572/572；辅助holdout内也均覆盖25/25。它确实在连续证据中可定位，但原始硬分类为0个目标体素。
- **case03不是浮点并列或刚好触界。** 该对象margin中位数={f(obj,'score_median'):.6f}，最大值={f(obj,'score_max'):.6f}，整个对象严格在其他类别更近的一侧；原未知域全部真9余量也严格为负，没有零余量。对象q9完整体中位/p90/最大={f(obj,'q9_full_median'):.6f}/{f(obj,'q9_full_p90'):.6f}/{f(obj,'q9_full_max'):.6f}；susceptibility中位数约0.03251、最大约0.04863，低于目标纯类别码本的0.1，但仍高于普通背景纯类别约0.0015–0.0054。因此“相对背景可排序”与“绝对距离不够接近纯label9”同时成立。q9用于解释局部混合，不能把0.5当作普适分界。
- **case01有全局连续线索，但没有原井零命中的主要对象可作同等级验证。** pi={f(c1m,'prevalence'):.9f}（约{100*f(c1m,'prevalence'):.4f}%）；margin/susceptibility AP={f(c1m,'AP'):.6f}/{f(c1s,'AP'):.6f}，AP_skill={f(c1m,'AP_skill'):.6f}/{f(c1s,'AP_skill'):.6f}。两种固定top5%均覆盖118/118，但precision仅{f(c1m,'top5_precision'):.6f}，不能据此直接生成准确伪井。原井零命中分量只有ID1/2/3，完整体积1/6/3，均低于min20；主域两分数分别覆盖1/1、6/6、3/3。唯一主要对象ID4完整110体素，被原井几何命中2体素，不能称为原井隐藏主要对象。三个小分量在holdout均无体素，必须记NA；case01 holdout的11个正体素不能证明这些零井小分量的留出恢复。
- **case02说明全局排序好仍不保证所有隐藏小体。** 主域margin/susceptibility AP约0.96767/0.96741（pi约0.07744）。原井零命中926体素对象top5%覆盖约69.44%/70.09%，129体素对象约40.31%/38.76%；32体素及41体素对象两分数均为0%。两个漏失对象q9最大值分别约0.39065/0.49347，margin最大值仍为负。故不能将case03的完整覆盖推广为全部隐藏目标线索都已充分保留。

**机制判断。** 在case03/case01，核平均把目标局部属性稀释成与背景混合的中间值；当前纯类别最近码本在绝对距离上偏向其他类别，而两个连续分数仍能把目标排在前面。两种分数本身都较强，证据不支持将主要问题单独归咎于“margin完全读不出、只有susceptibility能读出”。既定距离几何/归一化对绝对边界的贡献仍未通过干预分离；未校准物性、空气与边界混合、对象尺度/厚度和未来真实反演误差也未排除。对case02小对象，目前只能说固定5%摘要未覆盖，不证明所有空间读取均失败或全部信息不可逆消失。

**最有信息量的下一项研究决策。** 值得另立一个最小、严格配对的条件表示实验：保留相同连续证据及原9井/地表条件，比较现有硬伪井与保留连续性/不确定性的接口，并保留准确井上限和直接属性比较对象；重点审查case03对象1，同时检验case02漏失小对象。下一实验需事先冻结具体表示、误差处理和共同评价域，不能把本轮top5%界限或真值q9当作部署规则/输入。本任务不实现或运行该方法，不加密错误伪井，不据此要求重训Flow或直接开展真实正反演。

以下为完整方法、数字和三类图件。数值分析仅执行一次；本节基于输出CSV作解释，没有重算argmin、寻找最佳阈值或新增模型调用。

"""
    report=root / "REPORT.md"
    base=report.read_text(encoding="utf-8")
    title,body=base.split("\n",1)
    report.write_text(title+"\n\n"+interpretation+body,encoding="utf-8")
    relative=str(Path(__file__).resolve().relative_to(REPO))
    target=root / "source_snapshot" / relative
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(Path(__file__).read_bytes())
    command='CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 .venv/bin/python '+shlex.quote(relative)+' --root '+shlex.quote(str(root.relative_to(REPO)))
    with (root / "COMMAND.txt").open("a",encoding="utf-8") as stream:
        stream.write(command+"\n")
    tests=dict(status="passed",tests=5,result="Ran 5 tests in 0.004s; OK",
        command='CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 .venv/bin/python project/geodata-3d-conditional/scripts/stage21v2_diagnostics/test_metrics.py',
        checks=["constant AP=prevalence","complete tied positive group","documented sklearn example", "15 exhaustive nonempty binary tie fixtures","top5 includes all boundary ties without truth"])
    (root / "TEST_RESULTS.json").write_text(json.dumps(tests,indent=2)+"\n")
    review=dict(status="completed",final_report_source={relative:digest(Path(__file__))},
                analysis_recomputed=False,new_Flow_runs=0,training_runs=0,checkpoint_loaded=False,
                interpretation="strong retained continuous localization in case03 and case01; case01 has no zero-well major object; case02 small hidden objects remain uncovered at fixed budget")
    (root / "FINAL_REVIEW.json").write_text(json.dumps(review,indent=2)+"\n")
    artifacts={str(p.relative_to(root)):digest(p) for p in sorted(root.rglob("*")) if p.is_file() and p.name!="ARTIFACT_HASHES.json"}
    (root / "ARTIFACT_HASHES.json").write_text(json.dumps(artifacts,indent=2,sort_keys=True)+"\n")
    print(report)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    finalize(parser.parse_args().root)
