"""CPU-only, immutable offline summary of six sealed seed42 predictions."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys
sys.dont_write_bytecode = True
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from scripts.stage21.common import REPO, read_json, write_json, file_hash, verify_tensor
from scripts.stage21.evaluate import voxel_metrics, ratio

R21 = PROJECT / "experiments/stage21v2_dense_virtual_boreholes/run_20261016_v1"
R23 = PROJECT / "experiments/stage23_full_pseudo_wells/run_20261017_v2"
R24 = PROJECT / "experiments/stage24_multi_case/run_20261017_v1"
NAMES = ["Frozen Flow", "Proposed method"]
COLORS = ["#6b8196", "#158477"]
PALETTE = ["#ffffff", *plt.cm.tab20(np.linspace(0, 1, 14))]
CMAP = ListedColormap(PALETTE)
NORM = BoundaryNorm(np.arange(-1.5, 14.5, 1), 15)
METRICS = ["voxel_accuracy", "fixed_class_miou", "target_iou",
           "target_precision", "target_recall", "target_pred_truth_volume_ratio"]


def asset(path):
    return {"path": str(path.relative_to(REPO)), "file_sha256": file_hash(path)}


def csv_read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows({k: "NA" if v is None else v for k, v in row.items()} for row in rows)


def volume(record, labels=True):
    tensor = verify_tensor(record)
    if tuple(tensor.shape) != (1, 1, 64, 64, 64):
        raise ValueError(f"unexpected XYZ shape: {record['path']}")
    v = tensor.numpy()[0, 0]
    if labels and (not np.isin(v, np.arange(-1, 14)).all()):
        raise ValueError("invalid raw labels")
    return v


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=300, facecolor="white")
    fig.savefig(out / f"{name}.pdf", facecolor="white")
    plt.close(fig)


def slices(out, case, volumes, errors):
    nrows = 6 if errors else 3
    fig, axes = plt.subplots(nrows, 3, figsize=(10.4, 18.7 if errors else 10.3))
    fig.subplots_adjust(left=.115, right=.985, top=.965, bottom=.115 if errors else .175,
                        hspace=.30, wspace=.17)
    for col, (name, v) in enumerate(zip(["Ground truth", *NAMES], volumes)):
        axes[0, col].set_title(name, fontsize=14, pad=13)
        for axis, (horizontal, vertical) in enumerate([("Y", "Z"), ("X", "Z"), ("X", "Y")]):
            sl = np.take(v, 32, axis=axis).T
            tr = np.take(volumes[0], 32, axis=axis).T
            row = axis * (2 if errors else 1)
            for offset in range(2 if errors else 1):
                ax = axes[row + offset, col]
                if offset:
                    ax.imshow(sl != tr, origin="lower", interpolation="nearest",
                              cmap=ListedColormap(["white", "#d62728"]), vmin=0, vmax=1)
                else:
                    ax.imshow(sl, origin="lower", interpolation="nearest", cmap=CMAP, norm=NORM)
                ax.set(xticks=[0, 32, 63], yticks=[0, 32, 63], xlabel=horizontal, ylabel=vertical)
                ax.tick_params(labelsize=9, length=2)
                ax.xaxis.labelpad = 0
                ax.yaxis.labelpad = 0
                if col == 0:
                    label = "Error" if offset else f"{'XYZ'[axis]} = 32"
                    ax.text(-.38, .5, label, transform=ax.transAxes, rotation=90,
                            va="center", ha="center", fontsize=12)
    handles = [Patch(facecolor=PALETTE[i + 1], edgecolor="#cccccc", linewidth=.5,
                     label=f"Raw label {i}") for i in range(14)]
    handles.insert(0, Patch(facecolor="white", edgecolor="#cccccc", label="Air (−1)"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.54, .035 if errors else .045),
               ncol=5, frameon=False, fontsize=10, columnspacing=1.4, handlelength=1.2)
    if errors:
        fig.legend(handles=[Patch(facecolor="white", edgecolor="#cccccc", label="Correct"),
                            Patch(facecolor="#d62728", label="Error")],
                   loc="lower center", bbox_to_anchor=(.54, .011), ncol=2, frameon=False, fontsize=10)
    save(fig, out, f"{case}_center_slices_{'with_errors' if errors else 'clean'}")


def table(out, rows, name):
    fig, ax = plt.subplots(figsize=(13.7, 3.45))
    ax.axis("off")
    headers = ["Case", "Method", "OA", "mIoU", "Target IoU", "Precision", "Recall", "Volume ratio"]
    cells = []
    for r in rows:
        vals = [r["case"].replace("fullgeo_", ""), r["method"]]
        for k in METRICS:
            v = r[k]
            vals.append("NA" if v is None else
                        f"{v:.1%}" if k in ["voxel_accuracy", "target_precision", "target_recall"] else
                        f"{v:.2f}" if k == METRICS[-1] else f"{v:.3f}")
        cells.append(vals)
    t = ax.table(cellText=cells, colLabels=headers, cellLoc="center", loc="center",
                 colWidths=[.08, .19, .10, .10, .12, .11, .10, .13])
    t.auto_set_font_size(False)
    t.set_fontsize(12)
    t.scale(1, 2.0)
    for (i, j), cell in t.get_celld().items():
        cell.set_edgecolor("#d8dfe4")
        cell.set_linewidth(.6)
        cell.set_facecolor("#edf2f5" if i == 0 else "white")
        if i == 0:
            cell.set_text_props(weight="bold")
        if i > 0 and j == 1:
            cell.set_text_props(color=COLORS[(i - 1) % 2])
    for pair in range(0, 6, 2):
        for j, k in enumerate(METRICS[:-1], start=2):
            a, b = rows[pair][k], rows[pair + 1][k]
            if a is not None and b is not None and not np.isclose(a, b, rtol=0, atol=1e-12):
                cell = t[pair + (1 if a > b else 2), j]
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#eaf5f1")
    fig.subplots_adjust(left=.015, right=.985, top=.985, bottom=.015)
    save(fig, out, name)


def bars(out, rows):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
    for row, domain in enumerate(["D_global_nonair", "E_holdout"]):
        for col, metric in enumerate(["fixed_class_miou", "target_iou"]):
            ax = axes[row, col]
            for method, color, offset in zip(NAMES, COLORS, [-.19, .19]):
                vals = [next(r[metric] for r in rows if r["case"] == f"fullgeo_case{i:02}"
                             and r["method"] == method and r["domain"] == domain) for i in range(1, 4)]
                rects = ax.bar(np.arange(3) + offset, [np.nan if v is None else v for v in vals],
                               .35, label=method, color=color)
                ax.bar_label(rects, labels=["NA" if v is None else f"{v:.3f}" for v in vals],
                             fontsize=10, padding=3)
            ax.set(ylim=(0, 1), xticks=np.arange(3), xticklabels=["case01", "case02", "case03"],
                   yticks=np.linspace(0, 1, 6), ylabel="mIoU" if col == 0 else "Target IoU",
                   title=f"{domain} · {'mIoU' if col == 0 else 'Target IoU'}")
            ax.set_axisbelow(True)
            ax.grid(axis="y", alpha=.18)
            ax.spines[["top", "right"]].set_visible(False)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    fig.subplots_adjust(left=.08, right=.985, bottom=.10, top=.94, hspace=.35, wspace=.23)
    save(fig, out, "quantitative_comparison")


def build(out):
    if out.exists():
        raise FileExistsError(f"choose a new immutable output directory: {out}")
    out.mkdir(parents=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    manifest = {"role": "retrospective offline display only", "seed": 42,
                "shape": [1, 1, 64, 64, 64], "axis_order": "XYZ", "slice_index": 32,
                "script": asset(Path(__file__).resolve()), "cases": {},
                "reference_slice": asset(R21 / "report/figures/fullgeo_case01_fixed_center_slices.png"),
                "palette_source": asset(PROJECT / "scripts/stage21v2/report.py"),
                "evaluator": asset(PROJECT / "scripts/stage21/evaluate.py"),
                "existing_tables": [asset(R24 / n) for n in
                                    ["three_case_metrics.csv", "three_case_summary.csv", "PPT_ASSET_INDEX.md"]]}
    old_registry = read_json(R21 / "evaluation_registry.json")
    historical = csv_read(R24 / "three_case_metrics.csv")
    rows, classes_rows, checks, references = [], [], [], []
    protected = {}
    for i in range(1, 4):
        case = f"fullgeo_case{i:02}"
        root = R23 if i == 3 else R24 / "cases" / case
        execution = read_json(root / "EXECUTION.json")
        ev = read_json(root / "EVALUATION_INPUTS.json")
        registry = read_json(root / "inference_registry.json")
        protocol = read_json(root / "PROTOCOL.json")
        assert ev.get("case", case) == case and protocol["case"] == case
        original = old_registry["cases"][case]
        assert ev["truth"] == original["truth"] and ev["domains"] == original["domains"]
        truth = volume(ev["truth"])
        domains = {"D_global_nonair": (truth != -1, [int(k) for k in np.unique(truth[truth != -1])])}
        for name, d in ev["domains"].items():
            mask = volume(d["tensor"], labels=False).astype(bool)
            assert int(mask.sum()) == d["voxels"]
            domains[name] = mask, d["classes"]
        proposed_arm = "FLOW_STAGE23" if i == 3 else "FLOW_RESTORED_PSEUDO"
        selected = [next(s for s in execution["samples"] if s["arm"] == a)
                    for a in ["A_9", proposed_arm]]
        entry = {"truth": ev["truth"], "domains": ev["domains"],
                 "D_global_nonair": {"definition": "truth != -1", "classes": domains["D_global_nonair"][1]},
                 "evaluation_inputs": asset(root / "EVALUATION_INPUTS.json"),
                 "inference_registry": asset(root / "inference_registry.json"),
                 "protocol": asset(root / "PROTOCOL.json"), "predictions": {}}
        volumes = [truth]
        for method, s in zip(NAMES, selected):
            receipt_path = REPO / s["receipt"]["path"]
            assert file_hash(receipt_path) == s["receipt"]["file_sha256"]
            receipt = read_json(receipt_path)
            assert receipt["status"] == "completed" and receipt["prediction"] == s["prediction"]
            assert receipt.get("seed", receipt.get("source_seed")) == 42
            if method == NAMES[0]:
                assert receipt["case_id"] == case and receipt["arm"] == "A"
            else:
                assert receipt["arm"] == "FLOW_STAGE23"
                assert receipt["registry_sha256"] == file_hash(root / "inference_registry.json")
            pred = volume(s["prediction"])
            volumes.append(pred)
            entry["predictions"][method] = {"source_arm": s["arm"], "prediction": s["prediction"],
                                             "receipt": s["receipt"]}
            for domain, (mask, labels) in domains.items():
                m = voxel_metrics(pred, truth, mask, labels)
                m["target_pred_truth_volume_ratio"] = ratio(m["target_pred_volume"], m["target_truth_volume"])
                row = {"case": case, "seed": 42, "arm": s["arm"], "method": method, "domain": domain,
                       "miou_classes": json.dumps(labels),
                       "miou_defined_classes": json.dumps([k for k in labels if m[f"class_{k}_iou"] is not None]), **m}
                rows.append(row)
                for k in labels:
                    classes_rows.append({k2: row[k2] for k2 in ["case", "seed", "arm", "method", "domain"]} |
                                        {"raw_label": k, "iou": m[f"class_{k}_iou"],
                                         "truth_voxels": m[f"class_{k}_truth_voxels"]})
                if domain != "D_global_nonair":
                    old = next(r for r in historical if r["case"] == case and r["domain"] == domain and
                               r["arm"] == ("FLOW_RESTORED_PSEUDO" if method == NAMES[1] else "A_9"))
                    diffs = []
                    for key, val in m.items():
                        if key == "target_na_reason":
                            assert val == old[key]
                        elif val is None:
                            assert old.get(key, "") in ["", "NA", "None"]
                        else:
                            delta = abs(float(old[key]) - val)
                            assert delta <= 1e-12, (case, domain, key, delta)
                            diffs.append(delta)
                    checks.append({"case": case, "method": method, "domain": domain,
                                   "metrics_compared": len(m), "max_abs_difference": max(diffs)})
        for source in [root / "EXECUTION.json", root / "EVALUATION_INPUTS.json", root / "PROTOCOL.json",
                       root / "inference_registry.json", root / "sample_metrics.csv"]:
            protected[str(source)] = file_hash(source)
        for record in [ev["truth"], *[s["prediction"] for s in selected],
                       *[s["receipt"] for s in selected], *[d["tensor"] for d in ev["domains"].values()]]:
            protected[str(REPO / record["path"])] = record["file_sha256"]
        references.append({"case": case, "source": asset(root / "sample_metrics.csv"),
                           "retained_arms": [s["arm"] for s in execution["samples"] if s not in selected]})
        manifest["cases"][case] = entry
        for errors in [False, True]:
            slices(out, case.replace("fullgeo_", ""), volumes, errors)
    csv_write(out / "metrics_all_domains.csv", rows)
    csv_write(out / "class_iou.csv", classes_rows)
    for domain, name in [("D_global_nonair", "global_comparison"), ("E_holdout", "holdout_comparison")]:
        subset = [r for r in rows if r["domain"] == domain]
        csv_write(out / f"{name}.csv", [{k: r[k] for k in
                  ["case", "seed", "method", "domain", *METRICS, "domain_voxels", "miou_classes"]} for r in subset])
        table(out, subset, name)
    bars(out, rows)
    write_json(out / "reference_metrics_sources.json", {"sources": references,
               "note": "Unmodified historical tables retain B, C, COPY and direct classification. Direct full-field classification has a different information budget from sampled pseudo-well Flow."})
    for p in manifest["existing_tables"]:
        protected[str(REPO / p["path"])] = p["file_sha256"]
    assert all(file_hash(Path(p)) == h for p, h in protected.items())
    write_json(out / "asset_manifest.json", manifest)
    write_json(out / "verification.json", {"historical_comparisons": checks, "tolerance": 1e-12,
               "historical_input_files_unchanged": len(protected), "training": 0, "Flow_inference": 0,
               "property_restoration": 0, "device": "CPU", "metrics_rows": len(rows),
               "axis_contract": "XYZ; np.take(volume,32,axis=axis).T; origin=lower; nearest"})
    captions = ["# Figure captions", "",
                "Ground truth=真值；Frozen Flow=冻结Flow；Proposed method=改进方法。目标类为 raw9。",
                "三例均仅 seed42。Frozen Flow 是原9井与地表条件的 A_9；改进方法是恢复属性生成完整多岩性伪井后的冻结 Flow：case01/02 来自 Stage24，case03 来自 Stage23 v2。具体资产见 manifest。", ""]
    for i in range(1, 4):
        for style in ["clean", "with_errors"]:
            captions += [f"## case{i:02}_center_slices_{style}.png / .pdf", "",
                         "固定零基 X=32、Y=32、Z=32；横/纵轴依次为 Y/Z、X/Z、X/Y，坐标均为0–63体素索引。切片使用 np.take(volume,32,axis=axis).T、origin=lower、nearest。共享原参考图离散色表，空气−1白色，raw0…13固定映射。" +
                         (" 每个岩性行后是全截面 prediction != truth：包括空气，正确白色、错误红色；真值列全为正确。该图像域与非空气统计域不同。" if style == "with_errors" else " 三行分别为三个固定中心截面。"), ""]
    captions += ["## global_comparison.png / .pdf", "",
                 "D_global_nonair=truth!=-1；包含原条件与新增伪井位置，表示最终整体吻合，不能单独证明井外补全。OA/P/R为百分数，mIoU/IoU为0–1；仅较高且非平局的OA/mIoU/IoU/P/R加粗。体积比理想值1，不标为越大越好。", "",
                 "## holdout_comparison.png / .pdf", "",
                 "E_holdout沿用原注册mask、固定类别及空分母规则；无新增井条件直接写入，但属性参与全局代理，不是独立观测测试。数值格式与强调规则同上。", "",
                 "## quantitative_comparison.png / .pdf", "",
                 "上排全模型非空气域mIoU/目标IoU，下排E_holdout对应指标；统一0–1纵轴。每例seed42配对比较，无误差棒、置信区间或显著性检验。", ""]
    (out / "CAPTIONS.md").write_text("\n".join(captions))
    command = f"CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python {Path(__file__).relative_to(REPO)} --out {out.relative_to(REPO)}"
    files = sorted(p.name for p in out.iterdir()) + ["README.md"]
    (out / "README.md").write_text("# Three-case offline summary\n\n执行（服务器仓库根目录）：\n\n```bash\n" + command +
        "\n```\n\n输出目录不可覆盖；重跑请选择新版本目录。\n\n" +
        "D_global_nonair=truth!=-1；类别仅由各例真值非空气类确定，两方法相同。全模型含原条件及新增伪井，不能单独证明井外恢复。E_holdout/E_original_free直接读取原registry的mask/classes；后者包含新增伪井，前者虽无新增井直接写入，但属性参与全局代理，非独立观测测试。\n\n" +
        "IoU=TP/(TP+FP+FN)，OA=正确数/域体素数，mIoU=指定类别中有定义IoU的算术平均（严格复用原voxel_metrics，空并集跳过，未定义保留NA）；预测空气不剔除。miou_classes记录指定集合，miou_defined_classes记录实际参与均值集合。所有域指标与逐类IoU/真值数见CSV。\n\n" +
        f"12组历史域/方法重算均在1e-12容差内吻合Stage24原始CSV（逐类及目标计数一起检查），详见verification.json；{len(protected)}份输入文件哈希前后不变。18行主指标，6行global、6行holdout；9张图各PNG300dpi/PDF。\n\n" +
        "本次新增训练=0、Flow推理=0、属性恢复=0，仅CPU读既有标签与计算评价。历史结果、源码快照和认可3D图未覆盖。\n\n" +
        "限制：单seed、复用案例、人工可分属性代理、已知模糊核且无额外噪声，不构成泛化或真实物理反演证据。case01主要目标已有原井命中，小型零井目标仍遗漏；case02严重误报与对象合并仍在，体积偏大；case03也有误报和范围偏差。完整属性直接分类优于伪井Flow的历史数据保留于reference_metrics_sources.json索引，但其信息预算不同。\n\n文件：\n\n" +
        "\n".join(f"- {n}" for n in files) + "\n")
    print(json.dumps({"output": str(out), "rows": len(rows), "history_max_difference": max(c["max_abs_difference"] for c in checks)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    build(parser.parse_args().out.resolve())
