#!/usr/bin/env python3
"""Build the revised Chinese Stage15 conference manuscript as OOXML.

The source DOCX supplies the restrained A4 design system.  The scientific
content is rewritten around Phase1, Phase2 and Stage15-H and embeds exactly
two figures and one table.
"""

from __future__ import annotations

import html
import shutil
import struct
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PROJECT = ROOT / "project/geodata-3d-conditional"
SOURCE = ROOT / "面向三维地质地球物理联合推理的条件流匹配.docx"
REPORT = PROJECT / "experiments/stage15_binary_seismic_consensus/reports/conference_paper_v1"
OUTPUT = ROOT / "面向地质地球物理联合约束的三维条件流匹配推理_会议论文_最终版.docx"
FIG1 = REPORT / "figure1_progressive_guidance.png"
FIG2 = REPORT / "figure2_five_body_geophysical_guidance.png"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC = "http://schemas.openxmlformats.org/drawingml/2006/picture"


def esc(value: str) -> str:
    return html.escape(value, quote=False)


def run(text: str, *, bold: bool = False, italic: bool = False, size: int = 21, color: str = "000000", east_asia: str = "宋体", latin: str = "Times New Roman") -> str:
    props = [f'<w:rFonts w:ascii="{latin}" w:hAnsi="{latin}" w:eastAsia="{east_asia}"/>', f'<w:color w:val="{color}"/>', f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>']
    if bold:
        props.append("<w:b/><w:bCs/>")
    if italic:
        props.append("<w:i/><w:iCs/>")
    space = ' xml:space="preserve"' if text[:1].isspace() or text[-1:].isspace() else ""
    return f"<w:r><w:rPr>{''.join(props)}</w:rPr><w:t{space}>{esc(text)}</w:t></w:r>"


def paragraph(
    text: str = "",
    *,
    align: str = "both",
    first_line: int = 420,
    before: int = 0,
    after: int = 60,
    line: int = 310,
    size: int = 21,
    bold: bool = False,
    italic: bool = False,
    keep_next: bool = False,
    keep_lines: bool = False,
    page_break_before: bool = False,
    latin: str = "Times New Roman",
    east_asia: str = "宋体",
) -> str:
    keep = "<w:keepNext/>" if keep_next else ""
    lines = "<w:keepLines/>" if keep_lines else ""
    page = "<w:pageBreakBefore/>" if page_break_before else ""
    indent = f'<w:ind w:firstLine="{first_line}"/>' if first_line else ""
    ppr = f'<w:pPr>{keep}{lines}{page}<w:jc w:val="{align}"/>{indent}<w:spacing w:before="{before}" w:after="{after}" w:line="{line}" w:lineRule="auto"/></w:pPr>'
    return f"<w:p>{ppr}{run(text, bold=bold, italic=italic, size=size, latin=latin, east_asia=east_asia)}</w:p>"


def mixed_paragraph(parts: list[tuple[str, dict]], **kwargs) -> str:
    align = kwargs.get("align", "both")
    first_line = kwargs.get("first_line", 0)
    before = kwargs.get("before", 0)
    after = kwargs.get("after", 60)
    line = kwargs.get("line", 310)
    indent = f'<w:ind w:firstLine="{first_line}"/>' if first_line else ""
    ppr = f'<w:pPr><w:jc w:val="{align}"/>{indent}<w:spacing w:before="{before}" w:after="{after}" w:line="{line}" w:lineRule="auto"/></w:pPr>'
    return f"<w:p>{ppr}{''.join(run(text, **style) for text, style in parts)}</w:p>"


def heading(text: str, level: int = 1, *, page_break_before: bool = False) -> str:
    size = 24 if level == 1 else 22
    before = 220 if level == 1 else 120
    return paragraph(text, align="left", first_line=0, before=before, after=80, line=300, size=size, bold=True, keep_next=True, page_break_before=page_break_before, east_asia="黑体")


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        signature = stream.read(24)
    if signature[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")
    return struct.unpack(">II", signature[16:24])


def image_paragraph(path: Path, rel_id: str, drawing_id: int, width_inches: float = 5.72) -> str:
    px_w, px_h = png_size(path)
    cx = int(width_inches * 914400)
    cy = int(cx * px_h / px_w)
    name = path.name
    drawing = f'''<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">
      <wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>
      <wp:docPr id="{drawing_id}" name="{name}"/><wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>
      <a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
        <pic:pic><pic:nvPicPr><pic:cNvPr id="0" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr>
        <pic:blipFill><a:blip r:embed="{rel_id}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
        <pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>
      </a:graphicData></a:graphic>
    </wp:inline></w:drawing></w:r>'''
    return f'<w:p><w:pPr><w:keepNext/><w:jc w:val="center"/><w:spacing w:before="60" w:after="40"/></w:pPr>{drawing}</w:p>'


def caption(text: str, *, keep_next: bool = False) -> str:
    return paragraph(text, align="center", first_line=0, before=20, after=120, line=280, size=19, keep_next=keep_next, keep_lines=True)


def page_break() -> str:
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def document_xml() -> str:
    parts: list[str] = []
    parts.append(paragraph("面向地质—地球物理联合约束的三维条件流匹配推理", align="center", first_line=0, before=60, after=100, line=360, size=32, bold=True, east_asia="黑体"))
    parts.append(paragraph("3D Conditional Flow-Matching Inference with Geological and Geophysical Constraints", align="center", first_line=0, before=0, after=220, line=300, size=23, bold=True))
    parts.append(heading("摘  要", 1))
    parts.append(paragraph("地质先验与地球物理观测在联合推理中承担不同作用：前者限定结构可行域，后者在其中更新地下目标的位置与几何。本文以类别9异常体为对象，在冻结条件流匹配模型上构造由真值概率体、理想物性体至二值地震反演分数的递进证据链。Phase1检验生成模型对三维空间提示的响应，Phase2建立物性约束与类别结构之间的接口，Stage15将无噪声二值褶积地震转化为连续异常分数并用于推理期更新。五岩体对照中，三体由钻孔揭露、两体完全隐藏；地震制导使隐藏体召回率由0.138提高至0.902，但生成结果仍沿先验偏好的连续形态扩展。实验表明，地球物理空间证据能够补充先验遗漏的目标位置；边界和拓扑的校正则受证据表征与生成先验之间匹配程度的制约。"))
    parts.append(mixed_paragraph([("关键词：", {"bold": True, "size": 21, "east_asia": "黑体"}), ("条件流匹配；三维地质建模；地震反演；空间证据；生成先验", {"size": 21})], after=120))

    parts.append(heading("1  引言", 1))
    parts.append(paragraph("三维地质建模在有限观测条件下具有显著多解性。地表资料和钻孔对局部地层的约束准确，却很难确定未揭露地质体的地下展布。生成模型可由地层序列、接触关系和界面几何学习结构先验，为多解问题提供可行模型空间[1-3]；地震观测则覆盖地下区域，可通过数据响应更新异常体的位置和边界[4-5]。两类信息并非彼此替代：先验用于排除结构上不合理的解，地球物理观测用于在先验允许的模型之间进行区分。"))
    parts.append(paragraph("实现这一联合关系的难点，在于地震记录和反演物性并不等同于岩性类别。若将不可靠的物性异常直接解释为类别概率，生成结果可能被推向错误位置；若完全依赖生成先验，钻孔以外的异常体又容易遗漏。本文从推理期接口入手，按“空间可控性—物性传递—地震证据”组织实验，并以五个分离目标检验位置更新与拓扑保持。由此考察两个相互衔接的问题：地球物理证据能否改善先验对地下目标的恢复，以及这种更新在多大程度上受生成先验和证据分辨率制约。"))

    parts.append(heading("2  方法", 1))
    parts.append(heading("2.1  冻结条件流匹配及推理期制导", 2))
    parts.append(paragraph("研究对象为64×64×64离散地质体。条件流匹配学习由高斯噪声到类别嵌入的速度场vθ(xt,t)，其训练分布构成本文采用的地质可行域。推理时冻结网络及指数滑动平均权重，在ODE积分中叠加外部证据损失L的状态梯度：dx/dt=vθ(xt,t)−α(t)clip(∇xL)。式中，xt为时刻t的生成状态，vθ为冻结速度场，α(t)为制导强度，clip用于限制梯度幅值。地表和钻孔类别在每个积分步后重新投影；配对实验采用相同初始噪声、步数和求解器。"))
    parts.append(paragraph("概率制导采用类别9的交叉熵 Lp=−Σx c(x)[p9(x)log p̂9(x)+(1−p9(x))log(1−p̂9(x))]。其中p9(x)为外部概率体，p̂9(x)为当前生成状态的类别9概率，c(x)为地下有效区域掩膜。物性制导先计算期望物性 ŷ(x)=Σk p̂k(x)yk，再采用 Lm=Σs ws||c⊙(Gs*ŷ−Gs*y)||²2；p̂k和yk分别表示第k类的软概率和物性值，Gs为尺度s的高斯算子，ws为尺度权重，y为目标物性体。两种接口共享冻结Flow和硬条件投影。"))

    parts.append(heading("2.2  三维空间证据的递进构造", 2))
    parts.append(paragraph("证据构造分为三个层次。Phase1由类别9真值构造三维概率体p9(x)，给出空间提示作用于冻结Flow的上限；Phase2将同一目标转写为理想密度—磁化率体，以物性残差替代类别概率残差；Stage15-H再从地震观测获得空间证据。前两阶段采用12组严格配对样本。Stage15-H保留原始15类地质模型，仅在声学映射时将类别9设为高阻抗端元、其余地下类别统一为背景端元，空气保持独立。这样，实验逐步减少证据中直接可用的类别信息，同时保持目标和生成先验不变。"))
    parts.append(paragraph("地震正演采用法向入射叠后褶积模型。相邻界面反射系数为ri=(Zi+1−Zi)/(Zi+1+Zi)，Zi表示第i层声阻抗；反射序列经背景慢度时深映射后，与25 Hz零相位Ricker子波卷积。逐道反演求解线性化对数阻抗增量 δm=(GTG+λpI+λsDTD)−1GTr，其中r为地震残差，G=W(1/2D)为子波矩阵W与垂向差分算子D构成的线性算子，λp和λs分别控制阻尼和平滑。两次更新后，将阻抗Z投影到背景端元Zb与类别9端元Z9之间，得到连续异常分数q(x)=clip[(ln Z−ln Zb)/(ln Z9−ln Zb),0,1]，并以q加权类别9物性制导。"))

    parts.append(heading("2.3  五岩体先验—证据对照", 2))
    parts.append(paragraph("五岩体实验采用与当前checkpoint一致的背景，嵌入五个等体积、互不相交的类别9立方体，每体包含640个体素。九口钻井保持原有布局，其中三口分别穿过三个目标体，另两个目标体与全部硬条件无交集。Flow-only表示地质先验与钻井共同限定的结果，地震制导在此基础上加入二值地震空间证据。两组共享初始噪声、32步积分及seed 42、142、242。评价包括总体与隐藏体的IoU、precision、recall、预测体积及目标间连通关系。"))

    parts.append(heading("3  实验结果", 1))
    parts.append(heading("3.1  空间证据对异常体恢复的作用", 2))
    parts.append(paragraph("Phase1、Phase2与Stage15-H构成从直接空间提示到观测派生证据的递进关系。Phase1中，类别9交并比由0.031提高至0.810，说明冻结Flow能够在既有结构可行域内响应三维目标位置。将提示改写为理想物性约束后，Phase2达到0.481，表明类别—物性接口能够传递主要空间信息。Stage15-H使用地震反演分数，在3个固定种子上将交并比由0.038提高至0.267，平均召回率由0.056提高至0.463，质心距离由18.40降至9.05个体素；全部结果均精确满足地表和钻井条件。"))
    parts.append(image_paragraph(FIG1, "rId4", 1, width_inches=5.72))
    parts.append(caption("图1  三维空间证据的递进制导结果。第一行为同一cond_generation_0及seed42下的完整地质模型：(a)真值；(b)Flow-only；(c)Phase1概率制导；(d)Phase2理想物性制导；(e)二值地震反演证据制导。第二行为统一观测与类别9对照：(f)地表—钻井条件；(g)—(j)分别对应(b)—(e)的类别9结果。橙色为生成的类别9，浅灰蓝阴影为真实类别9，青色竖线为钻孔。"))
    parts.append(paragraph("图1显示，Flow-only给出了符合先验与稀疏条件的地质结构，但遗漏了钻孔未充分揭露的主要侵入体。概率证据恢复最完整，理想物性证据保留主要走向而出现局部缺口；由地震获得的异常分数仍能把生成结果推向真实目标区域，只是边界明显增厚。随着证据由类别概率转为物性和地震反演分数，异常体恢复精度逐级降低，但空间更新作用仍然保留。"))

    parts.append(heading("3.2  隐藏目标恢复与拓扑约束", 2))
    parts.append(image_paragraph(FIG2, "rId5", 2, width_inches=5.72))
    parts.append(caption("图2  五岩体案例的seed142结果。(a)五个相互分离的类别9真值；(b)Flow-only；(c)二值地震空间证据制导。洋红色粗线为穿过三个目标体的钻井，青色细线为其余钻井；橙色为类别9，灰色为真值参考。"))
    parts.append(paragraph("Flow-only优先响应三口命中类别9的钻井，两个完全隐藏目标的总体召回率中位数仅为0.138，且生成体在目标之间形成连续结构。加入地震证据后，隐藏体召回率提高到0.902，总体召回率达到0.947。地球物理观测由此补足了钻孔约束在地下空间覆盖上的不足。不过，预测类别9体素达到18 675，而真值为3 200；五个目标的10种两两组合均落入同一连通系统，总体IoU仅由0.115提高到0.161。"))
    parts.append(paragraph("地震异常分数提供了目标所在区域，却没有充分给出背景排除和界面几何，生成轨迹因而保留了先验偏好的连续形态。地震信息已能更新目标在哪里，尚不足以稳定更新目标到哪里结束，以及目标之间是否连通。"))

    parts.append(heading("4  结论", 1))
    parts.append(paragraph("实验得到两点认识。其一，冻结条件流匹配能够接收与目标位置一致的空间证据；地震反演形成的异常分数进入推理后，可补充先验和钻孔遗漏的类别9体，并改善召回率与空间定位。这验证了联合推理中由地球物理观测更新地下目标位置的作用。"))
    parts.append(paragraph("其二，地球物理证据对先验误差的纠正具有选择性。五岩体实验中，地震制导恢复了两个隐藏目标，却未能消除目标过度连接。当地震异常分数缺少可靠的背景排除和界面信息时，生成先验仍主导形态补全。因此，当前方法能够纠正位置遗漏，尚难稳定校正错误连接和边界扩张。地质先验限定结构可行性、地球物理证据更新目标位置，构成了推理期的单向联合约束；要达到双向联合推理，还需使地质先验参与反演模型空间限定，并以包含正异常、负背景和界面几何的地球物理后验更新地质模型。"))

    parts.append(heading("参考文献", 1))
    refs = [
        "[1] Song S, Mukerji T, Hou J, et al. GANSIM-3D for conditional geomodeling: theory and field application[J]. Water Resources Research, 2022, 58: e2021WR031865.",
        "[2] Mosser L, Dubrule O, Blunt M J. Stochastic seismic waveform inversion using generative adversarial networks as a geological prior[J]. Mathematical Geosciences, 2020, 52: 53-79.",
        "[3] Lipman Y, Chen R T Q, Ben-Hamu H, et al. Flow matching for generative modeling[C]//International Conference on Learning Representations. 2023.",
        "[4] Tarantola A. Inverse Problem Theory and Methods for Model Parameter Estimation[M]. Philadelphia: SIAM, 2005.",
        "[5] Simm R, Bacon M. Seismic Amplitude: An Interpreter's Handbook[M]. Cambridge: Cambridge University Press, 2014.",
        "[6] Ghyselincks S, Okhmak V, Zampini S, et al. Synthetic Geology: structural geology meets deep learning[J]. Journal of Geophysical Research: Machine Learning and Computation, 2026, 3: e2025JH000986.",
    ]
    for reference in refs:
        parts.append(paragraph(reference, align="both", first_line=0, before=0, after=20, line=225, size=16))

    body = "".join(parts)
    sect = '''<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:right="1800" w:bottom="1440" w:left="1800" w:header="851" w:footer="992" w:gutter="0"/><w:cols w:space="720"/><w:docGrid w:linePitch="312"/></w:sectPr>'''
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{W}" xmlns:r="{R}" xmlns:wp="{WP}" xmlns:a="{A}" xmlns:pic="{PIC}"><w:body>{body}{sect}</w:body></w:document>'''


def relationships_xml() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>
<Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image1.png"/>
<Relationship Id="rId5" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image2.png"/>
<Relationship Id="rId7" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/fontTable" Target="fontTable.xml"/>
</Relationships>'''


def build() -> None:
    for path in (SOURCE, FIG1, FIG2):
        if not path.is_file():
            raise FileNotFoundError(path)
    temp = OUTPUT.with_suffix(".tmp.docx")
    with zipfile.ZipFile(SOURCE, "r") as source, zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as target:
        skip = {"word/document.xml", "word/_rels/document.xml.rels", "word/media/image1.png", "word/media/image2.png", "word/media/image3.png"}
        for item in source.infolist():
            if item.filename not in skip and not item.filename.endswith("/"):
                target.writestr(item, source.read(item.filename))
        target.writestr("word/document.xml", document_xml())
        target.writestr("word/_rels/document.xml.rels", relationships_xml())
        target.write(FIG1, "word/media/image1.png")
        target.write(FIG2, "word/media/image2.png")
    shutil.move(temp, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build()
