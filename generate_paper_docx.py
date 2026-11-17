# -*- coding: utf-8 -*-
"""生成西安理工学术年会论文 Word 文档（含封皮 + 正文）。

原生公式由 docx_math 构建为 OMML，章节结构：1 引言 / 2 研究现状 / 3 主要贡献 /
4 方案创新点 / 5 模型框架 / 6 实验方案 / 7 结论。
"""
import docx
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

from docx_math import (
    mvar as V, mop as OP, mnum as NUM,
    sub as S, sup as P, subsup as SS,
    frac as F, d as D, nary as SIGMA, func as FN, rad as SQ,
    bar as BAR, acc as ACC, limlow as LIM, omath,
)

HAT = "\u0302"    # ̂ 组合抑扬符 COMBINING CIRCUMFLEX ACCENT（OMML m:acc 规范要求组合字符）
TILDE = "\u0303"  # ̃ 组合波浪符 COMBINING TILDE


def set_run(run, cn_font="宋体", size=12, bold=False, color=None, italic=False):
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), cn_font)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = RGBColor(*color)


def three_line_table(table, top=12, mid=6, bottom=12):
    """将表格设为三线表：仅顶线、表头下线、底线，无竖线。sz 单位为 1/8 pt。"""
    from docx.oxml import OxmlElement

    def _edge(cell, name, val, sz):
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        tcBorders = tcPr.find(qn("w:tcBorders"))
        if tcBorders is None:
            tcBorders = OxmlElement("w:tcBorders")
            tcPr.append(tcBorders)
        for old in tcBorders.findall(qn("w:" + name)):
            tcBorders.remove(old)
        el = OxmlElement("w:" + name)
        el.set(qn("w:val"), val)
        el.set(qn("w:sz"), str(sz))
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")
        tcBorders.append(el)

    rows = table.rows
    nrows = len(rows)
    for r_i, row in enumerate(rows):
        for cell in row.cells:
            _edge(cell, "left", "nil", 0)
            _edge(cell, "right", "nil", 0)
            if r_i == 0:
                _edge(cell, "top", "single", top)
                _edge(cell, "bottom", "single", mid)
            elif r_i == nrows - 1:
                _edge(cell, "top", "nil", 0)
                _edge(cell, "bottom", "single", bottom)
            else:
                _edge(cell, "top", "nil", 0)
                _edge(cell, "bottom", "nil", 0)


def add_para(doc, text="", cn="宋体", size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
             indent=True, space_after=4, line_spacing=1.5):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing
    if indent:
        p.paragraph_format.first_line_indent = Pt(size * 2)
    if text:
        r = p.add_run(text)
        set_run(r, cn_font=cn, size=size, bold=bold)
    return p


def para(doc, segments, cn="宋体", size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
         indent=True, space_after=4, line_spacing=1.5):
    """支持文本 + 行内原生公式混排的段落。segments 为 str 或 oMath 元素的列表。"""
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing
    if indent:
        p.paragraph_format.first_line_indent = Pt(size * 2)
    if isinstance(segments, str):
        segments = [segments]
    for seg in segments:
        if isinstance(seg, str):
            r = p.add_run(seg)
            set_run(r, cn_font=cn, size=size, bold=bold)
        else:
            p._p.append(seg)
    return p


def add_heading(doc, text, level=1):
    """章节标题：黑体。level 1=章(三号加粗) 2=节(四号加粗) 3=条(小四加粗)"""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.5
    if level == 1:
        size, cn = 15, "黑体"
    elif level == 2:
        size, cn = 13, "黑体"
    else:
        size, cn = 12, "黑体"
    r = p.add_run(text)
    set_run(r, cn_font=cn, size=size, bold=True)
    return p


def add_eq_math(doc, elements, tag):
    """插入居中显示的原生公式（编号在行尾）。"""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.5
    p._p.append(omath(elements))
    r = p.add_run("    " + tag)
    set_run(r, cn_font="宋体", size=11)
    return p


def cell_add(cell, segments, cn="宋体", size=10.5, bold=False,
             align=WD_ALIGN_PARAGRAPH.LEFT):
    p = cell.paragraphs[0]
    p.alignment = align
    if isinstance(segments, str):
        segments = [segments]
    for seg in segments:
        if isinstance(seg, str):
            r = p.add_run(seg)
            set_run(r, cn_font=cn, size=size, bold=bold)
        else:
            p._p.append(seg)


# ============ 公式定义 ============
# (1) z_0 = [t_a, t_n, t_c, p_1, …, p_N]
eq1 = [S(V("z"), NUM("0")), OP(" = "),
       D("[", "]",
         S(V("t"), V("a")), OP(", "),
         S(V("t"), V("n")), OP(", "),
         S(V("t"), V("c")), OP(", "),
         S(V("p"), NUM("1")), OP(", … , "),
         S(V("p"), V("N")))]

# (2) P_ℓ^pos = P_ℓ + E_pos^(ℓ)
eq2 = [SS(V("P"), V("ℓ"), OP("pos")), OP(" = "),
       S(V("P"), V("ℓ")), OP(" + "),
       SS(V("E"), OP("pos"), D("(", ")", V("ℓ")))]

# (3) A_ℓ = softmax( Q_anchor (P_ℓ^pos)^⊤ / √d ), U_ℓ = A_ℓ P_ℓ
eq3 = [S(V("A"), V("ℓ")), OP(" = "),
       FN(OP("softmax"), D("(", ")",
           F([S(V("Q"), OP("anchor")),
              P(D("(", ")", SS(V("P"), V("ℓ"), OP("pos"))), OP("⊤"))],
             SQ(V("d"))))),
       OP(",    "),
       S(V("U"), V("ℓ")), OP(" = "),
       S(V("A"), V("ℓ")), S(V("P"), V("ℓ"))]

# (4) t̃_ℓ = t + α Σ_{i=1}^{m} g_i(t) a_i
eq4 = [S(ACC(TILDE, V("t")), V("ℓ")), OP(" = "), V("t"), OP(" + "), V("α"), OP(" "),
       SIGMA("∑", [V("i"), OP(" = "), NUM("1")], V("m"),
             [S(V("g"), V("i")), D("(", ")", V("t")), OP(" "), S(V("a"), V("i"))])]

# (5) P̂_ℓ = F_ℓ(P_ℓ)
eq5 = [S(ACC(HAT, V("P")), V("ℓ")), OP(" = "),
       S(V("F"), V("ℓ")), D("(", ")", S(V("P"), V("ℓ")))]

# (6) s_i^(ℓ) = ⟨ p̂_i^(ℓ), t̄_a^(ℓ) ⟩ − ⟨ p̂_i^(ℓ), t̄_n^(ℓ) ⟩
# 注意：同一 lxml 元素被多处 append 会被“移动”，故 p̂_i^(ℓ) 需构建两个独立副本
_pi_l = P(ACC(HAT, S(V("p"), V("i"))), D("(", ")", V("ℓ")))
_pi_l2 = P(ACC(HAT, S(V("p"), V("i"))), D("(", ")", V("ℓ")))
_ta_l = P(BAR(S(V("t"), V("a"))), D("(", ")", V("ℓ")))
_tn_l = P(BAR(S(V("t"), V("n"))), D("(", ")", V("ℓ")))
eq6 = [SS(V("s"), V("i"), D("(", ")", V("ℓ"))), OP(" = "),
       D("⟨", "⟩", _pi_l, OP(", "), _ta_l), OP(" − "),
       D("⟨", "⟩", _pi_l2, OP(", "), _tn_l)]

# (7) Ĥ = H + α_sr · ( conv_{3×3}(H) − H )
_conv33 = S(OP("conv"), OP("3×3"))
eq7 = [ACC(HAT, V("H")), OP(" = "), V("H"), OP(" + "),
       S(V("α"), OP("sr")), OP(" · "),
       D("(", ")", _conv33, D("(", ")", V("H")), OP(" − "), V("H"))]

# (8) L_sep = max( 0, ⟨ t̄_a^(L), t̄_n^(L) ⟩ + τ )², τ = 0.5
_taL = P(BAR(S(V("t"), V("a"))), D("(", ")", V("L")))
_tnL = P(BAR(S(V("t"), V("n"))), D("(", ")", V("L")))
eq8 = [S(V("L"), OP("sep")), OP(" = "),
       P(LIM(OP("max"), D("(", ")",
             NUM("0"), OP(", "),
             D("⟨", "⟩", _taL, OP(", "), _tnL), OP(" + "), V("τ"))),
         NUM("2")),
       OP(",    "),
       V("τ"), OP(" = "), NUM("0.5")]

# (9) L_cls = −[ y·log σ(S) + (1−y)·log(1−σ(S)) ]
_log_sigmaS = FN(OP("log"), [V("σ"), D("(", ")", V("S"))])
_log_1msigma = FN(OP("log"), [NUM("1"), OP(" − "), V("σ"), D("(", ")", V("S"))])
eq9 = [S(V("L"), OP("cls")), OP(" = "), OP("−"),
       D("[", "]",
          V("y"), OP("·"), _log_sigmaS, OP(" + "),
          D("(", ")", [NUM("1"), OP(" − "), V("y")]), OP("·"), _log_1msigma)]

# (10) s̃_i = α s_i + (1−α) Σ_{j∈N_k(i)} w_ij s_j, α = 0.3
eq10 = [S(ACC(TILDE, V("s")), V("i")), OP(" = "), V("α"), OP(" "), S(V("s"), V("i")), OP(" + "),
        D("(", ")", [NUM("1"), OP(" − "), V("α")]), OP(" "),
        SIGMA("∑", [V("j"), OP(" ∈ "), S(V("N"), V("k")), D("(", ")", V("i"))],
              None,
              [S(V("w"), [V("i"), V("j")]), OP(" "), S(V("s"), V("j"))]),
        OP(",    "),
        V("α"), OP(" = "), NUM("0.3")]

# (11) c_ℓ = 1/(1 + Var_local(H_ℓ)), w_ℓ = softmax(c_ℓ / T)
_Varlocal = S(OP("Var"), OP("local"))
eq11 = [S(V("c"), V("ℓ")), OP(" = "),
        F(NUM("1"), [NUM("1"), OP(" + "), _Varlocal, D("(", ")", S(V("H"), V("ℓ")))]),
        OP(",    "),
        S(V("w"), V("ℓ")), OP(" = "),
        FN(OP("softmax"), F(S(V("c"), V("ℓ")), V("T")))]

# (12) s_mutual = 1/m Σ_{i=1}^m ( 1 − 1/k Σ_{j∈NN_k(q_i)} sim(q_i, f_j) )
_inner12 = SIGMA("∑", [V("j"), OP(" ∈ "), S(OP("NN"), V("k")), D("(", ")", S(V("q"), V("i")))],
                 None,
                 FN(OP("sim"), D("(", ")", [S(V("q"), V("i")), OP(", "), S(V("f"), V("j"))])))
eq12 = [S(V("s"), OP("mutual")), OP(" = "),
        F(NUM("1"), V("m")),
        SIGMA("∑", [V("i"), OP(" = "), NUM("1")], V("m"),
              D("(", ")", [NUM("1"), OP(" − "), F(NUM("1"), V("k")), _inner12]))]

# (13) C_refined = 1/(K+1) ( C + Σ_{k∈K} D_k^{−1} (M_k ⊙ W) C )
eq13 = [S(V("C"), OP("refined")), OP(" = "),
        F(NUM("1"), [V("K"), OP(" + "), NUM("1")]),
        D("(", ")",
          V("C"), OP(" + "),
          SIGMA("∑", [V("k"), OP(" ∈ "), V("K")], None,
                [P(S(V("D"), V("k")), OP("−1")),
                 D("(", ")", [S(V("M"), V("k")), OP(" ⊙ "), V("W")]),
                 V("C")]))]

# (14) s_fused = w·z(s_topk) + (1−w)·z(s_mutual)
eq14 = [S(V("s"), OP("fused")), OP(" = "), V("w"), OP("·"), V("z"), D("(", ")", S(V("s"), OP("topk"))),
        OP(" + "), D("(", ")", [NUM("1"), OP(" − "), V("w")]), OP("·"),
        V("z"), D("(", ")", S(V("s"), OP("mutual")))]


# ============ 正文 ============
doc = Document()
for sec in doc.sections:
    sec.top_margin = Cm(2.54)
    sec.bottom_margin = Cm(2.54)
    sec.left_margin = Cm(3.17)
    sec.right_margin = Cm(3.17)


# ---------- 封皮页 ----------
def blank(n=1, size=18):
    for _ in range(n):
        add_para(doc, "", indent=False, space_after=0)

blank(2, 24)
add_para(doc, "西安理工大学", cn="宋体", size=26, bold=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, space_after=14)
add_para(doc, "硕士研究生学术年会论文", cn="宋体", size=22, bold=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, space_after=36)
blank(1, 22)

fields = [
    ("题目", "＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿"),
    ("学      号", "＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿"),
    ("姓      名", "＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿"),
    ("学      科", "＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿"),
    ("导      师", "＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿"),
    ("日      期", "＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿"),
]
for label, line in fields:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(16)
    p.paragraph_format.line_spacing = 1.5
    r1 = p.add_run(label + "：")
    set_run(r1, cn_font="宋体", size=15, bold=True)
    r2 = p.add_run(line)
    set_run(r2, cn_font="宋体", size=15)

blank(1, 20)
add_para(doc, "指导教师签字：＿＿＿＿＿＿＿＿＿＿＿＿", cn="宋体", size=15, bold=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, space_after=36)
blank(1, 20)
add_para(doc, "西安理工大学研究生院", cn="宋体", size=16, bold=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, space_after=8)
add_para(doc, "2026年7月", cn="宋体", size=14, bold=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, space_after=0)

doc.add_page_break()

# ---------- 标题 ----------
add_para(doc, "解耦像素定位与图像分类的纯视觉零样本异常检测方法研究",
         cn="黑体", size=16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, space_after=8)
add_para(doc, "VisualAD-DINOv3: A Purely Visual Zero-Shot Anomaly Detection Method via Decoupled Pixel Localization and Image Classification",
         cn="Times New Roman", size=13, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, space_after=12)

# ---------- 摘要 ----------
add_para(doc, "摘  要", cn="黑体", size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, space_after=6)
add_para(doc,
    "零样本异常检测（Zero-Shot Anomaly Detection, ZSAD）要求模型在不接触目标类别异常样本的前提下，"
    "同时完成对未见类别的图像级异常判别与像素级异常定位，是工业质检与医学影像筛查走向“品类无关、快速冷启动”的关键共性技术。"
    "主流的零样本方法普遍依赖视觉—语言模型（VLM）中的文本分支，通过手工或可学习提示词刻画“正常/异常”语义并与图像特征对齐；"
    "这类范式虽然有效，但文本编码器与跨模态对齐带来了训练不稳定与参数冗余。针对上述问题，本文在纯视觉框架 VisualAD 的基础上提出 VisualAD-DINOv3，"
    "核心思路是将“像素级定位”与“图像级分类”两条任务解耦，分别交由两个完全冻结的大模型骨干，只在最轻量的模块上做训练。"
    "训练侧（仅用 VisA，共 11 轮）采用冻结的 CLIP ViT-L/14：前置两个随机初始化的可学习令牌（anomaly/normal token），经空间交叉注意力（SCA）"
    "与特征自对齐函数（SAF）多层增强后生成逐层异常热力图，再经可学习的空间细化层精炼，并通过“先令牌分离、后检测头微调”的双阶段课程式训练优化；"
    "推理侧（跨域迁移、完全无训练）引入第二个冻结骨干 DINOv3 ViT-L/16，仅用于三件后处理——patch 图传播去噪、空间一致性加权融合与 patch 稀缺度互评分，"
    "并将图像级分数经 CLS 近邻重评分（RsCIN）在纯互打分口径下输出，从而让像素图完全来自 CLIP、图像级分数完全来自 DINOv3 patch 稀缺度，两路互不干扰。"
    "该设计使跨域时像素定位基本稳定，而原先因域偏移崩溃的图像级分数（MVTec→VisA 仅 60.6%）被修复到 87.7%。"
    "所设计的框架不依赖任何文本分支，可直接泛化到未见类别与跨域数据集。")
add_para(doc, "关键词：零样本异常检测；视觉 Transformer；双令牌；DINOv3；空间交叉注意力；两阶段训练",
         cn="黑体", size=12, bold=False, indent=False, space_after=12)

# ---------- Abstract ----------
add_para(doc, "Abstract", cn="Times New Roman", size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, space_after=6)
add_para(doc,
    "Zero-shot anomaly detection (ZSAD) requires both image-level discrimination and pixel-level localization of anomalies in "
    "unseen categories without access to target-class anomaly samples. Mainstream approaches rely on the text branch of vision-language "
    "models (VLMs) to express normal/abnormal semantics and align them with visual features, which introduces training instability and "
    "parameter redundancy. Building upon the purely visual framework VisualAD, this paper proposes VisualAD-DINOv3, whose core idea is to "
    "decouple pixel-level localization from image-level classification and delegate the two tasks to two fully frozen large-model backbones, "
    "training only the lightest modules. On the training side (VisA only, 11 epochs in total), a frozen CLIP ViT-L/14 is used: two randomly "
    "initialized learnable tokens (anomaly/normal) are prepended, enhanced through multi-layer spatial cross-attention (SCA) and a "
    "self-alignment function (SAF) to produce per-layer anomaly heatmaps, refined by a learnable spatial refinement layer, and optimized "
    "under a two-stage curriculum that first separates the token directions and then fine-tunes the detection heads. On the inference side "
    "(cross-domain transfer, entirely training-free), a second frozen backbone DINOv3 ViT-L/16 is introduced for only three post-processing "
    "steps—patch-graph propagation denoising, spatial-consistency-weighted fusion, and patch-rarity mutual scoring—after which the image-level "
    "score is refined by CLS-neighborhood re-scoring (RsCIN) under the pure-mutual criterion. Hence the pixel map comes entirely from CLIP and "
    "the image-level score entirely from DINOv3 patch rarity, with the two paths decoupled. This design keeps pixel localization stable across "
    "domains and restores the image-level score from 60.6% to 87.7% on MVTec-to-VisA. The framework requires no text branch and generalizes "
    "directly to unseen categories and cross-domain datasets.", cn="Times New Roman", size=12)
add_para(doc, "Keywords: zero-shot anomaly detection; vision transformer; dual tokens; DINOv3; spatial cross-attention; two-stage training",
         cn="Times New Roman", size=12, bold=False, indent=False, space_after=12)

# ---------- 1 引言 ----------
add_heading(doc, "1  引言", 1)
add_heading(doc, "1.1  研究背景与意义", 2)
add_para(doc,
    "图像异常检测旨在识别出与预期正常模式发生偏离的图像与像素区域，在工业表面缺陷检测、医学影像病灶筛查、半导体质检等安全攸关场景中占据核心地位。"
    "传统方法多遵循无监督或半监督范式，即假设训练阶段能够获得足量、干净的类别专属正常样本[1][2]。然而在真实生产环境中，“冷启动”问题是普遍且棘手的："
    "为每一条新产品线或每一种疾病类别单独采集并清洗正常数据，成本高昂且常常不切实际；当缺陷形态不可枚举、类别长尾分布明显时，这类方法难以规模化扩展，"
    "并且在精心整理的数据与噪声纷杂的真实环境之间存在明显的域迁移脆弱性。")
add_para(doc,
    "零样本异常检测（ZSAD）正是为突破上述瓶颈而提出的新设定[3][4][5]：模型仅在辅助类别上接受训练，并直接对训练阶段从未见过的目标类别进行异常判别与定位，"
    "从而将异常检测能力与“逐类数据采集”这一负担解耦。这一能力高度契合工业质检与医疗筛查“品类快速更迭、缺陷样本稀缺”的现实需求，具有重要的应用价值与研究价值。")
add_para(doc,
    "近年来，以 CLIP 为代表的大规模预训练视觉—语言模型展现出优异的跨模态泛化能力，催生了一批基于提示词学习的零样本异常检测方法[6][7][8]。"
    "这类方法的基本思路是：将“正常”与“异常”两种语义分别编码为手工模板或可学习提示向量，再通过文本编码器与图像特征进行对齐打分，从而实现开集判别。"
    "尽管有效，该范式在实现上始终绑定着一个文本分支：文本编码器与跨模态对齐在前向与优化过程中均不可省略。")
add_para(doc,
    "然而，一个值得追问的问题是——如果最终的判别决策仅仅由“正常”与“异常”两组参考特征所主导，那么语言模态是否真的不可或缺？从纯视觉的角度看，"
    "异常本质上表现为纹理、形状、颜色层面的结构或统计偏差，这些信息完全可以被视觉域所捕捉。前期工作[9]通过一个探索性实验回答了这一问题："
    "移除 AnomalyCLIP 的文本编码器、直接在视觉空间中优化两个可学习向量以表征正常与异常，检测性能几乎无损失，而可训练参数量下降了 99% 以上。"
    "这说明，在基于 CLIP 的零样本异常检测流程中，文本提示很可能只是一种构造判别性视觉原型的间接途径，而非提供本质性语义锚定的必要手段。"
    "这一观察直接催生了纯视觉框架 VisualAD：在冻结的视觉 Transformer 中插入两个可学习令牌，通过多层自注意力与 patch 令牌交互，逐步习得“正常”与“异常”的高层概念，"
    "并借助空间交叉注意力（SCA）与特征自对齐函数（SAF）实现多层对齐与定位[9]。")
add_para(doc,
    "本文沿着“去文本、纯视觉、任务解耦”这一主线继续深入。虽然 VisualAD 已经充分证明了纯视觉双令牌机制的有效性，但它在三个层面仍存在可改进空间："
    "（1）训练范式层面，单阶段联合优化将令牌学习、检测头优化与空间对齐耦合在一起，缺失结构化的学习顺序；（2）空间建模层面，逐层融合采用简单求和，"
    "且输出异常图缺乏显式的局部空间约束，对纹理噪声与孤立高分点鲁棒性不足；（3）任务耦合层面，图像级分数与像素级分数同源于 CLIP 主干，"
    "跨域时图像级分数随域偏移而显著退化（MVTec→VisA 仅 60.6%），缺乏对图像级判别任务的独立建模。"
    "针对上述问题，本文在 VisualAD 的基础上提出 VisualAD-DINOv3：将像素级定位与图像级分类解耦，分别交由冻结的 CLIP 与冻结的 DINOv3 两个骨干承担，"
    "仅在轻量模块上训练，重构为双阶段课程式训练，并构建以空间细化、一致性融合、互评分与近邻重评分为核心的多级校准机制。"
    "实验上，本文沿用与 VisualAD 一致的零样本协议与评测体系，以验证各创新模块的有效性。")

# ---------- 2 研究现状 ----------
add_heading(doc, "2  研究现状", 1)
add_heading(doc, "2.1  无监督异常检测", 2)
add_para(doc,
    "无监督异常检测仅使用正常样本训练，现有方法大致可分为嵌入式、判别式与重建式三类。嵌入式方法如 PaDiM[10]、PatchCore[11]、CSFlow[12] 利用预训练模型的深度特征刻画正常分布，"
    "通过距离或分布建模区分异常；判别式方法如 CutPaste[13]、DRAEM[14] 通过合成异常将任务转化为监督问题；重建式方法包括自编码器[15][16][17] 与生成对抗网络[18][19][20]，"
    "假设异常因未见于正常数据而在重建时产生更大误差。此外，特征重建与特征对齐类方法通过将轻量网络对齐到预训练模型提取的特征，在效率与性能之间取得平衡。"
    "然而，上述方法普遍依赖类别专属的正常样本，对未见异常类型与域迁移场景适应能力有限，难以规模化部署。")
add_heading(doc, "2.2  基于视觉—语言模型的零样本异常检测", 2)
add_para(doc,
    "预训练视觉模型在工业异常检测中展现出强大的泛化与特征表达能力，零样本方法据此分为两类。第一类为免训练方法：WinCLIP[3] 对齐多粒度特征但需重复编码；"
    "SAA[21] 将 Grounding DINO[22] 与 SAM[23] 级联以提升定位与分割精度，代价是计算开销显著增加。第二类在保持目标域零样本的同时，在辅助数据上训练轻量模块以对齐视觉与文本："
    "AnomalyCLIP[6] 采用对象无关提示与“全局—局部”优化，但依赖双分支结构；AdaCLIP[8] 与 VCP-CLIP[24] 将视觉线索注入文本提示以提升适应性；"
    "CLIP-AD[7]、APRIL-GAN[25]、AA-CLIP[26]、AdaptCLIP[5]、Bayes-PFL[27] 等分别通过分阶段对齐、分割先验、视觉适配器等进一步改进。"
    "上述方法虽然取得了优异性能，却都保留了文本编码器与跨模态对齐，带来训练不稳定与参数冗余。")
add_para(doc,
    "与上述路线形成鲜明对比，VisualAD[9] 首次系统地论证并实现了“去文本”的零样本异常检测：直接移除文本分支，在冻结视觉 Transformer 中插入两个可学习令牌编码正常性与异常性，"
    "并通过 SCA 与 SAF 实现多层空间对齐，在工业与医学共 13 个基准上取得领先结果。本文正是沿 VisualAD 的纯视觉路线推进，并将其进一步扩展为像素定位与图像分类解耦的双骨干增强框架。")
add_heading(doc, "2.3  基于自监督视觉模型的异常检测", 2)
add_para(doc,
    "自监督视觉模型（尤其是 DINO 系列）因无需标注即可学到对结构与纹理敏感的特征，被广泛引入异常检测。DINOv2[28] 凭借强大的视觉特征已在多种视觉任务中作为特征提取器使用；"
    "其后续版本 DINOv3[29] 在特征质量与尺度鲁棒性上进一步提升。在异常检测领域，MuSc[30] 提出利用未标注测试集图像之间的互评分（mutual scoring）实现零样本工业异常分类与分割，"
    "其核心洞察是：正常 patch 在其它图像中能找到大量相似 patch，而异常 patch 则难以匹配；RsCIN 通过 CLS-token 近邻重评分抑制域偏移引起的假阳性；"
    "SPDL[31] 则借助 DINO 特征构建 patch 亲和度图对异常图进行标签传播。这些工作表明，自监督视觉模型所蕴含的、与标注无关的结构信息，能够为异常判别提供稳健且可迁移的先验。"
    "本文与上述工作的区别在于：不将 DINO 特征作为检测的主干替换，而是在冻结 CLIP 双令牌主干的检测流程之上，将 DINOv3 作为第二个完全冻结的骨干，"
    "仅在推理侧用于 patch 图传播、空间一致性融合与稀缺度互评分等后处理，实现图像级分数与像素级定位的解耦。")

# ---------- 3 主要贡献 ----------
add_heading(doc, "3  主要贡献", 1)
add_para(doc, "本文的主要贡献归纳如下：")
contribs = [
    "1）提出像素定位与图像分类解耦的双骨干架构：训练侧以冻结 CLIP ViT-L/14 承担像素级定位（仅训练 anomaly/normal 令牌与轻量检测头），"
    "推理侧引入第二个完全冻结的骨干 DINOv3 ViT-L/16 承担图像级分类的后处理（patch 图传播去噪、空间一致性融合、patch 稀缺度互评分），"
    "两路互不干扰，全程无需文本分支与跨模态对齐。",
    "2）提出双阶段课程式训练范式，解耦令牌学习与检测头优化：将 VisualAD 的单阶段联合目标重构为 Stage 1（令牌分离：仅更新 anomaly/normal 令牌与分离损失、"
    "弱分割损失）与 Stage 2（检测头微调：冻结令牌方向，集中优化 SAF、空间细化层并叠加图像级监督），使令牌方向先收敛、再据此训练定位能力。",
    "3）设计可学习空间细化层，增强异常图空间一致性：以 3×3 深度卷积 + 可学习残差系数 α 构建仅含 10 参数的轻量模块，初始化恒等映射，"
    "用于提升真实缺陷的局部连续性与抑制散乱纹理噪声。",
    "4）构建多级分数校准机制：以空间一致性加权融合替代逐层简单求和，结合同类别图像 DINO patch 稀有度互评分（MuSc）与 CLS-token 近邻重评分（RsCIN）的 z-score 融合，"
    "抑制域偏移带来的假阳性并放大真实异常。",
]
for c in contribs:
    add_para(doc, c)

# ---------- 4 方案创新点 ----------
add_heading(doc, "4  方案创新点", 1)
add_heading(doc, "4.1  问题定义", 2)
_Cs = S(V("C"), V("s"))
_Cu = S(V("C"), V("u"))
para(doc, ["本文遵循零样本异常检测（ZSAD）协议：模型在可见类别集合 ", omath([_Cs]),
           "（来源于工业数据集）上训练，并直接在未见类别集合 ", omath([_Cu]),
           "（来源于其它工业或医学数据集）上进行评估，且 ",
           omath([_Cs, OP(" ∩ "), _Cu, OP(" = "), OP("∅")]),
           "。训练与测试类别来自不同数据集，存在显著的域偏移，评估的是模型跨域泛化能力，且不进行任何类别专属微调。"])

add_heading(doc, "4.2  方法总览", 2)
para(doc, ["如图 1 所示，VisualAD-DINOv3 以冻结的 CLIP ViT-L/14@336px 为主干，输入图像经 patch 化后与可学习的 anomaly/normal 令牌及冻结的 CLS 令牌拼接为序列 ",
           omath(eq1),
           "，经 24 层 Transformer 前向传播后，在第 6、12、18、24 层提取多层 patch 表示。每层配置一个空间交叉注意力（SCA）模块与一个特征自对齐函数（SAF），"
           "分别对令牌与 patch 特征进行空间增强与重校准；patch-到-anomaly 与 patch-到-normal 的余弦相似度之差构成逐层异常图。在此基础上，本文引入以下增强模块："
           "可学习空间细化层（SpatialRefinement）对异常图进行局部空间约束；第二个完全冻结的骨干 DINOv3 ViT-L/16 仅在推理侧用于 patch 图传播去噪、空间一致性融合与稀缺度互评分，实现图像级分数与像素级定位的解耦；"
           "空间一致性融合将多层异常图加权聚合；DINO 图传播、MuSc 互评分与 RsCIN 重评分对图像级分数进行多级校准。整体框架无文本分支，训练与推理共享同一前向路径。"])

# 图 1
doc.add_picture(r"D:\visualad_dinov3\framework.png", width=Cm(14.5))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
add_para(doc, "图 1  VisualAD-DINOv3 总体框架", cn="黑体", size=10.5, bold=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, space_after=8)

add_heading(doc, "4.3  相对 VisualAD（基线）的创新点", 2)
add_para(doc, "本文方法在保留 VisualAD 双令牌机制与 SCA/SAF 多层对齐的基础上，做出的增量创新如下（见表 1）：")

add_para(doc, "表 1  VisualAD（基线）与 VisualAD-DINOv3（本文）创新点对比", cn="黑体", size=10.5, bold=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, space_after=4)
table = doc.add_table(rows=6, cols=3)
table.alignment = WD_TABLE_ALIGNMENT.CENTER
headers = ["维度", "VisualAD（基线）[9]", "VisualAD-DINOv3（本文）"]
for j, h in enumerate(headers):
    cell = table.rows[0].cells[j]
    cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cell.paragraphs[0].add_run(h)
    set_run(r, cn_font="黑体", size=10.5, bold=True)

_L_formula = [V("L"), OP(" = "), S(V("L"), OP("cls")), OP(" + "), S(V("L"), OP("seg")), OP(" + "), S(V("L"), OP("ctr"))]
_sumHl = SIGMA("∑", V("ℓ"), None, S(V("H"), V("ℓ")))
rows_data = [
    (["单阶段联合优化 ", omath(_L_formula)], "双阶段课程式：Stage 1 令牌分离 → Stage 2 检测头微调"),
    (["逐层异常图简单求和 ", omath([_sumHl])], "空间一致性 softmax 加权融合"),
    ("无显式空间约束", ["可学习 3×3 空间细化层（残差 + ", omath([V("α")]), "）"]),
    ("像素级与图像级分数同源于 CLIP 骨干", "解耦 DINOv3 骨干：推理侧图传播 + 一致性融合 + 稀缺度互评分"),
    ("仅 top-k 均值", "MuSc 互评分 + RsCIN 近邻重评分 + z-score 融合"),
]
for i, (col2, col3) in enumerate(rows_data, start=1):
    cell0 = table.rows[i].cells[0]
    cell0.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    # 第一列（维度名）——用维度标签
    dim_labels = ["训练范式", "多层融合", "空间建模", "任务解耦", "图像级重评分"]
    r = cell0.paragraphs[0].add_run(dim_labels[i - 1])
    set_run(r, cn_font="宋体", size=10.5)
    cell_add(table.rows[i].cells[1], col2)
    cell_add(table.rows[i].cells[2], col3)
three_line_table(table)
add_para(doc, "", indent=False, space_after=2)

add_para(doc, "具体地，本文的创新点可以从以下三个层面展开：")
add_para(doc,
    "（一）任务解耦创新——双冻结骨干分工。VisualAD 的像素级定位与图像级分类均由同一 CLIP 主干输出，二者高度耦合，跨域时图像级分数随域偏移而退化。"
    "本文将两条任务解耦：训练侧沿用冻结 CLIP ViT-L/14 完成像素级定位，仅在 anomaly/normal 令牌与轻量检测头上做训练；"
    "推理侧引入第二个完全冻结的骨干 DINOv3 ViT-L/16，专门承担图像级分类的三件后处理——基于深层 patch 特征的图传播去噪、基于局部方差的空间一致性融合、"
    "以及基于跨图像 patch 稀缺度的互评分（MuSc）。DINOv3 不参与任何训练，仅在推理时提供与 CLIP 互补、对域偏移更稳健的结构特征，"
    "实现“像素图来自 CLIP、图像级分数来自 DINOv3”的解耦。")
add_para(doc,
    "（二）训练范式创新——双阶段课程式训练。VisualAD 采用单阶段联合优化，令牌学习与检测头优化相互耦合。本文将其解耦为两阶段：Stage 1（“令牌分离”）仅更新 anomaly/normal 令牌"
    "与最终层归一化，以平方合页分离损失（配合弱分割损失）强制两个令牌在余弦空间中拉开到大于 120° 的角距离，先收敛出稳定、可判别的令牌方向；Stage 2（“检测头微调”）冻结令牌方向，"
    "转而集中优化 SAF 与空间细化层，并叠加图像级二值交叉熵监督，使定位能力建立在已收敛的令牌方向之上。这一“先方向、后定位”的课程式安排，提升了训练稳定性与最终定位精度。")
add_para(doc,
    "（三）识别机制创新——多级分数校准。VisualAD 的图像级分数仅依赖异常图 top-1% 均值的单一来源，且多层融合采用简单求和。本文构建了由三个递进模块组成的校准机制："
    "其一，空间一致性融合以每层异常图的局部方差度量其空间紧凑程度，据此生成 softmax 权重，抑制分数散乱、疑似噪声的层；其二，MuSc 互评分在同类测试图像构成的 DINO 特征库上，"
    "度量每张图像最异常 patch 的跨图像稀有度，从“样本集合一致性”角度补充个体判别的不足；其三，RsCIN 重评分以 CLIP CLS-token 相似度构建 k 近邻图，"
    "通过多窗口（k=5,10,20）标签传播，抑制因域偏移而“独立高分”的假阳性。三者与基础分数的 z-score 融合，共同提升图像级判别的鲁棒性。")

# ---------- 5 模型框架 ----------
add_heading(doc, "5  模型框架", 1)

add_heading(doc, "5.1  冻结骨干与双令牌序列", 2)
para(doc, ["给定骨干（如 CLIP ViT-L/14@336px），输入图像被切分为 ", omath([V("N")]), " 个 patch，与两个可学习全局令牌 ",
           omath([S(V("t"), V("a"))]), "（anomaly）、", omath([S(V("t"), V("n"))]), "（normal）及冻结的类令牌 ",
           omath([S(V("t"), V("c"))]), " 拼接，构成输入序列："])
add_eq_math(doc, eq1, "(1)")
para(doc, ["其中 ", omath([S(V("p"), V("i"))]), " 为第 ", omath([V("i")]), " 个图像 patch 令牌，所有令牌均为 ",
           omath([V("d")]), " 维（默认 1024）。主干保持冻结，仅新增的 anomaly/normal 令牌（及其位置编码）与各检测头参与更新。"
           "为捕获多样化的空间与语义线索，从一组中间层 ", omath([V("L"), OP(" = "), OP("{"), OP("6,12,18,24"), OP("}")]),
           " 提取多层 patch 表示。"])

add_heading(doc, "5.2  空间交叉注意力（SCA，继承自基线）", 2)
para(doc, ["全局令牌编码高层语义，但缺乏显式空间定位。为在不破坏 ViT 管线的前提下注入局部证据，在每个选定层 ", omath([V("ℓ")]),
           " 应用一个 SCA 模块。给定该层 patch 特征 ", omath([S(V("P"), V("ℓ")), OP(" ∈ "), P(V("ℝ"), [V("N"), OP("×"), V("d")])]),
           "，先叠加层专属的可学习位置编码："])
add_eq_math(doc, eq2, "(2)")
para(doc, ["SCA 采用 ", omath([V("m")]), "（默认 ", omath([V("m"), OP(" = "), NUM("4")]), "）个可学习锚点查询 ",
           omath([S(V("Q"), OP("anchor")), OP(" ∈ "), P(V("ℝ"), [V("m"), OP("×"), V("d")])]),
           "，计算跨注意力分布与锚点聚合特征："])
add_eq_math(doc, eq3, "(3)")
para(doc, ["其中 ", omath([S(V("A"), V("ℓ")), OP(" ∈ "), P(V("ℝ"), [V("m"), OP("×"), V("N")])]),
           " 为空间注意力分布，", omath([S(V("U"), V("ℓ")), OP(" ∈ "), P(V("ℝ"), [V("m"), OP("×"), V("d")])]),
           " 的第 ", omath([V("i")]), " 行为锚点聚合特征 ", omath([S(V("a"), V("i"))]),
           "。随后，SCA 通过令牌引导的门控机制将锚点特征适配到每个令牌，计算门控向量 ", omath([V("g"), D("(", ")", V("t")), OP(" = "), V("σ"), D("(", ")", [S(V("W"), V("g")), V("t")]), OP(" ∈ "), P(V("ℝ"), [V("m")])]),
           "，并得到增强令牌："])
add_eq_math(doc, eq4, "(4)")
para(doc, ["其中 ", omath([V("α")]), " 为可学习残差尺度。SCA 对每一层独立实例化，输出增强后的 ",
           omath([P(ACC(TILDE, S(V("t"), V("a"))), D("(", ")", V("ℓ")))]), "、",
           omath([P(ACC(TILDE, S(V("t"), V("n"))), D("(", ")", V("ℓ")))]),
           "，用以与重校准后的 patch 特征对齐，形成逐层异常图。"])

add_heading(doc, "5.3  特征自对齐函数（SAF，继承自基线）", 2)
para(doc, ["除 SCA 外，采用可学习 SAF 对 patch 表示进行重校准。在每个选定层 ", omath([V("ℓ")]),
           "，SAF 实现为单隐层 MLP（隐层维度与输入相同）："])
add_eq_math(doc, eq5, "(5)")
para(doc, ["该非线性重校准使 patch 特征与不断演化的 normal/anomaly 令牌对齐。逐层 patch 级异常分数通过自对齐余弦对比得到（",
           omath([BAR(V("x")), OP(" = "), F(V("x"), S(D("‖", "‖", V("x")), NUM("2")))]),
           " 为 ", omath([S(V("ℓ"), NUM("2"))]), " 归一化）："])
add_eq_math(doc, eq6, "(6)")

add_heading(doc, "5.4  可学习空间细化层（SpatialRefinement，创新）", 2)
para(doc, ["逐层异常图经上采样后，在融合前通过本文提出的可学习空间细化层进行局部空间约束。该模块由一个 3×3 单通道深度卷积与一个可学习残差系数 ",
           omath([S(V("α"), OP("sr"))]),
           " 构成，共计约 10 个参数，并初始化为恒等映射（中心权重为 1、其余为 0，",
           omath([S(V("α"), OP("sr"))]), " 初值为 0.1），使训练起始时近似为无操作："])
add_eq_math(doc, eq7, "(7)")
add_para(doc,
    "其设计动机是：真实缺陷往往在空间上连续、其高分响应具有邻域一致性，而纹理噪声的高分点则孤立、散乱。通过可学习卷积系数，该层能够在训练中自适应地"
    "“增强被正常区域包围的孤立真缺陷高分”、“抑制缺乏空间一致性的散乱噪声”，且极少的参数量保证了最小化的过拟合风险。")

add_heading(doc, "5.5  双阶段课程式训练（创新）", 2)
para(doc, ["本文重构训练过程为两阶段。设总训练轮次由 ", omath([S(V("E"), NUM("1"))]), "（Stage 1）与 ",
           omath([S(V("E"), NUM("2"))]), "（Stage 2）构成。"])
para(doc, "Stage 1（令牌分离）：仅 anomaly/normal 令牌与最终层归一化 ln_post 参与更新，SAF 与空间细化层保持冻结。为显式分离两个令牌，在深层施加余弦间隔惩罚：")
add_eq_math(doc, eq8, "(8)")
para(doc, ["该损失鼓励两个令牌的余弦相似度降至 −0.5 以下（对应角距离大于 120°）。Stage 1 的总损失为 ",
           omath([V("L"), OP(" = "), S(V("w"), OP("sep")), OP("·"), S(V("L"), OP("sep")), OP(" + "), NUM("0.1"), OP("·"), S(V("L"), OP("seg"))]),
           "，其中弱权重分割损失为令牌提供轻微的空间接地。"])
para(doc, "Stage 2（检测头微调）：令牌方向被冻结，SAF 与空间细化层参与优化，损失由分割损失与图像级二值交叉熵构成（分离损失从 Stage 2 剔除，因为令牌方向已收敛）。图像级监督作用于异常图的 top-k 聚合分数：")
add_eq_math(doc, eq9, "(9)")
para(doc, ["其中 ", omath([V("S")]), " 为最终异常图 top-1% 像素的均值分数。像素级监督沿用 Focal Loss 与 Dice Loss 的组合 ",
           omath([S(V("L"), OP("seg")), OP(" = "), SIGMA("∑", [V("ℓ"), OP(" ∈ "), V("L")], None,
                  D("(", ")", [P(S(V("L"), OP("focal")), D("(", ")", V("ℓ"))), OP(" + "),
                               P(S(V("L"), OP("dice")), D("(", ")", V("ℓ")))]))]),
           "。"])

add_heading(doc, "5.6  DINOv3 推理后处理与空间一致性融合（创新）", 2)
para(doc, ["DINOv3 骨干（仅推理）：第二个完全冻结的骨干 DINOv3 ViT-L/16 不参与任何训练与梯度更新，仅在推理侧承担图像级分类的后处理。"
           "其深层 patch 特征 ", omath([S(V("F"), OP("dino")), OP(" ∈ "), P(V("ℝ"), [V("C"), OP("×"), V("h"), OP("×"), V("w")])]),
           " 被用于以下三处：patch 图传播去噪、空间一致性融合的层可靠性度量、以及跨图像稀缺度互评分（MuSc）的特征库。"])
para(doc, "DINO patch 图传播（SPDL 式，仅测试）：以 DINO 特征构建最近邻亲和度，对异常图进行标签传播：")
add_eq_math(doc, eq10, "(10)")
para(doc, ["其中 ", omath([S(V("N"), V("k")), D("(", ")", V("i"))]), " 为 patch ", omath([V("i")]),
           " 在 DINO 特征空间中的 ", omath([V("k")]), " 近邻，", omath([S(V("w"), [V("i"), V("j")])]),
           " 为 softmax 归一化的亲和度，", omath([V("α"), OP(" = "), NUM("0.3")]),
           " 为原始异常图的保留权重（对应传播强度 ", omath([NUM("1"), OP(" − "), V("α")]), "）。"
           "该传播平滑了异常图中的噪声并强化了结构一致的缺陷区域。"])
para(doc, "空间一致性融合（仅测试阶段）：以每层异常图的空间一致性（局部方差的倒数）评价该层的可靠性，进而生成 softmax 权重：")
add_eq_math(doc, eq11, "(11)")
para(doc, ["最终异常图 ", omath([V("H"), OP(" = "), SIGMA("∑", [V("ℓ"), OP(" ∈ "), V("L")], None, [S(V("w"), V("ℓ")), S(V("H"), V("ℓ"))])]),
           "，分数紧凑、空间一致的层获得更高权重，替代了简单的逐层求和（训练阶段的多层融合仍采用逐层简单求和）。"])

add_heading(doc, "5.7  MuSc 互评分与 RsCIN 重评分（创新）", 2)
para(doc, "MuSc 互评分：对每个类别，将同类测试图像的全部 DINO 深层 patch 特征（经 ℓ₂ 归一化）汇总为特征库。对每张图像的最终异常图，取 top-m（约 1%）最异常 patch，"
           "在“排除自身图像”的同类特征库中检索其 top-k 最近邻，以平均稀有度作为图像级互评分：")
add_eq_math(doc, eq12, "(12)")
para(doc, "其直觉是：正常 patch 在同类其它图像中易于找到相似者，而异常 patch 则普遍稀有。")
para(doc, ["RsCIN 重评分：以 CLIP CLS-token 相似度构建图像级 k 近邻图，通过多窗口（", omath([V("k"), OP(" ∈ "), OP("{"), OP("5,10,20"), OP("}")]),
           "）标签传播精炼图像分数："])
add_eq_math(doc, eq13, "(13)")
para(doc,
    "该机制的动机是：CLS-token 距离对局部噪声稳健，两张 CLS 相似度高的图像应具有相近的异常分数；因此，一张图像仅当其自身与其近邻的异常分数都较低时，才被判定为正常，"
    "从而抑制域偏移引入的孤立假阳性、放大真实异常。")
para(doc, "z-score 融合：基础 top-k 分数与 MuSc 互评分分别经 z-score 归一化后加权融合，消除量纲差异：")
add_eq_math(doc, eq14, "(14)")
para(doc, "融合后的分数再经 RsCIN 传播，得到最终图像级异常分数。")

# ---------- 6 实验方案 ----------
add_heading(doc, "6  实验方案与初步结果", 1)
add_heading(doc, "6.1  数据集", 2)
add_para(doc,
    "沿用 VisualAD 的零样本评测体系[9]，覆盖工业与医学两大领域、共 13 个真实数据集。工业域包括 MVTec-AD[32]、VisA[33]、BTAD[34]、KSDD2[35]、DAGM[36]、DTD-Synthetic[37]；"
    "医学域包括 OCT17、BrainMRI、BrainAD、HIS、CVC-ClinicDB、Endo、Kvasir 等，覆盖视网膜 OCT、脑 MRI、组织病理与结肠镜等多种模态与任务。按照约定，以 VisA 作为辅助训练集"
    "（其各类别与其它数据集不重叠），评测 VisA 自身时则在 MVTec-AD 上微调。")
add_heading(doc, "6.2  评价指标", 2)
add_para(doc, "分类层面报告图像级 AUROC、F1-max、AP；分割层面报告像素级 AUROC、F1-max、AP 与 PRO，以全面评估定位能力。")
add_heading(doc, "6.3  对比方法与消融设计", 2)
add_para(doc, "对比基线：WinCLIP[3]、APRIL-GAN[25]、CLIP-AD[7]、AnomalyCLIP[6]、AdaCLIP[8]，以及基线方法 VisualAD（CLIP / DINOv2 两种实例）[9]。")
add_para(doc,
    "消融项（逐项验证各创新模块的贡献）：1）双阶段训练 vs 单阶段联合优化；2）空间细化层（SpatialRefinement）的有无；3）DINOv3 推理后处理骨干（图传播 / 一致性融合 / 稀缺度互评分）的有无；"
    "4）空间一致性融合 vs 逐层求和；5）MuSc 互评分、RsCIN 重评分与 z-score 融合的逐步叠加；6）不同层组合 {6},{12},{18},{24},{6,12,18,24} 的影响。")
add_heading(doc, "6.4  实现细节", 2)
add_para(doc,
    "采用公开骨干 CLIP ViT-L/14@336px 与 DINOv3 ViT-L/16；输入图像缩放到 518×518；从 24 层编码器中提取第 6、12、18、24 层 patch 特征；SCA 锚点数 m=4；"
    "训练共 11 轮（Stage 1 共 5 轮、Stage 2 共 6 轮），仅优化 CLIP 侧的 anomaly/normal 令牌与轻量检测头；优化器为 Adam，初始学习率 1×10⁻³，批大小 8；"
    "DINOv3 ViT-L/16 完全冻结，仅在推理侧提供 patch 特征用于后处理，不使用任何文本分支。")
add_heading(doc, "6.5  主要结果与局限", 2)
add_para(doc,
    "表 2～3 分别给出图像级 AUROC 与像素级 AUROC 在十个基准数据集上的对比结果。"
    "其中“ours”为本文最终配置（开启 RsCIN 重评分与 DINOv3 稀缺度互评分），“baseline”为早期以 VisA 作为训练集的配置；"
    "“—”表示该数据集无有效评测结果；每列中较优的指标以加粗标示。")

_datasets = ["MVTec", "VisA", "BTAD", "DAGM", "DTD", "ClinicDB", "Endo", "Kvasir", "BrainMRI", "SDD"]
_result_tables = [
    ("表 2  图像级 AUROC（Sample-AUROC）对比", [
        ("ours", ["95.9", "87.7", "97.3", "99.4", "98.6", "—", "—", "—", "75.2", "87.5"]),
        ("baseline", ["92.2", "84.5", "94.8", "99.5", "97.8", "—", "—", "—", "92.5", "95.2"]),
    ]),
    ("表 3  像素级 AUROC（Pixel-AUROC）对比", [
        ("ours", ["91.6", "94.5", "94.8", "95.5", "98.0", "85.9", "90.0", "87.6", "—", "97.6"]),
        ("baseline", ["90.9", "95.6", "91.7", "93.0", "98.0", "78.6", "84.6", "80.4", "—", "97.7"]),
    ]),
]

def _better_flags(a, b):
    """返回 (a加粗, b加粗)；a/b 为数值字符串或 '—'。"""
    if a == "—" and b == "—":
        return False, False
    if a == "—":
        return False, True
    if b == "—":
        return True, False
    fa, fb = float(a), float(b)
    if fa == fb:
        return False, False
    return fa > fb, fb > fa

for _title, _rows in _result_tables:
    add_para(doc, _title, cn="黑体", size=10.5, bold=True,
             align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, space_after=4)
    _table = doc.add_table(rows=len(_rows) + 1, cols=len(_datasets) + 1)
    _table.alignment = WD_TABLE_ALIGNMENT.CENTER
    _hdr = ["方法"] + _datasets
    for _j, _h in enumerate(_hdr):
        _cell = _table.rows[0].cells[_j]
        _cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        _r = _cell.paragraphs[0].add_run(_h)
        set_run(_r, cn_font="黑体", size=9, bold=True)
    _ours_vals = _rows[0][1]
    _base_vals = _rows[1][1]
    _flags = [_better_flags(_ours_vals[k], _base_vals[k]) for k in range(len(_datasets))]
    for _i, (_label, _vals) in enumerate(_rows, start=1):
        _c0 = _table.rows[_i].cells[0]
        _c0.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        _r0 = _c0.paragraphs[0].add_run(_label)
        set_run(_r0, cn_font="宋体", size=9)
        for _j, _v in enumerate(_vals, start=1):
            _cell = _table.rows[_i].cells[_j]
            _cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            _r = _cell.paragraphs[0].add_run(_v)
            set_run(_r, cn_font="宋体", size=9, bold=_flags[_j - 1][_i - 1])
    three_line_table(_table)
    add_para(doc, "", indent=False, space_after=4)

add_para(doc,
    "初步实验验证了上述解耦设计的有效性。在跨域图像级判别上，纯 CLIP 方案因域偏移导致图像级分数严重退化（MVTec→VisA 仅 60.6%），"
    "而引入 DINOv3 稀缺度互评分与 RsCIN 重评分后，图像级分数被修复至 87.7%，同时像素级定位基本保持稳定——这印证了“像素图来自 CLIP、图像级分数来自 DINOv3”的解耦分工："
    "CLIP 的像素定位泛化性好，而 DINOv3 的 patch 稀缺度对域偏移更稳健。")
add_para(doc,
    "但该设计依赖“正常 patch 彼此相似”这一隐含假设。在类内方差极大的域（如 BrainMRI、SDD）中，正常样本自身即呈现高度异构的纹理，"
    "导致其 patch 被误判为“稀缺”，假阳性随之暴涨，图像级分数分别下滑 17.3 与 7.7 个百分点。这一失效模式提示，基于稀缺度的图像级评分需要辅以类别内方差自适应，"
    "是后续改进的重点方向。")
add_para(doc, "注：与对比基线的完整对比、消融分析与定性可视化小节待训练结束后补全；本节先给出本文两版配置在两个核心指标上的初步汇总。", indent=True)

# ---------- 7 结论 ----------
add_heading(doc, "7  结论", 1)
add_para(doc,
    "本文在纯视觉零样本异常检测框架 VisualAD 的基础上，提出将像素级定位与图像级分类解耦的双骨干异常检测方法 VisualAD-DINOv3。方法保留了双可学习令牌与 SCA/SAF 多层空间对齐机制，"
    "训练侧由冻结 CLIP ViT-L/14 承担像素定位，重构单阶段优化为“先令牌分离、后检测头微调”的双阶段课程式训练并新增可学习空间细化层；"
    "推理侧由第二个完全冻结的骨干 DINOv3 ViT-L/16 承担图像级评分，通过 patch 图传播去噪、空间一致性融合、patch 稀缺度互评分（MuSc）与 RsCIN 重评分构成的多级校准机制输出最终图像级分数。"
    "实验表明，该解耦设计使跨域时像素定位基本稳定，并将 MVTec→VisA 的图像级分数由 60.6% 修复至 87.7%；但“正常 patch 彼此相似”的假设在类内方差极大的域（BrainMRI、SDD）中失效，"
    "导致图像级分数分别下滑 17.3 与 7.7 个百分点。后续工作将围绕类别内方差自适应的稀缺度评分、更细粒度的结构化令牌设计，以及面向强域偏移场景的轻量不确定性估计展开，"
    "以进一步验证各创新模块在更多基准上的有效性与可推广性。")

# ---------- 参考文献 ----------
add_heading(doc, "参考文献", 1)
refs = [
    "[1] Bergmann P, Fauser M, Sattlegger D, et al. MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection[C]//CVPR, 2019.",
    "[2] Roth K, Pemula L, Zepeda J, et al. Towards Total Recall in Industrial Anomaly Detection[C]//CVPR, 2022.",
    "[3] Jeong J, Zou Y, Kim T, et al. WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation[C]//CVPR, 2023.",
    "[4] Cao Y, Xu X, Sun C, et al. Segment Any Anomaly Without Training via Hybrid Prompt Regularization[J]. CoRR, 2023.",
    "[5] Gao B-B, Zhou Y, Yan J, et al. AdaptCLIP: Adapting CLIP for Universal Visual Anomaly Detection[J]. CoRR, abs/2505.09926, 2025.",
    "[6] Zhou Q, Pang G, Tian Y, et al. AnomalyCLIP: Object-Agnostic Prompt Learning for Zero-Shot Anomaly Detection[C]//ICLR, 2024.",
    "[7] Chen X, Zhang J, Tian G, et al. CLIP-AD: A Language-Guided Staged Dual-Path Model for Zero-Shot Anomaly Detection[J]. CoRR, abs/2311.00453, 2023.",
    "[8] Cao Y, Zhang J, Frittoli L, et al. AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for Zero-Shot Anomaly Detection[J]. 2024.",
    "[9] Hou Y, Li P, Liu Z, et al. VisualAD: Language-Free Zero-Shot Anomaly Detection via Vision Transformer[J]. arXiv:2603.07952, 2026.",
    "[10] Defard T, Setkov A, Loesch A, et al. PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization[C]//ICPR Workshops, 2020.",
    "[11] Roth K, Pemula L, Zepeda J, et al. Towards Total Recall in Industrial Anomaly Detection[C]//CVPR, 2022.",
    "[12] Rudolph M, Wehrbein T, Rosenhahn B, et al. Fully Convolutional Cross-Scale-Flows for Image-Based Defect Detection[C]//WACV, 2022.",
    "[13] Li C-L, Sohn K, Yoon J, et al. CutPaste: Self-Supervised Learning for Anomaly Detection and Localization[C]//CVPR, 2021.",
    "[14] Zavrtanik V, Kristan M, Skočaj D. DRAEM — A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection[C]//ICCV, 2021.",
    "[15] Bengio Y, Yao L, Alain G, et al. Generalized Denoising Auto-Encoders as Generative Models[J]. arXiv:1305.6663, 2013.",
    "[16] Gong D, Liu L, Le V, et al. Memorizing Normality to Detect Anomaly: Memory-Augmented Deep Autoencoder for Unsupervised Anomaly Detection[C]//ICCV, 2019.",
    "[17] Chen Z, et al. Autoencoders for Anomaly Detection (Surveys & Variants).",
    "[18] Perera P, Nallapati R, Xiang B. OCGAN: One-Class Novelty Detection Using GANs with Constrained Latent Representations[C]//CVPR, 2019.",
    "[19] Yan X, Zhang H, Xu X, et al. Learning Semantic Context from Normal Samples for Unsupervised Anomaly Detection[C]//AAAI, 2021.",
    "[20] Zaheer M, Lee J, Astrid M, et al. Old Is Gold: Redefining the Adversarially Learned One-Class Classifier Training Paradigm[C]//CVPR, 2020.",
    "[21] Cao Y, Xu X, Sun C, et al. Segment Any Anomaly Without Training via Hybrid Prompt Regularization[J]. CoRR, abs/2305.10724, 2023.",
    "[22] Liu S, Zeng Z, Ren T, et al. Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection[J]. CoRR, 2023.",
    "[23] Kirillov A, et al. Segment Anything (SAM)[C]//ICCV, 2023.",
    "[24] Qu Z, Tao X, Prasad M, et al. VCP-CLIP: A Visual Context Prompting Model for Zero-Shot Anomaly Segmentation[C]//ECCV, 2024.",
    "[25] Chen X, et al. A Zero-/Few-Shot Anomaly Classification and Segmentation Method for CVPR 2023 VAND Workshop Challenge[J]. CoRR, abs/2305.17382, 2023.",
    "[26] Ma W, Zhang X, Yao Q, et al. AA-CLIP: Enhancing Zero-Shot Anomaly Detection via Anomaly-Aware CLIP[C]//CVPR, 2025.",
    "[27] Qu Z, Tao X, Gong X, et al. Bayesian Prompt Flow Learning for Zero-Shot Anomaly Detection[C]//CVPR, 2025.",
    "[28] Oquab M, Darcet T, Moutakanni T, et al. DINOv2: Learning Robust Visual Features without Supervision[J]. CoRR, 2023.",
    "[29] Oquab M, et al. DINOv3: Transformers, Self-Supervised with Feature Magnitudes, 2025.",
    "[30] Li X, et al. MuSc: Zero-Shot Industrial Anomaly Classification and Segmentation with Mutual Scoring of Unlabeled Images[C]//ICLR, 2024.",
    "[31] SPDL: Self-Supervised Patches Denoising (DINO 图传播相关工作), 2024.",
    "[32] Bergmann P, Fauser M, Sattlegger D, et al. MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection[C]//CVPR, 2019.",
    "[33] Zou Y, Jeong J, Pemula L, et al. Spot-the-Difference Self-Supervised Pre-Training for Anomaly Detection and Segmentation[C]//ECCV, 2022.",
    "[34] Mishra P, Verk R, Fornasier D, et al. VT-ADL: A Vision Transformer Network for Image Anomaly Detection and Localization[C]//ISIE, 2021.",
    "[35] Tabernik D, Sela S, Skvarč J, et al. Segmentation-Based Deep-Learning Approach for Surface-Defect Detection[J]. J. Intell. Manuf., 2020.",
    "[36] Wieler M, Hahn T. Weakly Supervised Learning for Industrial Optical Inspection[C]//DAGM Symposium, 2007.",
    "[37] Aota T, Tong L T T, Okatani T. Zero-Shot versus Many-Shot: Unsupervised Texture Anomaly Detection[C]//WACV, 2023.",
]
for ref in refs:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.3
    r = p.add_run(ref)
    set_run(r, cn_font="宋体", size=10.5)

out = r"D:\visualad_dinov3\学术年会论文.docx"
doc.save(out)
print("saved:", out)
