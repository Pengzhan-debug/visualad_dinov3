# -*- coding: utf-8 -*-
"""测试 docx_math 生成的原生公式能否被 Word 正确渲染。"""
import docx_math as M
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH


def add_eq(doc, elements, tag):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(6)
    om = M.omath(elements)
    p._p.append(om)
    r = p.add_run("    " + tag)
    r.font.size = Pt(11)
    return p


doc = Document()

# eq1: z_0 = [t_a, t_n, t_c, p_1, …, p_N]
add_eq(doc, [
    M.sub(M.mvar("z"), M.mnum("0")),
    M.mop(" = "),
    M.d("[", "]",
        M.sub(M.mvar("t"), M.mvar("a")), M.mop(", "),
        M.sub(M.mvar("t"), M.mvar("n")), M.mop(", "),
        M.sub(M.mvar("t"), M.mvar("c")), M.mop(", "),
        M.sub(M.mvar("p"), M.mnum("1")), M.mop(", … , "),
        M.sub(M.mvar("p"), M.mvar("N"))),
], "(1)")

# eq3: A_ℓ = softmax( Q_anchor (P_ℓ^pos)^⊤ / √d ), U_ℓ = A_ℓ P_ℓ
add_eq(doc, [
    M.sub(M.mvar("A"), M.mvar("ℓ")),
    M.mop(" = "),
    M.func(M.mop("softmax"),
           M.d("(", ")",
               M.frac([M.sub(M.mvar("Q"), M.mop("anchor")),
                       M.sup(M.d("(", ")", M.subsup(M.mvar("P"), M.mvar("ℓ"), M.mop("pos"))), M.mop("⊤"))],
                      M.rad(M.mvar("d"))))),
    M.mop(",    "),
    M.sub(M.mvar("U"), M.mvar("ℓ")),
    M.mop(" = "),
    M.sub(M.mvar("A"), M.mvar("ℓ")),
    M.sub(M.mvar("P"), M.mvar("ℓ")),
], "(3)")

# eq8: L_sep = max(0, ⟨t̄_a^(L), t̄_n^(L)⟩ + τ)², τ = 0.5
add_eq(doc, [
    M.sub(M.mvar("L"), M.mop("sep")),
    M.mop(" = "),
    M.sup(M.limlow(M.mop("max"),
                   M.d("(", ")",
                       M.mnum("0"), M.mop(", "),
                       M.d("⟨", "⟩",
                           M.subsup(M.bar(M.mvar("t")), M.mvar("a"), M.d("(", ")", M.mvar("L"))),
                           M.mop(", "),
                           M.subsup(M.bar(M.mvar("t")), M.mvar("n"), M.d("(", ")", M.mvar("L"))),
                       ), M.mop(" + "), M.mvar("τ"))), M.mnum("2")),
    M.mop(",    "),
    M.mvar("τ"), M.mop(" = "), M.mnum("0.5"),
], "(8)")

# eq12: s_mutual = 1/m Σ_{i=1}^m ( 1 − 1/k Σ_{j∈NN_k(q_i)} sim(q_i, f_j) )
inner_sum = M.nary("∑",
                   [M.mvar("j"), M.mop(" ∈ "), M.sub(M.mop("NN"), M.mvar("k")),
                    M.d("(", ")", M.sub(M.mvar("q"), M.mvar("i")))],
                   None,
                   M.func(M.mop("sim"),
                          M.d("(", ")",
                              M.sub(M.mvar("q"), M.mvar("i")), M.mop(", "),
                              M.sub(M.mvar("f"), M.mvar("j")))))
outer = M.nary("∑",
               M.sub(M.mvar("i"), M.mnum("1")),
               M.mvar("m"),
               M.d("(", ")",
                   M.mnum("1"), M.mop(" − "),
                   M.frac(M.mnum("1"), M.mvar("k")),
                   inner_sum))
add_eq(doc, [
    M.sub(M.mvar("s"), M.mop("mutual")),
    M.mop(" = "),
    M.frac(M.mnum("1"), M.mvar("m")),
    outer,
], "(12)")

# eq13: C_refined = 1/(K+1) ( C + Σ_{k∈K} D_k^{-1} (M_k ⊙ W) C )
add_eq(doc, [
    M.sub(M.mvar("C"), M.mop("refined")),
    M.mop(" = "),
    M.frac(M.mnum("1"), [M.mvar("K"), M.mop(" + "), M.mnum("1")]),
    M.d("(", ")",
        M.mvar("C"), M.mop(" + "),
        M.nary("∑", [M.mvar("k"), M.mop(" ∈ "), M.mvar("K")], None,
               [M.sup(M.sub(M.mvar("D"), M.mvar("k")), M.mop("−1")),
                M.d("(", ")", [M.sub(M.mvar("M"), M.mvar("k")), M.mop(" ⊙ "), M.mvar("W")]),
                M.mvar("C")])),
], "(13)")

out = r"D:\visualad_dinov3\__math_test.docx"
doc.save(out)
print("saved:", out)
