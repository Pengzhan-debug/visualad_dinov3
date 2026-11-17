# -*- coding: utf-8 -*-
"""OMML (Office Math Markup Language) 原生公式构建助手。

用于在 python-docx 生成的 Word 文档中插入规范的原生公式（上下标、分式、
求和、定界符、函数、根式、上划线、重音、极限等），替代伪 Unicode 文本公式。

用法示例::

    from docx_math import mvar, mop, mnum, sub, frac, nary, rad, d, omath
    # x_i = 1/2 * Σ_{k=1}^{n} y_k
    elems = [sub(mvar("x"), mvar("i")), mop(" = "), frac(mnum("1"), mnum("2")),
             mop(" "), nary("∑", sub(mvar("k"), mnum("1")), mvar("n"),
                            sub(mvar("y"), mvar("k")))]
    om = omath(elems)
"""
from lxml import etree
import copy

M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
XML_NS = "http://www.w3.org/XML/1998/namespace"


def _m(tag):
    return "{%s}%s" % (M_NS, tag)


def _el(tag):
    return etree.Element(_m(tag))


def _as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def _run(text, sty=None):
    """数学文本 run。sty: 'i' 斜体(变量)、'p' 正体(函数/符号/数字)、'b' 粗体、'bi' 粗斜体。"""
    r = _el("r")
    if sty:
        rPr = _el("rPr")
        s = _el("sty")
        s.set(_m("val"), sty)
        rPr.append(s)
        r.append(rPr)
    t = _el("t")
    t.set("{%s}space" % XML_NS, "preserve")
    t.text = text
    r.append(t)
    return r


def mvar(text):
    """变量（斜体）。"""
    return _run(text, "i")


def mop(text):
    """运算符/函数名/符号（正体）。"""
    return _run(text, "p")


def mnum(text):
    """数字（正体）。"""
    return _run(text, "p")


def _arg(tag, content):
    """构造一个参数容器（m:e / m:sub / m:sup / m:num / m:den / m:fName / m:lim / m:deg）。"""
    e = _el(tag)
    for c in _as_list(content):
        e.append(c)
    return e


def sub(base, subc):
    """下标。"""
    s = _el("sSub")
    s.append(_arg("e", base))
    s.append(_arg("sub", subc))
    return s


def sup(base, supc):
    """上标。"""
    s = _el("sSup")
    s.append(_arg("e", base))
    s.append(_arg("sup", supc))
    return s


def subsup(base, subc, supc):
    """上下标。"""
    s = _el("sSubSup")
    s.append(_arg("e", base))
    s.append(_arg("sub", subc))
    s.append(_arg("sup", supc))
    return s


def frac(num, den):
    """分式。"""
    f = _el("f")
    f.append(_arg("num", num))
    f.append(_arg("den", den))
    return f


def d(open_chr, close_chr, *contents):
    """定界符（括号）。contents 为多个参数。"""
    dd = _el("d")
    dPr = _el("dPr")
    if open_chr:
        b = _el("begChr")
        b.set(_m("val"), open_chr)
        dPr.append(b)
    if close_chr:
        c = _el("endChr")
        c.set(_m("val"), close_chr)
        dPr.append(c)
    dd.append(dPr)
    for c in contents:
        dd.append(_arg("e", c))
    return dd


def nary(chr_char, subc, supc, expr, limloc="undOvr"):
    """n 元运算符（求和/积分等）。limloc: undOvr 上下限置上下方。"""
    n = _el("nary")
    nPr = _el("naryPr")
    ch = _el("chr")
    ch.set(_m("val"), chr_char)
    nPr.append(ch)
    ll = _el("limLoc")
    ll.set(_m("val"), limloc)
    nPr.append(ll)
    gr = _el("grow")
    gr.set(_m("val"), "1")
    nPr.append(gr)
    n.append(nPr)
    if subc is not None:
        n.append(_arg("sub", subc))
    if supc is not None:
        n.append(_arg("sup", supc))
    n.append(_arg("e", expr))
    return n


def func(name, arg):
    """函数应用，name 为函数名元素（正体文本）。"""
    f = _el("func")
    fName = _el("fName")
    for c in _as_list(name):
        fName.append(c)
    f.append(fName)
    f.append(_arg("e", arg))
    return f


def rad(expr, deg=None):
    """根式。deg=None 表示隐藏次数（平方根）。"""
    r = _el("rad")
    rPr = _el("radPr")
    if deg is None:
        dh = _el("degHide")
        dh.set(_m("val"), "1")
        rPr.append(dh)
    r.append(rPr)
    if deg is not None:
        r.append(_arg("deg", deg))
    else:
        r.append(_el("deg"))
    r.append(_arg("e", expr))
    return r


def bar(expr):
    """上划线（均值等）。"""
    b = _el("bar")
    bPr = _el("barPr")
    pos = _el("pos")
    pos.set(_m("val"), "top")
    bPr.append(pos)
    b.append(bPr)
    b.append(_arg("e", expr))
    return b


def acc(chr_char, expr):
    """重音符号（hat ̂ / tilde ̃）。"""
    a = _el("acc")
    aPr = _el("accPr")
    ch = _el("chr")
    ch.set(_m("val"), chr_char)
    aPr.append(ch)
    a.append(aPr)
    a.append(_arg("e", expr))
    return a


def limlow(base, lim):
    """下极限（max/min 等）。"""
    l = _el("limLow")
    l.append(_arg("e", base))
    l.append(_arg("lim", lim))
    return l


def omath(elements):
    """生成 m:oMath 内联公式元素。

    注意：对每个子元素做 deepcopy，避免同一 lxml 元素被多次复用（如 eq1 同时
    用于正文与显示公式）时被“移动”导致前一处变空。
    """
    om = _el("oMath")
    for e in _as_list(elements):
        om.append(copy.deepcopy(e))
    return om
