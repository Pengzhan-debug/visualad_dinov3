"""CVPR-style method figure for the improved VisualAD framework.

The script is self-contained and writes framework.pdf, framework.svg, and
framework.png beside itself. Edit the centralized constants below to adjust
the canvas, colors, font sizes, boxes, paths, and labels.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


# =============================================================================
# CENTRALIZED CONSTANTS
# =============================================================================
CANVAS = {
    "size_in": (7.205, 7.00),       # 183 mm CVPR two-column width
    "dpi": 300,
    "axes": (0.0, 0.0, 1.0, 1.0),
    "xlim": (0.0, 1.0),
    "ylim": (0.0, 1.0),
}

COLORS = {
    "paper": "#ffffff",
    "ink": "#20262e",
    "muted": "#6f7c89",
    "line": "#4e5964",
    "divider": "#c9d0d7",
    "panel": "#fbfaf7",
    "frozen_fill": "#eef4fb",
    "frozen_edge": "#2b6cb0",
    "learn_fill": "#ffffff",
    "learn_edge": "#c05621",
    "post_fill": "#f5f0ff",
    "post_edge": "#6b46c1",
    "loss_fill": "#f7fafc",
    "loss_edge": "#a0aec0",
    "output_fill": "#1a202c",
    "output_text": "#ffffff",
    "token_a": "#efa39a",
    "token_a_edge": "#b94f48",
    "token_n": "#f4e7ae",
    "token_n_edge": "#aa8b2f",
    "token_cls": "#dceaf7",
    "token_cls_edge": "#477eae",
    "patch": "#acd889",
    "patch_edge": "#739e4e",
    "sca_fill": "#eaf5f5",
    "attention": "#3f8f96",
    "gold": "#bd9634",
}

FONTS = {
    "panel": 10.5,
    "section": 9.0,
    "module": 8.0,
    "detail": 8.0,
    "tiny": 8.0,
    "micro": 7.0,
    "token": 8.0,
    "legend": 8.0,
    "output": 8.0,
}

STYLE = {
    "box_lw": 0.85,
    "inner_lw": 0.65,
    "arrow_lw": 0.85,
    "dash": (4.5, 3.0),
    "aux_dash": (5.0, 3.3),
    "rounding": 0.009,
    "padding": 0.0025,
    "arrow_scale": 7.5,
    "line_step": 0.021,
    "panel_divider_y": 0.455,
}

ICON = {
    "title_y": 0.81,
    "param_y": 0.14,
    "node_r": 0.030,
    "icon_lw": 0.60,
    "score_bar_h": 0.080,
}


# All absolute boxes use normalized figure coordinates (x, y, width, height).
BOX = {
    # Training panel
    "train_input": (0.020, 0.725, 0.072, 0.130),
    "train_clip": (0.190, 0.735, 0.275, 0.135),
    "train_sca": (0.205, 0.555, 0.260, 0.145),
    "train_saf": (0.485, 0.585, 0.110, 0.080),
    "train_dino": (0.500, 0.780, 0.215, 0.066),
    "train_refiner": (0.765, 0.575, 0.140, 0.100),
    "train_stage1": (0.020, 0.463, 0.410, 0.078),
    "train_stage2": (0.450, 0.463, 0.530, 0.078),

    # Inference branches
    "infer_input": (0.020, 0.245, 0.068, 0.120),
    "infer_clip": (0.120, 0.300, 0.180, 0.095),
    "infer_head": (0.128, 0.255, 0.164, 0.038),
    "infer_dino": (0.120, 0.135, 0.180, 0.082),

    # Inference post-processing
    "step1": (0.370, 0.275, 0.145, 0.090),
    "step2": (0.545, 0.275, 0.145, 0.090),
    "pixel_output": (0.720, 0.275, 0.140, 0.090),
    "step3": (0.385, 0.070, 0.200, 0.105),
    "step5": (0.605, 0.070, 0.130, 0.105),
    "step4": (0.755, 0.070, 0.115, 0.105),
    "image_output": (0.885, 0.070, 0.105, 0.105),

    # Compact legend
    "legend_frozen": (0.650, 0.935, 0.020, 0.022),
    "legend_learn": (0.750, 0.935, 0.020, 0.022),
    "legend_post": (0.840, 0.935, 0.020, 0.022),
    "legend_output": (0.925, 0.935, 0.020, 0.022),
}


# Text anchors not derived from a box center.
LABEL = {
    "training": (0.020, 0.958),
    "training_note": (0.020, 0.928),
    "inference": (0.020, 0.432),
    "inference_note": (0.020, 0.407),
    "tokens": (0.112, 0.708),
    "clip_layers": (0.327, 0.890),
    "readout": (0.250, 0.715),
    "cosine": (0.640, 0.695),
    "cosine_2": (0.640, 0.675),
    "maps": (0.742, 0.720),
    "map_formula": (0.690, 0.595),
    "train_output": (0.950, 0.690),
    "clip_maps": (0.325, 0.342),
    "dino_features": (0.325, 0.190),
    "pixel_note": (0.797, 0.375),
    "decoupled": (0.830, 0.235),
    "aux_dino": (0.342, 0.105),
    "aux_cls": (0.700, 0.392),
    "legend_title": (0.590, 0.946),
}


# Arrow polylines; only the final segment receives an arrowhead.
PATH = {
    # Training
    "train_input_tokens": [(0.094, 0.790), (0.112, 0.790)],
    "train_tokens_clip": [(0.170, 0.790), (0.188, 0.790)],
    "train_clip_sca": [(0.330, 0.735), (0.330, 0.702)],
    "train_clip_saf": [(0.465, 0.790), (0.478, 0.790), (0.478, 0.675), (0.540, 0.675), (0.540, 0.667)],
    "train_sca_cos": [(0.465, 0.625), (0.475, 0.570), (0.610, 0.570), (0.638, 0.640)],
    "train_saf_cos": [(0.597, 0.625), (0.638, 0.650)],
    "train_cos_maps": [(0.670, 0.650), (0.695, 0.650)],
    "train_maps_refine": [(0.748, 0.630), (0.763, 0.630)],
    "train_refine_output": [(0.907, 0.625), (0.923, 0.625)],
    "train_to_stage2": [(0.850, 0.575), (0.850, 0.545), (0.715, 0.545), (0.715, 0.543)],
    "train_input_dino": [(0.056, 0.857), (0.056, 0.906), (0.607, 0.906), (0.607, 0.848)],
    "train_dino_loss": [(0.717, 0.813), (0.952, 0.813), (0.952, 0.543)],

    # Inference main CLIP/DINO flow
    "infer_input_clip": [(0.090, 0.310), (0.108, 0.348), (0.118, 0.348)],
    "infer_input_dino": [(0.090, 0.300), (0.108, 0.182), (0.118, 0.182)],
    "infer_clip_head": [(0.210, 0.300), (0.210, 0.295)],
    "infer_head_maps": [(0.294, 0.274), (0.313, 0.330)],
    "infer_maps_step1": [(0.347, 0.330), (0.368, 0.313)],
    "step1_step2": [(0.517, 0.320), (0.543, 0.320)],
    "step2_pixel": [(0.692, 0.320), (0.718, 0.315)],
    "step2_step3": [(0.618, 0.275), (0.618, 0.215), (0.485, 0.215), (0.485, 0.177)],
    "step3_step5": [(0.587, 0.155), (0.594, 0.155), (0.594, 0.190), (0.670, 0.190), (0.670, 0.177)],
    "step5_step4": [(0.737, 0.123), (0.753, 0.123)],
    "step4_image": [(0.872, 0.123), (0.883, 0.123)],

    # Auxiliary flows routed through whitespace
    "dino_step1": [(0.300, 0.176), (0.335, 0.176), (0.335, 0.252), (0.442, 0.252), (0.442, 0.273)],
    "dino_step3": [(0.300, 0.165), (0.350, 0.165), (0.350, 0.045), (0.485, 0.045), (0.485, 0.068)],
    "cls_step4": [(0.300, 0.350), (0.350, 0.350), (0.350, 0.415), (0.705, 0.415), (0.705, 0.195), (0.812, 0.195), (0.812, 0.177)],
}


FLOW_TEXT = {
    "clip_tensor": (0.620, 0.752, "patch feat [1024,37,37]"),
    "four_maps": (0.400, 0.390, "4× maps"),
    "dino_tensor": (0.400, 0.225, "[1024,h,w]"),
    "graph_maps": (0.530, 0.375, "4 maps"),
    "fused_map": (0.745, 0.370, "[518,518]"),
    "dino_weight": (0.838, 0.830, "regularity weight"),
    "patch_query": (0.550, 0.235, "top-1% patches"),
    "cls": (0.570, 0.430, "CLIP CLS [1024]"),
}


TEXT = {
    "stage1_title": "Stage 1 · epochs 1–5",
    "stage1_lines": [
        "freeze CLIP · train a, n, ln_post (angle >120°)",
        "L = 0.1·Lsep + 0.1·Lseg",
        "Lsep = ReLU(cos(a,n)+0.5)²;  Lseg = Focal + Dice",
    ],
    "stage2_title": "Stage 2 · epochs 6–11",
    "stage2_lines": [
        "freeze a, n · train SAF, ln_post, Refinement (SCA lr=0)",
        "L = Lfocal + Ldice + LBCE(top-1% mean)",
    ],
}


# =============================================================================
# DRAWING HELPERS
# =============================================================================
def rounded_box(ax, key, fill, edge, dashed=False, lw=None, z=3):
    x, y, w, h = BOX[key]
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={STYLE['padding']},rounding_size={STYLE['rounding']}",
        facecolor=fill,
        edgecolor=edge,
        linewidth=lw or STYLE["box_lw"],
        linestyle=(0, STYLE["dash"]) if dashed else "solid",
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def add_text(ax, x, y, text, size="detail", weight="normal", color="ink", ha="center", va="center", z=12):
    ax.text(
        x, y, text,
        fontsize=FONTS[size],
        fontweight=weight,
        color=COLORS[color],
        ha=ha,
        va=va,
        zorder=z,
    )


def arrow(ax, points, auxiliary=False, color=None, z=8):
    col = color or (COLORS["post_edge"] if auxiliary else COLORS["ink"])
    line_style = (0, STYLE["aux_dash"]) if auxiliary else "solid"
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot(
            [start[0], end[0]], [start[1], end[1]],
            color=col, linewidth=STYLE["arrow_lw"], linestyle=line_style,
            solid_capstyle="round", zorder=z,
        )
    start, end = points[-2], points[-1]
    patch = FancyArrowPatch(
        start, end,
        arrowstyle="-|>", mutation_scale=STYLE["arrow_scale"],
        linewidth=STYLE["arrow_lw"], linestyle=line_style,
        color=col, shrinkA=1.2, shrinkB=1.2, zorder=z,
    )
    ax.add_patch(patch)


def box_text(ax, key, title, details, title_color="ink", detail_color="muted", title_size="module"):
    x, y, w, h = BOX[key]
    cx, cy = x + w / 2.0, y + h / 2.0
    n = len(details)
    top = cy + n * STYLE["line_step"] / 2.0
    add_text(ax, cx, top, title, size=title_size, weight="bold", color=title_color)
    for i, line in enumerate(details, start=1):
        add_text(ax, cx, top - i * STYLE["line_step"], line, size="detail", color=detail_color)


def draw_input(ax, key):
    rounded_box(ax, key, COLORS["loss_fill"], COLORS["line"])
    x, y, w, h = BOX[key]
    ax.add_patch(Rectangle((x + 0.14*w, y + 0.14*h), 0.72*w, 0.72*h,
                           facecolor="#cbd4da", edgecolor="none", zorder=4))
    ax.add_patch(Rectangle((x + 0.35*w, y + 0.20*h), 0.30*w, 0.58*h,
                           facecolor="#8198a5", edgecolor=COLORS["line"],
                           linewidth=STYLE["inner_lw"], zorder=5))
    ax.add_patch(Circle((x + 0.58*w, y + 0.46*h), 0.09*w,
                        facecolor=COLORS["token_a"], edgecolor=COLORS["paper"],
                        linewidth=STYLE["inner_lw"], zorder=6))
    for frac in (0.33, 0.66):
        ax.plot([x + frac*w, x + frac*w], [y + 0.14*h, y + 0.86*h],
                color=COLORS["paper"], linewidth=0.35, zorder=7)
        ax.plot([x + 0.14*w, x + 0.86*w], [y + frac*h, y + frac*h],
                color=COLORS["paper"], linewidth=0.35, zorder=7)


def token_chip(ax, x, y, label, fill, edge, w=0.026, h=0.035):
    patch = FancyBboxPatch(
        (x - w/2.0, y - h/2.0), w, h,
        boxstyle=f"round,pad={STYLE['padding']},rounding_size={STYLE['rounding']/2.0}",
        facecolor=fill, edgecolor=edge, linewidth=STYLE["inner_lw"], zorder=7,
    )
    ax.add_patch(patch)
    add_text(ax, x, y, label, size="token", weight="bold", z=9)


def patch_row(ax, x, y, count=4, cell_w=0.014, cell_h=0.024, gap=0.004):
    for i in range(count):
        ax.add_patch(Rectangle(
            (x + i*(cell_w + gap), y), cell_w, cell_h,
            facecolor=COLORS["patch"], edgecolor=COLORS["patch_edge"],
            linewidth=STYLE["inner_lw"], zorder=6,
        ))


def draw_tokens(ax):
    token_chip(ax, 0.120, 0.815, "A", COLORS["token_a"], COLORS["token_a_edge"])
    token_chip(ax, 0.150, 0.815, "N", COLORS["token_n"], COLORS["token_n_edge"])
    token_chip(ax, 0.120, 0.770, "C", COLORS["token_cls"], COLORS["token_cls_edge"])
    patch_row(ax, 0.140, 0.758, count=3, cell_w=0.010, cell_h=0.024, gap=0.003)
    add_text(ax, *LABEL["tokens"], "[A, N, CLS, 37×37 patches]", size="micro", color="muted")


def draw_clip_backbone(ax):
    rounded_box(ax, "train_clip", COLORS["frozen_fill"], COLORS["frozen_edge"], dashed=True)
    x, y, w, h = BOX["train_clip"]
    layer_names = ["L1", "L6", "L12", "L18", "L24"]
    for i, name in enumerate(layer_names):
        bx = x + 0.025*w + i*0.190*w
        bw = 0.145*w
        by = y + 0.22*h
        bh = 0.55*h
        ax.add_patch(FancyBboxPatch(
            (bx, by), bw, bh,
            boxstyle=f"round,pad={STYLE['padding']/2.0},rounding_size={STYLE['rounding']/2.0}",
            facecolor="#dcebc8", edgecolor="#7b9d4f",
            linewidth=STYLE["inner_lw"], zorder=5,
        ))
        add_text(ax, bx + bw/2.0, by + bh/2.0, name, size="tiny", weight="bold", z=7)
    add_text(ax, *LABEL["clip_layers"], "Frozen CLIP ViT-L/14@336px · layers {6,12,18,24}",
             size="module", weight="bold")


def draw_matrix(ax, x, y, w, h, rows=5, cols=7, cmap="YlGnBu", z=7):
    cm = plt.get_cmap(cmap)
    for r in range(rows):
        for c in range(cols):
            value = 0.18 + 0.74*((np.sin((r+1)*1.4 + (c+1)*0.9) + 1.0)/2.0)
            ax.add_patch(Rectangle(
                (x + c*w/cols, y + r*h/rows), w/cols, h/rows,
                facecolor=cm(value), edgecolor=COLORS["paper"],
                linewidth=0.18, zorder=z,
            ))
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=COLORS["line"],
                           linewidth=STYLE["inner_lw"], zorder=z+1))


def draw_sca(ax):
    rounded_box(ax, "train_sca", COLORS["sca_fill"], COLORS["frozen_edge"], dashed=True)
    x, y, w, h = BOX["train_sca"]
    add_text(ax, x + 0.50*w, y + 0.89*h, "SCA · layer-wise · frozen (lr=0)", size="module", weight="bold")
    add_text(ax, x + 0.50*w, y + 0.73*h, "m=4 · pos-enc · gate · res 0.01", size="tiny", color="muted")

    patch_row(ax, x + 0.05*w, y + 0.42*h, count=3, cell_w=0.035*w, cell_h=0.22*h, gap=0.012*w)
    for i in range(4):
        ax.add_patch(Circle((x + (0.25 + 0.045*i)*w, y + 0.25*h), 0.012*w,
                            facecolor=COLORS["gold"], edgecolor="none", zorder=7))
    add_text(ax, x + 0.31*w, y + 0.10*h, "Q × 4", size="tiny", color="muted")
    draw_matrix(ax, x + 0.46*w, y + 0.25*h, 0.20*w, 0.35*h, rows=5, cols=6)
    add_text(ax, x + 0.57*w, y + 0.14*h, "Attn", size="tiny", color="muted")
    token_chip(ax, x + 0.73*w, y + 0.49*h, "A", COLORS["token_a"], COLORS["token_a_edge"],
               w=0.080*w, h=0.28*h)
    token_chip(ax, x + 0.84*w, y + 0.49*h, "N", COLORS["token_n"], COLORS["token_n_edge"],
               w=0.080*w, h=0.28*h)
    gate = FancyBboxPatch(
        (x + 0.70*w, y + 0.18*h), 0.18*w, 0.19*h,
        boxstyle=f"round,pad={STYLE['padding']/2.0},rounding_size={STYLE['rounding']/2.0}",
        facecolor=COLORS["paper"], edgecolor=COLORS["attention"],
        linewidth=STYLE["inner_lw"], zorder=6,
    )
    ax.add_patch(gate)
    add_text(ax, x + 0.79*w, y + 0.275*h, "gate", size="tiny", weight="bold", color="attention")
    arrow(ax, [(x + 0.20*w, y + 0.50*h), (x + 0.44*w, y + 0.43*h)], color=COLORS["attention"])
    arrow(ax, [(x + 0.67*w, y + 0.43*h), (x + 0.69*w, y + 0.30*h)], color=COLORS["attention"])
    arrow(ax, [(x + 0.88*w, y + 0.28*h), (x + 0.94*w, y + 0.50*h)], color=COLORS["attention"])


def draw_saf(ax):
    rounded_box(ax, "train_saf", COLORS["learn_fill"], COLORS["learn_edge"])
    x, y, w, h = BOX["train_saf"]
    add_text(ax, x + w/2.0, y + 0.73*h, "SAF", size="module", weight="bold")
    add_text(ax, x + w/2.0, y + 0.44*h, "1024→1024 MLP", size="tiny")
    add_text(ax, x + w/2.0, y + 0.18*h, "per-layer · GELU", size="tiny", color="muted")


def draw_dino_teacher(ax):
    rounded_box(ax, "train_dino", COLORS["frozen_fill"], COLORS["frozen_edge"], dashed=True)
    box_text(ax, "train_dino", "DINOv3 ViT-L/16 · frozen",
             ["regularity w = 1/(1+var)", "optional (dino_reg_weight>0)"])


def draw_similarity(ax):
    add_text(ax, *LABEL["cosine"], "S-l = cos(p,a)", size="tiny", weight="bold")
    add_text(ax, *LABEL["cosine_2"], "− cos(p,n)", size="tiny", weight="bold")
    ax.add_patch(Circle((0.655, 0.645), 0.017, facecolor=COLORS["paper"],
                        edgecolor=COLORS["line"], linewidth=STYLE["box_lw"], zorder=7))
    add_text(ax, 0.655, 0.645, "−", size="module", weight="bold", z=9)


def draw_heatmap(ax, x, y, w, h, phase=0.0, z=6):
    n = 9
    grid = np.linspace(-1.0, 1.0, n)
    xx, yy = np.meshgrid(grid, grid)
    blob = np.exp(-((xx - 0.30 - phase)**2 + (yy + 0.18)**2)/0.10)
    blob += 0.18*np.exp(-((xx + 0.42)**2 + (yy - 0.45 + phase)**2)/0.16)
    blob = (blob - blob.min())/(blob.max() - blob.min() + 1e-9)
    cm = plt.get_cmap("turbo")
    for r in range(n):
        for c in range(n):
            ax.add_patch(Rectangle(
                (x + c*w/n, y + r*h/n), w/n + 0.0001, h/n + 0.0001,
                facecolor=cm(0.05 + 0.90*blob[r, c]), edgecolor="none", zorder=z,
            ))
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=COLORS["line"],
                           linewidth=STYLE["inner_lw"], zorder=z+1))


def draw_map_stack(ax, x, y, w, h, count=4, dx=0.010, dy=0.010):
    for i in range(count):
        draw_heatmap(ax, x + i*dx, y + i*dy, w, h, phase=(i-1.5)*0.025, z=6+i)


def draw_train_output(ax):
    add_text(ax, *LABEL["maps"], "4 maps", size="tiny", weight="bold")
    draw_map_stack(ax, 0.700, 0.600, 0.043, 0.072, count=4, dx=0.010, dy=0.008)
    rounded_box(ax, "train_refiner", COLORS["learn_fill"], COLORS["learn_edge"])
    box_text(ax, "train_refiner", "SpatialRefinement", ["3×3 conv", "identity init + α"])
    draw_heatmap(ax, 0.930, 0.600, 0.047, 0.075, phase=0.0)
    add_text(ax, *LABEL["train_output"], "refined H-l", size="tiny", weight="bold")


def draw_stage_boxes(ax):
    for key in ("train_stage1", "train_stage2"):
        rounded_box(ax, key, COLORS["loss_fill"], COLORS["loss_edge"])
    for key, title_key, lines_key in [
        ("train_stage1", "stage1_title", "stage1_lines"),
        ("train_stage2", "stage2_title", "stage2_lines"),
    ]:
        box_text(ax, key, TEXT[title_key], TEXT[lines_key])


def draw_legend(ax):
    add_text(ax, *LABEL["legend_title"], "Legend", size="legend", weight="bold", ha="right")
    items = [
        ("legend_frozen", "frozen", "frozen_fill", "frozen_edge", True),
        ("legend_learn", "learn", "learn_fill", "learn_edge", False),
        ("legend_post", "test", "post_fill", "post_edge", False),
        ("legend_output", "output", "output_fill", "output_fill", False),
    ]
    for key, text, fill, edge, dashed in items:
        rounded_box(ax, key, COLORS[fill], COLORS[edge], dashed=dashed, lw=STYLE["inner_lw"])
        x, y, w, h = BOX[key]
        add_text(ax, x + w + 0.005, y + h/2.0, text, size="legend", ha="left")


def draw_training(ax):
    add_text(ax, *LABEL["training"], "a  Two-stage training on VisA", size="panel", weight="bold", ha="left")
    add_text(ax, *LABEL["training_note"], "11 epochs = 5 + 6 · frozen CLIP backbone · optional DINOv3 weighting", size="tiny", color="muted", ha="left")
    draw_legend(ax)
    draw_input(ax, "train_input")
    draw_tokens(ax)
    draw_clip_backbone(ax)
    draw_sca(ax)
    draw_saf(ax)
    draw_dino_teacher(ax)
    draw_similarity(ax)
    draw_train_output(ax)
    draw_stage_boxes(ax)

    for key in [
        "train_input_tokens", "train_tokens_clip", "train_clip_sca", "train_clip_saf",
        "train_sca_cos", "train_saf_cos", "train_cos_maps", "train_maps_refine",
        "train_refine_output",
    ]:
        arrow(ax, PATH[key])
    arrow(ax, PATH["train_to_stage2"], color=COLORS["loss_edge"])
    arrow(ax, PATH["train_input_dino"], auxiliary=True)
    arrow(ax, PATH["train_dino_loss"], auxiliary=True)
    add_text(ax, *FLOW_TEXT["clip_tensor"][:2], FLOW_TEXT["clip_tensor"][2], size="tiny", color="muted")
    add_text(ax, *FLOW_TEXT["dino_weight"][:2], FLOW_TEXT["dino_weight"][2], size="tiny", color="muted")


def draw_infer_branch(ax):
    draw_input(ax, "infer_input")

    rounded_box(ax, "infer_clip", COLORS["frozen_fill"], COLORS["frozen_edge"], dashed=True)
    box_text(ax, "infer_clip", "CLIP branch", ["Frozen ViT-L/14", "A/N · SCA · SAF", "+ SpatialRefinement"])
    rounded_box(ax, "infer_head", COLORS["learn_fill"], COLORS["learn_edge"])
    box_text(ax, "infer_head", "L6 · L12 · L18 · L24", [], title_size="tiny")
    rounded_box(ax, "infer_dino", COLORS["frozen_fill"], COLORS["frozen_edge"], dashed=True)
    box_text(ax, "infer_dino", "DINOv3 branch", ["Frozen ViT-L/16", "deepest patches"])

    draw_map_stack(ax, 0.315, 0.305, 0.028, 0.052, count=4, dx=0.006, dy=0.006)
    draw_matrix(ax, 0.315, 0.160, 0.038, 0.052, rows=5, cols=5, cmap="Purples")

    for key in ["infer_input_clip", "infer_input_dino", "infer_clip_head", "infer_head_maps", "infer_maps_step1"]:
        arrow(ax, PATH[key])


def draw_post_box(ax, key, title, details):
    rounded_box(ax, key, COLORS["post_fill"], COLORS["post_edge"])
    box_text(ax, key, title, details)


def post_title_and_param(ax, key, title, param):
    x, y, w, h = BOX[key]
    add_text(ax, x + 0.50*w, y + ICON["title_y"]*h, title,
             size="module", weight="bold")
    add_text(ax, x + 0.50*w, y + ICON["param_y"]*h, param,
             size="tiny", color="muted")


def draw_graph_icon(ax, key, node_fill=None):
    x, y, w, h = BOX[key]
    pts = np.array([
        [0.33, 0.42], [0.47, 0.61], [0.58, 0.36],
        [0.69, 0.58], [0.76, 0.36],
    ])
    pts[:, 0] = x + pts[:, 0] * w
    pts[:, 1] = y + pts[:, 1] * h
    for a, b in [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]:
        ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]],
                color=COLORS["post_edge"], linewidth=ICON["icon_lw"], zorder=7)
    radius = ICON["node_r"] * min(w, h)
    for i, (px, py) in enumerate(pts):
        fill = node_fill or (COLORS["token_a"] if i == 2 else COLORS["paper"])
        ax.add_patch(Circle((px, py), radius, facecolor=fill,
                            edgecolor=COLORS["post_edge"],
                            linewidth=STYLE["inner_lw"], zorder=8))


def draw_graph_refine(ax):
    rounded_box(ax, "step1", COLORS["post_fill"], COLORS["post_edge"])
    post_title_and_param(ax, "step1", "① DINO Graph", "k=8 · α=0.3")
    x, y, w, h = BOX["step1"]
    draw_matrix(ax, x + 0.08*w, y + 0.31*h, 0.16*w, 0.28*h,
                rows=3, cols=3, cmap="Purples", z=7)
    arrow(ax, [(x + 0.27*w, y + 0.45*h), (x + 0.31*w, y + 0.45*h)],
          color=COLORS["post_edge"], z=9)
    draw_graph_icon(ax, "step1")


def draw_consistency_icon(ax):
    rounded_box(ax, "step2", COLORS["post_fill"], COLORS["post_edge"])
    post_title_and_param(ax, "step2", "② Consistency", "5×5 · softmax")
    x, y, w, h = BOX["step2"]
    cm = plt.get_cmap("turbo")
    for i in range(3):
        ax.add_patch(Rectangle(
            (x + (0.10 + 0.055*i)*w, y + (0.34 + 0.035*i)*h),
            0.13*w, 0.25*h, facecolor=cm(0.25 + 0.25*i),
            edgecolor=COLORS["line"], linewidth=STYLE["inner_lw"], zorder=7+i,
        ))
    arrow(ax, [(x + 0.38*w, y + 0.46*h), (x + 0.52*w, y + 0.46*h)],
          color=COLORS["post_edge"], z=9)
    draw_heatmap(ax, x + 0.59*w, y + 0.29*h, 0.25*w, 0.34*h, phase=0.0, z=7)


def draw_mutual_icon(ax):
    rounded_box(ax, "step3", COLORS["post_fill"], COLORS["post_edge"])
    post_title_and_param(ax, "step3", "③ Mutual Scoring", "top-1% · 5-NN")
    x, y, w, h = BOX["step3"]
    query = (x + 0.22*w, y + 0.46*h)
    ax.add_patch(Rectangle((query[0] - 0.045*w, query[1] - 0.08*h),
                           0.09*w, 0.16*h, facecolor=COLORS["token_a"],
                           edgecolor=COLORS["token_a_edge"],
                           linewidth=STYLE["inner_lw"], zorder=8))
    bank = [(0.53, 0.58), (0.66, 0.61), (0.78, 0.52),
            (0.58, 0.35), (0.72, 0.32), (0.84, 0.38)]
    for i, (bx, by) in enumerate(bank):
        px, py = x + bx*w, y + by*h
        if i < 5:
            ax.plot([query[0], px], [query[1], py], color=COLORS["post_edge"],
                    linewidth=ICON["icon_lw"], alpha=0.72, zorder=6)
        ax.add_patch(Circle((px, py), ICON["node_r"]*min(w, h),
                            facecolor=COLORS["paper"], edgecolor=COLORS["post_edge"],
                            linewidth=STYLE["inner_lw"], zorder=8))


def draw_score_fusion_icon(ax):
    rounded_box(ax, "step5", COLORS["post_fill"], COLORS["post_edge"])
    post_title_and_param(ax, "step5", "⑤ Score Fusion", "z-score · w=0")
    x, y, w, h = BOX["step5"]
    ys = [0.55, 0.37]
    widths = [0.25, 0.42]
    fills = [COLORS["frozen_edge"], COLORS["token_a_edge"]]
    for yy, ww, fill in zip(ys, widths, fills):
        ax.add_patch(Rectangle((x + 0.14*w, y + yy*h), ww*w,
                               ICON["score_bar_h"]*h,
                               facecolor=fill, edgecolor="none", zorder=8))
    arrow(ax, [(x + 0.60*w, y + 0.46*h), (x + 0.78*w, y + 0.46*h)],
          color=COLORS["post_edge"], z=9)
    ax.add_patch(Circle((x + 0.83*w, y + 0.46*h), 0.055*h,
                        facecolor=COLORS["paper"], edgecolor=COLORS["post_edge"],
                        linewidth=STYLE["inner_lw"], zorder=8))


def draw_rscin_icon(ax):
    rounded_box(ax, "step4", COLORS["post_fill"], COLORS["post_edge"])
    post_title_and_param(ax, "step4", "④ RsCIN", "k={5,10,20}")
    draw_graph_icon(ax, "step4", node_fill=COLORS["token_cls"])


def draw_outputs(ax):
    rounded_box(ax, "pixel_output", COLORS["output_fill"], COLORS["output_fill"])
    x, y, w, h = BOX["pixel_output"]
    add_text(ax, x + 0.50*w, y + 0.82*h, "Pixel-level Map", size="output",
             weight="bold", color="output_text")
    draw_heatmap(ax, x + 0.40*w, y + 0.32*h, 0.20*w, 0.30*h, phase=0.0, z=7)
    add_text(ax, x + 0.50*w, y + 0.13*h, "Pixel-AUROC", size="tiny",
             color="output_text")
    rounded_box(ax, "image_output", COLORS["output_fill"], COLORS["output_fill"])
    x, y, w, h = BOX["image_output"]
    add_text(ax, x + 0.50*w, y + 0.74*h, "Image-level", size="output",
             weight="bold", color="output_text")
    add_text(ax, x + 0.50*w, y + 0.55*h, "Score", size="output",
             weight="bold", color="output_text")
    ax.add_patch(Rectangle((x + 0.16*w, y + 0.20*h), 0.68*w, 0.09*h,
                           facecolor="#596579", edgecolor="none", zorder=7))
    ax.add_patch(Rectangle((x + 0.16*w, y + 0.20*h), 0.49*w, 0.09*h,
                           facecolor=COLORS["token_a"], edgecolor="none", zorder=8))


def draw_inference(ax):
    add_text(ax, *LABEL["inference"], "b  Zero-shot cross-domain inference", size="panel", weight="bold", ha="left")
    add_text(ax, *LABEL["inference_note"], "frozen CLIP + test-time frozen DINOv3", size="tiny", color="muted", ha="left")
    draw_infer_branch(ax)

    draw_graph_refine(ax)
    draw_consistency_icon(ax)
    draw_mutual_icon(ax)
    draw_score_fusion_icon(ax)
    draw_rscin_icon(ax)
    draw_outputs(ax)

    for key in ["step1_step2", "step2_pixel", "step2_step3", "step3_step5", "step5_step4", "step4_image"]:
        arrow(ax, PATH[key])
    for key in ["dino_step1", "dino_step3", "cls_step4"]:
        arrow(ax, PATH[key], auxiliary=True)

    for key in ["four_maps"]:
        x, y, value = FLOW_TEXT[key]
        add_text(ax, x, y, value, size="tiny", color="muted")
    add_text(ax, *LABEL["decoupled"], "decoupled image scoring", size="tiny", weight="bold", color="post_edge")


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "font.size": FONTS["detail"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": COLORS["paper"],
        "savefig.facecolor": COLORS["paper"],
    })

    fig = plt.figure(figsize=CANVAS["size_in"], facecolor=COLORS["paper"])
    ax = fig.add_axes(CANVAS["axes"])
    ax.set_xlim(*CANVAS["xlim"])
    ax.set_ylim(*CANVAS["ylim"])
    ax.axis("off")

    ax.plot([0.015, 0.985], [STYLE["panel_divider_y"], STYLE["panel_divider_y"]],
            color=COLORS["divider"], linewidth=STYLE["inner_lw"], zorder=1)
    draw_training(ax)
    draw_inference(ax)

    out = Path(__file__).resolve().parent
    metadata = {"Title": "Improved VisualAD Method Framework", "Creator": "Python + Matplotlib"}
    fig.savefig(out / "framework.pdf", format="pdf", metadata=metadata)
    fig.savefig(out / "framework.svg", format="svg", metadata=metadata)
    fig.savefig(out / "framework.png", format="png", dpi=CANVAS["dpi"], metadata=metadata)
    plt.close(fig)


if __name__ == "__main__":
    main()
