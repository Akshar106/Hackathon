"""
Generate CNN architecture diagram for CharClassifier (4-block ResVGG).
Saves to docs/architecture.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

os.makedirs("docs", exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis("off")
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#1a1a2e")

# ── Color palette ─────────────────────────────────────────────────────────────
C_INPUT   = "#4cc9f0"
C_CONV    = "#4361ee"
C_RES     = "#7209b7"
C_SKIP    = "#f72585"
C_POOL    = "#3a0ca3"
C_GAP     = "#480ca8"
C_FC      = "#560bad"
C_OUTPUT  = "#4cc9f0"
C_ARROW   = "#aaaaaa"
C_TEXT    = "white"
C_DIM     = "#aaaacc"

def box(ax, x, y, w, h, color, text, subtext="", alpha=0.85, fontsize=9):
    rect = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor="white",
        linewidth=0.8, alpha=alpha, zorder=3,
    )
    ax.add_patch(rect)
    ax.text(x, y + (0.12 if subtext else 0), text,
            ha="center", va="center", fontsize=fontsize,
            color=C_TEXT, fontweight="bold", zorder=4)
    if subtext:
        ax.text(x, y - 0.22, subtext,
                ha="center", va="center", fontsize=7,
                color=C_DIM, zorder=4)

def arrow(ax, x1, y1, x2, y2, color=C_ARROW, style="->", lw=1.2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0"))

def res_block(ax, cx, top_y, label, in_ch, out_ch, spatial, color=C_CONV):
    """Draw one ResVGGBlock with conv1, conv2, skip, and residual add."""
    bh = 0.55   # box height
    bw = 1.6    # box width
    gap = 0.72  # vertical gap between boxes

    y0 = top_y
    y1 = y0 - gap
    y2 = y1 - gap
    y3 = y2 - gap * 0.8   # pool

    # Block label
    ax.text(cx, y0 + 0.5, label, ha="center", va="center",
            fontsize=9, color="#ffdd57", fontweight="bold")
    ax.text(cx, y0 + 0.2, f"{spatial}", ha="center", va="center",
            fontsize=7.5, color=C_DIM)

    # conv1
    box(ax, cx, y0, bw, bh, C_CONV,
        f"Conv2d({in_ch}→{out_ch}, 3×3)",
        "BN → ReLU", fontsize=7.5)
    # conv2
    box(ax, cx, y1, bw, bh, C_CONV,
        f"Conv2d({out_ch}→{out_ch}, 3×3)",
        "BN", fontsize=7.5)
    # skip
    skip_x = cx + bw/2 + 0.55
    box(ax, skip_x, (y0 + y1)/2, 1.2, bh*0.75, C_SKIP,
        f"Skip 1×1", f"({in_ch}→{out_ch})", fontsize=7, alpha=0.9)

    # residual add circle
    add_x, add_y = cx, y1 - 0.35
    circle = plt.Circle((add_x, add_y), 0.18, color="#f72585",
                         zorder=4, alpha=0.9)
    ax.add_patch(circle)
    ax.text(add_x, add_y, "+", ha="center", va="center",
            fontsize=11, color="white", fontweight="bold", zorder=5)

    # MaxPool + Dropout
    box(ax, cx, y3, bw, bh*0.75, C_POOL,
        "MaxPool2d(2) + Dropout", fontsize=7.5)

    # Arrows inside block
    arrow(ax, cx, y0 - bh/2, cx, y1 + bh/2)        # conv1 → conv2
    arrow(ax, cx, y1 - bh/2, add_x, add_y + 0.18)  # conv2 → add
    arrow(ax, skip_x - 0.6, (y0+y1)/2,
          add_x + 0.18, add_y, color=C_SKIP)        # skip → add
    arrow(ax, cx, y0 - bh/2 - 0.05,
          skip_x - 0.6, (y0+y1)/2,
          color=C_SKIP, style="-")                   # input → skip
    arrow(ax, add_x, add_y - 0.18, cx, y3 + bh*0.375)  # add → pool

    return y3 - bh*0.375   # bottom y of block


# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(8, 9.6, "CharClassifier — 4-Block ResVGG (5.15M params)",
        ha="center", va="center", fontsize=13, color="white",
        fontweight="bold")
ax.text(8, 9.25, "EMNIST byclass · 62 classes (0–9, A–Z, a–z) · 32×32 grayscale input",
        ha="center", va="center", fontsize=9, color=C_DIM)

# ── Input ─────────────────────────────────────────────────────────────────────
box(ax, 1.3, 8.5, 1.6, 0.5, C_INPUT, "Input", "32×32 × 1ch", fontsize=8)

# ── Four blocks ───────────────────────────────────────────────────────────────
block_specs = [
    (3.2,   8.2, "Block 1",   1,  64, "32×32 → 16×16"),
    (5.8,   8.2, "Block 2",  64, 128, "16×16 →  8×8"),
    (8.9,   8.2, "Block 3", 128, 256, " 8×8  →  4×4"),
    (12.0,  8.2, "Block 4", 256, 512, " 4×4  →  2×2"),
]

prev_x, prev_y = 1.3, 8.25
block_bottoms = []
for (cx, ty, lbl, ic, oc, sp) in block_specs:
    bot = res_block(ax, cx, ty, lbl, ic, oc, sp)
    block_bottoms.append((cx, bot))

# Inter-block arrows
for i in range(len(block_specs) - 1):
    x1 = block_specs[i][0] + 0.8
    x2 = block_specs[i+1][0] - 0.8
    y  = block_bottoms[i][1] + 0.55
    arrow(ax, x1, y, x2, y, lw=1.5)

# Input → Block1
arrow(ax, 1.3 + 0.8, 8.5, block_specs[0][0] - 0.8, 8.5, lw=1.5)

# ── GAP ───────────────────────────────────────────────────────────────────────
gap_x = 14.2
gap_y = 5.8
box(ax, gap_x, gap_y, 1.5, 0.55, C_GAP,
    "GlobalAvgPool", "→ 512-dim", fontsize=8)
arrow(ax, block_specs[-1][0] + 0.8, gap_y, gap_x - 0.75, gap_y, lw=1.5)

# ── FC head ───────────────────────────────────────────────────────────────────
fc1_y = gap_y - 1.1
fc2_y = fc1_y - 1.0
out_y = fc2_y - 1.0

box(ax, gap_x, fc1_y, 1.5, 0.55, C_FC,
    "Linear(512→512)", "ReLU + Dropout(0.5)", fontsize=7.5)
box(ax, gap_x, fc2_y, 1.5, 0.5, C_FC,
    "Linear(512→62)", fontsize=8)
box(ax, gap_x, out_y, 1.5, 0.5, C_OUTPUT,
    "Output logits", "62 classes", fontsize=8)

arrow(ax, gap_x, gap_y - 0.28, gap_x, fc1_y + 0.28, lw=1.5)
arrow(ax, gap_x, fc1_y - 0.28, gap_x, fc2_y + 0.25, lw=1.5)
arrow(ax, gap_x, fc2_y - 0.25, gap_x, out_y + 0.25, lw=1.5)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    (C_CONV,  "Conv2d + BN"),
    (C_SKIP,  "Residual skip (1×1 Conv)"),
    (C_POOL,  "MaxPool + Dropout"),
    (C_GAP,   "Global Avg Pool"),
    (C_FC,    "Fully Connected"),
]
lx, ly = 0.3, 3.2
ax.text(lx, ly + 0.4, "Legend", color="white", fontsize=8,
        fontweight="bold")
for i, (c, label) in enumerate(legend_items):
    rect = FancyBboxPatch((lx, ly - i*0.45 - 0.15), 0.3, 0.28,
                           boxstyle="round,pad=0.02",
                           facecolor=c, edgecolor="white",
                           linewidth=0.5, zorder=3)
    ax.add_patch(rect)
    ax.text(lx + 0.45, ly - i*0.45 - 0.01, label,
            color=C_DIM, fontsize=7.5, va="center")

# ── Parameter count annotation ────────────────────────────────────────────────
ax.text(8, 0.35,
        "Trained on EMNIST byclass · 200k samples · 30 epochs · "
        "OneCycleLR · AdamW · Label smoothing=0.05 · Val acc: 87.86%",
        ha="center", va="center", fontsize=8, color=C_DIM)

plt.tight_layout(pad=0.3)
plt.savefig("docs/architecture.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved: docs/architecture.png")
plt.close()
