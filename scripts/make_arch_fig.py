"""System architecture figure, laid out for a single IEEE column."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = ROOT / "paper" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7,
})

FILL, FILL_HI, EDGE = "#f2f2f2", "#cfcfcf", "#1a1a1a"


def box(ax, x, y, w, h, label, sub=None, hi=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.03",
        linewidth=0.8, edgecolor=EDGE, facecolor=FILL_HI if hi else FILL))
    ax.text(x + w / 2, y + h * (0.62 if sub else 0.5), label,
            ha="center", va="center", fontsize=7.2)
    if sub:
        ax.text(x + w / 2, y + h * 0.25, sub, ha="center", va="center",
                fontsize=5.9, color="#3d3d3d")


def arrow(ax, x1, y1, x2, y2, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=6, linewidth=0.8,
                                 color=EDGE, shrinkA=0, shrinkB=0))


fig, ax = plt.subplots(figsize=(3.4, 2.75))
ax.set_xlim(0, 6.0)
ax.set_ylim(0, 5.0)
ax.axis("off")

h = 0.60
# row 1
box(ax, 0.05, 4.20, 1.75, h, "Documents", "PDF / DOCX / ZIP")
box(ax, 2.30, 4.20, 1.75, h, "Text layer", "pypdf, PyMuPDF")
box(ax, 4.20, 4.20, 1.70, h, "OCR fallback", "cached")
arrow(ax, 1.80, 4.50, 2.30, 4.50)
arrow(ax, 4.05, 4.42, 4.20, 4.42)
arrow(ax, 4.20, 4.62, 4.05, 4.62)

# row 2
box(ax, 0.05, 3.10, 2.60, h, "Structured extraction", "skills, YOE, degree")
box(ax, 3.20, 3.10, 2.70, h, "Bi-encoder retrieval", "MiniLM + FAISS")
arrow(ax, 3.17, 4.20, 1.35, 3.70)
arrow(ax, 2.65, 3.40, 3.20, 3.40)

# row 3
box(ax, 0.05, 2.00, 2.60, h, "Cross-encoder", "re-rank top 30")
box(ax, 3.20, 2.00, 2.70, h, "Adaptive weighting", "active components only")
arrow(ax, 4.55, 3.10, 1.35, 2.60)
arrow(ax, 2.65, 2.30, 3.20, 2.30)

# row 4: the contribution
box(ax, 0.60, 0.80, 4.80, 0.66,
    "Rendering-layer integrity verification",
    "per-glyph colour, size, render mode", hi=True)
arrow(ax, 0.92, 4.20, 1.60, 1.46)          # raw bytes in
arrow(ax, 4.60, 1.46, 4.55, 2.00)          # penalty out

ax.text(0.05, 4.92, "Conventional text-layer pipeline",
        fontsize=6.8, style="italic")
ax.text(0.60, 0.46, "Proposed addition (reads original bytes)",
        fontsize=6.4, style="italic")

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_architecture.{ext}", bbox_inches="tight",
                dpi=600 if ext == "png" else None)
print("wrote fig_architecture (single column)")
