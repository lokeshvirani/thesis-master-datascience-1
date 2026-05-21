"""Draw the grounded-reconstruction method pipeline as a figure for the thesis.

Two rows: (top) generation path real->caption->inpaint->generated; (bottom)
analysis path patches->features->t-SNE. Saves results/method/pipeline.png.

Run anywhere with matplotlib (no GPU/data needed):
    python src/analysis/plot_pipeline.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "results/method/pipeline.png"

C_GOOD = "#dff0d8"; C_BAD = "#f2dede"; C_MASK = "#e7e7e7"
C_LLM = "#d9e8fb"; C_GEN = "#fde9d9"; C_FEAT = "#e8e0f5"; C_OUT = "#fff3cd"
EC = "#3b3b3b"


def box(ax, x, y, w, h, title, sub="", fc="#eef"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.4, edgecolor=EC, facecolor=fc))
    ax.text(x + w / 2, y + h / 2 + (0.16 if sub else 0), title,
            ha="center", va="center", fontsize=10, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.24, sub, ha="center", va="center",
                fontsize=8, color="#333")


def arrow(ax, x0, y0, x1, y1, text="", rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=15, lw=1.5,
        color="#444", connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.2, text, ha="center",
                fontsize=8, color="#444", style="italic")


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.5, 7))
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")

    # ---- top row: generation path ----
    box(ax, 0.3, 6.3, 2.3, 1.0, "Bad image", "real defect", C_BAD)
    box(ax, 0.3, 4.5, 2.3, 1.0, "Good image", "clean / normal", C_GOOD)
    box(ax, 0.3, 2.7, 2.3, 1.0, "GT mask", "defect location", C_MASK)

    box(ax, 3.4, 6.3, 2.5, 1.0, "Stage 1", "Vision LLM caption", C_LLM)
    box(ax, 6.7, 6.3, 3.0, 1.0, "Caption", "“holographic foil patch…”", C_OUT)

    box(ax, 10.4, 4.6, 2.7, 1.6, "Stage 2", "SDXL inpainting", C_LLM)
    box(ax, 13.4, 4.9, 2.3, 1.0, "Generated", "synthetic defect", C_GEN)

    arrow(ax, 2.6, 6.8, 3.4, 6.8)                                  # bad -> LLM
    arrow(ax, 5.9, 6.8, 6.7, 6.8)                                  # LLM -> caption
    arrow(ax, 9.7, 6.6, 10.6, 6.1, "what")                        # caption -> inpaint
    arrow(ax, 2.6, 5.0, 10.4, 5.5, "base (clean object)")        # good -> inpaint
    arrow(ax, 2.6, 3.2, 10.4, 4.9, "where (mask)")               # mask -> inpaint
    arrow(ax, 13.1, 5.4, 13.4, 5.4)                               # inpaint -> generated

    # ---- bottom row: analysis path ----
    box(ax, 0.3, 0.6, 3.3, 1.3,
        "Patches", "real · normal · generated", C_GOOD)
    box(ax, 4.4, 0.6, 3.6, 1.3,
        "Stage 3", "WideResNet50 L2+L3 (1536-D)", C_FEAT)
    box(ax, 8.8, 0.6, 3.0, 1.3, "Stage 4", "PCA(50) → t-SNE", C_FEAT)
    box(ax, 12.6, 0.6, 3.1, 1.3, "t-SNE figure", "real vs gen vs normal", C_OUT)

    arrow(ax, 3.6, 1.25, 4.4, 1.25)
    arrow(ax, 8.0, 1.25, 8.8, 1.25)
    arrow(ax, 11.8, 1.25, 12.6, 1.25)

    # generated -> analysis patches (routed curve)
    arrow(ax, 14.5, 4.9, 2.0, 1.9, "generated images", rad=0.18)

    fig.suptitle("Grounded reconstruction pipeline: real → caption → "
                 "regenerate → compare in feature space",
                 fontsize=13, fontweight="bold", y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
