"""
Generates plots from benchmark_results.json.
Run this after run_benchmark.py finishes.

Saves to results/plots/:
    bar_f1_comparison.png
    bar_all_metrics.png
    scatter_tradeoff.png
    heatmap_metrics.png
"""

import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "benchmark_results.json")
PLOTS_DIR    = os.path.join(PROJECT_ROOT, "results", "plots")

# Colour palette — one colour per vectorizer, consistent across all plots
PALETTE = {
    "word2vec": "#4C72B0",
    "fasttext": "#DD8452",
    "glove"   : "#55A868",
    "bert"    : "#C44E52",
}

plt.rcParams.update({
    "font.family"   : "DejaVu Sans",
    "font.size"     : 11,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "figure.dpi"    : 150,
})


def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Plot 1 — F1-macro bar chart
# ---------------------------------------------------------------------------
def plot_f1_comparison(results):
    # bar chart of F1-macro scores across all vectorizers
    names  = [r["vectorizer"] for r in results]
    f1s    = [r["f1_macro"] for r in results]
    colors = [PALETTE.get(n, "#888") for n in names]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, f1s, color=colors, width=0.55, edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, f1s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    ax.set_title("F1-Macro Score by Vectorizer\n(Toxic Comment Classification – Jigsaw Dataset)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("F1-Macro Score")
    ax.set_ylim(0, min(1.0, max(f1s) + 0.08))
    ax.set_xlabel("Vectorizer")

    out = os.path.join(PLOTS_DIR, "bar_f1_comparison.png")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2 — Grouped bar: all four metrics
# ---------------------------------------------------------------------------
def plot_all_metrics(results):
    # grouped bar chart showing accuracy, precision, recall, f1 side by side
    metrics     = ["accuracy", "precision", "recall", "f1_macro"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Macro"]
    names       = [r["vectorizer"] for r in results]
    x           = np.arange(len(metrics))
    width       = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (name, r) in enumerate(zip(names, results)):
        vals   = [r[m] for m in metrics]
        offset = (i - (len(results) - 1) / 2) * width
        bars   = ax.bar(x + offset, vals, width, label=name.capitalize(),
                        color=PALETTE.get(name, "#888"), edgecolor="white")

    ax.set_title("Classification Metrics by Vectorizer",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Vectorizer", frameon=False)

    out = os.path.join(PLOTS_DIR, "bar_all_metrics.png")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 3 — Trade-off scatter: F1-macro vs total time
# ---------------------------------------------------------------------------
def plot_tradeoff(results):
    # scatter plot of F1 vs total time - the main trade-off visual
    fig, ax = plt.subplots(figsize=(7, 5))

    for r in results:
        name      = r["vectorizer"]
        total_t   = r["fit_time_s"] + r["transform_time_s"]
        f1        = r["f1_macro"]
        color     = PALETTE.get(name, "#888")

        ax.scatter(total_t, f1, s=160, color=color, zorder=5, edgecolors="white", linewidths=1.5)
        ax.annotate(
            name.capitalize(),
            (total_t, f1),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=10,
            color=color,
            fontweight="bold",
        )

    ax.set_title("Performance vs Computational Cost\n(F1-Macro  ×  Fit + Transform Time)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Total Time (fit + transform) — seconds")
    ax.set_ylabel("F1-Macro Score")

    # Annotate ideal quadrant
    ax.text(
        0.02, 0.97, "← Faster, Better →",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=9, color="grey", style="italic"
    )

    out = os.path.join(PLOTS_DIR, "scatter_tradeoff.png")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 4 — Heatmap
# ---------------------------------------------------------------------------
def plot_heatmap(results):
    # heatmap of all metrics, useful for spotting patterns at a glance
    import matplotlib.colors as mcolors

    metrics = ["accuracy", "precision", "recall", "f1_macro", "f1_weighted"]
    names   = [r["vectorizer"].capitalize() for r in results]
    data    = np.array([[r[m] for m in metrics] for r in results])

    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0.5, vmax=1.0)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1-Macro", "F1-Wtd"],
                       rotation=30, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)

    for i in range(len(names)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                    fontsize=9, color="black" if data[i, j] < 0.85 else "white",
                    fontweight="bold")

    ax.set_title("Metrics Heatmap — Vectorizer Comparison",
                 fontsize=12, fontweight="bold", pad=10)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Score")

    out = os.path.join(PLOTS_DIR, "heatmap_metrics.png")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(
            f"Results file not found at {RESULTS_PATH}.\n"
            "Run  python experiments/run_benchmark.py  first."
        )

    os.makedirs(PLOTS_DIR, exist_ok=True)
    results = load_results()

    print(f"Generating plots for {len(results)} vectorizers …\n")
    # run all four plots
    plot_f1_comparison(results)
    plot_all_metrics(results)
    plot_tradeoff(results)
    plot_heatmap(results)
    print(f"\nAll plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
