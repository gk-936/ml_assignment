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
import argparse

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_ROOT   = os.path.join(PROJECT_ROOT, "results", "plots")

# Colour palette — one colour per vectorizer, consistent across all plots
PALETTE = {
    "word2vec": "#1f77b4", "fasttext": "#ff7f0e", "glove": "#2ca02c",
    "distilbert": "#d62728", "roberta": "#9467bd",
    "bow": "#8c564b", "tfidf": "#e377c2"
}

plt.rcParams.update({
    "font.family"     : "DejaVu Sans",
    "font.size"       : 11,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "figure.dpi"      : 150,
})


def load_results(dataset: str):
    path = os.path.join(PROJECT_ROOT, "results", f"benchmark_{dataset}.json")
    if not os.path.exists(path):
        # fallback to benchmark_results.json if jigsaw and specific file missing
        if dataset == "jigsaw":
            path = os.path.join(PROJECT_ROOT, "results", "benchmark_results.json")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results file not found for dataset '{dataset}' at {path}")
        
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Plot 1 — F1-macro bar chart
# ---------------------------------------------------------------------------
def plot_f1_comparison(results, dataset: str, plots_dir: str):
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

    title = f"F1-Macro Score by Vectorizer\n({dataset.capitalize()} Dataset)"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("F1-Macro Score")
    ax.set_ylim(0, min(1.0, max(f1s) + 0.08))
    ax.set_xlabel("Vectorizer")

    out = os.path.join(plots_dir, "bar_f1_comparison.png")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2 — Grouped bar: all four metrics
# ---------------------------------------------------------------------------
def plot_all_metrics(results, plots_dir: str):
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

    out = os.path.join(plots_dir, "bar_all_metrics.png")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 3 — Trade-off scatter: F1-macro vs total time
# ---------------------------------------------------------------------------
def plot_tradeoff(results, plots_dir: str):
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

    ax.set_title("Performance vs Computational Cost\n(F1-Macro  ×  Total Time)",
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

    out = os.path.join(plots_dir, "scatter_tradeoff.png")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 4 — Heatmap
# ---------------------------------------------------------------------------
def plot_heatmap(results, plots_dir: str):
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

    out = os.path.join(plots_dir, "heatmap_metrics.png")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument(
        "--dataset", choices=["jigsaw", "steam"], default="jigsaw",
        help="Which dataset results to plot (jigsaw or steam)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    try:
        results = load_results(args.dataset)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Run  python experiments/run_benchmark.py --dataset {args.dataset}  first.")
        return

    plots_dir = os.path.join(PLOTS_ROOT, args.dataset)
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Generating plots for {len(results)} vectorizers (Dataset: {args.dataset.upper()}) …\n")
    
    plot_f1_comparison(results, args.dataset, plots_dir)
    plot_all_metrics(results, plots_dir)
    plot_tradeoff(results, plots_dir)
    plot_heatmap(results, plots_dir)
    
    print(f"\nAll plots saved to {plots_dir}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
