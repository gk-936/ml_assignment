"""
t-SNE plots of each vectorizer's embedding space.
Shows whether toxic and non-toxic comments separate in the embedding space.

Runs on a 2000 sample subset (t-SNE is slow on large datasets).
Saves individual plots + a combined 2x2 figure to results/plots/.

Usage: python analysis/tsne_visualizer.py
"""

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from vectorizers.word2vec_vectorizer import Word2VecVectorizer
from vectorizers.fasttext_vectorizer import FastTextVectorizer
from vectorizers.glove_vectorizer    import GloVeVectorizer
from vectorizers.contextual_vectorizer     import ContextualVectorizer

PLOTS_ROOT  = os.path.join(PROJECT_ROOT, "results", "plots")

# Use a smaller subset just for t-SNE — it's O(n²) in memory
TSNE_SAMPLES    = 2_000
RANDOM_STATE    = 42
TSNE_PERPLEXITY = 40

PALETTE_COLORS = {0: "#4C72B0", 1: "#C44E52"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size"  : 10,
    "figure.dpi" : 150,
})


def load_subset(path, n=TSNE_SAMPLES):
    # load equal samples from each class for t-SNE
    df = pd.read_csv(path)
    # Ensure there are enough samples
    counts = df["label"].value_counts()
    n_per_class = min(n // 2, counts.min())
    
    return (
        df.groupby("label")
          .sample(n=n_per_class, random_state=RANDOM_STATE)
          .sample(frac=1, random_state=RANDOM_STATE)
          .reset_index(drop=True)
    )


def compute_tsne(vectors: np.ndarray) -> np.ndarray:
    """Reduce vectors to 2D using t-SNE."""
    print(f"  Computing t-SNE on {vectors.shape} …")
    tsne = TSNE(
        n_components = 2,
        perplexity   = TSNE_PERPLEXITY,
        random_state = RANDOM_STATE,
        n_iter       = 1000,
        init         = "pca",   # PCA init is faster and more stable than random
        learning_rate= "auto",
    )
    return tsne.fit_transform(vectors)


def plot_tsne(embedded, labels, title, out_path, target_labels):
    """Save a scatter plot of the 2D embeddings coloured by class."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for cls, color in PALETTE_COLORS.items():
        mask = labels == cls
        ax.scatter(
            embedded[mask, 0], embedded[mask, 1],
            c=color, s=8, alpha=0.5,
            label=target_labels[cls],
        )
    ax.set_title(f"t-SNE: {title}", fontsize=12, fontweight="bold", pad=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="t-SNE visualization")
    parser.add_argument(
        "--dataset", choices=["jigsaw", "steam"], default="jigsaw",
        help="Which dataset to visualize (jigsaw or steam)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.dataset == "jigsaw":
        data_path = os.path.join(PROJECT_ROOT, "data", "processed", "jigsaw_binary.csv")
        target_labels = ["Non-toxic", "Toxic"]
    else:  # steam
        data_path = os.path.join(PROJECT_ROOT, "data", "processed", "steam_binary.csv")
        target_labels = ["Negative", "Positive"]

    if not os.path.exists(data_path):
        prep_script = "data/prepare_dataset.py" if args.dataset == "jigsaw" else "data/scrape_steam.py"
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run python {prep_script} first.")

    plots_dir = os.path.join(PLOTS_ROOT, args.dataset)
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Loading subset from {args.dataset.upper()} for t-SNE …")
    df     = load_subset(data_path)
    texts  = df["text"].tolist()
    labels = df["label"].values

    vectorizers = [
        ("Word2Vec", Word2VecVectorizer(vector_size=100, epochs=10)),
        ("FastText", FastTextVectorizer(vector_size=100, epochs=10)),
        ("GloVe",    GloVeVectorizer(vector_size=100)),
        ("BERT",     ContextualVectorizer(batch_size=32, max_length=128)),
    ]

    # fit each vectorizer and compute t-SNE on the embeddings
    for name, vec in vectorizers:
        print(f"\n{'='*50}")
        print(f"  Vectorizer: {name}")
        print(f"{'='*50}")
        vectors  = vec.fit(texts).transform(texts)
        embedded = compute_tsne(vectors)
        out_path = os.path.join(plots_dir, f"tsne_{name.lower()}.png")
        plot_tsne(embedded, labels, title=name, out_path=out_path, target_labels=target_labels)

    # stitch individual plots into one combined figure
    print("\nGenerating combined 2×2 t-SNE figure …")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes      = axes.flatten()
    tsne_files = [
        ("Word2Vec", os.path.join(plots_dir, "tsne_word2vec.png")),
        ("FastText", os.path.join(plots_dir, "tsne_fasttext.png")),
        ("GloVe",    os.path.join(plots_dir, "tsne_glove.png")),
        ("BERT",     os.path.join(plots_dir, "tsne_bert.png")),
    ]
    import matplotlib.image as mpimg
    for ax, (name, fpath) in zip(axes, tsne_files):
        if os.path.exists(fpath):
            ax.imshow(mpimg.imread(fpath))
            ax.set_title(name, fontsize=13, fontweight="bold")
        ax.axis("off")

    fig.suptitle(f"t-SNE Embedding Spaces: {args.dataset.capitalize()} Dataset",
                 fontsize=15, fontweight="bold", y=1.01)
    out = os.path.join(plots_dir, "tsne_combined.png")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
