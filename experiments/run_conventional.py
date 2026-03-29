"""
run_conventional.py
====================
Purpose:
    Experiment runner for the conventional word vectorization methods:
    Bag-of-Words (BoW) and TF-IDF. Loads the processed Jigsaw dataset,
    trains each vectorizer + Logistic Regression classifier, evaluates
    on the held-out test set, and saves all results to
    results/conventional_results.json.

    This script is one half of the combined benchmark — your partner's
    run_benchmark.py covers Word2Vec, FastText, GloVe, and BERT. Both
    JSON files are merged by analysis/merge_results.py.

Usage:
    python experiments/run_conventional.py
"""

import json
import os
import time
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from vectorizers.bow_vectorizer import BoWVectorizer
from vectorizers.tfidf_vectorizer import TFIDFVectorizer

# ── Constants ────────────────────────────────────────────────────────────────
DATA_PATH = "data/processed/jigsaw_binary.csv"
RESULTS_PATH = "results/conventional_results.json"
TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_data(path: str):
    """
    Load the processed Jigsaw binary classification dataset.

    Parameters
    ----------
    path : str
        Path to jigsaw_binary.csv (columns: text, label).

    Returns
    -------
    tuple
        (texts: list[str], labels: np.ndarray)
    """
    print(f"[Data] Loading dataset from {path}...")
    df = pd.read_csv(path)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].values
    print(f"[Data] {len(texts)} samples loaded. "
          f"Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    return texts, labels


def run_experiment(
    vectorizer,
    X_train: list,
    X_test: list,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """
    Run a single vectorizer experiment end-to-end.

    Fits the vectorizer on training data, transforms both splits,
    trains a Logistic Regression classifier, evaluates on the test set,
    and returns a results dict compatible with benchmark_results.json.

    Parameters
    ----------
    vectorizer : BaseVectorizer
        An unfitted BoWVectorizer or TFIDFVectorizer instance.
    X_train : list[str]
        Raw training texts.
    X_test : list[str]
        Raw test texts.
    y_train : np.ndarray
        Training labels.
    y_test : np.ndarray
        Test labels.

    Returns
    -------
    dict
        Results dict with keys matching the shared JSON schema.
    """
    name = vectorizer.name
    print(f"\n{'='*60}")
    print(f"[{name}] Starting experiment")
    print(f"{'='*60}")

    # ── Vectorizer fit ────────────────────────────────────────────────
    t0 = time.perf_counter()
    vectorizer.fit(X_train)
    fit_time = time.perf_counter() - t0
    print(f"[{name}] fit() completed in {fit_time:.3f}s")

    # ── Vectorizer transform ──────────────────────────────────────────
    t0 = time.perf_counter()
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    transform_time = time.perf_counter() - t0
    print(f"[{name}] transform() (train+test) completed in {transform_time:.3f}s")
    print(f"[{name}] Train matrix shape: {X_train_vec.shape}")

    # ── Classifier ───────────────────────────────────────────────────
    print(f"[{name}] Training Logistic Regression...")
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        C=1.0,
    )
    t0 = time.perf_counter()
    clf.fit(X_train_vec, y_train)
    clf_train_time = time.perf_counter() - t0
    print(f"[{name}] Classifier trained in {clf_train_time:.3f}s")

    # ── Evaluation ───────────────────────────────────────────────────
    y_pred = clf.predict(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n[{name}] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["non-toxic", "toxic"]))

    return {
        "vectorizer": name,
        "embedding_dim": vectorizer.max_features,
        "fit_time_s": round(fit_time, 4),
        "transform_time_s": round(transform_time, 4),
        "classifier_train_time_s": round(clf_train_time, 4),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
    }


def print_summary_table(results: list) -> None:
    """
    Print a formatted summary table of all results to stdout.

    Parameters
    ----------
    results : list[dict]
        List of result dicts produced by run_experiment().
    """
    header = f"\n{'Vectorizer':<12} {'Dim':>6} {'Fit(s)':>8} {'Xfm(s)':>8} " \
             f"{'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1-Mac':>8} {'F1-Wt':>8}"
    print("\n" + "=" * 80)
    print("CONVENTIONAL METHODS — SUMMARY")
    print("=" * 80)
    print(header)
    print("-" * 80)
    for r in results:
        print(
            f"{r['vectorizer']:<12} {r['embedding_dim']:>6} "
            f"{r['fit_time_s']:>8.3f} {r['transform_time_s']:>8.3f} "
            f"{r['accuracy']:>8.4f} {r['precision']:>8.4f} "
            f"{r['recall']:>8.4f} {r['f1_macro']:>8.4f} "
            f"{r['f1_weighted']:>8.4f}"
        )
    print("=" * 80)
    print("PRIMARY METRIC: F1-Macro (treats both classes equally)")
    print("=" * 80 + "\n")


def main():
    # ── Load data ────────────────────────────────────────────────────
    texts, labels = load_data(DATA_PATH)

    # ── Train / test split ───────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=RANDOM_STATE,
    )
    print(f"[Data] Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    # ── Run experiments ───────────────────────────────────────────────
    vectorizers = [
        BoWVectorizer(max_features=10000, ngram_range=(1, 2)),
        TFIDFVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True, min_df=2),
    ]

    all_results = []
    for vec in vectorizers:
        result = run_experiment(vec, X_train, X_test, y_train, y_test)
        all_results.append(result)

    # ── Print summary ────────────────────────────────────────────────
    print_summary_table(all_results)

    # ── Save results ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[Results] Saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
