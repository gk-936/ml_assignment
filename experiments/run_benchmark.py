"""
experiments/run_benchmark.py
------------------------------
Main experiment script. Runs all five vectorizers against the same
dataset, classifier, and evaluation protocol — saves results to JSON.

Vectorizers
-----------
    Word2Vec   — Skip-gram, trained on corpus
    FastText   — Skip-gram + subword n-grams, trained on corpus
    GloVe      — Pretrained Stanford vectors
    DistilBERT — Distilled BERT, frozen [CLS] token (66M params, ~2x faster)
    RoBERTa    — Optimised BERT retraining, frozen [CLS] token (125M params)

Experiment protocol
-------------------
1.  Load  data/processed/jigsaw_binary.csv
2.  Train/test split: 80/20, stratified, random_state=42
3.  For each vectorizer:
    a. Time the fit() call on train set
    b. Time the transform() call on train + test sets
    c. Train LogisticRegression (class_weight='balanced') on train vectors
    d. Evaluate on test vectors → accuracy, precision, recall, F1
4.  Save all results to results/benchmark_results.json
5.  Print formatted comparison table

Usage
-----
    python experiments/run_benchmark.py
    python experiments/run_benchmark.py --vectorizers word2vec fasttext glove
    python experiments/run_benchmark.py --vectorizers word2vec fasttext glove distilbert roberta
"""

import argparse
import json
import os
import sys
import time
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from vectorizers.word2vec_vectorizer  import Word2VecVectorizer
from vectorizers.fasttext_vectorizer  import FastTextVectorizer
from vectorizers.glove_vectorizer     import GloVeVectorizer
from vectorizers.contextual_vectorizer import ContextualVectorizer
from vectorizers.bow_vectorizer        import BoWVectorizer
from vectorizers.tfidf_vectorizer      import TFIDFVectorizer

DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "processed", "jigsaw_binary.csv")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "benchmark_results.json")
RANDOM_STATE = 42
TEST_SIZE    = 0.20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(path: str):
    df = pd.read_csv(path)
    # Ensure columns exist; Steam uses 'text' and 'label' as well.
    return df["text"].tolist(), df["label"].tolist()


def evaluate(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy"    : round(accuracy_score(y_true, y_pred), 4),
        "precision"   : round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall"      : round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_macro"    : round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_weighted" : round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
    }


def run_vectorizer(
    name: str,
    vectorizer,
    X_train: List[str],
    X_test: List[str],
    y_train: List[int],
    y_test: List[int],
    target_names: List[str],
) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    print(f"{'='*60}")

    # Fit
    t0 = time.perf_counter()
    vectorizer.fit(X_train)
    fit_time = time.perf_counter() - t0
    print(f"  Fit time    : {fit_time:.2f}s")

    # Transform
    t0 = time.perf_counter()
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)
    transform_time = time.perf_counter() - t0
    print(f"  Transform   : {transform_time:.2f}s  |  shape: {X_train_vec.shape}")

    # Classify
    clf = LogisticRegression(
        max_iter     = 1000,
        class_weight = "balanced",
        random_state = RANDOM_STATE,
        C            = 1.0,
    )
    t0 = time.perf_counter()
    clf.fit(X_train_vec, y_train)
    train_time = time.perf_counter() - t0

    # Evaluate
    y_pred  = clf.predict(X_test_vec)
    metrics = evaluate(y_test, y_pred)

    print(f"\n  Classification report:\n")
    print(classification_report(y_test, y_pred, target_names=target_names))

    return {
        "vectorizer"              : name,
        "embedding_dim"           : int(X_train_vec.shape[1]),
        "fit_time_s"              : round(fit_time, 3),
        "transform_time_s"        : round(transform_time, 3),
        "classifier_train_time_s" : round(train_time, 3),
        **metrics,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run word vectorizer benchmark")
    parser.add_argument(
        "--dataset", choices=["jigsaw", "steam"], default="jigsaw",
        help="Which dataset to use (jigsaw or steam)"
    )
    parser.add_argument(
        "--vectorizers", nargs="+",
        choices=["word2vec", "fasttext", "glove", "distilbert", "roberta", "bow", "tfidf"],
        default=["word2vec", "fasttext", "glove", "distilbert", "roberta", "bow", "tfidf"],
        help="Which vectorizers to run (default: all seven)",
    )
    return parser.parse_args()


def build_vectorizers(names: List[str]):
    registry = {
        "word2vec"  : lambda: Word2VecVectorizer(vector_size=100, epochs=10),
        "fasttext"  : lambda: FastTextVectorizer(vector_size=100, epochs=10),
        "glove"     : lambda: GloVeVectorizer(vector_size=100),
        "distilbert": lambda: ContextualVectorizer(model_name="distilbert", batch_size=32, max_length=128),
        "roberta"   : lambda: ContextualVectorizer(model_name="roberta",    batch_size=32, max_length=128),
        "bow"       : lambda: BoWVectorizer(max_features=10000),
        "tfidf"     : lambda: TFIDFVectorizer(max_features=10000),
    }
    return [(name, registry[name]()) for name in names]


def main():
    args = parse_args()

    # Dynamic paths and labels based on dataset
    if args.dataset == "jigsaw":
        data_path = os.path.join(PROJECT_ROOT, "data", "processed", "jigsaw_binary.csv")
        results_path = os.path.join(PROJECT_ROOT, "results", f"benchmark_jigsaw.json")
        target_names = ["non-toxic", "toxic"]
    else:  # steam
        data_path = os.path.join(PROJECT_ROOT, "data", "processed", "steam_binary.csv")
        results_path = os.path.join(PROJECT_ROOT, "results", f"benchmark_steam.json")
        target_names = ["negative", "positive"]

    print(f"\n[run_benchmark] Dataset: {args.dataset.upper()}")
    print(f"[run_benchmark] Loading dataset …")
    
    if not os.path.exists(data_path):
        prep_script = "data/prepare_dataset.py" if args.dataset == "jigsaw" else "data/scrape_steam.py"
        raise FileNotFoundError(
            f"Processed dataset not found at {data_path}.\n"
            f"Run  python {prep_script}  first."
        )

    texts, labels = load_data(data_path)
    print(f"  Total samples: {len(texts):,}")
    print(f"  Positive/Toxic: {sum(labels):,}  ({sum(labels)/len(labels):.1%})")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size    = TEST_SIZE,
        stratify     = labels,
        random_state = RANDOM_STATE,
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    vectorizers = build_vectorizers(args.vectorizers)

    all_results = []
    for name, vec in vectorizers:
        result = run_vectorizer(name, vec, X_train, X_test, y_train, y_test, target_names)
        all_results.append(result)

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[run_benchmark] Results saved → {results_path}")

    # Summary table
    print("\n" + "="*72)
    print(f"  {'Vectorizer':<12} {'Dim':>5} {'Fit(s)':>8} {'Trans(s)':>10} "
          f"{'Acc':>7} {'F1-macro':>10} {'F1-wtd':>8}")
    print("-"*72)
    for r in all_results:
        print(
            f"  {r['vectorizer']:<12} {r['embedding_dim']:>5} "
            f"{r['fit_time_s']:>8.1f} {r['transform_time_s']:>10.1f} "
            f"{r['accuracy']:>7.4f} {r['f1_macro']:>10.4f} {r['f1_weighted']:>8.4f}"
        )
    print("="*72)


if __name__ == "__main__":
    main()
