"""
merge_results.py
=================
Purpose:
    Combines conventional_results.json (BoW + TF-IDF, produced by
    experiments/run_conventional.py) with benchmark_results.json
    (Word2Vec, FastText, GloVe, BERT, produced by partner's
    experiments/run_benchmark.py) into a single comparison table.

    Outputs:
        results/combined_results.json   — merged JSON for further analysis
        Prints a formatted comparison table to stdout

Usage:
    python analysis/merge_results.py

    Both JSON files must exist before running this script. Run your
    partner's run_benchmark.py and your run_conventional.py first.
"""

import json
import os

CONVENTIONAL_PATH = "results/conventional_results.json"
BENCHMARK_PATH = "results/benchmark_results.json"
COMBINED_PATH = "results/combined_results.json"


def load_json(path: str) -> list:
    """
    Load a JSON results file.

    Parameters
    ----------
    path : str
        Path to the JSON file.

    Returns
    -------
    list[dict]
        List of result dicts.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Results file not found: {path}\n"
            f"Make sure you have run the corresponding experiment script first."
        )
    with open(path, "r") as f:
        return json.load(f)


def print_comparison_table(results: list) -> None:
    """
    Print a formatted comparison table for all six vectorizers.

    Parameters
    ----------
    results : list[dict]
        Combined list of result dicts (all vectorizers).
    """
    # Define display order
    order = ["BoW", "TF-IDF", "Word2Vec", "FastText", "GloVe", "BERT"]
    result_map = {r["vectorizer"]: r for r in results}

    header = (
        f"\n{'Vectorizer':<12} {'Type':<12} {'Dim':>6} "
        f"{'Fit(s)':>8} {'Xfm(s)':>8} "
        f"{'Accuracy':>10} {'F1-Macro':>10} {'F1-Weighted':>12}"
    )
    type_map = {
        "BoW": "Sparse",
        "TF-IDF": "Sparse",
        "Word2Vec": "Dense DL",
        "FastText": "Subword DL",
        "GloVe": "Dense DL",
        "BERT": "Contextual",
    }

    print("\n" + "=" * 82)
    print("COMBINED BENCHMARK RESULTS — ALL SIX VECTORIZERS")
    print("=" * 82)
    print(header)
    print("-" * 82)

    for name in order:
        if name not in result_map:
            vtype = type_map.get(name, "—")
            print(f"{name:<12} {vtype:<12} {'—':>6} {'—':>8} {'—':>8} "
                  f"{'—':>10} {'—':>10} {'—':>12}  ← results not available")
            continue
        r = result_map[name]
        vtype = type_map.get(name, "—")
        print(
            f"{r['vectorizer']:<12} {vtype:<12} {r['embedding_dim']:>6} "
            f"{r['fit_time_s']:>8.3f} {r['transform_time_s']:>8.3f} "
            f"{r['accuracy']:>10.4f} {r['f1_macro']:>10.4f} "
            f"{r['f1_weighted']:>12.4f}"
        )

    print("=" * 82)
    print("PRIMARY METRIC: F1-Macro  |  Classifier: Logistic Regression (all methods)")
    print("=" * 82 + "\n")


def main():
    print("[Merge] Loading conventional results (BoW, TF-IDF)...")
    conventional = load_json(CONVENTIONAL_PATH)

    print("[Merge] Loading benchmark results (Word2Vec, FastText, GloVe, BERT)...")
    try:
        benchmark = load_json(BENCHMARK_PATH)
    except FileNotFoundError as e:
        print(f"[Merge] WARNING: {e}")
        print("[Merge] Proceeding with conventional results only.")
        benchmark = []

    combined = conventional + benchmark
    print(f"[Merge] Combined {len(combined)} vectorizer results.")

    # Print comparison table
    print_comparison_table(combined)

    # Save combined JSON
    os.makedirs(os.path.dirname(COMBINED_PATH), exist_ok=True)
    with open(COMBINED_PATH, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"[Merge] Combined results saved to {COMBINED_PATH}")


if __name__ == "__main__":
    main()
