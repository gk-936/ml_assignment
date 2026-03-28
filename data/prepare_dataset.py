"""
prepare_dataset.py
------------------
Loads the Jigsaw Toxic Comment dataset from Kaggle, subsamples it to a
manageable size while preserving class balance, and saves a clean CSV.

Dataset: Jigsaw Toxic Comment Classification Challenge
Source  : https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
License : CC0 (public domain)

How to get the raw data
-----------------------
Option A (recommended):
    kaggle competitions download -c jigsaw-toxic-comment-classification-challenge
    unzip the archive into  data/raw/

Option B (manual):
    Download train.csv from Kaggle and place it at  data/raw/train.csv

The script produces  data/processed/jigsaw_binary.csv  with columns:
    text  (str)  – cleaned raw comment text
    label (int)  – 1 = toxic, 0 = non-toxic
"""

import os
import pandas as pd
from sklearn.utils import resample

RAW_PATH       = os.path.join(os.path.dirname(__file__), "raw", "train.csv")
PROCESSED_DIR  = os.path.join(os.path.dirname(__file__), "processed")
OUTPUT_PATH    = os.path.join(PROCESSED_DIR, "jigsaw_binary.csv")

# How many samples per class to keep.
# 10 000 each → 20 000 total.  Plenty for meaningful benchmarks.
SAMPLES_PER_CLASS = 10_000
RANDOM_STATE      = 42


def load_and_binarise(path: str) -> pd.DataFrame:
    """Read raw Jigsaw CSV and collapse six toxic labels into one binary label."""
    df = pd.read_csv(path)
    toxic_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    df["label"] = (df[toxic_cols].sum(axis=1) > 0).astype(int)
    return df[["comment_text", "label"]].rename(columns={"comment_text": "text"})


def balanced_subsample(df: pd.DataFrame, n: int, random_state: int) -> pd.DataFrame:
    """Return n samples from each class, concatenated and shuffled."""
    majority = df[df["label"] == 0]
    minority = df[df["label"] == 1]

    maj_sample = resample(majority, n_samples=n, random_state=random_state, replace=False)
    # Minority class may have fewer than n rows — sample with replace if needed
    replace_flag = len(minority) < n
    min_sample = resample(minority, n_samples=n, random_state=random_state, replace=replace_flag)

    return (
        pd.concat([maj_sample, min_sample])
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )


def main():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"Raw data not found at {RAW_PATH}.\n"
            "Download train.csv from Kaggle and place it at data/raw/train.csv"
        )

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Loading raw data …")
    df = load_and_binarise(RAW_PATH)
    print(f"  Total rows : {len(df):,}")
    print(f"  Toxic      : {df['label'].sum():,}  ({df['label'].mean():.1%})")

    print(f"Subsampling to {SAMPLES_PER_CLASS:,} per class …")
    df_balanced = balanced_subsample(df, SAMPLES_PER_CLASS, RANDOM_STATE)
    print(f"  Final size : {len(df_balanced):,}")

    df_balanced.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
