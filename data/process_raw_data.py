import os
import pandas as pd

RAW_DIR       = os.path.join(os.path.dirname(__file__), "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")
RAW_IN        = os.path.join(RAW_DIR, "steam_reviews_raw.csv")
PROCESSED_OUT = os.path.join(PROCESSED_DIR, "steam_binary.csv")

SAMPLES_PER_CLASS = 20000
RANDOM_STATE      = 42
MIN_LEN           = 20
MAX_LEN           = 512

def main():
    if not os.path.exists(RAW_IN):
        print(f"Error: Raw data file not found at {RAW_IN}")
        return

    print(f"Loading raw data from {RAW_IN}...")
    df = pd.read_csv(RAW_IN)
    print(f"Loaded {len(df):,} total samples.")

    # basic cleanup (though scraper should have done this)
    df = df.dropna(subset=['text'])
    df['text_len'] = df['text'].astype(str).apply(len)
    df = df[(df['text_len'] >= MIN_LEN) & (df['text_len'] <= MAX_LEN)]
    print(f"After length filtering ({MIN_LEN}-{MAX_LEN}): {len(df):,} samples.")

    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]

    print(f"Total Positive: {len(pos):,}")
    print(f"Total Negative: {len(neg):,}")

    if len(pos) < SAMPLES_PER_CLASS or len(neg) < SAMPLES_PER_CLASS:
        print(f"Warning: Not enough samples to reach {SAMPLES_PER_CLASS} per class.")
        n = min(len(pos), len(neg))
        print(f"Balancing to {n:,} per class instead.")
    else:
        n = SAMPLES_PER_CLASS
        print(f"Balancing to {n:,} per class.")

    pos_sample = pos.sample(n=n, random_state=RANDOM_STATE)
    neg_sample = neg.sample(n=n, random_state=RANDOM_STATE)

    df_out = (
        pd.concat([pos_sample, neg_sample])
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    # select only required columns
    df_out = df_out[["text", "label"]]

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df_out.to_csv(PROCESSED_OUT, index=False)

    print(f"Successfully processed and saved to {PROCESSED_OUT}")
    print(f"Final Count: {len(df_out):,} samples.")
    print(f"  Pos: {len(df_out[df_out['label']==1]):,}")
    print(f"  Neg: {len(df_out[df_out['label']==0]):,}")

if __name__ == "__main__":
    main()
