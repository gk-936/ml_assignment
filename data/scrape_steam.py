"""
data/scrape_steam.py
---------------------
Scrapes Steam game reviews using Steam's public storefront API.
No API key or login needed.

Pulls from 12 games across different genres for vocabulary diversity.
Label comes from voted_up: True = positive (1), False = negative (0).

Usage:
    python data/scrape_steam.py

Output:
    data/raw/steam_reviews_raw.csv     - all scraped reviews with game column
    data/processed/steam_binary.csv    - balanced, cleaned, ready for benchmark

Requirements:
    pip install requests pandas tqdm
"""

import os
import time
import requests
import pandas as pd
from tqdm import tqdm

RAW_DIR       = os.path.join(os.path.dirname(__file__), "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")
RAW_OUT       = os.path.join(RAW_DIR, "steam_reviews_raw.csv")
PROCESSED_OUT = os.path.join(PROCESSED_DIR, "steam_binary.csv")

GAMES = {
    "Delta Force":      2507950,
    "Where Winds Meet": 3564740,
    "7 Days to Die":    251570,
    "NFS Heat":         1222680,
    "NFS Most Wanted":  24870,
    "Forza Horizon 4":  1293830,
    "CS2":              730,
    "Elden Ring":       1245620,
    "Stardew Valley":   413150,
    "Cyberpunk 2077":   1091500,
    "Among Us":         945360,
    "GTA V":            271590,
}

REVIEWS_PER_GAME  = 10000
MIN_LEN           = 20
MAX_LEN           = 512
SAMPLES_PER_CLASS = 20000
RANDOM_STATE      = 42


def fetch_reviews(app_id: int, game_name: str, target: int) -> list:
    """pull reviews from steam API using cursor pagination"""
    url    = f"https://store.steampowered.com/appreviews/{app_id}"
    cursor = "*"
    params = {
        "json"         : 1,
        "language"     : "english",
        "filter"       : "recent",
        "review_type"  : "all",
        "purchase_type": "all",
        "num_per_page" : 100,
    }

    collected = []
    pbar      = tqdm(total=target, desc=f"  {game_name:<22}", unit="rev")
    retries   = 0

    while len(collected) < target:
        params["cursor"] = cursor

        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            retries = 0
        except requests.exceptions.RequestException as e:
            retries += 1
            wait = retries * 3
            print(f"\n  [{game_name}] Request error: {e}. Retry {retries} in {wait}s...")
            if retries > 5:
                print(f"  [{game_name}] Too many retries, skipping.")
                break
            time.sleep(wait)
            continue

        if data.get("success") != 1:
            print(f"\n  [{game_name}] API returned success=0. Stopping.")
            break

        reviews = data.get("reviews", [])
        if not reviews:
            print(f"\n  [{game_name}] No more reviews available.")
            break

        for r in reviews:
            text = r.get("review", "").strip()
            if len(text) < MIN_LEN or len(text) > MAX_LEN:
                continue
            label = 1 if r.get("voted_up") else 0
            collected.append({
                "text" : text,
                "label": label,
                "game" : game_name,
            })
            pbar.update(1)
            if len(collected) >= target:
                break

        # pass cursor as-is — do NOT url-encode it
        cursor = data.get("cursor", "")
        if not cursor:
            break

        time.sleep(0.4)

    pbar.close()
    return collected


def balance_classes(df: pd.DataFrame, n_per_class: int, random_state: int) -> pd.DataFrame:
    """downsample to equal positive and negative"""
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]

    n = min(len(pos), len(neg), n_per_class)
    if n < n_per_class:
        print(f"  Warning: only {n} samples available per class (wanted {n_per_class})")

    pos_sample = pos.sample(n=n, random_state=random_state)
    neg_sample = neg.sample(n=n, random_state=random_state)

    return (
        pd.concat([pos_sample, neg_sample])
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("=" * 60)
    print("  Steam Review Scraper")
    print(f"  {len(GAMES)} games x ~{REVIEWS_PER_GAME} reviews each")
    print(f"  Final output: {SAMPLES_PER_CLASS * 2:,} balanced samples")
    print("=" * 60)

    all_reviews = []

    for game_name, app_id in GAMES.items():
        reviews = fetch_reviews(app_id, game_name, REVIEWS_PER_GAME)
        all_reviews.extend(reviews)
        print(f"  {game_name}: {len(reviews):,} collected")

    df_raw = pd.DataFrame(all_reviews)

    print(f"\nRaw totals: {len(df_raw):,} reviews")
    print(f"  Positive : {df_raw['label'].sum():,}  ({df_raw['label'].mean():.1%})")
    print(f"  Negative : {(df_raw['label']==0).sum():,}  ({1-df_raw['label'].mean():.1%})")

    df_raw.to_csv(RAW_OUT, index=False)
    print(f"Raw saved → {RAW_OUT}")

    print("\nPer-game breakdown:")
    for game, grp in df_raw.groupby("game"):
        pos = grp["label"].sum()
        neg = len(grp) - pos
        print(f"  {game:<22} total={len(grp):>5}  pos={pos:>5}  neg={neg:>5}")

    print(f"\nBalancing to {SAMPLES_PER_CLASS:,} per class...")
    df_balanced = balance_classes(df_raw, SAMPLES_PER_CLASS, RANDOM_STATE)
    df_out      = df_balanced[["text", "label"]].copy()
    df_out.to_csv(PROCESSED_OUT, index=False)

    print(f"\nProcessed saved → {PROCESSED_OUT}")
    print(f"  Final size : {len(df_out):,}")
    print(f"  Positive   : {df_out['label'].sum():,}  ({df_out['label'].mean():.1%})")
    print(f"  Negative   : {(df_out['label']==0).sum():,}")
    print("\nDone. Run the benchmark with:")
    print("  python experiments/run_benchmark.py")


if __name__ == "__main__":
    main()
