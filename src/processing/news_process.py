import json
from pathlib import Path

import pandas as pd


RAW_NEWS_DIR = Path("data/raw/news")
PROCESSED_DATA_DIR = Path("data/processed")
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = PROCESSED_DATA_DIR / "news_clean.csv"


def load_raw_articles() -> list[dict]:
    articles = []

    for json_file in RAW_NEWS_DIR.glob("news_*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            articles.extend(json.load(f))

    return articles


def process_articles(articles: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(articles)

    if df.empty:
        raise ValueError("No articles found in raw data.")

    # Drop duplicates across runs
    df = df.drop_duplicates(subset="id")

    # Combine text fields
    df["text"] = (
        df["title"].fillna("") + " "
        + df["description"].fillna("") + " "
        + df["content"].fillna("")
    )

    # Basic cleaning
    df["text"] = df["text"].str.strip()
    df = df[df["text"].str.len() > 50]

    # Select final columns
    return df[
        ["id", "source", "published_at", "ingested_at", "text"]
    ]


def main():
    raw_articles = load_raw_articles()
    df = process_articles(raw_articles)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved processed dataset with {len(df)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
