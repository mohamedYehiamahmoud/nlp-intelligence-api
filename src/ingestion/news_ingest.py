import os
import json
import hashlib
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

NEWS_API_KEY = os.getenv("NEWSAPI_KEY")
BASE_URL = "https://newsapi.org/v2/top-headlines"

RAW_DATA_DIR = Path("data/raw/news")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def generate_article_id(title: str, published_at: str) -> str:
    raw = f"{title}_{published_at}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def fetch_news(country: str = "us", page_size: int = 20) -> list[dict]:
    if not NEWS_API_KEY:
        raise RuntimeError("NEWSAPI_KEY not found in environment variables")

    params = {
        "apiKey": NEWS_API_KEY,
        "country": country,
        "pageSize": page_size,
    }

    response = requests.get(BASE_URL, params=params, timeout=10)
    response.raise_for_status()

    return response.json().get("articles", [])


def normalize_articles(articles: list[dict]) -> list[dict]:
    normalized = []
    ingested_at = datetime.utcnow().isoformat()

    for article in articles:
        title = article.get("title")
        published_at = article.get("publishedAt")

        if not title or not published_at:
            continue

        article_id = generate_article_id(title, published_at)

        normalized.append({
            "id": article_id,
            "source": article.get("source", {}).get("id"),
            "title": title,
            "description": article.get("description"),
            "content": article.get("content"),
            "published_at": published_at,
            "ingested_at": ingested_at,
        })

    return normalized


def save_raw_articles(articles: list[dict]) -> None:
    if not articles:
        print("No articles to save.")
        return

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = RAW_DATA_DIR / f"news_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(articles)} articles to {output_path}")


def main():
    articles = fetch_news()
    normalized_articles = normalize_articles(articles)
    save_raw_articles(normalized_articles)


if __name__ == "__main__":
    main()
