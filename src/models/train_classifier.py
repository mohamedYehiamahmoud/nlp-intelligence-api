from pathlib import Path
import re
import joblib

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


DATA_PATH = Path("data/processed/news_clean.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def weak_label(text: str) -> str:
    """
    Weak, rule-based topic labeling.
    This is a heuristic baseline, not ground truth.
    """
    text = text.lower()

    tech_keywords = [
        "ai", "artificial intelligence", "machine learning",
        "technology", "tech", "software", "startup", "chip"
    ]
    business_keywords = [
        "market", "stock", "economy", "business",
        "finance", "trade", "inflation", "company"
    ]
    sports_keywords = [
        "match", "tournament", "league",
        "goal", "sports", "football", "nba", "fifa"
    ]

    if any(k in text for k in tech_keywords):
        return "technology"
    if any(k in text for k in business_keywords):
        return "business"
    if any(k in text for k in sports_keywords):
        return "sports"

    return "general"



def load_and_label_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["label"] = df["text"].apply(weak_label)

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    return df



def train():
    df = load_and_label_data()

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)

    print("\nClassification Report:\n")
    print(classification_report(y_test, preds))

    # Save artifacts
    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(model, MODEL_DIR / "logreg_classifier.joblib")

    print("\nSaved model artifacts to /models")


if __name__ == "__main__":
    train()
