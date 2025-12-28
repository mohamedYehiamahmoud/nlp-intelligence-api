from pathlib import Path
import joblib

MODEL_DIR = Path("models")

vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
model = joblib.load(MODEL_DIR / "logreg_classifier.joblib")


def classify_text(text: str) -> dict:
    X = vectorizer.transform([text])
    proba = model.predict_proba(X)[0]
    label = model.classes_[proba.argmax()]
    confidence = float(proba.max())

    return {
        "label": label,
        "confidence": round(confidence, 3),
    }
