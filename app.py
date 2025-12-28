from fastapi import FastAPI
from pydantic import BaseModel

from src.inference.classifier import classify_text
from src.inference.summarizer import summarize_text

app = FastAPI(title="Real-Time NLP Intelligence API")


class TextRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/classify")
def classify(request: TextRequest):
    return classify_text(request.text)


@app.post("/summarize")
def summarize(request: TextRequest):
    return summarize_text(request.text)
