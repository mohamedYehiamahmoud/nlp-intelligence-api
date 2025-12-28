from fastapi import FastAPI

app = FastAPI(title="Real-Time NLP Intelligence API")

@app.get("/health")
def health():
    return {"status": "ok"}
