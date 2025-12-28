from transformers import pipeline

# Lightweight, CPU-friendly summarization model
summarizer = pipeline(
    task="summarization",
    model="t5-small",
)


def summarize_text(text: str, max_length: int = 60) -> dict:
    """
    Generate a short summary for the input text.
    Optimized for speed and deployment.
    """
    if not text or len(text.strip()) < 50:
        return {
            "summary": "",
            "note": "Text too short to summarize reliably.",
        }

    result = summarizer(
        "summarize: " + text,
        max_length=max_length,
        min_length=25,
        do_sample=False,
    )

    return {
        "summary": result[0]["summary_text"],
    }
