"""Builds the initial sentiment report for one piece of text."""
from datetime import datetime, timezone

from backend.services.preprocessing import preprocess
from backend.services.sentiment import analyze_sentiment
from backend.services.validation import validate_text


def build_report(text) -> dict:
    """Validate -> VADER on original text -> preprocessing demo -> report.

    Raises ValidationError if the input is bad.
    """
    text = validate_text(text)
    sentiment = analyze_sentiment(text)  # original text, keeps negation cues
    processed = preprocess(text)         # shown separately

    summary = (
        f"The text is {sentiment['label']} "
        f"(compound score {sentiment['compound']:.3f})."
    )
    return {
        "original_text": text,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "char_count": len(text),
        "word_count": len(text.split()),
        "sentiment": sentiment,
        "preprocessing": processed,
        "summary": summary,
    }