"""Builds the initial sentiment report for one piece of text."""
from datetime import datetime, timezone

from backend.services.preprocessing import preprocess
from backend.services.sentiment import analyze_sentiment, classify_compound
from backend.services.validation import ValidationError, validate_text


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
def build_csv_report(rows, column: str) -> dict:
    """Analyze each CSV row; skip bad rows; add an overall summary."""
    results, skipped = [], []

    for row_number, raw in rows:
        try:
            text = validate_text(raw)
        except ValidationError as exc:
            skipped.append({"row": row_number, "reason": str(exc)})
            continue
        sentiment = analyze_sentiment(text)
        results.append({"row": row_number, "text": text[:200], **sentiment})

    if not results:
        raise ValidationError(
            f"No valid text found in column '{column}'. "
            "Every row was empty or invalid."
        )

    average = sum(r["compound"] for r in results) / len(results)
    counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for r in results:
        counts[r["label"]] += 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "column_used": column,
        "rows_analyzed": len(results),
        "rows_skipped": len(skipped),
        "overall": {
            "average_compound": round(average, 4),
            "label": classify_compound(average),
            "counts": counts,
        },
        "results": results,
        "skipped": skipped,
    }