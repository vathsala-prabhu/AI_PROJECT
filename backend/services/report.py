
"""Builds sentiment reports for single text and CSV inputs."""

from collections import Counter
from datetime import datetime, timezone

from backend.services.emotion import detect_emotion
from backend.services.emotion_model import predict as predict_transformer
from backend.services.preprocessing import preprocess
from backend.services.sentiment import (
    analyze_sentiment,
    classify_sentiment,
)
from backend.services.validation import ValidationError, validate_text


def get_emotion(text, compound) -> dict:
    """
    Use DAIR-AI DistilBERT when available.
    Fall back to keyword-based emotion detection otherwise.
    """

    baseline = detect_emotion(text, compound)

    model_result = predict_transformer(text)

    if model_result is None:
        baseline["engine"] = "keyword"
        return baseline

    # Retain keyword prediction for comparison.
    model_result["baseline"] = baseline["label"]

    return model_result


def build_report(text) -> dict:
    """
    Validate text -> VADER sentiment -> emotion detection
    -> preprocessing demonstration.

    Raises ValidationError if input is invalid.
    """

    text = validate_text(text)

    # Analyze original text to preserve negation cues.
    sentiment = analyze_sentiment(text)

    # DAIR-AI transformer or keyword fallback.
    emotion = get_emotion(text, sentiment["compound"])

    # Preprocessing is displayed separately.
    processed = preprocess(text)

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
        "emotion": emotion,
        "preprocessing": processed,
        "summary": summary,
    }


def build_csv_report(rows, column: str) -> dict:
    """
    Analyze CSV rows, skip invalid entries,
    and generate an overall summary.
    """

    results = []
    skipped = []

    for row_number, raw in rows:

        try:
            text = validate_text(raw)

        except ValidationError as exc:
            skipped.append({
                "row": row_number,
                "reason": str(exc),
            })
            continue

        sentiment = analyze_sentiment(text)

        emotion_result = get_emotion(
            text,
            sentiment["compound"],
        )

        results.append({
            "row": row_number,
            "text": text[:200],
            "emotion": emotion_result["label"],
            **sentiment,
        })

    if not results:
        raise ValidationError(
            f"No valid text found in column '{column}'. "
            "Every row was empty or invalid."
        )

    # Calculate average compound score.
    average = sum(
        result["compound"] for result in results
    ) / len(results)

    # Calculate average sentiment proportions.
    average_scores = {
        "pos": sum(
            result["positive"] for result in results
        ) / len(results),

        "neu": sum(
            result["neutral"] for result in results
        ) / len(results),

        "neg": sum(
            result["negative"] for result in results
        ) / len(results),
    }

    # Classify overall CSV sentiment using the highest
    # average positive, neutral or negative proportion.
    overall_label = classify_sentiment(average_scores)

    counts = {
        "Positive": 0,
        "Negative": 0,
        "Neutral": 0,
    }

    for result in results:
        counts[result["label"]] += 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),

        "column_used": column,

        "rows_analyzed": len(results),

        "rows_skipped": len(skipped),

        "overall": {
            "average_compound": round(average, 4),

            "label": overall_label,

            "counts": counts,

            "emotion_counts": dict(
                Counter(
                    result["emotion"] for result in results
                )
            ),
        },

        "results": results,

        "skipped": skipped,
    }
