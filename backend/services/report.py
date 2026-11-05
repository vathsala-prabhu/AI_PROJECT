
"""Report generation for MoodMentor."""

from datetime import datetime, timezone
from collections import Counter

from backend.services.sentiment import (
    analyze_sentiment,
    classify_sentiment,
)
from backend.services.emotion import detect_emotion
from backend.services.emotion_model import predict as transformer_emotion
from backend.services.validation import validate_text, ValidationError
from backend.services.preprocessing import preprocess


def get_emotion(text: str, compound: float) -> dict:
    """Get transformer emotion predictions and keyword-based baseline."""

    baseline_result = detect_emotion(text, compound)

    if isinstance(baseline_result, dict):
        baseline_label = baseline_result.get(
            "label", "Mixed / unclear"
        )
    else:
        baseline_label = str(baseline_result)

    transformer_result = transformer_emotion(text)

    if transformer_result is None:
        if isinstance(baseline_result, dict):
            return {
                **baseline_result,
                "engine": "keyword",
                "baseline": baseline_label,
            }

        return {
            "engine": "keyword",
            "label": baseline_label,
            "detected": [baseline_label],
            "baseline": baseline_label,
            "scores": {},
        }

    transformer_result["baseline"] = baseline_label

    # Ensure the detected-emotions field is always available.
    if "detected" not in transformer_result:
        transformer_result["detected"] = [
            transformer_result.get("label", "Mixed / unclear")
        ]

    return transformer_result


def build_report(text: str) -> dict:
    """Generate a complete report for one text."""

    cleaned_text = validate_text(text)

    sentiment = analyze_sentiment(cleaned_text)

    emotion = get_emotion(
        cleaned_text,
        sentiment["compound"],
    )

    preprocessing = preprocess(cleaned_text)

    word_count = len(cleaned_text.split())

    detected_emotions = emotion.get(
        "detected",
        [emotion.get("label", "Mixed / unclear")]
    )

    emotion_text = ", ".join(detected_emotions)

    summary = (
        f"The text expresses {sentiment['label']} sentiment "
        f"(compound score: {sentiment['compound']:.3f}). "
        f"Dominant emotion: {emotion['label']}. "
        f"Detected emotions: {emotion_text}."
    )

    return {
        "original_text": cleaned_text,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "char_count": len(cleaned_text),
        "word_count": word_count,
        "sentiment": sentiment,
        "emotion": emotion,
        "preprocessing": preprocessing,
        "summary": summary,
    }


def build_csv_report(rows: list, column_used=None) -> dict:
    """Generate individual and aggregate reports for CSV records."""

    reports = []
    results = []
    skipped = []

    sentiment_counts = Counter()
    emotion_counts = Counter()
    detected_emotion_counts = Counter()

    compound_scores = []
    positive_scores = []
    negative_scores = []
    neutral_scores = []

    for index, row in enumerate(rows, start=2):

        row_number = index

        if isinstance(row, dict):
            if column_used and column_used in row:
                text = row[column_used]
            else:
                text = row.get("text", "")

        elif isinstance(row, (tuple, list)):
            if len(row) >= 2:
                row_number = row[0]
                text = row[1]
            elif len(row) == 1:
                text = row[0]
            else:
                skipped.append({
                    "row": row_number,
                    "reason": "Empty row",
                })
                continue

        else:
            text = row

        try:
            report = build_report(text)

        except (ValidationError, ValueError, TypeError) as exc:
            skipped.append({
                "row": row_number,
                "reason": str(exc),
            })
            continue

        reports.append(report)

        sentiment = report["sentiment"]
        emotion = report["emotion"]

        emotion_label = emotion.get(
            "label", "Mixed / unclear"
        )

        detected_emotions = emotion.get(
            "detected",
            [emotion_label]
        )

        sentiment_counts[sentiment["label"]] += 1

        # Preserve primary emotion counts for existing dashboards.
        emotion_counts[emotion_label] += 1

        # Count every detected emotion separately.
        for detected in detected_emotions:
            detected_emotion_counts[detected] += 1

        compound_scores.append(sentiment["compound"])
        positive_scores.append(sentiment["positive"])
        negative_scores.append(sentiment["negative"])
        neutral_scores.append(sentiment["neutral"])

        # Flat structure expected by crud.save_csv_rows().
        result_row = {
            "text": report["original_text"],
            "compound": sentiment["compound"],
            "positive": sentiment["positive"],
            "negative": sentiment["negative"],
            "neutral": sentiment["neutral"],
            "label": sentiment["label"],
            "emotion": emotion_label,
            "detected_emotions": detected_emotions,
            "emotion_scores": emotion.get("scores", {}),
        }

        if isinstance(row, (tuple, list)) and len(row) >= 2:
            result_row["row_number"] = row_number

        results.append(result_row)

    total = len(results)
    rows_skipped = len(skipped)

    if total == 0:
        average_compound = 0.0

        average_scores = {
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 0.0,
        }

        overall_sentiment = "Neutral"

    else:
        average_compound = sum(compound_scores) / total

        average_scores = {
            "positive": sum(positive_scores) / total,
            "negative": sum(negative_scores) / total,
            "neutral": sum(neutral_scores) / total,
        }

        # Use the highest average VADER proportion.
        overall_sentiment = classify_sentiment({
            "pos": average_scores["positive"],
            "neu": average_scores["neutral"],
            "neg": average_scores["negative"],
        })

    overall = {
        "sentiment": overall_sentiment,
        "counts": dict(sentiment_counts),
        "emotion_counts": dict(emotion_counts),
        "detected_emotion_counts": dict(detected_emotion_counts),
        "average_compound": round(average_compound, 4),
        "average_scores": average_scores,
    }

    return {
        "total_records": total,
        "rows_analyzed": total,
        "rows_skipped": rows_skipped,
        "skipped": skipped,
        "sentiment_counts": dict(sentiment_counts),
        "emotion_counts": dict(emotion_counts),
        "detected_emotion_counts": dict(detected_emotion_counts),
        "average_compound": round(average_compound, 4),
        "overall_sentiment": overall_sentiment,
        "average_scores": average_scores,
        "overall": overall,
        "column_used": column_used,
        "results": results,
        "reports": reports,
    }
