
"""VADER sentiment analysis using the highest sentiment proportion."""

from functools import lru_cache
from nltk.sentiment.vader import SentimentIntensityAnalyzer


@lru_cache(maxsize=1)
def get_analyzer() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


def classify_sentiment(scores: dict) -> str:
    """Choose the sentiment with the highest VADER proportion."""

    values = {
        "Positive": scores["pos"],
        "Neutral": scores["neu"],
        "Negative": scores["neg"],
    }

    highest = max(values.values())

    # Prefer Neutral if there is a tie.
    if values["Neutral"] == highest:
        return "Neutral"

    if values["Positive"] == highest:
        return "Positive"

    return "Negative"


def analyze_sentiment(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text must be a non-empty string.")

    scores = get_analyzer().polarity_scores(text)

    label = classify_sentiment(scores)

    return {
        "compound": scores["compound"],
        "positive": scores["pos"],
        "negative": scores["neg"],
        "neutral": scores["neu"],
        "label": label,
    }
