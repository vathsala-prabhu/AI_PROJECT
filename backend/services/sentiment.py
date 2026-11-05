
"""VADER sentiment analysis using compound score thresholds."""

from functools import lru_cache
from nltk.sentiment.vader import SentimentIntensityAnalyzer


@lru_cache(maxsize=1)
def get_analyzer():
    return SentimentIntensityAnalyzer()


def classify_sentiment(scores):
    """
    Classify sentiment.

    Numeric input or a dictionary with compound:
        Uses VADER compound thresholds.

    Dictionary with only positive, neutral and negative proportions:
        Uses the highest proportion. Neutral wins ties.
    """

    if isinstance(scores, dict):

        if "compound" in scores:
            compound = scores["compound"]

        else:
            positive = scores.get("pos", scores.get("positive", 0))
            neutral = scores.get("neu", scores.get("neutral", 0))
            negative = scores.get("neg", scores.get("negative", 0))

            values = {
                "Positive": positive,
                "Neutral": neutral,
                "Negative": negative,
            }

            highest = max(values.values())

            if neutral == highest:
                return "Neutral"
            elif positive == highest:
                return "Positive"
            else:
                return "Negative"

    else:
        compound = scores

    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def analyze_sentiment(text: str) -> dict:
    """Analyze text using VADER."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text must be a non-empty string.")

    scores = get_analyzer().polarity_scores(text)

    return {
        "compound": scores["compound"],
        "positive": scores["pos"],
        "negative": scores["neg"],
        "neutral": scores["neu"],
        "label": classify_sentiment(scores["compound"]),
    }
