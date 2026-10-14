"""VADER sentiment analysis with standard compound-score thresholds.

VADER is run on the ORIGINAL text (not the preprocessed text) so that
negation, capitalization and punctuation cues are kept.
"""
from functools import lru_cache

from nltk.sentiment.vader import SentimentIntensityAnalyzer

POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


@lru_cache(maxsize=1)
def get_analyzer() -> SentimentIntensityAnalyzer:
    """Create the VADER analyzer once and reuse it."""
    return SentimentIntensityAnalyzer()


def classify_compound(compound: float) -> str:
    """Map a compound score to Positive / Negative / Neutral."""
    if compound >= POSITIVE_THRESHOLD:
        return "Positive"
    if compound <= NEGATIVE_THRESHOLD:
        return "Negative"
    return "Neutral"


def analyze_sentiment(text: str) -> dict:
    """Return compound, pos, neg, neu scores and the sentiment label."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text must be a non-empty string.")

    scores = get_analyzer().polarity_scores(text)
    return {
        "compound": scores["compound"],
        "positive": scores["pos"],
        "negative": scores["neg"],
        "neutral": scores["neu"],
        "label": classify_compound(scores["compound"]),
    }