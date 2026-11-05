
import pytest

from backend.services.sentiment import (
    analyze_sentiment,
    classify_sentiment,
)


def test_positive_text():
    result = analyze_sentiment("I am extremely happy and excited today!")

    assert result["label"] == "Positive"
    assert result["compound"] > 0


def test_negative_text():
    result = analyze_sentiment("I feel terrible and disappointed.")

    assert result["label"] == "Negative"
    assert result["compound"] < 0


def test_neutral_text():
    result = analyze_sentiment(
        "The meeting is scheduled for tomorrow at 10 AM."
    )

    assert result["label"] == "Neutral"


def test_negation_is_preserved():
    result = analyze_sentiment("I am not happy with the results.")

    assert result["compound"] < 0
    assert result["label"] == "Negative"

    # Verify that the label follows compound score thresholds.
    assert result["label"] == classify_sentiment(result["compound"])


def test_scores_present_and_sum_to_one():
    result = analyze_sentiment("I am happy today.")

    assert "compound" in result
    assert "positive" in result
    assert "negative" in result
    assert "neutral" in result
    assert "label" in result

    total = (
        result["positive"]
        + result["negative"]
        + result["neutral"]
    )

    assert total == pytest.approx(1.0, abs=0.01)
    assert -1 <= result["compound"] <= 1


def test_highest_positive_score():
    scores = {
        "pos": 0.8,
        "neu": 0.1,
        "neg": 0.1,
    }

    assert classify_sentiment(scores) == "Positive"


def test_highest_neutral_score():
    scores = {
        "pos": 0.1,
        "neu": 0.8,
        "neg": 0.1,
    }

    assert classify_sentiment(scores) == "Neutral"


def test_highest_negative_score():
    scores = {
        "pos": 0.1,
        "neu": 0.1,
        "neg": 0.8,
    }

    assert classify_sentiment(scores) == "Negative"


def test_tie_prefers_neutral():
    scores = {
        "pos": 0.5,
        "neu": 0.5,
        "neg": 0.0,
    }

    assert classify_sentiment(scores) == "Neutral"


@pytest.mark.parametrize("text", ["", "   ", None])
def test_empty_or_invalid_raises(text):
    with pytest.raises(ValueError):
        analyze_sentiment(text)
