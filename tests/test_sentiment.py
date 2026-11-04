
import pytest

from backend.services.sentiment import (
    analyze_sentiment,
    classify_sentiment,
)


def test_positive_text():
    result = analyze_sentiment("I love this, it is wonderful!")

    assert result["label"] == "Positive"
    assert result["positive"] > result["neutral"]
    assert result["positive"] > result["negative"]


def test_negative_text():
    result = analyze_sentiment("I hate this, it is terrible.")

    assert result["label"] == "Negative"
    assert result["negative"] > result["neutral"]
    assert result["negative"] > result["positive"]


def test_neutral_text():
    result = analyze_sentiment("The table is in the room.")

    assert result["label"] == "Neutral"
    assert result["neutral"] > result["positive"]
    assert result["neutral"] > result["negative"]


def test_negation_is_preserved():
    result = analyze_sentiment("I am not happy with the results.")

    # Compound score is still calculated independently.
    assert result["compound"] < 0

    # Label must match the highest VADER proportion.
    expected = classify_sentiment({
        "pos": result["positive"],
        "neu": result["neutral"],
        "neg": result["negative"],
    })

    assert result["label"] == expected


def test_scores_present_and_sum_to_one():
    result = analyze_sentiment("This is a good day.")

    total = (
        result["positive"]
        + result["negative"]
        + result["neutral"]
    )

    assert total == pytest.approx(1.0, abs=0.01)
    assert -1 <= result["compound"] <= 1


def test_highest_positive_score():
    scores = {
        "pos": 0.7,
        "neu": 0.2,
        "neg": 0.1,
    }

    assert classify_sentiment(scores) == "Positive"


def test_highest_neutral_score():
    scores = {
        "pos": 0.17,
        "neu": 0.632,
        "neg": 0.198,
    }

    assert classify_sentiment(scores) == "Neutral"


def test_highest_negative_score():
    scores = {
        "pos": 0.1,
        "neu": 0.2,
        "neg": 0.7,
    }

    assert classify_sentiment(scores) == "Negative"


def test_tie_prefers_neutral():
    scores = {
        "pos": 0.4,
        "neu": 0.4,
        "neg": 0.2,
    }

    assert classify_sentiment(scores) == "Neutral"


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_empty_or_invalid_raises(bad):
    with pytest.raises(ValueError):
        analyze_sentiment(bad)
