import pytest

from backend.services.sentiment import analyze_sentiment, classify_compound


def test_positive_text():
    result = analyze_sentiment("I love this, it is wonderful!")
    assert result["label"] == "Positive"
    assert result["compound"] >= 0.05


def test_negative_text():
    result = analyze_sentiment("I hate this, it is terrible.")
    assert result["label"] == "Negative"
    assert result["compound"] <= -0.05


def test_neutral_text():
    result = analyze_sentiment("The table is in the room.")
    assert result["label"] == "Neutral"


def test_negation_is_preserved():
    result = analyze_sentiment("I am not happy with the results.")
    assert result["label"] == "Negative"


def test_scores_present_and_sum_to_one():
    result = analyze_sentiment("This is a good day.")
    total = result["positive"] + result["negative"] + result["neutral"]
    assert total == pytest.approx(1.0, abs=0.01)
    assert -1 <= result["compound"] <= 1


def test_threshold_boundaries():
    assert classify_compound(0.05) == "Positive"
    assert classify_compound(0.0499) == "Neutral"
    assert classify_compound(0.0) == "Neutral"
    assert classify_compound(-0.0499) == "Neutral"
    assert classify_compound(-0.05) == "Negative"


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_empty_or_invalid_raises(bad):
    with pytest.raises(ValueError):
        analyze_sentiment(bad)