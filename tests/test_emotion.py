
from backend.services.emotion import UNCLEAR, detect_emotion
from backend.services.report import build_csv_report, build_report


def label(text, compound):
    return detect_emotion(text, compound)["label"]


def test_joy():
    assert label("I am so happy and excited today", 0.7) == "Joy"


def test_sadness():
    assert label("I feel sad and lonely", -0.5) == "Sadness"


def test_anger():
    assert label("I am furious and I hate this", -0.6) == "Anger"


def test_fear():
    assert label("I am scared and anxious about the exam", -0.5) == "Fear"


def test_love():
    assert label("I feel love and affection", 0.8) == "Love"


def test_surprise():
    assert label("I am surprised and shocked", 0.0) == "Surprise"


def test_no_keywords_is_unclear():
    assert label("The table is in the room", 0.0) == UNCLEAR


def test_empty_is_unclear():
    assert label("", 0.0) == UNCLEAR


def test_negation_is_not_joy():
    assert label("I am not happy", -0.46) == UNCLEAR


def test_sentiment_check_blocks_contradiction():
    assert label("Overwhelmed by the sermon given", 0.052) == UNCLEAR


def test_tie_is_unclear():
    assert label("I feel sad and angry", -0.5) == UNCLEAR


def test_report_contains_emotion(monkeypatch):
    # Test keyword fallback independently of the transformer.
    monkeypatch.setattr(
        "backend.services.report.transformer_emotion",
        lambda text: None,
    )

    report = build_report("I am so happy and excited today!")

    assert "emotion" in report
    assert report["emotion"]["label"] == "Joy"
    assert "detected" in report["emotion"]


def test_csv_rows_have_emotion(monkeypatch):
    # Make CSV tests deterministic by using keyword fallback.
    monkeypatch.setattr(
        "backend.services.report.transformer_emotion",
        lambda text: None,
    )

    rows = [
        (2, "I am happy and excited"),
        (3, "I hate this, I am furious"),
    ]

    data = build_csv_report(rows, "text")

    assert data["results"][0]["emotion"] == "Joy"
    assert data["results"][1]["emotion"] == "Anger"

    assert data["overall"]["emotion_counts"] == {
        "Joy": 1,
        "Anger": 1,
    }

    assert data["results"][0]["detected_emotions"]
    assert data["results"][1]["detected_emotions"]
