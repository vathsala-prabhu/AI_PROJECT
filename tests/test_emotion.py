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


def test_calm():
    assert label("I feel calm and peaceful", 0.6) == "Calm"


def test_no_keywords_is_unclear():
    assert label("The table is in the room", 0.0) == UNCLEAR


def test_empty_is_unclear():
    assert label("", 0.0) == UNCLEAR


def test_negation_is_not_joy():
    assert label("I am not happy", -0.46) == UNCLEAR


def test_sentiment_check_blocks_contradiction():
    # Fear word, but VADER says positive: no contradictory label.
    assert label("Overwhelmed by the sermon given", 0.052) == UNCLEAR


def test_tie_is_unclear():
    assert label("I feel sad and angry", -0.5) == UNCLEAR


def test_report_contains_emotion():
    report = build_report("I am so happy and excited today!")
    assert report["emotion"]["label"] == "Joy"


def test_csv_rows_have_emotion():
    rows = [(2, "I love this wonderful day"), (3, "I hate this, I am furious")]
    data = build_csv_report(rows, "text")
    assert data["results"][0]["emotion"] == "Joy"
    assert data["results"][1]["emotion"] == "Anger"
    assert data["overall"]["emotion_counts"] == {"Joy": 1, "Anger": 1}
