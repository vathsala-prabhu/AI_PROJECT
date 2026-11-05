
import pytest

from backend.services.report import build_report
from backend.services.validation import ValidationError


VALID_EMOTIONS = {
    "Joy",
    "Sadness",
    "Love",
    "Anger",
    "Fear",
    "Surprise",
}


def test_report_has_all_sections():
    report = build_report("I love this wonderful day!")

    for key in [
        "original_text",
        "generated_at",
        "char_count",
        "word_count",
        "sentiment",
        "emotion",
        "preprocessing",
        "summary",
    ]:
        assert key in report


def test_report_label_matches_sentiment():
    report = build_report("I am not happy with the results.")

    assert report["sentiment"]["label"] == "Negative"
    assert "Negative" in report["summary"]


def test_report_counts():
    report = build_report("one two three")

    assert report["word_count"] == 3
    assert report["char_count"] == 13


def test_report_invalid_input_raises():
    with pytest.raises(ValidationError):
        build_report("   ")


def test_report_keeps_original_text():
    report = build_report("  Great job!  ")

    assert report["original_text"] == "Great job!"


def test_report_contains_valid_emotion():
    report = build_report("I am so happy and excited today!")

    assert report["emotion"]["label"] in VALID_EMOTIONS


def test_report_contains_emotion_engine():
    report = build_report("I am so happy and excited today!")

    assert report["emotion"]["engine"] in {
        "transformer",
        "keyword",
    }

def test_report_contains_multiple_emotion_fields():
    report = build_report(
        "I am happy about my new job but nervous about the responsibilities."
    )

    emotion = report["emotion"]

    assert "detected" in emotion
    assert isinstance(emotion["detected"], list)

    assert "scores" in emotion
    assert set(emotion["scores"].keys()) == VALID_EMOTIONS


def test_report_summary_contains_detected_emotions():
    report = build_report(
        "I am excited about my new opportunity."
    )

    assert "Detected emotions:" in report["summary"]
