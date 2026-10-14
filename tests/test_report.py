import pytest

from backend.services.report import build_report
from backend.services.validation import ValidationError


def test_report_has_all_sections():
    report = build_report("I love this wonderful day!")
    for key in ["original_text", "generated_at", "char_count",
                "word_count", "sentiment", "preprocessing", "summary"]:
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