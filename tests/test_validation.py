import pytest

from backend.services.validation import (
    MAX_TEXT_CHARS,
    ValidationError,
    validate_text,
)


def test_valid_text_is_stripped():
    assert validate_text("  hello world  ") == "hello world"


def test_empty_string_rejected():
    with pytest.raises(ValidationError):
        validate_text("")


def test_whitespace_only_rejected():
    with pytest.raises(ValidationError):
        validate_text("   \n\t  ")


def test_none_rejected():
    with pytest.raises(ValidationError):
        validate_text(None)


def test_non_string_rejected():
    with pytest.raises(ValidationError):
        validate_text(12345)


def test_too_long_rejected():
    with pytest.raises(ValidationError):
        validate_text("a" * (MAX_TEXT_CHARS + 1))


def test_no_letters_rejected():
    with pytest.raises(ValidationError):
        validate_text("!!! 123 ???")


def test_null_byte_rejected():
    with pytest.raises(ValidationError):
        validate_text("hello\x00world")