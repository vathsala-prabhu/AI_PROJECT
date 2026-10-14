"""Input validation: rejects empty, invalid, or oversized text."""

MAX_TEXT_CHARS = 50_000


class ValidationError(ValueError):
    """Raised when user input is not acceptable. The message is safe to show."""


def validate_text(text) -> str:
    """Return the stripped text, or raise ValidationError with a clear reason."""
    if text is None:
        raise ValidationError("No text was provided.")
    if not isinstance(text, str):
        raise ValidationError("Input must be text.")
    if "\x00" in text:
        raise ValidationError("Text contains invalid characters.")

    text = text.strip()
    if not text:
        raise ValidationError("Text is empty. Please enter some text.")
    if len(text) > MAX_TEXT_CHARS:
        raise ValidationError(
            f"Text is too long ({len(text):,} characters). "
            f"The limit is {MAX_TEXT_CHARS:,}."
        )
    if not any(ch.isalpha() for ch in text):
        raise ValidationError("Text must contain at least one letter.")
    return text
