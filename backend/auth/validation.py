"""Validation for registration input."""
import re

from backend.services.validation import ValidationError

USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{3,30}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD = 8
MAX_PASSWORD = 72  # bcrypt only uses the first 72 bytes


def validate_registration(username, email, password):
    """Return (username, email, password) cleaned, or raise ValidationError."""
    if not username or not str(username).strip():
        raise ValidationError("Username is required.")
    if not email or not str(email).strip():
        raise ValidationError("Email is required.")
    if not password:
        raise ValidationError("Password is required.")

    username = str(username).strip()
    email = str(email).strip().lower()

    if not USERNAME_RE.match(username):
        raise ValidationError(
            "Username must be 3-30 characters: letters, numbers or underscore."
        )
    if not EMAIL_RE.match(email):
        raise ValidationError("Please enter a valid email address.")
    if len(password) < MIN_PASSWORD:
        raise ValidationError(f"Password must be at least {MIN_PASSWORD} characters.")
    if len(password.encode("utf-8")) > MAX_PASSWORD:
        raise ValidationError(f"Password must be at most {MAX_PASSWORD} bytes.")
    return username, email, password