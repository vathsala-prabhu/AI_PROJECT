"""Validation and saving of the editable profile fields."""
from sqlalchemy.orm import Session

from backend.db.models import User
from backend.services.validation import ValidationError

FREQUENCIES = ["Daily", "A few times a week", "Weekly", "Occasionally"]
TIMES = ["Morning", "Afternoon", "Evening", "Night", "No preference"]

TEXT_FIELDS = {
    "full_name": ("Full name", 100),
    "bio": ("About me", 500),
    "journaling_goal": ("Journaling goal", 300),
}
CHOICE_FIELDS = {
    "journaling_frequency": ("Writing frequency", FREQUENCIES),
    "preferred_time": ("Best time to write", TIMES),
}


def _clean_text(value, label, limit):
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    if "\x00" in value:
        raise ValidationError(f"{label} contains invalid characters.")
    if len(value) > limit:
        raise ValidationError(f"{label} must be at most {limit} characters.")
    return value


def _clean_choice(value, label, options):
    if value is None or not str(value).strip():
        return None
    value = str(value).strip()
    if value not in options:
        raise ValidationError(f"{label} must be one of: {', '.join(options)}.")
    return value


def update_profile(db: Session, user: User, data: dict) -> User:
    """Validate every field first, then save. Blank values clear a field."""
    cleaned = {}
    for field, (label, limit) in TEXT_FIELDS.items():
        cleaned[field] = _clean_text(data.get(field), label, limit)
    for field, (label, options) in CHOICE_FIELDS.items():
        cleaned[field] = _clean_choice(data.get(field), label, options)

    for field, value in cleaned.items():
        setattr(user, field, value)
    db.commit()
    db.refresh(user)
    return user