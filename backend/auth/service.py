"""User registration logic."""
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.auth.security import hash_password
from backend.auth.validation import validate_registration
from backend.db.models import User
from backend.services.validation import ValidationError


def register_user(db: Session, username, email, password) -> User:
    username, email, password = validate_registration(username, email, password)

    if db.scalar(select(User).where(func.lower(User.username) == username.lower())):
        raise ValidationError("That username is already taken.")
    if db.scalar(select(User).where(User.email == email)):
        raise ValidationError("An account with that email already exists.")

    user = User(username=username, email=email, password_hash=hash_password(password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user