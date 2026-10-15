"""User registration and login logic."""
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from backend.auth.security import hash_password, verify_password
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


def authenticate_user(db: Session, identifier, password):
    """Return the User if the credentials are right, otherwise None."""
    if not identifier or not str(identifier).strip() or not password:
        raise ValidationError("Username (or email) and password are required.")

    identifier = str(identifier).strip().lower()
    user = db.scalar(
        select(User).where(
            or_(func.lower(User.username) == identifier, User.email == identifier)
        )
    )
    if user is None or not verify_password(password, user.password_hash):
        return None
    return user