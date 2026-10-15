"""Pydantic models describing the data the API accepts."""
from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    # Optional so a missing field gets our friendly validation message
    # instead of a generic framework error.
    text: str | None = None


class RegisterRequest(BaseModel):
    username: str | None = None
    email: str | None = None
    password: str | None = None
class LoginRequest(BaseModel):
    username: str | None = None  # username OR email
    password: str | None = None