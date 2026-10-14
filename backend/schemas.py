"""Pydantic models describing the data the API accepts."""
from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    # Optional so a missing field gets our friendly validation message
    # instead of a generic framework error.
    text: str | None = None