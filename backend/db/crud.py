"""Database helpers for saved analyses."""
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.db.models import Analysis


def save_analysis(db: Session, user_id: int, report: dict, source: str = "text") -> Analysis:
    s = report["sentiment"]
    row = Analysis(
        user_id=user_id,
        source=source,
        input_text=report["original_text"],
        compound=s["compound"],
        positive=s["positive"],
        negative=s["negative"],
        neutral=s["neutral"],
        label=s["label"],
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def list_analyses(db: Session, user_id: int, limit: int = 50) -> list[Analysis]:
    stmt = (
        select(Analysis)
        .where(Analysis.user_id == user_id)
        .order_by(Analysis.created_at.desc(), Analysis.id.desc())
        .limit(limit)
    )
    return list(db.scalars(stmt))