"""Database helpers for saved analyses."""
from sqlalchemy import func, select
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


def save_csv_rows(db: Session, user_id: int, results: list[dict]) -> int:
    """Save every analyzed CSV row as its own journal entry."""
    rows = [
        Analysis(
            user_id=user_id,
            source="csv",
            input_text=r["text"],
            compound=r["compound"],
            positive=r["positive"],
            negative=r["negative"],
            neutral=r["neutral"],
            label=r["label"],
        )
        for r in results
    ]
    db.add_all(rows)
    db.commit()
    return len(rows)


def list_analyses(db: Session, user_id: int, limit: int = 50) -> list[Analysis]:
    stmt = (
        select(Analysis)
        .where(Analysis.user_id == user_id)
        .order_by(Analysis.created_at.desc(), Analysis.id.desc())
        .limit(limit)
    )
    return list(db.scalars(stmt))


def overview_stats(db: Session, user_id: int, recent: int = 20) -> dict:
    """Totals, label counts, average score and the latest scores (oldest first)."""
    total = db.scalar(
        select(func.count(Analysis.id)).where(Analysis.user_id == user_id)
    ) or 0
    average = db.scalar(
        select(func.avg(Analysis.compound)).where(Analysis.user_id == user_id)
    )
    label_rows = db.execute(
        select(Analysis.label, func.count(Analysis.id))
        .where(Analysis.user_id == user_id)
        .group_by(Analysis.label)
    ).all()
    counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for label, n in label_rows:
        counts[label] = n

    latest = list_analyses(db, user_id, limit=recent)
    latest.reverse()  # oldest first, so the chart reads left to right
    return {
        "total": total,
        "average_compound": round(average, 4) if average is not None else None,
        "counts": counts,
        "recent": [
            {"compound": r.compound, "label": r.label, "created_at": r.created_at.isoformat()}
            for r in latest
        ],
    }