
"""Database helpers for saved analyses."""

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.db.models import Analysis


def save_analysis(
    db: Session,
    user_id: int,
    report: dict,
    source: str = "text",
) -> Analysis:
    """Save a single text analysis to the database."""

    sentiment = report["sentiment"]
    emotion = report.get("emotion", {}).get("label")

    row = Analysis(
        user_id=user_id,
        source=source,
        input_text=report["original_text"],
        compound=sentiment["compound"],
        positive=sentiment["positive"],
        negative=sentiment["negative"],
        neutral=sentiment["neutral"],
        label=sentiment["label"],
        emotion=emotion,
    )

    db.add(row)
    db.commit()
    db.refresh(row)

    return row


def save_csv_rows(
    db: Session,
    user_id: int,
    results: list[dict],
) -> int:
    """Save every analyzed CSV row as its own journal entry."""

    rows = [
        Analysis(
            user_id=user_id,
            source="csv",
            input_text=result["text"],
            compound=result["compound"],
            positive=result["positive"],
            negative=result["negative"],
            neutral=result["neutral"],
            label=result["label"],
            emotion=result.get("emotion"),
        )
        for result in results
    ]

    if not rows:
        return 0

    db.add_all(rows)
    db.commit()

    return len(rows)


def list_analyses(
    db: Session,
    user_id: int,
    limit: int = 50,
) -> list[Analysis]:
    """Retrieve the latest analyses for a user."""

    stmt = (
        select(Analysis)
        .where(Analysis.user_id == user_id)
        .order_by(
            Analysis.created_at.desc(),
            Analysis.id.desc(),
        )
        .limit(limit)
    )

    return list(db.scalars(stmt))


def overview_stats(
    db: Session,
    user_id: int,
    recent: int = 20,
) -> dict:
    """Return totals, sentiment counts, emotion counts and recent scores."""

    # Total number of analyses
    total = (
        db.scalar(
            select(func.count(Analysis.id)).where(
                Analysis.user_id == user_id
            )
        )
        or 0
    )

    # Average sentiment compound score
    average = db.scalar(
        select(func.avg(Analysis.compound)).where(
            Analysis.user_id == user_id
        )
    )

    # Sentiment label counts
    label_rows = db.execute(
        select(
            Analysis.label,
            func.count(Analysis.id),
        )
        .where(Analysis.user_id == user_id)
        .group_by(Analysis.label)
    ).all()

    counts = {
        "Positive": 0,
        "Neutral": 0,
        "Negative": 0,
    }

    for label, count in label_rows:
        counts[label] = count

    # Emotion counts
    # Supports the new DAIR-AI six-class labels
    emotion_rows = db.execute(
        select(
            Analysis.emotion,
            func.count(Analysis.id),
        )
        .where(
            Analysis.user_id == user_id,
            Analysis.emotion.is_not(None),
        )
        .group_by(Analysis.emotion)
    ).all()

    emotion_counts = {
        emotion: count
        for emotion, count in emotion_rows
    }

    # Latest analyses
    latest = list_analyses(
        db,
        user_id,
        limit=recent,
    )

    # Reverse so the chart displays oldest to newest
    latest.reverse()

    return {
        "total": total,
        "average_compound": (
            round(float(average), 4)
            if average is not None
            else None
        ),
        "counts": counts,
        "emotion_counts": emotion_counts,
        "recent": [
            {
                "compound": row.compound,
                "label": row.label,
                "created_at": row.created_at.isoformat(),
            }
            for row in latest
        ],
    }
