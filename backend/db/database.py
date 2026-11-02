"""Database engine and session helper."""
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from backend.config import DATABASE_URL

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. Create a .env file (see .env.example)."
    )


class Base(DeclarativeBase):
    pass


engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def ensure_database_schema() -> None:
    """Create tables and backfill any missing columns from older schemas."""
    Base.metadata.create_all(bind=engine)
    inspector = inspect(engine)

    if not inspector.has_table("analyses"):
        return

    analyses_columns = {column["name"] for column in inspector.get_columns("analyses")}
    if "emotion" not in analyses_columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE analyses ADD COLUMN emotion VARCHAR(20)"))


ensure_database_schema()

# Shows in the backend terminal which database is really being used
print(f"[MoodMentor] Database: {engine.url.render_as_string(hide_password=True)}")


def get_db():
    """FastAPI dependency: one database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()