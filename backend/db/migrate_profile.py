"""One-time migration: adds the profile columns to the existing users table."""
from sqlalchemy import text

from backend.db.database import engine

COLUMNS = [
    ("full_name", "VARCHAR(100)"),
    ("bio", "VARCHAR(500)"),
    ("journaling_goal", "VARCHAR(300)"),
    ("journaling_frequency", "VARCHAR(30)"),
    ("preferred_time", "VARCHAR(30)"),
]

with engine.begin() as conn:
    for name, sql_type in COLUMNS:
        conn.execute(text(f"ALTER TABLE users ADD COLUMN IF NOT EXISTS {name} {sql_type}"))

print("Profile columns are ready.")