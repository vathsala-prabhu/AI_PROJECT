"""Checks the connection and creates the tables. Run once."""
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from backend.db import models  # noqa: F401  (registers the tables)
from backend.db.database import Base, engine


def main():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError as exc:
        print("Could not connect to PostgreSQL.")
        print("Check: PostgreSQL is running, the database 'moodmentor' exists,")
        print("and DATABASE_URL in .env has the right password.")
        print(f"\nDetails: {exc.orig}")
        raise SystemExit(1)

    print("Connected to PostgreSQL.")
    Base.metadata.create_all(engine)
    print("Tables ready:", ", ".join(Base.metadata.tables))


if __name__ == "__main__":
    main()