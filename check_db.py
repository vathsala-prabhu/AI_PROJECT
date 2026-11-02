"""Dev helper: shows which database the backend uses and tests a login."""
import getpass

from sqlalchemy import func, select, text

from backend.auth.security import verify_password
from backend.db.database import SessionLocal, engine
from backend.db.models import User

print("Backend database:", engine.url.render_as_string(hide_password=True))
with engine.connect() as c:
    print("Server says:", c.execute(text("select current_database(), inet_server_port()")).all())

db = SessionLocal()
users = db.scalars(select(User).order_by(User.id)).all()
print(f"Users found: {len(users)}")
for u in users:
    print(" ", u.id, u.username, u.email, "hash length", len(u.password_hash))

name = input("Username to test (Enter to skip): ").strip()
if name:
    user = db.scalar(select(User).where(func.lower(User.username) == name.lower()))
    if user is None:
        print("No such user in this database.")
    else:
        pw = getpass.getpass("Password: ")
        print("Password matches:", verify_password(pw, user.password_hash))
db.close()
