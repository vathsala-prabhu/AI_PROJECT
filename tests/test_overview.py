import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from backend.db.database import SessionLocal
from backend.db.models import Analysis, User
from backend.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def cleanup_test_users():
    yield
    db = SessionLocal()
    test_ids = select(User.id).where(User.username.like("testuser_%"))
    db.execute(delete(Analysis).where(Analysis.user_id.in_(test_ids)))
    db.execute(delete(User).where(User.username.like("testuser_%")))
    db.commit()
    db.close()


def logged_in_headers():
    name = "testuser_" + uuid.uuid4().hex[:8]
    creds = {"username": name, "email": f"{name}@example.com", "password": "password123"}
    assert client.post("/register", json=creds).status_code == 201
    token = client.post(
        "/login", json={"username": name, "password": creds["password"]}
    ).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_overview_requires_login():
    assert client.get("/my/overview").status_code == 401


def test_overview_empty():
    body = client.get("/my/overview", headers=logged_in_headers()).json()
    assert body["total"] == 0
    assert body["average_compound"] is None
    assert body["recent"] == []


def test_overview_counts_text_entries():
    h = logged_in_headers()
    client.post("/my/analyze", json={"text": "I love this wonderful day!"}, headers=h)
    client.post("/my/analyze", json={"text": "I hate this, it is terrible."}, headers=h)
    body = client.get("/my/overview", headers=h).json()
    assert body["total"] == 2
    assert body["counts"]["Positive"] == 1
    assert body["counts"]["Negative"] == 1
    assert len(body["recent"]) == 2


def test_csv_upload_is_saved_per_row():
    h = logged_in_headers()
    r = client.post(
        "/my/analyze-file",
        files={"file": ("d.csv", b"text\nI love it\nI hate it\n", "text/csv")},
        headers=h,
    )
    assert r.status_code == 200
    assert r.json()["saved_entries"] == 2
    assert client.get("/my/overview", headers=h).json()["total"] == 2