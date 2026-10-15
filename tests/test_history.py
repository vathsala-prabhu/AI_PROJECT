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


def make_logged_in_user():
    name = "testuser_" + uuid.uuid4().hex[:8]
    creds = {"username": name, "email": f"{name}@example.com", "password": "password123"}
    assert client.post("/register", json=creds).status_code == 201
    token = client.post(
        "/login", json={"username": name, "password": creds["password"]}
    ).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_save_requires_login():
    r = client.post("/my/analyze", json={"text": "I love this"})
    assert r.status_code == 401


def test_history_requires_login():
    assert client.get("/my/history").status_code == 401


def test_analysis_is_saved_and_listed():
    headers = make_logged_in_user()
    r = client.post("/my/analyze", json={"text": "I am not happy with the results."}, headers=headers)
    assert r.status_code == 200
    assert r.json()["sentiment"]["label"] == "Negative"
    assert "saved_id" in r.json()

    history = client.get("/my/history", headers=headers).json()
    assert len(history) == 1
    assert history[0]["label"] == "Negative"


def test_history_is_private_per_user():
    a = make_logged_in_user()
    b = make_logged_in_user()
    client.post("/my/analyze", json={"text": "I love this wonderful day!"}, headers=a)
    assert len(client.get("/my/history", headers=a).json()) == 1
    assert client.get("/my/history", headers=b).json() == []


def test_invalid_text_not_saved():
    headers = make_logged_in_user()
    r = client.post("/my/analyze", json={"text": "   "}, headers=headers)
    assert r.status_code == 400
    assert client.get("/my/history", headers=headers).json() == []