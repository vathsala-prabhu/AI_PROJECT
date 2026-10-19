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


def test_profile_requires_login():
    assert client.put("/me/profile", json={"bio": "hi"}).status_code == 401


def test_profile_update_and_read_back():
    h = logged_in_headers()
    payload = {
        "full_name": "Test Person",
        "bio": "I like writing.",
        "journaling_goal": "Stay calm",
        "journaling_frequency": "Daily",
        "preferred_time": "Evening",
    }
    r = client.put("/me/profile", json=payload, headers=h)
    assert r.status_code == 200
    me = client.get("/me", headers=h).json()
    for key, value in payload.items():
        assert me[key] == value
    assert "password_hash" not in me


def test_blank_values_clear_fields():
    h = logged_in_headers()
    client.put("/me/profile", json={"bio": "something"}, headers=h)
    client.put("/me/profile", json={"bio": "   "}, headers=h)
    assert client.get("/me", headers=h).json()["bio"] is None


def test_bio_too_long_rejected():
    h = logged_in_headers()
    r = client.put("/me/profile", json={"bio": "a" * 501}, headers=h)
    assert r.status_code == 400
    assert "500" in r.json()["detail"]


def test_bad_frequency_rejected():
    h = logged_in_headers()
    r = client.put("/me/profile", json={"journaling_frequency": "Hourly"}, headers=h)
    assert r.status_code == 400


def test_profile_is_private_per_user():
    a = logged_in_headers()
    b = logged_in_headers()
    client.put("/me/profile", json={"full_name": "Alice A"}, headers=a)
    assert client.get("/me", headers=b).json()["full_name"] is None