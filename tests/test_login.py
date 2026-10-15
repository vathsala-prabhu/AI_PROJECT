import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from backend.auth.security import create_access_token
from backend.db.database import SessionLocal
from backend.db.models import User
from backend.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def cleanup_test_users():
    yield
    db = SessionLocal()
    db.execute(delete(User).where(User.username.like("testuser_%")))
    db.commit()
    db.close()


def make_user(password="password123"):
    name = "testuser_" + uuid.uuid4().hex[:8]
    data = {"username": name, "email": f"{name}@example.com", "password": password}
    r = client.post("/register", json=data)
    assert r.status_code == 201
    return data, r.json()["id"]


def test_login_ok():
    data, _ = make_user()
    r = client.post("/login", json={"username": data["username"], "password": data["password"]})
    assert r.status_code == 200
    body = r.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]
    assert "password_hash" not in body["user"]


def test_login_with_email():
    data, _ = make_user()
    r = client.post("/login", json={"username": data["email"], "password": data["password"]})
    assert r.status_code == 200


def test_wrong_password():
    data, _ = make_user()
    r = client.post("/login", json={"username": data["username"], "password": "wrongpass"})
    assert r.status_code == 401
    assert "invalid" in r.json()["detail"].lower()


def test_unknown_user():
    r = client.post("/login", json={"username": "testuser_nobody", "password": "password123"})
    assert r.status_code == 401


def test_missing_fields():
    r = client.post("/login", json={})
    assert r.status_code == 400


def test_me_with_token():
    data, _ = make_user()
    token = client.post(
        "/login", json={"username": data["username"], "password": data["password"]}
    ).json()["access_token"]
    r = client.get("/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["username"] == data["username"]


def test_me_without_token():
    r = client.get("/me")
    assert r.status_code == 401


def test_me_bad_token():
    r = client.get("/me", headers={"Authorization": "Bearer not-a-real-token"})
    assert r.status_code == 401


def test_me_expired_token():
    _, user_id = make_user()
    expired = create_access_token(user_id, expires_minutes=-1)
    r = client.get("/me", headers={"Authorization": f"Bearer {expired}"})
    assert r.status_code == 401