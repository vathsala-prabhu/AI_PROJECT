import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from backend.auth.security import hash_password, verify_password
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


def new_name():
    return "testuser_" + uuid.uuid4().hex[:8]


def payload(**overrides):
    name = new_name()
    data = {"username": name, "email": f"{name}@example.com", "password": "password123"}
    data.update(overrides)
    return data


def test_hash_and_verify():
    hashed = hash_password("secret123")
    assert hashed != "secret123"
    assert verify_password("secret123", hashed)
    assert not verify_password("wrong", hashed)


def test_register_ok():
    r = client.post("/register", json=payload())
    assert r.status_code == 201
    assert "password" not in r.json()
    assert "password_hash" not in r.json()


def test_duplicate_username():
    data = payload()
    assert client.post("/register", json=data).status_code == 201
    other = payload(username=data["username"])
    r = client.post("/register", json=other)
    assert r.status_code == 400
    assert "username" in r.json()["detail"].lower()


def test_duplicate_email():
    data = payload()
    assert client.post("/register", json=data).status_code == 201
    other = payload(email=data["email"])
    r = client.post("/register", json=other)
    assert r.status_code == 400
    assert "email" in r.json()["detail"].lower()


def test_short_password():
    r = client.post("/register", json=payload(password="short"))
    assert r.status_code == 400
    assert "password" in r.json()["detail"].lower()


def test_bad_email():
    r = client.post("/register", json=payload(email="not-an-email"))
    assert r.status_code == 400


def test_bad_username():
    r = client.post("/register", json=payload(username="a b"))
    assert r.status_code == 400


def test_missing_fields():
    r = client.post("/register", json={})
    assert r.status_code == 400
