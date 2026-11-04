
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

    try:
        test_ids = select(User.id).where(
            User.username.like("testuser_%")
        )

        db.execute(
            delete(Analysis).where(
                Analysis.user_id.in_(test_ids)
            )
        )

        db.execute(
            delete(User).where(
                User.username.like("testuser_%")
            )
        )

        db.commit()

    finally:
        db.close()


def make_logged_in_user():
    name = "testuser_" + uuid.uuid4().hex[:8]

    creds = {
        "username": name,
        "email": f"{name}@example.com",
        "password": "password123",
    }

    assert client.post(
        "/register",
        json=creds,
    ).status_code == 201

    token = client.post(
        "/login",
        json={
            "username": name,
            "password": creds["password"],
        },
    ).json()["access_token"]

    return {
        "Authorization": f"Bearer {token}"
    }


def test_save_requires_login():
    response = client.post(
        "/my/analyze",
        json={"text": "I love this"},
    )

    assert response.status_code == 401


def test_history_requires_login():
    response = client.get("/my/history")

    assert response.status_code == 401


def test_analysis_is_saved_and_listed():
    headers = make_logged_in_user()

    response = client.post(
        "/my/analyze",
        json={
            "text": "I am not happy with the results."
        },
        headers=headers,
    )

    assert response.status_code == 200
    assert response.json()["sentiment"]["label"] == "Negative"
    assert "saved_id" in response.json()

    history = client.get(
        "/my/history",
        headers=headers,
    ).json()

    assert len(history) == 1
    assert history[0]["label"] == "Negative"

    # Verify that the emotion is also saved.
    assert history[0]["emotion"] is not None


def test_history_is_private_per_user():
    user_a = make_logged_in_user()
    user_b = make_logged_in_user()

    response = client.post(
        "/my/analyze",
        json={
            "text": "I love this wonderful day!"
        },
        headers=user_a,
    )

    assert response.status_code == 200

    history_a = client.get(
        "/my/history",
        headers=user_a,
    ).json()

    history_b = client.get(
        "/my/history",
        headers=user_b,
    ).json()

    assert len(history_a) == 1
    assert history_b == []


def test_invalid_text_not_saved():
    headers = make_logged_in_user()

    response = client.post(
        "/my/analyze",
        json={"text": "   "},
        headers=headers,
    )

    assert response.status_code == 400

    history = client.get(
        "/my/history",
        headers=headers,
    ).json()

    assert history == []


def test_new_emotion_labels_are_saved():
    headers = make_logged_in_user()

    valid_emotions = {
        "Joy",
        "Sadness",
        "Love",
        "Anger",
        "Fear",
        "Surprise",
    }

    response = client.post(
        "/my/analyze",
        json={
            "text": "I am so happy and excited today!"
        },
        headers=headers,
    )

    assert response.status_code == 200

    history = client.get(
        "/my/history",
        headers=headers,
    ).json()

    assert len(history) == 1
    assert history[0]["emotion"] in valid_emotions
