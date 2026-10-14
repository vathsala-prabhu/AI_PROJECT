from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_positive():
    response = client.post("/analyze", json={"text": "I love this wonderful day!"})
    assert response.status_code == 200
    assert response.json()["sentiment"]["label"] == "Positive"


def test_analyze_negation():
    response = client.post("/analyze", json={"text": "I am not happy with the results."})
    assert response.status_code == 200
    assert response.json()["sentiment"]["label"] == "Negative"


def test_empty_text_returns_400():
    response = client.post("/analyze", json={"text": ""})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_whitespace_text_returns_400():
    response = client.post("/analyze", json={"text": "    "})
    assert response.status_code == 400


def test_missing_text_returns_400():
    response = client.post("/analyze", json={})
    assert response.status_code == 400


def test_too_long_text_returns_400():
    response = client.post("/analyze", json={"text": "a" * 50_001})
    assert response.status_code == 400
    assert "too long" in response.json()["detail"].lower()


def test_wrong_type_returns_422():
    response = client.post("/analyze", json={"text": 12345})
    assert response.status_code == 422