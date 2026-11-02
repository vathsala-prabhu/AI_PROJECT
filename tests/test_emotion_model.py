import pytest

from backend.services import emotion_model


def test_keyword_mode_skips_transformer():
    assert emotion_model.predict("I am so happy") is None


def test_transformer_predicts_when_model_is_present(monkeypatch):
    if not (emotion_model.MODEL_DIR / "config.json").exists():
        pytest.skip("model folder not installed")
    monkeypatch.setenv("MOODMENTOR_EMOTION_ENGINE", "auto")
    result = emotion_model.predict("I am so happy and grateful today")
    assert result["engine"] == "transformer"
    assert result["label"] == "Joy"
    assert result["top"]