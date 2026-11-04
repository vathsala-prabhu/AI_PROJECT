
import pytest

from backend.services import emotion_model


VALID_EMOTIONS = {
    "Joy",
    "Sadness",
    "Love",
    "Anger",
    "Fear",
    "Surprise",
}


def test_keyword_mode_skips_transformer(monkeypatch):
    monkeypatch.setenv("MOODMENTOR_EMOTION_ENGINE", "keyword")

    assert emotion_model.predict("I am so happy") is None


def test_transformer_predicts_when_model_is_present(monkeypatch):
    if not (emotion_model.MODEL_DIR / "config.json").exists():
        pytest.skip("DAIR-AI model folder not installed")

    monkeypatch.setenv("MOODMENTOR_EMOTION_ENGINE", "auto")

    result = emotion_model.predict(
        "I am so happy and grateful today"
    )

    assert result is not None
    assert result["engine"] == "transformer"
    assert result["label"] in VALID_EMOTIONS
    assert result["top"]

    # Check that all six emotion scores are returned.
    assert set(result["scores"].keys()) == VALID_EMOTIONS

    # Check that probabilities are valid.
    probabilities = list(result["scores"].values())

    assert all(0 <= score <= 1 for score in probabilities)
    assert sum(probabilities) == pytest.approx(1.0, abs=0.01)

    # Check that the predicted emotion is the top prediction.
    assert result["top"][0][0].capitalize() == result["label"]
