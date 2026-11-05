
"""Multi-label transformer emotion engine: DistilBERT fine-tuned on DAIR-AI Emotion."""

import os
from functools import lru_cache
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

MODEL_DIR = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "moodmentor_emotion"
)

MAX_LENGTH = 128

LABEL_NAMES = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise",
]

DEFAULT_THRESHOLD = 0.30


def engine_enabled() -> bool:
    """Check whether transformer emotion detection is enabled."""
    return os.getenv(
        "MOODMENTOR_EMOTION_ENGINE", "auto"
    ).lower() != "keyword"


def get_threshold() -> float:
    """Read and validate the emotion detection threshold."""
    try:
        threshold = float(
            os.getenv(
                "MOODMENTOR_EMOTION_THRESHOLD",
                str(DEFAULT_THRESHOLD),
            )
        )

        if 0.0 <= threshold <= 1.0:
            return threshold

    except (ValueError, TypeError):
        pass

    return DEFAULT_THRESHOLD


@lru_cache(maxsize=1)
def _load():
    """Load the multi-label model and tokenizer once."""

    if not (MODEL_DIR / "config.json").exists():
        print(f"Emotion model not found at: {MODEL_DIR}")
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

        model = AutoModelForSequenceClassification.from_pretrained(
            str(MODEL_DIR)
        )

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        model.to(device)
        model.eval()

        if model.config.num_labels != len(LABEL_NAMES):
            raise ValueError(
                f"Expected {len(LABEL_NAMES)} emotion outputs, "
                f"but model has {model.config.num_labels}."
            )

        if model.config.problem_type != "multi_label_classification":
            print(
                "Warning: Model config does not specify "
                "multi_label_classification."
            )

        return tokenizer, model, LABEL_NAMES.copy(), device

    except Exception as exc:
        print(f"Error loading emotion model: {exc}")
        return None


def predict(text: str):
    """Predict multiple emotions using sigmoid probabilities."""

    if not engine_enabled():
        return None

    if not isinstance(text, str) or not text.strip():
        return None

    loaded = _load()

    if loaded is None:
        return None

    tokenizer, model, names, device = loaded
    threshold = get_threshold()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.sigmoid(logits)[0]

    scores = {
        name.capitalize(): round(
            float(probabilities[i].item()), 4
        )
        for i, name in enumerate(names)
    }

    top_predictions = sorted(
        [
            (name, float(probabilities[i].item()))
            for i, name in enumerate(names)
        ],
        key=lambda item: item[1],
        reverse=True,
    )

    detected = [
        name.capitalize()
        for name, score in top_predictions
        if score >= threshold
    ]

    if not detected:
        detected = [top_predictions[0][0].capitalize()]

    primary_emotion = top_predictions[0][0].capitalize()

    return {
        "engine": "transformer",
        "label": primary_emotion,
        "scores": scores,
        "top": [
            [name.lower(), round(score, 4)]
            for name, score in top_predictions
        ],
        "detected": detected,
        "matched": {},
        "threshold": threshold,
    }
