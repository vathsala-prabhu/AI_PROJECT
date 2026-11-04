
"""Transformer emotion engine: DistilBERT fine-tuned on DAIR-AI Emotion."""

import os
from functools import lru_cache
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# backend/models/moodmentor_emotion/
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


def engine_enabled() -> bool:
    """Check whether transformer emotion detection is enabled."""
    return os.getenv(
        "MOODMENTOR_EMOTION_ENGINE", "auto"
    ).lower() != "keyword"


@lru_cache(maxsize=1)
def _load():
    """Load the fine-tuned model and tokenizer once."""

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

        names = [
            model.config.id2label.get(i, LABEL_NAMES[i])
            for i in range(model.config.num_labels)
        ]

        return tokenizer, model, names, device

    except Exception as exc:
        print(f"Error loading emotion model: {exc}")
        return None


def predict(text):
    """Predict one of six emotions using DAIR-AI DistilBERT."""

    if not engine_enabled():
        return None

    if not text or not text.strip():
        return None

    loaded = _load()

    if loaded is None:
        return None

    tokenizer, model, names, device = loaded

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
        probabilities = torch.softmax(logits, dim=-1)[0]

    predicted_id = int(torch.argmax(probabilities).item())

    predicted_emotion = names[predicted_id].lower()

    scores = {
        name.capitalize(): round(
            float(probabilities[i].item()), 4
        )
        for i, name in enumerate(names)
    }

    top_predictions = sorted(
        [
            (name.lower(), float(probabilities[i].item()))
            for i, name in enumerate(names)
        ],
        key=lambda item: item[1],
        reverse=True,
    )

    return {
        "engine": "transformer",
        "label": predicted_emotion.capitalize(),

        "scores": scores,

        "top": [
            [name, round(score, 4)]
            for name, score in top_predictions[:5]
        ],

        "detected": [predicted_emotion],

        "matched": {},
    }
