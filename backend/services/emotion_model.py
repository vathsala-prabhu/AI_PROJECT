
"""Transformer emotion engine: DistilBERT fine-tuned on GoEmotions."""

import json
import os
from functools import lru_cache
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

GROUPS = {
    "joy": [
        "admiration", "amusement", "approval", "caring", "desire",
        "excitement", "gratitude", "joy", "love", "optimism",
        "pride", "relief"
    ],
    "sadness": [
        "sadness", "disappointment", "embarrassment", "grief", "remorse"
    ],
    "anger": [
        "anger", "annoyance", "disapproval"
    ],
    "fear": [
        "fear", "nervousness"
    ],
    "disgust": [
        "disgust"
    ],
    "surprise": [
        "surprise", "realization", "confusion", "curiosity"
    ],
    "neutral": [
        "neutral"
    ],
}


def engine_enabled() -> bool:
    return os.getenv("MOODMENTOR_EMOTION_ENGINE", "auto").lower() != "keyword"


@lru_cache(maxsize=1)
def _load():
    """Load the trained model and thresholds once."""

    if not (MODEL_DIR / "config.json").exists():
        print(f"Emotion model not found at: {MODEL_DIR}")
        return None

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR)
    ).eval()

    names = [
        model.config.id2label[i]
        for i in range(model.config.num_labels)
    ]

    thresholds_path = MODEL_DIR / "thresholds.json"
    thresholds = json.loads(thresholds_path.read_text())

    # Group thresholds are optional.
    group_thresholds_path = MODEL_DIR / "group_thresholds.json"

    if group_thresholds_path.exists():
        group_thresholds = json.loads(
            group_thresholds_path.read_text()
        )
    else:
        group_thresholds = {}

    return tokenizer, model, names, thresholds, group_thresholds


def predict(text):
    """Return emotion predictions using the fine-tuned DistilBERT model."""

    if not engine_enabled():
        return None

    loaded = _load()

    if loaded is None:
        return None

    import torch

    tokenizer, model, names, thresholds, group_thresholds = loaded

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        probs = torch.sigmoid(
            model(**inputs).logits
        )[0].cpu().numpy()

    # Fine-grained GoEmotions predictions.
    fine = {
        name: float(probability)
        for name, probability in zip(names, probs)
    }

    # Combine fine-grained emotions into broader groups.
    groups = {
        group: max(fine[emotion] for emotion in members)
        for group, members in GROUPS.items()
    }

    # Select the group with the highest confidence score.
    highest_group = max(groups, key=groups.get)
    label = highest_group.capitalize()

    return {
        "engine": "transformer",
        "label": label,

        "scores": {
            group.capitalize(): round(score, 3)
            for group, score in groups.items()
        },

        "top": [
            [name, round(probability, 3)]
            for name, probability in sorted(
                fine.items(),
                key=lambda item: -item[1]
            )[:5]
        ],

        "detected": [
            name
            for name, probability in fine.items()
            if probability >= thresholds.get(name, 0.5)
        ],

        "matched": {},
    }
