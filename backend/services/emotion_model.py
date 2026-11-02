"""Transformer emotion engine: DistilBERT fine-tuned on GoEmotions."""
import json
import os
from functools import lru_cache
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "moodmentor-distilbert"
UNCLEAR = "Mixed / unclear"

GROUPS = {
    "joy": ["admiration", "amusement", "approval", "caring", "desire",
            "excitement", "gratitude", "joy", "love", "optimism", "pride", "relief"],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "anger": ["anger", "annoyance", "disapproval"],
    "fear": ["fear", "nervousness"],
    "disgust": ["disgust"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
    "neutral": ["neutral"],
}


def engine_enabled() -> bool:
    return os.getenv("MOODMENTOR_EMOTION_ENGINE", "auto").lower() != "keyword"


@lru_cache(maxsize=1)
def _load():
    """Load once. Returns None if the model folder is not there."""
    if not (MODEL_DIR / "config.json").exists():
        return None
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).eval()
    names = [model.config.id2label[i] for i in range(model.config.num_labels)]
    thresholds = json.loads((MODEL_DIR / "thresholds.json").read_text())
    group_thresholds = json.loads((MODEL_DIR / "group_thresholds.json").read_text())
    return tokenizer, model, names, thresholds, group_thresholds


def predict(text):
    """Return an emotion dict, or None when the transformer is unavailable."""
    if not engine_enabled():
        return None
    loaded = _load()
    if loaded is None:
        return None

    import torch

    tokenizer, model, names, thresholds, group_thresholds = loaded
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        probs = torch.sigmoid(model(**inputs).logits)[0].numpy()

    fine = {n: float(p) for n, p in zip(names, probs)}
    groups = {g: max(fine[m] for m in members) for g, members in GROUPS.items()}
    active = {g: s for g, s in groups.items() if s >= group_thresholds.get(g, 0.5)}

    non_neutral = {g: s for g, s in active.items() if g != "neutral"}
    if non_neutral:
        label = max(non_neutral, key=non_neutral.get).capitalize()
    elif "neutral" in active:
        label = "Neutral"
    else:
        label = UNCLEAR

    return {
        "engine": "transformer",
        "label": label,
        "scores": {g.capitalize(): round(s, 3) for g, s in groups.items()},
        "top": [[n, round(p, 3)] for n, p in sorted(fine.items(), key=lambda x: -x[1])[:5]],
        "detected": [n for n, p in fine.items() if p >= thresholds.get(n, 0.5)],
        "matched": {},
    }