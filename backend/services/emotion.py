"""Lightweight emotion detection: keyword rules + a VADER sanity check."""
from backend.services.preprocessing import clean_text, get_nlp

EMOTIONS = ["Joy", "Sadness", "Anger", "Fear", "Calm"]
UNCLEAR = "Mixed / unclear"

LEXICON = {
    "Joy": {
        "happy", "joy", "joyful", "glad", "delight", "delighted", "excited",
        "excite", "love", "loved", "wonderful", "great", "amazing", "awesome",
        "proud", "thrilled", "cheerful", "celebrate", "enjoy", "enjoyed",
        "fantastic", "pleased", "grateful", "thankful", "relieved",
    },
    "Sadness": {
        "sad", "unhappy", "depressed", "cry", "cried", "tears", "lonely",
        "heartbroken", "miserable", "grief", "hopeless", "down", "disappointed",
        "disappoint", "hurt", "sorrow", "gloomy", "upset", "empty", "lost",
    },
    "Anger": {
        "angry", "anger", "furious", "mad", "hate", "hated", "annoy", "annoyed",
        "irritated", "irritate", "frustrate", "frustrated", "rage", "outraged",
        "resent", "rude", "fed",
    },
    "Fear": {
        "afraid", "scared", "scare", "fear", "anxious", "anxiety", "worry",
        "worried", "nervous", "panic", "terrified", "frightened", "dread",
        "overwhelmed", "stress", "stressed", "uneasy", "tense",
    },
    "Calm": {
        "calm", "peaceful", "relaxed", "relax", "serene", "tranquil", "rested",
        "steady", "balanced", "comfortable", "soothed", "peace",
    },
}

NEGATORS = {"not", "no", "never", "n't", "nothing", "hardly"}
NEGATION_WINDOW = 3
POSITIVE_EMOTIONS = {"Joy", "Calm"}
NEGATIVE_EMOTIONS = {"Sadness", "Anger", "Fear"}

WORD_TO_EMOTION = {
    word: emotion for emotion, words in LEXICON.items() for word in words
}


def detect_emotion(text, compound: float) -> dict:
    """Return the main emotion, per-emotion word counts and the matched words."""
    doc = get_nlp()(clean_text(text or ""))
    tokens = [t for t in doc if not t.is_space]

    scores = {e: 0 for e in EMOTIONS}
    matched = {e: [] for e in EMOTIONS}

    for i, tok in enumerate(tokens):
        emotion = (
            WORD_TO_EMOTION.get(tok.text.lower())
            or WORD_TO_EMOTION.get(tok.lemma_.lower())
        )
        if not emotion:
            continue
        before = tokens[max(0, i - NEGATION_WINDOW):i]
        if any(t.text.lower() in NEGATORS for t in before):
            continue  # "not happy" must not count as Joy
        scores[emotion] += 1
        matched[emotion].append(tok.text.lower())

    # VADER sanity check: the emotion may not contradict the sentiment.
    if compound >= 0.05:
        blocked = NEGATIVE_EMOTIONS
    elif compound <= -0.05:
        blocked = POSITIVE_EMOTIONS
    else:
        blocked = set()
    for e in blocked:
        scores[e] = 0
        matched[e] = []

    top = max(scores.values())
    winners = [e for e, s in scores.items() if s == top]
    label = winners[0] if top > 0 and len(winners) == 1 else UNCLEAR

    return {
        "label": label,
        "scores": scores,
        "matched": {e: w for e, w in matched.items() if w},
    }