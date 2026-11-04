
"""Lightweight emotion detection: keyword rules + VADER sanity check."""

from backend.services.preprocessing import clean_text, get_nlp

EMOTIONS = [
    "Joy",
    "Sadness",
    "Love",
    "Anger",
    "Fear",
    "Surprise",
]

UNCLEAR = "Mixed / unclear"

LEXICON = {
    "Joy": {
        "happy", "joy", "joyful", "glad", "delight", "delighted",
        "excited", "excite", "wonderful", "great", "amazing",
        "awesome", "proud", "thrilled", "cheerful", "celebrate",
        "enjoy", "enjoyed", "fantastic", "pleased", "grateful",
        "thankful", "relieved", "ecstatic", "blissful",
    },

    "Sadness": {
        "sad", "unhappy", "depressed", "cry", "cried", "tears",
        "lonely", "heartbroken", "miserable", "grief", "hopeless",
        "down", "disappointed", "disappoint", "hurt", "sorrow",
        "gloomy", "upset", "empty", "lost",
    },

    "Love": {
        "love", "loved", "loving", "affection", "affectionate",
        "adore", "adored", "cherish", "cherished", "romantic",
        "caring", "devoted", "passionate",
    },

    "Anger": {
        "angry", "anger", "furious", "mad", "hate", "hated",
        "annoy", "annoyed", "irritated", "irritate",
        "frustrate", "frustrated", "rage", "outraged",
        "resent", "rude", "infuriated",
    },

    "Fear": {
        "afraid", "scared", "scare", "fear", "anxious",
        "anxiety", "worry", "worried", "nervous", "panic",
        "terrified", "frightened", "dread", "overwhelmed",
        "stress", "stressed", "uneasy", "tense",
    },

    "Surprise": {
        "surprised", "surprise", "shocked", "astonished",
        "amazed", "unexpected", "unbelievable", "suddenly",
        "astonishing", "stunned", "startled", "wow",
    },
}

NEGATORS = {"not", "no", "never", "n't", "nothing", "hardly"}
NEGATION_WINDOW = 3

POSITIVE_EMOTIONS = {"Joy", "Love"}
NEGATIVE_EMOTIONS = {"Sadness", "Anger", "Fear"}

WORD_TO_EMOTION = {
    word: emotion
    for emotion, words in LEXICON.items()
    for word in words
}


def detect_emotion(text, compound: float) -> dict:
    """Return the main emotion, emotion scores and matched words."""

    doc = get_nlp()(clean_text(text or ""))
    tokens = [t for t in doc if not t.is_space]

    scores = {emotion: 0 for emotion in EMOTIONS}
    matched = {emotion: [] for emotion in EMOTIONS}

    for i, tok in enumerate(tokens):

        emotion = (
            WORD_TO_EMOTION.get(tok.text.lower())
            or WORD_TO_EMOTION.get(tok.lemma_.lower())
        )

        if not emotion:
            continue

        before = tokens[max(0, i - NEGATION_WINDOW):i]

        if any(t.text.lower() in NEGATORS for t in before):
            continue

        scores[emotion] += 1
        matched[emotion].append(tok.text.lower())

    # VADER sentiment sanity check.
    if compound >= 0.05:
        blocked = NEGATIVE_EMOTIONS
    elif compound <= -0.05:
        blocked = POSITIVE_EMOTIONS
    else:
        blocked = set()

    for emotion in blocked:
        scores[emotion] = 0
        matched[emotion] = []

    top = max(scores.values())

    winners = [
        emotion
        for emotion, score in scores.items()
        if score == top
    ]

    label = (
        winners[0]
        if top > 0 and len(winners) == 1
        else UNCLEAR
    )

    return {
        "label": label,
        "scores": scores,
        "matched": {
            emotion: words
            for emotion, words in matched.items()
            if words
        },
    }
