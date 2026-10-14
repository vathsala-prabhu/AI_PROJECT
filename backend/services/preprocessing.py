"""Text preprocessing: clean -> tokenize -> remove stop-words -> lemmatize.

NOTE: VADER sentiment is run on the ORIGINAL text, not on this output.
This module exists to show the preprocessing pipeline separately.
"""
import re
from functools import lru_cache

import spacy

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
HTML_RE = re.compile(r"<[^>]+>")
NON_LETTER_RE = re.compile(r"[^a-z\s']")
SPACES_RE = re.compile(r"\s+")


@lru_cache(maxsize=1)
def get_nlp():
    """Load the spaCy model once and reuse it (loading is slow)."""
    return spacy.load("en_core_web_sm")


def clean_text(text: str) -> str:
    """Lowercase, remove URLs/emails/HTML/numbers/punctuation, fix spaces."""
    text = text.lower()
    text = HTML_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = NON_LETTER_RE.sub(" ", text)
    text = SPACES_RE.sub(" ", text).strip()
    return text


def preprocess(text: str) -> dict:
    """Run the full pipeline and return every intermediate result."""
    cleaned = clean_text(text)
    doc = get_nlp()(cleaned)

    tokens = [t.text for t in doc if t.is_alpha]
    tokens_no_stopwords = [t.text for t in doc if t.is_alpha and not t.is_stop]
    lemmas = [t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop]

    return {
        "cleaned_text": cleaned,
        "tokens": tokens,
        "tokens_no_stopwords": tokens_no_stopwords,
        "lemmas": lemmas,
    }