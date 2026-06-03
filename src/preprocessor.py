import re
from typing import Optional

import nltk
from nltk.corpus import stopwords


_whitespace_re = re.compile(r"\s+")
_non_alnum_re = re.compile(r"[^a-zA-Z0-9\s]")


def _get_stopwords() -> set[str]:
    """Load English stopwords, downloading the corpus if needed."""
    # Lazy-load and fallback download for NLTK English stopwords.
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))


def _normalize_whitespace(text: str) -> str:
    """Collapse consecutive whitespace and trim."""
    return _whitespace_re.sub(" ", text).strip()


def clean_text(text: Optional[str]) -> str:
    """
    Lowercase, remove special characters, and normalize spacing.

    Edge cases: None, empty, or very short inputs return "".
    """
    if not text or not text.strip():
        return ""

    # Normalize text: lowercase, remove special chars, and collapse whitespace.
    cleaned = text.lower()
    cleaned = _non_alnum_re.sub(" ", cleaned)
    cleaned = _normalize_whitespace(cleaned)

    if len(cleaned) < 2:
        return ""

    return cleaned


def remove_stopwords(text: Optional[str]) -> str:
    """
    Remove English stopwords using NLTK.

    Edge cases: None, empty, or very short inputs return "".
    """
    if not text or not text.strip():
        return ""

    # Remove English stopwords to reduce noise.
    stops = _get_stopwords()
    words = [word for word in text.split() if word.lower() not in stops]
    filtered = " ".join(words)

    if len(filtered) < 2:
        return ""

    return filtered


def preprocess(text: Optional[str]) -> str:
    """Full preprocessing for modeling: clean then remove stopwords."""
    return remove_stopwords(clean_text(text))


def preprocess_for_display(text: Optional[str]) -> str:
    """
    Preprocess for UI display: remove special characters and stopwords,
    but preserve the original casing of kept words.
    """
    if not text or not text.strip():
        return ""

    # UI preprocessing: strip special chars and stopwords while preserving casing.
    cleaned = _non_alnum_re.sub(" ", text)
    cleaned = _normalize_whitespace(cleaned)
    if len(cleaned) < 2:
        return ""

    stops = _get_stopwords()
    words = [word for word in cleaned.split() if word.lower() not in stops]
    filtered = " ".join(words)

    if len(filtered) < 2:
        return ""

    return filtered
