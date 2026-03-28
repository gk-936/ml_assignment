"""
preprocessing/text_cleaner.py
------------------------------
A single, consistent preprocessing pipeline used by ALL vectorizers.
Keeping preprocessing identical across methods ensures that performance
differences are attributable to the vectorizer — not to cleaning choices.

Steps applied:
    1. Lowercase
    2. Remove URLs
    3. Remove HTML tags
    4. Remove non-alphanumeric characters (keep spaces)
    5. Collapse multiple spaces
    6. Tokenise (split on whitespace)
    7. Remove stopwords  (NLTK English stopwords)
    8. Drop single-character tokens
    9. Re-join to string  (returned as str, not list)

NOTE: We deliberately do NOT stem or lemmatise.
      Word2Vec / FastText / GloVe / BERT all depend on real word forms
      for their embeddings.  Stemming would corrupt the vocabulary lookups.
"""

import re
import string
from functools import lru_cache
from typing import List

import nltk

# Download stopwords quietly if not already present
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

_STOPWORDS = set(stopwords.words("english"))

# Regex patterns — compiled once at module load for speed
_URL_RE    = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE   = re.compile(r"<[^>]+>")
_NONALPHA  = re.compile(r"[^a-z0-9\s]")
_SPACES_RE = re.compile(r"\s+")


def clean(text: str) -> str:
    """
    Full cleaning pipeline.  Returns a single whitespace-separated string.

    Parameters
    ----------
    text : str
        Raw input text (e.g., a Jigsaw comment).

    Returns
    -------
    str
        Cleaned, tokenised, stopword-removed text joined as a string.

    Examples
    --------
    >>> clean("You're an IDIOT!! Visit http://spam.com for more")
    'idiot visit'
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _NONALPHA.sub(" ", text)
    text = _SPACES_RE.sub(" ", text).strip()

    tokens = [
        tok for tok in text.split()
        if tok not in _STOPWORDS and len(tok) > 1
    ]

    return " ".join(tokens)


def tokenize(text: str) -> List[str]:
    """
    Returns the cleaned text as a list of tokens.
    Used by gensim-based vectorizers (Word2Vec, FastText) which expect
    a list of lists for training.

    Parameters
    ----------
    text : str
        Raw or already-cleaned text.

    Returns
    -------
    List[str]
        List of tokens after cleaning.
    """
    return clean(text).split()


def preprocess_corpus(texts, verbose: bool = True):
    """
    Apply clean() to an iterable of texts.

    Parameters
    ----------
    texts : iterable of str
    verbose : bool
        Print progress every 5 000 documents.

    Returns
    -------
    List[str]
        Cleaned texts (as strings, not token lists).
    """
    cleaned = []
    for i, text in enumerate(texts):
        cleaned.append(clean(text))
        if verbose and (i + 1) % 5_000 == 0:
            print(f"  Preprocessed {i + 1:,} documents …")
    return cleaned
