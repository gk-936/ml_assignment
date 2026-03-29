"""
Bag-of-Words (BoW) Vectorizer
==============================
Purpose:
    Converts raw text documents into sparse count vectors using the
    Bag-of-Words model. Each dimension in the output vector corresponds
    to a vocabulary term; the value is the raw term count in that document.

Theory:
    BoW originates from the distributional hypothesis (Harris, 1954):
    meaning can be inferred from term co-occurrence statistics. For a
    vocabulary of size V, document d is represented as a V-dimensional
    vector where position w holds c(w, d) — the count of word w in d.
    Word order and grammar are deliberately discarded; only frequency
    matters. This makes BoW extremely fast and memory-efficient, at the
    cost of losing all sequential context.

Reference:
    Harris, Z. S. (1954). Distributional structure.
    Word, 10(2-3), 146-162.

Usage:
    from vectorizers.bow_vectorizer import BoWVectorizer
    vec = BoWVectorizer()
    X_train = vec.fit_transform(train_texts)
    X_test  = vec.transform(test_texts)
"""

from typing import List

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from preprocessing.text_cleaner import clean
from .base import BaseVectorizer


class BoWVectorizer(BaseVectorizer):
    """
    Bag-of-Words vectorizer wrapping sklearn's CountVectorizer.

    Inherits from BaseVectorizer and follows the shared .fit() /
    .transform() interface used by all vectorizers in this benchmark.

    BoW represents each document as a sparse count vector over a fixed
    vocabulary. It ignores word order entirely — 'not toxic' and 'toxic
    not' produce identical vectors. Despite this limitation, BoW is a
    strong baseline because raw term frequency still carries substantial
    discriminative signal for toxicity detection.

    Bigrams (ngram_range=(1,2)) are included to capture limited local
    context such as 'not bad' or 'very toxic', partially compensating
    for the loss of word order.

    Parameters
    ----------
    max_features : int, optional
        Maximum vocabulary size (default 10000). Only the top
        max_features terms by corpus frequency are kept.
    ngram_range : tuple, optional
        Range of n-gram sizes to extract (default (1, 2) — unigrams
        and bigrams).
    """

    def __init__(self, max_features: int = 10000, ngram_range: tuple = (1, 2)):
        super().__init__(name="BoW")
        self.max_features = max_features
        self.ngram_range = ngram_range
        self._model: CountVectorizer = None

    def fit(self, texts: List[str]) -> "BoWVectorizer":
        """
        Learn the vocabulary from a list of training texts.

        Applies the shared text cleaner before fitting so the vocabulary
        matches exactly what transform() will see at inference time.

        Parameters
        ----------
        texts : List[str]
            Raw (uncleaned) training documents.

        Returns
        -------
        BoWVectorizer
            self, to allow method chaining (vec.fit(X).transform(X)).
        """
        print("[BoW] Cleaning training texts...")
        cleaned = [clean(t) for t in texts]

        print(f"[BoW] Fitting CountVectorizer (max_features={self.max_features}, "
              f"ngram_range={self.ngram_range})...")
        self._model = CountVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
        )
        self._model.fit(cleaned)
        self._fitted = True
        vocab_size = len(self._model.vocabulary_)
        print(f"[BoW] Vocabulary fitted — {vocab_size} terms retained.")
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts into dense BoW count vectors.

        Parameters
        ----------
        texts : List[str]
            Raw (uncleaned) documents to vectorize.

        Returns
        -------
        np.ndarray
            Dense array of shape (n_samples, max_features) containing
            raw term counts. dtype is int64.

        Raises
        ------
        RuntimeError
            If called before fit().
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("BoWVectorizer must be fitted before calling transform().")

        print(f"[BoW] Transforming {len(texts)} documents...")
        cleaned = [clean(t) for t in texts]
        # .toarray() converts scipy sparse matrix → dense numpy array.
        # This is required by the shared analysis scripts.
        return self._model.transform(cleaned).toarray()
