"""
Word2Vec vectorizer using gensim Skip-gram.
Trains on the given corpus and encodes sentences by averaging word vectors.
"""

from typing import List

import numpy as np
from gensim.models import Word2Vec

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from preprocessing.text_cleaner import tokenize



class Word2VecVectorizer:
    

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        workers: int = 4,
        epochs: int = 10,
    ):
        self.name = "Word2Vec"
        self._fitted = False
        self.vector_size = vector_size
        self.window      = window
        self.min_count   = min_count
        self.workers     = workers
        self.epochs      = epochs
        self._model      = None

    def fit(self, texts: List[str]) -> "Word2VecVectorizer":
        """Train Word2Vec on the corpus."""
        print(f"[Word2Vec] Tokenising {len(texts):,} documents …")
        tokenised = [tokenize(t) for t in texts]

        print(f"[Word2Vec] training... dim={self.vector_size}, epochs={self.epochs}")
        self._model = Word2Vec(
            sentences  = tokenised,
            vector_size= self.vector_size,
            window     = self.window,
            min_count  = self.min_count,
            workers    = self.workers,
            epochs     = self.epochs,
            sg         = 1,   # 1 = Skip-gram, 0 = CBOW
        )
        vocab_size = len(self._model.wv)
        print(f"[Word2Vec] Done.  Vocabulary: {vocab_size:,} tokens.")
        self._fitted = True
        return self

    def _embed_sentence(self, tokens: List[str]) -> np.ndarray:
        # mean-pool word vectors for a single tokenised sentence
        wv = self._model.wv
        vecs = [wv[tok] for tok in tokens if tok in wv]
        if not vecs:
            return np.zeros(self.vector_size, dtype=np.float32)
        return np.mean(vecs, axis=0)

    def transform(self, texts: List[str]) -> np.ndarray:
        """Return mean-pooled word vectors for each text."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        tokenised = [tokenize(t) for t in texts]
        return np.vstack([self._embed_sentence(toks) for toks in tokenised])
