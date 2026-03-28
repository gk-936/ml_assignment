"""
FastText vectorizer using gensim.
Like Word2Vec but uses character n-grams so it can handle OOV words.
Reference: Bojanowski et al. 2017
"""

from typing import List

import numpy as np
from gensim.models import FastText

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from preprocessing.text_cleaner import tokenize



class FastTextVectorizer:
    

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        min_n: int = 3,
        max_n: int = 6,
        workers: int = 4,
        epochs: int = 10,
    ):
        self.name = "FastText"
        self._fitted = False
        self.vector_size = vector_size
        self.window      = window
        self.min_count   = min_count
        self.min_n       = min_n
        self.max_n       = max_n
        self.workers     = workers
        self.epochs      = epochs
        self._model      = None

    def fit(self, texts: List[str]) -> "FastTextVectorizer":
        """Train FastText model on corpus."""
        print(f"[FastText] Tokenising {len(texts):,} documents …")
        tokenised = [tokenize(t) for t in texts]

        print(f"[FastText] training... dim={self.vector_size}, n-grams {self.min_n}-{self.max_n}")
        self._model = FastText(
            sentences  = tokenised,
            vector_size= self.vector_size,
            window     = self.window,
            min_count  = self.min_count,
            min_n      = self.min_n,
            max_n      = self.max_n,
            workers    = self.workers,
            epochs     = self.epochs,
            sg         = 1,   # Skip-gram backbone
        )
        vocab_size = len(self._model.wv)
        print(f"[FastText] Done.  Vocabulary: {vocab_size:,} tokens  "
              f"(OOV words will still get subword-derived vectors).")
        self._fitted = True
        return self

    def _embed_sentence(self, tokens: List[str]) -> np.ndarray:
        # fasttext handles OOV automatically unlike word2vec
        if not tokens:
            return np.zeros(self.vector_size, dtype=np.float32)
        vecs = [self._model.wv[tok] for tok in tokens]
        return np.mean(vecs, axis=0)

    def transform(self, texts: List[str]) -> np.ndarray:
        """Encode texts as mean-pooled FastText vectors. OOV words handled via subword n-grams."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        tokenised = [tokenize(t) for t in texts]
        return np.vstack([self._embed_sentence(toks) for toks in tokenised])
