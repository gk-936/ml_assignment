"""
GloVe vectorizer using pretrained Stanford vectors (glove.6B.100d).
Loads the txt file and does mean pooling. No training needed.
Note: GloVe was trained on Wikipedia so it struggles with internet slang.
"""

import os
from typing import List, Dict

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from preprocessing.text_cleaner import tokenize


# Default path — override via constructor if your file is elsewhere
_DEFAULT_GLOVE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "glove", "glove.6B.100d.txt"
)


class GloVeVectorizer:
    

    def __init__(self, glove_path: str = _DEFAULT_GLOVE_PATH, vector_size: int = 100):
        self.name = "GloVe"
        self._fitted = False
        self.glove_path  = glove_path
        self.vector_size = vector_size
        self._vectors: Dict[str, np.ndarray] = {}

    def _load_glove(self) -> None:
        """Read the glove txt file into a dict."""
        if not os.path.exists(self.glove_path):
            raise FileNotFoundError(
                f"GloVe file not found at {self.glove_path}.\n"
                "Download glove.6B.zip from https://nlp.stanford.edu/projects/glove/\n"
                "and place glove.6B.100d.txt at data/glove/glove.6B.100d.txt"
            )

        print(f"[GloVe] Loading vectors from {self.glove_path} …")
        vectors = {}
        with open(self.glove_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                word  = parts[0]
                vec   = np.array(parts[1:], dtype=np.float32)
                vectors[word] = vec

        self._vectors = vectors
        print(f"[GloVe] Loaded {len(vectors):,} word vectors  (dim={self.vector_size}).")

    def fit(self, texts: List[str]) -> "GloVeVectorizer":
        """Load GloVe vectors from file. texts param not used, just here for consistency."""
        self._load_glove()
        self._fitted = True
        return self

    def _embed_sentence(self, tokens: List[str]) -> np.ndarray:
        # mean-pool GloVe vectors; OOV tokens are skipped
        vecs = [self._vectors[tok] for tok in tokens if tok in self._vectors]
        if not vecs:
            return np.zeros(self.vector_size, dtype=np.float32)
        return np.mean(vecs, axis=0)

    def transform(self, texts: List[str]) -> np.ndarray:
        """Look up GloVe vector for each token and average them."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        tokenised = [tokenize(t) for t in texts]
        return np.vstack([self._embed_sentence(toks) for toks in tokenised])


