"""
Contextual vectorizer for DistilBERT and RoBERTa using HuggingFace.
Extracts the [CLS] token from the last hidden layer as sentence embedding.
Models are frozen - no fine tuning, just using pretrained weights.

Models supported:
    distilbert -> distilbert-base-uncased (66M params, faster)
    roberta    -> roberta-base (125M params, slightly better)

Note: RoBERTa gives a warning about pooler weights not being initialized.
This is fine - we use last_hidden_state directly, not the pooler.
"""
import os
from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))



MODEL_REGISTRY = {
    "bert":       "bert-base-uncased",
    "distilbert": "distilbert-base-uncased",
    "roberta":    "roberta-base",
}


class ContextualVectorizer:

    def __init__(
        self,
        model_name: str = "distilbert",
        batch_size: int = 32,
        max_length: int = 128,
        device: str = None,
    ):
        # Resolve short name → full HF model ID
        resolved = MODEL_REGISTRY.get(model_name.lower(), model_name)
        # Use the short name as the display name if it maps cleanly
        display_name = model_name.upper() if model_name.lower() in MODEL_REGISTRY else model_name
        self.name = display_name
        self._fitted = False

        self.model_id   = resolved
        self.batch_size = batch_size
        self.max_length = max_length

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._tokenizer = None
        self._model     = None

    def fit(self, texts: List[str]) -> "ContextualVectorizer":
        """Load the model. texts not used since these are pretrained."""
        print(f"[{self.name}] Loading '{self.model_id}' on {self.device} …")

        # AutoTokenizer and AutoModel handle all three architectures
        # automatically — no need for model-specific imports.
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model     = AutoModel.from_pretrained(self.model_id)
        self._model.eval()
        self._model.to(self.device)

        n_params = sum(p.numel() for p in self._model.parameters())
        print(f"[{self.name}] loaded. {n_params:,} params")
        self._fitted = True
        return self

    @torch.no_grad()
    def _encode_batch(self, batch: List[str]) -> np.ndarray:
        # encode one batch, returns numpy array of shape (batch_size, 768)
        encoded = self._tokenizer(
            batch,
            padding        = True,
            truncation     = True,
            max_length     = self.max_length,
            return_tensors = "pt",
        )

        # Move all tensors to device
        inputs = {k: v.to(self.device) for k, v in encoded.items()}

        outputs = self._model(**inputs)

        # [CLS] token = index 0 of last hidden state
        cls_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return cls_vectors  # shape: (batch_size, hidden_size)

    def transform(self, texts: List[str]) -> np.ndarray:
        """Run inference in batches and return CLS token embeddings."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        n_batches   = (len(texts) + self.batch_size - 1) // self.batch_size
        all_vectors = []

        print(f"[{self.name}] Encoding {len(texts):,} texts in {n_batches} batches "
              f"(batch_size={self.batch_size}, max_len={self.max_length}) …")

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vecs  = self._encode_batch(batch)
            all_vectors.append(vecs)

            batches_done = (i // self.batch_size) + 1
            if batches_done % 50 == 0 or batches_done == n_batches:
                print(f"  Batch {batches_done}/{n_batches} …")

        result = np.vstack(all_vectors)
        print(f"[{self.name}] Done.  Output shape: {result.shape}")
        return result
