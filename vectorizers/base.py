from abc import ABC, abstractmethod
from typing import List
import numpy as np

class BaseVectorizer(ABC):
    def __init__(self, name: str):
        self.name = name
        self._fitted = False

    @abstractmethod
    def fit(self, texts: List[str]) -> "BaseVectorizer": ...

    @abstractmethod
    def transform(self, texts: List[str]) -> np.ndarray: ...

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        return self.fit(texts).transform(texts) 
