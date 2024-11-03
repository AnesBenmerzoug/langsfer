from abc import ABC, abstractmethod

from sklearn.metrics.pairwise import cosine_similarity
from numpy.typing import NDArray


__all__ = ["CosineSimilarity"]


class SimilarityStrategy(ABC):
    @abstractmethod
    def apply(self, v: NDArray, w: NDArray) -> NDArray: ...


class CosineSimilarity(SimilarityStrategy):
    def apply(self, v: NDArray, w: NDArray) -> NDArray:
        return cosine_similarity(v, w)
