from abc import ABC, abstractmethod

from fastdist.fastdist import cosine_matrix_to_matrix
from numpy.typing import NDArray


__all__ = ["CosineSimilarity"]


class SimilarityStrategy(ABC):
    @abstractmethod
    def apply(self, v: NDArray, w: NDArray) -> NDArray: ...


class CosineSimilarity(SimilarityStrategy):
    def apply(self, v: NDArray, w: NDArray) -> NDArray:
        return cosine_matrix_to_matrix(v, w)
