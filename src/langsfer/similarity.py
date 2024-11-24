"""This module provides classes for computing similarities between 2D arrays."""

from abc import ABC, abstractmethod

from sklearn.metrics.pairwise import cosine_similarity
from numpy.typing import NDArray


__all__ = ["CosineSimilarity"]


class SimilarityStrategy(ABC):
    """Abstract base class for similarity computation strategies.

    This class defines the interface for computing similarities between two 2D arrays.
    Concrete subclasses must implement the `apply` method to define their specific similarity computation logic.
    """

    @abstractmethod
    def apply(self, v: NDArray, w: NDArray) -> NDArray:
        """Abstract method to compute similarity between two 2D arrays.

        Args:
            v: 2D array.
            w: 2D array.

        Returns:
            A 2D array containing the cosine similarity between the input arrays.
        """
        ...


class CosineSimilarity(SimilarityStrategy):
    """Concrete implementation of the `SimilarityStrategy` that computes the cosine similarity
    between two 2D arrays.

    Cosine similarity is a measure of similarity between two vectors based on the cosine of
    the angle between them. It is widely used in tasks such as document comparison, word
    embedding similarity, and more.
    """

    def apply(self, v: NDArray, w: NDArray) -> NDArray:
        """Compute the cosine similarity between two vectors `v` and `w`.

        Cosine similarity is calculated as the dot product of `v` and `w` divided by
        the product of their magnitudes. It returns a value between -1 and 1, where
        1 indicates that the vectors are identical, and -1 indicates that they are
        diametrically opposed.

        Args:
            v: 2D array.
            w: 2D array.

        Returns:
            A 2D array containing the cosine similarity between the input arrays.
        """
        return cosine_similarity(v, w)
