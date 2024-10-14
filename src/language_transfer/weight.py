from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy.special import softmax
from numpy.typing import NDArray

__all__ = ["IdentityWeights", "SoftmaxWeights"]


class WeightsStrategy(ABC):
    _next_strategy: Optional["WeightsStrategy"] = None

    def apply(self, scores: NDArray) -> NDArray:
        weights = self._compute_weights(scores)
        if self._next_strategy is not None:
            weights = self._next_strategy(weights)
        return weights

    @abstractmethod
    def _compute_weights(self, scores: NDArray) -> NDArray: ...

    def compose(self, other: "WeightsStrategy") -> "WeightsStrategy":
        if not isinstance(other, WeightsStrategy):
            raise ValueError(
                f"other must be an instance of WeightsStrategy instead of {type(other)}"
            )
        self._next_strategy = other


class IdentityWeights(WeightsStrategy):
    def _compute_weights(self, scores: NDArray) -> NDArray:
        return scores


class SoftmaxWeights(WeightsStrategy):
    def __init__(self, temperature: float = 0.3) -> None:
        self._epsilon = 1e-7
        self.temperature = temperature + self._epsilon

    def _compute_weights(self, scores: NDArray) -> NDArray:
        weights = softmax(scores / self.temperature, axis=1)
        return weights


class TopKWeights(WeightsStrategy):
    """Weight strategy that keeps the top-k highest input scores per row
    and sets all the other ones to zero.

    Examples:
        >>> from language_transfer.weights import TopKWeights
        >>> import numpy as np
        >>> weight_strategy = TopKWeights(k=1)
        >>> weight_strategy.apply(np.array([3, 1, 10]))
        array([0, 0, 10])

    Args:
        k: Number of highest values per row to keep
    """

    def __init__(self, k: int = 10) -> None:
        self.k = k

    def _compute_weights(self, scores: NDArray) -> NDArray:
        """
        This method is heavily inspired by the one provided in the following
        stackoverflow answer: https://stackoverflow.com/a/59405060
        The original code is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
        """
        if scores.ndim != 2:
            raise ValueError(
                f"Expected score to have 2 dimensions instead of {scores.ndim}"
            )
        # get unsorted indices of top-k values
        topk_indices = np.argpartition(scores, -self.k, axis=1)[:, -self.k :]
        rows, _ = np.indices((scores.shape[0], self.k))
        kth_vals = scores[rows, topk_indices].min(axis=1, keepdims=True)
        # get boolean mask of values smaller than k-th
        is_smaller_than_kth = scores < kth_vals
        weights = np.where(is_smaller_than_kth, 0, scores)
        return weights
