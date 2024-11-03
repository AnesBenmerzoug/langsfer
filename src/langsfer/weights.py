from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy.special import softmax
from numpy.typing import NDArray

__all__ = ["IdentityWeights", "SoftmaxWeights", "SparsemaxWeights", "TopKWeights"]


class WeightsStrategy(ABC):
    _next_strategy: Optional["WeightsStrategy"] = None

    def apply(self, scores: NDArray) -> NDArray:
        if scores.ndim != 2:
            raise ValueError(f"scores must have 2 dimensions instead of {scores.ndim}")
        weights = self._compute_weights(scores)
        if self._next_strategy is not None:
            weights = self._next_strategy.apply(weights)
        if weights.ndim != 2:
            raise RuntimeError(
                f"expected weights to have 2 dimensions instead of {weights.ndim}"
            )
        return weights

    @abstractmethod
    def _compute_weights(self, scores: NDArray) -> NDArray: ...

    def compose(self, other: "WeightsStrategy") -> "WeightsStrategy":
        if not isinstance(other, WeightsStrategy):
            raise ValueError(
                f"other must be an instance of WeightsStrategy instead of {type(other)}"
            )
        self._next_strategy = other
        return self


class IdentityWeights(WeightsStrategy):
    def _compute_weights(self, scores: NDArray) -> NDArray:
        return scores


class SoftmaxWeights(WeightsStrategy):
    def __init__(self, temperature: float = 1.0) -> None:
        self._epsilon = 1e-7
        self.temperature = temperature + self._epsilon

    def _compute_weights(self, scores: NDArray) -> NDArray:
        weights = softmax(scores / self.temperature, axis=1)
        return weights


class SparsemaxWeights(WeightsStrategy):
    """Implements Sparsemax weight strategy.

    Described in Martins, Andre, and Ramon Astudillo.
    [From softmax to sparsemax: A sparse model of attention and multi-label classification.](https://proceedings.mlr.press/v48/martins16)
    International conference on machine learning. PMLR, 2016.

    The implementation is a slightly modified version of this code:
    https://github.com/AndreasMadsen/course-02456-sparsemax/blob/cd73efc1267b5c3b319fb3dc77774c99c10d5d82/python_reference/sparsemax.py#L4
    The original code is license under the [MIT license.](https://github.com/AndreasMadsen/course-02456-sparsemax/blob/cd73efc1267b5c3b319fb3dc77774c99c10d5d82/LICENSE.md)

    Examples:
        >>> from langsfer.weight import SparsemaxWeights
        >>> import numpy as np
        >>> weights_strategy = SparsemaxWeights()
        >>> scores = np.array([[0.0, 1.0, 2.0], [10, 20, 30]])
        >>> weights_strategy.apply(scores).tolist()
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
    """

    def _compute_weights(self, scores: NDArray) -> NDArray:
        # Translate by max for numerical stability
        scores = scores - scores.max(axis=-1, keepdims=True)

        # Sort scores in descending order
        scores_sorted = np.sort(scores, axis=1)[:, ::-1]

        # Compute k
        scores_cumsum = np.cumsum(scores_sorted, axis=1)
        k_range = np.arange(1, scores_sorted.shape[1] + 1)
        scores_check = 1 + k_range * scores_sorted > scores_cumsum
        k = scores.shape[1] - np.argmax(scores_check[:, ::-1], axis=1)

        # Compute tau(z)
        tau_sum = scores_cumsum[np.arange(0, scores.shape[0]), k - 1]
        tau = ((tau_sum - 1) / k).reshape(-1, 1)

        # Compute weights elementwise as either scores - tau, order 0.0 when the former is negative
        weights = np.maximum(0, scores - tau)
        return weights


class TopKWeights(WeightsStrategy):
    """Weight strategy that keeps the top-k highest input scores per row
    and sets all the other ones to -np.inf in order for them to be ignored in future computations
    e.g. if this strategy is followed by the softmax strategy then those values become 0.

    This implementation method is heavily inspired by the one provided in the following
    stackoverflow answer: https://stackoverflow.com/a/59405060
    The original code is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

    Examples:
        >>> from langsfer.weights import TopKWeights
        >>> import numpy as np
        >>> weight_strategy = TopKWeights(k=1)
        >>> weight_strategy.apply(np.array([[3, 1, 10]])).tolist()
        [[-np.inf, -np.inf, 10]]

    Args:
        k: Number of highest values per row to keep
    """

    def __init__(self, k: int = 10) -> None:
        self.k = k

    def _compute_weights(self, scores: NDArray) -> NDArray:
        # Get unsorted indices of top-k values
        topk_indices = np.argpartition(scores, -self.k, axis=1)[:, -self.k :]
        rows, _ = np.indices((scores.shape[0], self.k))
        kth_vals = scores[rows, topk_indices].min(axis=1, keepdims=True)
        # Get boolean mask of values smaller than k-th
        is_smaller_than_kth = scores < kth_vals
        # Replace smaller values with -np.inf
        weights = np.where(is_smaller_than_kth, -np.inf, scores)
        return weights
