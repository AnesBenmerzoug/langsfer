"""This module provides various weight strategies for transforming input scores
into weights that are used to compute target embedding vectors as weighted averages.
The strategies are designed to modify the input scores based on different
criteria, such as ranking, sparsity, or normalization.

The `WeightsStrategy` class is an abstract base class that defines the interface for weight strategies.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy.special import softmax
from numpy.typing import NDArray

__all__ = ["IdentityWeights", "SoftmaxWeights", "SparsemaxWeights", "TopKWeights"]


class WeightsStrategy(ABC):
    """Abstract base class for weight computation strategies.

    This class defines the interface for applying a weight transformation to input scores.
    Subclasses must implement the `_compute_weights` method to define their specific weight computation logic.

    Additionally, the `apply` is a concrete method that handles the application of the transformation,
    with optional chaining of multiple weight strategies via the `compose` method.

    Attributes:
        _next_strategy: Optionally references another `WeightsStrategy` that can be applied after the current one.
    """

    _next_strategy: Optional["WeightsStrategy"] = None

    def apply(self, scores: NDArray) -> NDArray:
        """Applies the weight transformation to the input scores.

        This method first checks that the input scores are two-dimensional, computes the weights using
        the `_compute_weights` method, and optionally applies a next strategy if defined.

        Args:
            scores: A 2D array of input scores to be transformed.

        Returns:
            A 2D array of transformed weights.

        Raises:
            RuntimeError: If the input scores or output weights are not 2-dimensional.
        """
        if scores.ndim != 2:
            raise RuntimeError(
                f"scores must have 2 dimensions instead of {scores.ndim}"
            )
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
        """Chains another weight strategy to apply after the current strategy.

        The resulting strategy will apply the current strategy first, then apply the
        `other` strategy to the transformed weights.

        Args:
            other: Another `WeightsStrategy` to apply after the current one.

        Returns:
            The current strategy, now with a chained `other` strategy.

        Raises:
            ValueError: If `other` is not an instance of `WeightsStrategy`.
        """
        if not isinstance(other, WeightsStrategy):
            raise ValueError(
                f"other must be an instance of WeightsStrategy instead of {type(other)}"
            )
        self._next_strategy = other
        return self


class IdentityWeights(WeightsStrategy):
    """Weight strategy that returns the input scores unchanged.

    This strategy applies no transformation to the input scores and simply returns them as is.

    Example:
        >>> weight_strategy = IdentityWeights()
        >>> scores = np.array([[1.0, 2.0, 3.0], [4.0, 4.0, 4.0]])
        >>> weight_strategy.apply(scores).tolist()
        [[1.0, 2.0, 3.0], [4.0, 4.0, 4.0]]
    """

    def _compute_weights(self, scores: NDArray) -> NDArray:
        """Returns the input scores without any modification.

        Args:
            scores: A 2D array of input scores.

        Returns:
            The same input scores as the output.
        """
        return scores


class SoftmaxWeights(WeightsStrategy):
    """Weight strategy that applies the softmax transformation to the input scores.

    Softmax normalizes the scores into a probability distribution, where each score is divided by the
    sum of the exponentials of all scores in the row, resulting in values between 0 and 1.

    Args:
        temperature: A scaling factor applied to the scores before applying softmax.

    Example:
        >>> weight_strategy = SoftmaxWeights()
        >>> scores = np.array([[1.0, 2.0, 3.0], [4.0, 4.0, 4.0]])
        >>> weight_strategy.apply(scores).tolist()
        [[0.09003058735208934, 0.24472848513183193, 0.6652409275160788], [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]]
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self._epsilon = 1e-7
        self.temperature = temperature + self._epsilon

    def _compute_weights(self, scores: NDArray) -> NDArray:
        """Computes the softmax weights for the input scores.

        Softmax is applied by dividing the scores by the specified temperature to control the spread
        of the probability distribution.

        Args:
            scores: A 2D array of input scores.

        Returns:
            A 2D array of transformed scores that represent a probability distribution.
        """
        weights = softmax(scores / self.temperature, axis=1)
        return weights


class SparsemaxWeights(WeightsStrategy):
    """Weight strategy that applies the Sparsemax transformation to the input scores.

    Sparsemax is a sparse alternative to softmax, where less significant values are set to zero.
    This implementation follows the method described in the paper:
    [From softmax to sparsemax: A sparse model of attention and multi-label classification.](https://proceedings.mlr.press/v48/martins16)

    The implementation is a slightly modified version of this code:
    https://github.com/AndreasMadsen/course-02456-sparsemax/blob/cd73efc1267b5c3b319fb3dc77774c99c10d5d82/python_reference/sparsemax.py#L4
    The original code is license under the [MIT license.](https://github.com/AndreasMadsen/course-02456-sparsemax/blob/cd73efc1267b5c3b319fb3dc77774c99c10d5d82/LICENSE.md)

    Examples:
        >>> weight_strategy = SparsemaxWeights()
        >>> scores = np.array([[1.0, 2.0, 3.0], [4.0, 4.0, 4.0]])
        >>> weight_strategy.apply(scores).tolist()
        [[0.0, 0.0, 1.0], [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]]
    """

    def _compute_weights(self, scores: NDArray) -> NDArray:
        """Computes the Sparsemax weights for the input scores.

        Sparsemax is computed by first sorting the scores, calculating a threshold `tau`, and
        then zeroing out values that are below this threshold.

        Args:
            scores: A 2D array of input scores.

        Returns:
            A 2D array of transformed weights with sparsity enforced.
        """
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
    """Weight strategy that retains only the top-k highest values per row and sets all other values to -np.inf.

    This strategy is useful in situations where only the top-k scores are relevant, and the remaining values
    should be ignored in subsequent computations, e.g. if this strategy is followed by the softmax strategy then those values become 0.

    This implementation method is heavily inspired by the one provided in the following
    stackoverflow answer: https://stackoverflow.com/a/59405060
    The original code is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

    Args:
        k: The number of top values to retain for each row.

    Examples:
        >>> scores = np.array([[1.0, 2.0, 3.0], [4.0, 4.0, 4.0]])
        >>> weight_strategy = TopKWeights(k=1)
        >>> weight_strategy.apply(scores).tolist()
        [[-inf, -inf, 3.0], [4.0, 4.0, 4.0]]
    """

    def __init__(self, k: int = 10) -> None:
        self.k = k

    def _compute_weights(self, scores: NDArray) -> NDArray:
        """Computes the Top-K weights for the input scores.

        The scores are first partitioned to find the top-k values per row. All other values are set to -np.inf.

        Args:
            scores: A 2D array of input scores.

        Returns:
            A 2D array of weights with only the top-k values per row kept and others replaced with -np.inf.
        """
        # Get unsorted indices of top-k values
        topk_indices = np.argpartition(scores, -self.k, axis=1)[:, -self.k :]
        rows, _ = np.indices((scores.shape[0], self.k))
        kth_vals = scores[rows, topk_indices].min(axis=1, keepdims=True)
        # Get boolean mask of values smaller than k-th
        is_smaller_than_kth = scores < kth_vals
        # Replace smaller values with -np.inf
        weights = np.where(is_smaller_than_kth, -np.inf, scores)
        return weights
