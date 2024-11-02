import numpy as np
import pytest
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as numpy_st
from numpy.typing import NDArray

from langsfer.weights import (
    IdentityWeights,
    SoftmaxWeights,
    SparsemaxWeights,
    TopKWeights,
)


@given(
    input_scores=numpy_st.arrays(
        dtype=st.one_of(st.just(np.float32), st.just(np.float64)),
        shape=numpy_st.array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=5),
        fill=st.floats(allow_nan=False, allow_infinity=False, width=32),
    ),
)
def test_identity_weights(input_scores: NDArray):
    weight_strategy = IdentityWeights()
    np.testing.assert_equal(weight_strategy.apply(input_scores), input_scores)


@pytest.mark.parametrize(
    "input_scores, temperature",
    [
        (np.array([[0.0]]), 0.0),
        (np.array([[0.0]]), 1.0),
        (np.array([[100.0]]), 0.0),
        (np.array([[100.0]]), 1.0),
    ],
)
def test_softmax_weights_single_value_per_row(
    input_scores: NDArray, temperature: float
):
    weight_strategy = SoftmaxWeights(temperature)
    np.testing.assert_equal(
        weight_strategy.apply(input_scores), np.ones_like(input_scores)
    )


@pytest.mark.parametrize(
    "input_scores, temperature, expected_weights",
    [
        (np.array([[0.0, 1.0]]), 1.0, np.array([[0.26, 0.73]])),
        (np.array([[0.0, 1.0]]), 0.0, np.array([[0.0, 1.0]])),
        (np.array([[0.0, 1.0], [1.0, 1.0]]), 0.5, np.array([[0.11, 0.88], [0.5, 0.5]])),
    ],
)
def test_softmax_weights(
    input_scores: NDArray, temperature: float, expected_weights: NDArray
):
    weight_strategy = SoftmaxWeights(temperature)
    weights = weight_strategy.apply(input_scores)
    np.testing.assert_almost_equal(weights, expected_weights, decimal=2)
    np.testing.assert_allclose(np.sum(weights, axis=1), np.ones(weights.shape[0]))


@pytest.mark.parametrize(
    "input_scores, expected_weights",
    [
        (np.array([[0, 1.0]]), np.array([[0, 1.0]])),
        (np.array([[0, 1.0], [1.0, 1.0]]), np.array([[0, 1.0], [0.5, 0.5]])),
        (np.array([[-2, 0, 0.5]]), np.array([[0, 0.25, 0.75]])),
        (
            np.array([[0.49, -0.13, 0.64, 1.52, -0.23]]),
            np.array([[0, 0, 0.06, 0.94, 0]]),
        ),
        (np.array([[0, 1.0, 2.0], [10, 20, 30]]), np.array([[0, 0, 1.0], [0, 0, 1.0]])),
    ],
)
def test_sparsemax_weights(input_scores: NDArray, expected_weights: NDArray):
    weight_strategy = SparsemaxWeights()
    weights = weight_strategy.apply(input_scores)
    np.testing.assert_almost_equal(weights, expected_weights, decimal=2)
    np.testing.assert_allclose(np.sum(weights, axis=1), np.ones(weights.shape[0]))


@pytest.mark.parametrize(
    "input_scores, k, expected_weights",
    [
        (np.array([[3, 5, 10]]), 1, np.array([[-np.inf, -np.inf, 10]])),
        (np.array([[3, 5, 10]]), 2, np.array([[-np.inf, 5, 10]])),
        (np.array([[3, 5, 10]]), 3, np.array([[3, 5, 10]])),
        (
            np.array([[3, 5, 10], [100, -1, 11]]),
            2,
            np.array([[-np.inf, 5, 10], [100, -np.inf, 11]]),
        ),
    ],
)
def test_topk_weights(input_scores: NDArray, k: int, expected_weights: NDArray):
    weight_strategy = TopKWeights(k)
    np.testing.assert_equal(weight_strategy.apply(input_scores), expected_weights)
