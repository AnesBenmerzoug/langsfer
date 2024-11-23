"""This module provides classes for embedding initialization methods used
in cross-lingual transfer learning and language model specialization.

These classes implement various strategies for initializing the embedding
layers of target language models based on source language embeddings.
They provide flexible and efficient ways to transfer knowledge from
pre-trained models in one language to models in another language.

Classes in this module allow for fine-tuned control over the embedding
initialization process through several configurable strategies:
- Alignment strategies (e.g., bilingual dictionaries, identity alignment)
- Similarity measures (e.g., cosine similarity)
- Weight computation techniques (e.g., softmax, sparsemax)
- Token overlap strategies (e.g., exact match, fuzzy match)
"""

import logging
from abc import ABC, abstractmethod

import numpy as np
import torch
from more_itertools import chunked
from numpy.typing import NDArray
from transformers import PreTrainedTokenizerBase
from tqdm.auto import tqdm

from langsfer.alignment import AlignmentStrategy, IdentityAlignment
from langsfer.embeddings import AuxiliaryEmbeddings
from langsfer.similarity import SimilarityStrategy, CosineSimilarity
from langsfer.weights import WeightsStrategy, IdentityWeights
from langsfer.token_overlap import TokenOverlapStrategy, NoTokenOverlap

__all__ = [
    "EmbeddingInitializer",
    "RandomEmbeddingsInitialization",
    "WeightedAverageEmbeddingsInitialization",
]

logger = logging.getLogger(__name__)


class EmbeddingInitializer(ABC):
    """Abstract base class for initializing embeddings.

    This class serves as the base for various embedding initialization strategies.
    Subclasses should implement the `initialize` method to compute embeddings based on specific strategies.
    """

    @abstractmethod
    def initialize(
        self, *, seed: int | None = None, show_progress: bool = False
    ) -> NDArray:
        """Abstract method to initialize the embeddings of target tokens.

        This method should be implemented by subclasses to compute and return embeddings.

        Args:
            seed: An optional seed for the random number generator.
            show_progress: If True, displays a progress bar for the initialization process.

        Returns:
            A 2D array containing the initialized target embeddings.
        """
        ...


class RandomEmbeddingsInitialization(EmbeddingInitializer):
    """Random initialization of embeddings using a normal distribution.

    This class initializes embeddings by generating random values based on the mean and
    standard deviation of the source embeddings.

    Args:
        source_embeddings_matrix: A 2D array containing the source embeddings matrix.
        target_tokenizer: A tokenizer for the target language.
    """

    def __init__(
        self,
        source_embeddings_matrix: NDArray,
        target_tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self.source_embeddings_matrix = source_embeddings_matrix
        self.target_tokenizer = target_tokenizer

    @torch.no_grad()
    def initialize(
        self, *, seed: int | None = None, show_progress: bool = False
    ) -> NDArray:
        """Initialize the target embeddings using random values.

        Generates a random target embeddings matrix based on the mean and standard deviation of the source embeddings matrix.

        Args:
            seed: An optional seed for the random number generator.
            show_progress: If True, displays a progress bar for the initialization process.

        Returns:
            A 2D array containing the randomly initialized target embeddings.
        """
        rng = np.random.default_rng(seed)
        target_embeddings_matrix = rng.normal(
            np.mean(self.source_embeddings_matrix, axis=0),
            np.std(self.source_embeddings_matrix, axis=0),
            (
                len(self.target_tokenizer),
                self.source_embeddings_matrix.shape[1],
            ),
        ).astype(self.source_embeddings_matrix.dtype)

        return target_embeddings_matrix


class WeightedAverageEmbeddingsInitialization(EmbeddingInitializer):
    """Weighted average initialization of embeddings based on source embeddings.

    This class computes the target embeddings by first copying the embeddings of overlapping tokens
    from the source model and then computing the embeddings of non-overlapping tokens as a weighted
    average of the source tokens based on similarity.

    Args:
        source_tokenizer: The tokenizer of the source language.
        source_embeddings_matrix: A 2D array containing the source embeddings matrix.
        target_tokenizer: The tokenizer of the target language.
        target_auxiliary_embeddings: FastText auxiliary embeddings for the target language.
        source_auxiliary_embeddings: Optional FastText auxiliary embeddings for the source language.
        alignment_strategy: The strategy used to align source and target embeddings.
        similarity_strategy: The strategy used to compute token similarities.
        weights_strategy: The strategy used to compute token weights.
        token_overlap_strategy: The strategy used to determine token overlap.
        batch_size: The size of batches for non-overlapping token computations.
    """

    def __init__(
        self,
        source_tokenizer: PreTrainedTokenizerBase,
        source_embeddings_matrix: NDArray,
        target_tokenizer: PreTrainedTokenizerBase,
        target_auxiliary_embeddings: AuxiliaryEmbeddings,
        source_auxiliary_embeddings: AuxiliaryEmbeddings | None = None,
        *,
        alignment_strategy: AlignmentStrategy = IdentityAlignment(),
        similarity_strategy: SimilarityStrategy = CosineSimilarity(),
        weights_strategy: WeightsStrategy = IdentityWeights(),
        token_overlap_strategy: TokenOverlapStrategy = NoTokenOverlap(),
        batch_size: int = 1024,
    ) -> None:
        self.source_tokenizer = source_tokenizer
        self.source_embeddings_matrix = source_embeddings_matrix
        self.target_tokenizer = target_tokenizer
        self.source_auxiliary_embeddings = source_auxiliary_embeddings
        self.target_auxiliary_embeddings = target_auxiliary_embeddings
        self.alignment_strategy = alignment_strategy
        self.similarity_strategy = similarity_strategy
        self.weights_strategy = weights_strategy
        self.token_overlap_strategy = token_overlap_strategy
        self.batch_size = batch_size

    @torch.no_grad()
    def initialize(
        self, *, seed: int | None = None, show_progress: bool = False
    ) -> NDArray:
        """Initialize target embeddings using weighted averages.

        This method computes the target embeddings by first copying the embeddings of
        overlapping tokens from the source model and then calculating the embeddings for
        non-overlapping tokens as a weighted average of source tokens based on cosine similarity.

        Args:
            seed: An optional seed for the random number generator.
            show_progress: If True, displays a progress bar for the initialization process.

        Returns:
            A 2D array containing the initialized target embeddings.
        """
        rng = np.random.default_rng(seed)

        # Initialize target embeddings as random
        target_embeddings_matrix = rng.normal(
            np.mean(self.source_embeddings_matrix, axis=0),
            np.std(self.source_embeddings_matrix, axis=0),
            (
                len(self.target_tokenizer),
                self.source_embeddings_matrix.shape[1],
            ),
        ).astype(self.source_embeddings_matrix.dtype)

        # Find overlapping and non-overlapping tokens using token overlap strategy
        overlapping_tokens, non_overlapping_tokens = self.token_overlap_strategy.apply(
            self.source_tokenizer, self.target_tokenizer
        )
        overlapping_source_token_ids = list(
            self.source_tokenizer.convert_tokens_to_ids(overlapping_tokens)
        )
        overlapping_target_token_ids = list(
            self.target_tokenizer.convert_tokens_to_ids(overlapping_tokens)
        )
        non_overlapping_target_token_ids = list(
            self.target_tokenizer.convert_tokens_to_ids(non_overlapping_tokens)
        )

        # Copy overlapping token embedding vectors
        # shape of assigned: (n_target_tokens, n_overlapping_tokens)
        target_embeddings_matrix[overlapping_target_token_ids] = (
            self.source_embeddings_matrix[overlapping_source_token_ids]
        )

        # Compute target embedding vectors of non overlapping tokens
        # as weighted average of source tokens
        target_embeddings_matrix[non_overlapping_target_token_ids] = (
            self._compute_non_overlapping_token_embeddings(
                overlapping_source_token_ids=overlapping_source_token_ids,
                overlapping_target_token_ids=overlapping_target_token_ids,
                non_overlapping_target_token_ids=non_overlapping_target_token_ids,
                show_progress=show_progress,
            )
        )
        return target_embeddings_matrix

    def _compute_non_overlapping_token_embeddings(
        self,
        overlapping_target_token_ids: list[int],
        overlapping_source_token_ids: list[int],
        non_overlapping_target_token_ids: list[int],
        *,
        show_progress: bool = False,
    ) -> NDArray:
        """Compute embeddings for non-overlapping tokens as weighted averages.

        This method calculates the embeddings for non-overlapping target tokens based on a weighted
        average of the source embeddings, using cosine similarity to determine the weights.

        Args:
            overlapping_target_token_ids: List of token IDs for overlapping target tokens.
            overlapping_source_token_ids: List of token IDs for overlapping source tokens.
            non_overlapping_target_token_ids: List of token IDs for non-overlapping target tokens.
            show_progress: If True, displays a progress bar for the initialization process.

        Returns:
            A 2D array containing the embeddings for the non-overlapping target tokens.
        """
        # Map source and target subword tokens to auxiliary token space
        target_subword_embeddings = self._map_tokens_into_auxiliary_embedding_space(
            self.target_tokenizer,
            self.target_auxiliary_embeddings,
        )
        # TODO: investigate why this is needed
        target_subword_embeddings /= (
            np.linalg.norm(target_subword_embeddings, axis=1)[:, np.newaxis] + 1e-8
        )

        if self.source_auxiliary_embeddings is None:
            reference_subword_embeddings = target_subword_embeddings[
                overlapping_target_token_ids
            ].copy()
            source_embeddings_matrix = self.source_embeddings_matrix[
                overlapping_source_token_ids
            ]
        else:
            reference_subword_embeddings = (
                self._map_tokens_into_auxiliary_embedding_space(
                    self.source_tokenizer,
                    self.source_auxiliary_embeddings,
                )
            )

            # Align source to target
            reference_subword_embeddings = self.alignment_strategy.apply(
                reference_subword_embeddings
            )

            # TODO: investigate why this is needed
            reference_subword_embeddings /= (
                np.linalg.norm(reference_subword_embeddings, axis=1)[:, np.newaxis]
                + 1e-8
            )

            source_embeddings_matrix = self.source_embeddings_matrix

        # Compute target embedding vectors of non overlapping tokens
        # as weighted average of source tokens
        target_embedding_vec_batches = []

        for token_batch_ids in tqdm(
            chunked(non_overlapping_target_token_ids, self.batch_size),
            desc="Non-Overlapping Tokens",
            disable=not show_progress,
        ):
            # Compute similarities
            # shape: (batch_size, n_reference_embeddings)
            similarities = self.similarity_strategy.apply(
                target_subword_embeddings[token_batch_ids],
                reference_subword_embeddings,
            )
            # compute weights
            # shape: (batch_size, n_reference_embeddings)
            weights = self.weights_strategy.apply(similarities)

            # weighted average of source model's overlapping token embeddings
            # with weight from cosine similarity in target token embedding space
            # shape: (batch_size,)
            weights_row_sum = weights.sum(axis=1)
            # shape: (batch_size, source_embedding_dim)
            non_overlapping_embedding_vectors = (
                weights @ source_embeddings_matrix / weights_row_sum[:, np.newaxis]
            )

            target_embedding_vec_batches.append(non_overlapping_embedding_vectors)
        return np.concatenate(target_embedding_vec_batches, axis=0)

    @staticmethod
    def _map_tokens_into_auxiliary_embedding_space(
        tokenizer: PreTrainedTokenizerBase,
        embeddings: AuxiliaryEmbeddings,
    ) -> NDArray:
        """Map tokens into the auxiliary embedding space.

        This method converts token IDs from a tokenizer into their corresponding vector
        representations in an auxiliary embedding space. The embeddings are retrieved
        from the provided auxiliary embeddings source.

        Args:
            tokenizer: A pre-trained tokenizer.
            embeddings: An object containing the auxiliary embeddings, which provides
                vector representations for tokens.

        Returns:
            A 2D array of shape (n_tokens, embedding_dim),
            where each row corresponds to the embedding of a token from the tokenizer
            in the auxiliary embedding space.
        """
        embeddings_matrix = np.zeros(
            (len(tokenizer), embeddings.embeddings_matrix.shape[1])
        )

        for i in range(len(tokenizer)):
            # Unlike in the WECHSEL code, we use `convert_ids_to_tokens`
            # instead of `decode` to avoid empty strings
            token: str = tokenizer.convert_ids_to_tokens(i)
            embeddings_matrix[i] = embeddings.get_vector_for_token(token)

        return embeddings_matrix
