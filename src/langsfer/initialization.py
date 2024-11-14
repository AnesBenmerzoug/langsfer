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
    @abstractmethod
    def initialize(
        self, *, seed: int | None = None, show_progress: bool = False
    ) -> NDArray: ...


class RandomEmbeddingsInitialization(EmbeddingInitializer):
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
        embeddings_matrix = np.zeros(
            (len(tokenizer), embeddings.embeddings_matrix.shape[1])
        )

        for i in range(len(tokenizer)):
            # Unlike in the WECHSEL code, we use `convert_ids_to_tokens`
            # instead of `decode` to avoid empty strings
            token: str = tokenizer.convert_ids_to_tokens(i)
            embeddings_matrix[i] = embeddings.get_vector_for_token(token)

        return embeddings_matrix
