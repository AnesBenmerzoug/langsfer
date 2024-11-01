import logging
from abc import ABC, abstractmethod

import numpy as np
import torch
from numpy.typing import NDArray
from transformers import PreTrainedTokenizerBase
from tqdm.auto import tqdm

from langsfer.alignment import AlignmentStrategy, IdentityAlignment
from langsfer.embeddings import TransformersEmbeddings, FastTextEmbeddings
from langsfer.similarity import SimilarityStrategy, CosineSimilarity
from langsfer.weight import WeightsStrategy, IdentityWeights
from langsfer.token_overlap import TokenOverlapStrategy, NoTokenOverlap

__all__ = ["EmbeddingInitializer"]

logger = logging.getLogger(__name__)


class EmbeddingInitializer(ABC):
    @abstractmethod
    def initialize(self, *, seed: int | None = None) -> TransformersEmbeddings: ...


class WeightedAverageEmbeddingsInitialization(EmbeddingInitializer):
    def __init__(
        self,
        source_embeddings: TransformersEmbeddings,
        target_tokenizer: PreTrainedTokenizerBase,
        target_auxiliary_embeddings: FastTextEmbeddings,
        source_auxiliary_embeddings: FastTextEmbeddings | None = None,
        *,
        alignment_strategy: AlignmentStrategy = IdentityAlignment(),
        similarity_strategy: SimilarityStrategy = CosineSimilarity(),
        weights_strategy: WeightsStrategy = IdentityWeights(),
        token_overlap_strategy: TokenOverlapStrategy = NoTokenOverlap(),
    ) -> None:
        self.source_embeddings = source_embeddings
        self.target_tokenizer = target_tokenizer
        self.source_auxiliary_embeddings = source_auxiliary_embeddings
        self.target_auxiliary_embeddings = target_auxiliary_embeddings
        self.alignment_strategy = alignment_strategy
        self.similarity_strategy = similarity_strategy
        self.weights_strategy = weights_strategy
        self.token_overlap_strategy = token_overlap_strategy

    @torch.no_grad()
    def initialize(
        self, *, seed: int | None = None, show_progress: bool = False
    ) -> TransformersEmbeddings:
        rng = np.random.default_rng(seed)

        # Map source and target subword tokens to auxiliary token space
        source_subword_embeddings = self._map_tokens_into_embedding_space(
            self.source_embeddings.tokenizer,
            self.source_auxiliary_embeddings,
        )
        print(f"{source_subword_embeddings.shape=}")
        target_subword_embeddings = self._map_tokens_into_embedding_space(
            self.target_tokenizer,
            self.target_auxiliary_embeddings,
        )
        print(f"{target_subword_embeddings.shape=}")

        # Align source to target
        source_subword_embeddings = self.alignment_strategy.apply(
            source_subword_embeddings
        )
        print(f"{source_subword_embeddings.shape=}")

        # TODO: investigate why this is needed
        source_subword_embeddings /= (
            np.linalg.norm(source_subword_embeddings, axis=1)[:, np.newaxis] + 1e-8
        )
        target_subword_embeddings /= (
            np.linalg.norm(target_subword_embeddings, axis=1)[:, np.newaxis] + 1e-8
        )

        # Initialize target embeddings as random
        target_embeddings_matrix = rng.normal(
            np.mean(self.source_embeddings.embeddings_matrix, axis=0),
            np.std(self.source_embeddings.embeddings_matrix, axis=0),
            (
                len(self.target_tokenizer.vocab),
                self.source_embeddings.embeddings_matrix.shape[1],
            ),
        ).astype(self.source_embeddings.embeddings_matrix.dtype)
        print(f"{target_embeddings_matrix.shape=}")

        # Find overlapping and non-overlapping tokens using token overlap strategy
        overlapping_tokens, non_overlapping_tokens = self.token_overlap_strategy.apply(
            self.source_embeddings.tokenizer, self.target_tokenizer
        )
        print(f"{len(overlapping_tokens)=}")
        print(f"{len(non_overlapping_tokens)=}")

        # Copy overlapping token embedding vectors
        for token in tqdm(
            overlapping_tokens, desc="Overlapping Tokens", disable=not show_progress
        ):
            target_token_id = self.target_tokenizer.convert_tokens_to_ids(token)
            target_embeddings_matrix[target_token_id] = (
                self.source_embeddings.get_vector_for_token(token)
            )

        # Compute weights
        target_non_overlapping_tokens_ids: list[int] = [
            self.target_tokenizer.convert_tokens_to_ids(token)
            for token in non_overlapping_tokens
        ]

        # shape: (n_non_overlapping_tokens, n_source_tokens)
        similarities = self.similarity_strategy.apply(
            target_subword_embeddings[target_non_overlapping_tokens_ids],
            source_subword_embeddings,
        )
        # shape: (n_non_overlapping_tokens, n_source_tokens)
        weights = self.weights_strategy.apply(similarities)
        print(f"{weights.shape=}")

        # Compute remaining target embedding vectors
        # as weighted average of source tokens

        # weighted average of source model's overlapping token embeddings
        # with weight from cosine similarity in target token embedding space
        # shape: (n_non_overlapping_tokens,)
        weights_row_sum = weights.sum(axis=1)
        # shape: (n_non_overlapping_tokens, source_embedding_dim)
        non_overlapping_embedding_vectors = (
            weights
            @ self.source_embeddings.embeddings_matrix
            / weights_row_sum[:, np.newaxis]
        )
        print(f"{non_overlapping_embedding_vectors.shape=}")

        for i, token in tqdm(
            enumerate(non_overlapping_tokens),
            desc="Non Overlapping Tokens",
            disable=not show_progress,
        ):
            target_token_id = self.target_tokenizer.convert_tokens_to_ids(token)
            target_embeddings_matrix[target_token_id] = (
                non_overlapping_embedding_vectors[i]
            )

        # Create target embeddings object
        target_embeddings = TransformersEmbeddings(
            embeddings_matrix=target_embeddings_matrix,
            tokenizer=self.target_tokenizer,
        )
        return target_embeddings

    @staticmethod
    def _map_tokens_into_embedding_space(
        tokenizer: PreTrainedTokenizerBase,
        embeddings: FastTextEmbeddings,
    ) -> NDArray:
        embeddings_matrix = np.zeros(
            (len(tokenizer.vocab), embeddings.embeddings_matrix.shape[1])
        )

        for i in range(len(tokenizer.vocab)):
            token: str = tokenizer.convert_ids_to_tokens(i)

            # `get_word_vector` returns zeros if not able to decompose
            embeddings_matrix[i] = embeddings.get_vector_for_token(token)

        return embeddings_matrix
