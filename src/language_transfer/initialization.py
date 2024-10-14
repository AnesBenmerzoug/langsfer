import logging
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from transformers import PreTrainedTokenizerBase
from tqdm.auto import tqdm

from language_transfer.alignment import AlignmentStrategy, IdentityAlignment
from language_transfer.embeddings import TransformersEmbeddings, FastTextEmbeddings
from language_transfer.similarity import SimilarityStrategy, CosineSimilarity
from language_transfer.weight import WeightsStrategy, IdentityWeights
from language_transfer.token_overlap import TokenOverlapStrategy, NoTokenOverlap

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

    def initialize(
        self, *, seed: int | None = None, show_progress: bool = False
    ) -> TransformersEmbeddings:
        rng = np.random.default_rng(seed)

        # Map source and target subword tokens to auxiliary token space
        source_subword_embeddings = (
            self._map_subword_embeddings_in_word_embedding_space(
                self.source_main_embeddings._tokenizer,
                self.source_auxiliary_embeddings,
            )
        )
        target_subword_embeddings = (
            self._map_subword_embeddings_in_word_embedding_space(
                self.target_tokenizer,
                self.target_auxiliary_embeddings,
            )
        )

        # Align source to target
        source_subword_embeddings = self.alignment_strategy(source_subword_embeddings)

        # TODO: investigate why this is needed
        source_subword_embeddings /= (
            np.linalg.norm(source_subword_embeddings, axis=1)[:, np.newaxis] + 1e-8
        )
        target_subword_embeddings /= (
            np.linalg.norm(target_subword_embeddings, axis=1)[:, np.newaxis] + 1e-8
        )

        # Compute weights
        similarities = self.similarity_strategy(
            target_subword_embeddings,
            source_subword_embeddings,
        )
        weights = self.weights_strategy(similarities)

        # Initialize target embeddings as random
        target_embeddings_matrix = rng.normal(
            np.mean(self.source_embeddings.embeddings_matrix, axis=0),
            np.std(self.source_embeddings.embeddings_matrix, axis=0),
            (
                len(self.target_tokenizer.vocabulary),
                self.source_main_embeddings.embeddings_matrix.shape[1],
            ),
        ).astype(self.source_embeddings.embeddings_matrix.dtype)

        # Find overlapping and non-overlapping tokens using token overlap strategy
        overlapping_tokens, non_overlapping_tokens = self.token_overlap_strategy.apply(
            self.source_embeddings._tokenizer, self.target_tokenizer
        )

        # Copy overlapping token embedding vectors
        for token in tqdm(
            overlapping_tokens, desc="Overlapping Tokens", disable=not show_progress
        ):
            target_token_id = self.target_tokenizer.convert_tokens_to_ids(token)
            target_embeddings_matrix[target_token_id] = (
                self.source_embeddings.get_vector_for_token(token)
            )

        # Compute remaining target embedding vectors
        # as weighted average of source tokens
        embedding_vectors = np.average(
            self.source_embeddings.embeddings_matrix,
            weights=weights,
            axis=0,
            keepdims=True,
        )

        for token in tqdm(
            non_overlapping_tokens,
            desc="Non Overlapping Tokens",
            disable=not show_progress,
        ):
            target_token_id = self.target_tokenizer.convert_tokens_to_ids(token)
            target_embeddings_matrix[target_token_id] = embedding_vectors[
                self.source_embeddings.get_id_for_token(token)
            ]

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
            (len(tokenizer.vocabulary), embeddings.embeddings_matrix.shape[1])
        )

        for i in range(len(tokenizer.vocabulary)):
            token = tokenizer.get_token_for_id(i)

            # `get_word_vector` returns zeros if not able to decompose
            embeddings_matrix[i] = embeddings.get_vector_for_token(token)

        return embeddings_matrix
