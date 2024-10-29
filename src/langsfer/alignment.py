import logging
import os
import warnings
from abc import ABC, abstractmethod
from itertools import product

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import orthogonal_procrustes

from langsfer.embeddings import FastTextEmbeddings

__all__ = ["IdentityAlignment", "BilingualDictionaryAlignment"]

logger = logging.getLogger(__name__)


class AlignmentStrategy(ABC):
    @abstractmethod
    def apply(self, embedding_matrix: NDArray) -> NDArray: ...


class IdentityAlignment(AlignmentStrategy):
    def apply(self, embedding_matrix: NDArray) -> NDArray:
        return embedding_matrix


class BilingualDictionaryAlignment(AlignmentStrategy):
    """Alignment strategy that uses a bilingual dictionary to compute the alignment matrix.

    The bilingual dictionary maps words in the source language to words in the target language
    and is expected to be of the form:

    ```
    english_word1 \t target_word1\n
    english_word2 \t target_word2\n
    ...
    english_wordn \t target_wordn\n
    ```

    Args:
        source_word_embeddings: Word embeddings of the source language
        target_word_embeddings: Word embeddings of the target language
        bilingual_dictionary: Dictionary mapping words in source language to words in target language
        bilingual_dictionary_file: Path to a bilingual dictionary file
    """

    def __init__(
        self,
        source_word_embeddings: FastTextEmbeddings,
        target_word_embeddings: FastTextEmbeddings,
        bilingual_dictionary: dict[str, list[str]] | None,
        bilingual_dictionary_file: str | os.PathLike | None = None,
    ) -> None:
        if bilingual_dictionary is None and bilingual_dictionary_file is None:
            raise ValueError(
                "At least one of bilingual dictionary or file must be provided"
            )

        if bilingual_dictionary is not None and bilingual_dictionary_file is not None:
            warnings.warn(
                "Both bilingual dictionary and file were provided. Using dictionary."
            )

        self.source_word_embeddings = source_word_embeddings
        self.target_word_embeddings = target_word_embeddings
        self.bilingual_dictionary = bilingual_dictionary
        self.bilingual_dictionary_file = bilingual_dictionary_file
        if self.bilingual_dictionary is None:
            self.bilingual_dictionary = self._load_bilingual_dictionary(
                self.bilingual_dictionary_file
            )

    @staticmethod
    def _load_bilingual_dictionary(
        file_path: str | os.PathLike,
    ) -> dict[str, list[str]]:
        bilingual_dictionary: dict[str, list[str]] = {}

        for line in open(file_path):
            line = line.strip()
            try:
                source_word, target_word = line.split("\t")
            except ValueError:
                source_word, target_word = line.split()

            if source_word not in bilingual_dictionary:
                bilingual_dictionary[source_word] = list()
            bilingual_dictionary[source_word].append(target_word)

        return bilingual_dictionary

    @staticmethod
    def _compute_alignment_matrix(
        source_word_embeddings: FastTextEmbeddings,
        target_word_embeddings: FastTextEmbeddings,
        bilingual_dictionary: dict[str, list[str]],
    ) -> NDArray:
        logger.info(
            "Computing word embedding alignment matrix from bilingual dictionary"
        )
        correspondences = []

        for source_word in bilingual_dictionary:
            for target_word in bilingual_dictionary[source_word]:
                source_word_variants = (
                    source_word,
                    source_word.lower(),
                    source_word.title(),
                )
                target_word_variants = (
                    target_word,
                    target_word.lower(),
                    target_word.title(),
                )

                for src_w, tgt_w in product(source_word_variants, target_word_variants):
                    try:
                        src_word_vector = source_word_embeddings.get_vector_for_token(
                            src_w
                        )
                    except KeyError:
                        logger.debug(
                            f"Could not find source word embedding for word '{src_w}'"
                        )
                        continue

                    try:
                        tgt_word_vector = target_word_embeddings.get_vector_for_token(
                            tgt_w
                        )
                    except KeyError:
                        logger.debug(
                            f"Could not find target word embedding for word '{tgt_w}'"
                        )
                        continue

                    correspondences.append([src_word_vector, tgt_word_vector])

        correspondences = np.array(correspondences)

        alignment_matrix, _ = orthogonal_procrustes(
            correspondences[:, 0], correspondences[:, 1]
        )

        return alignment_matrix

    def apply(self, embedding_matrix: NDArray) -> NDArray:
        alignment_matrix = self._compute_alignment_matrix(
            self.source_word_embeddings,
            self.target_word_embeddings,
            self.bilingual_dictionary,
        )
        aligned_embedding_matrix = embedding_matrix @ alignment_matrix
        return aligned_embedding_matrix
