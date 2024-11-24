"""This module provides strategies for aligning embedding matrices using different techniques.

The `AlignmentStrategy` class is an abstract base class that defines the interface for embedding alignment strategies.
"""

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
    """Abstract base class for defining strategies to align embedding matrices.

    Subclasses must implement the `apply` method to define the logic for aligning
    the embedding matrix based on their specific alignment technique.
    """

    @abstractmethod
    def apply(self, embedding_matrix: NDArray) -> NDArray: ...


class IdentityAlignment(AlignmentStrategy):
    """Alignment strategy that does not alter the input embedding matrix.

    This strategy simply returns the input embedding matrix unchanged.

    Example:
        >>> identity_alignment = IdentityAlignment()
        >>> aligned_embeddings = identity_alignment.apply(embedding_matrix)
        >>> # aligned_embeddings will be the same as embedding_matrix
    """

    def apply(self, embedding_matrix: NDArray) -> NDArray:
        """Returns the input embedding matrix unchanged.

        Args:
            embedding_matrix: 2D embedding matrix to be aligned.

        Returns:
            The same embedding matrix as the output, without any modifications.
        """
        return embedding_matrix


class BilingualDictionaryAlignment(AlignmentStrategy):
    """Alignment strategy that uses a bilingual dictionary to compute the alignment matrix.

    This strategy uses word pairs from a bilingual dictionary to compute an alignment
    matrix between the source and target embedding matrices. The dictionary maps words in the
    source language to words in the target language. The alignment matrix is computed by
    applying orthogonal Procrustes analysis to the word vector correspondences.

    The bilingual dictionary maps words in the source language to words in the target language
    and is expected to be of the form:

    ```
    english_word1 \t target_word1\n
    english_word2 \t target_word2\n
    ...
    english_wordn \t target_wordn\n
    ```

    Args:
        source_word_embeddings: Word embeddings of the source language.
        target_word_embeddings: Word embeddings of the target language.
        bilingual_dictionary: Dictionary mapping words in source language to words in target language.
        bilingual_dictionary_file: Path to a bilingual dictionary file containing word pairs.
    """

    def __init__(
        self,
        source_word_embeddings: FastTextEmbeddings,
        target_word_embeddings: FastTextEmbeddings,
        bilingual_dictionary: dict[str, list[str]] | None = None,
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
        """Loads a bilingual dictionary from a file.

        The file is expected to contain word pairs, one per line, separated by tabs, e.g.:

        ```
        english_word1 \t target_word1\n
        english_word2 \t target_word2\n
        ...
        english_wordn \t target_wordn\n
        ```

        Args:
            file_path: Path to the bilingual dictionary file.

        Returns:
            A dictionary where the keys are source language words, and the values are lists of target language words.
        """
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

    def _compute_alignment_matrix(self) -> NDArray:
        """Computes the alignment matrix using the bilingual dictionary.

        The method iterates over the bilingual dictionary, retrieving word vector correspondences from the
        source and target language embeddings. It uses orthogonal Procrustes analysis to compute the
        transformation matrix that aligns the source word embeddings with the target word embeddings.

        Returns:
            A 2D array representing the alignment matrix.
        """
        logger.info(
            "Computing word embedding alignment matrix from bilingual dictionary"
        )
        correspondences = []

        for source_word in self.bilingual_dictionary:
            for target_word in self.bilingual_dictionary[source_word]:
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
                    # Check if 'src_w' is a valid token in source word embeddings
                    try:
                        self.source_word_embeddings.get_id_for_token(src_w)
                    except KeyError:
                        logger.debug(
                            "Could not find source embedding id for word '%s'", src_w
                        )
                        continue

                    # Check if 'tgt_w' is a valid token in target word embeddings
                    try:
                        self.target_word_embeddings.get_id_for_token(tgt_w)
                    except KeyError:
                        logger.debug(
                            "Could not find source embedding id for word '%s'", tgt_w
                        )
                        continue

                    src_word_vector = self.source_word_embeddings.get_vector_for_token(
                        src_w
                    )
                    tgt_word_vector = self.target_word_embeddings.get_vector_for_token(
                        tgt_w
                    )
                    correspondences.append([src_word_vector, tgt_word_vector])

        correspondences = np.array(correspondences)

        alignment_matrix, _ = orthogonal_procrustes(
            correspondences[:, 0], correspondences[:, 1]
        )

        return alignment_matrix

    def apply(self, embedding_matrix: NDArray) -> NDArray:
        """Applies the computed alignment matrix to the given embedding matrix.

        The embedding matrix is transformed by multiplying it with the alignment matrix
        obtained from the bilingual dictionary.

        Args:
            embedding_matrix: 2D embedding matrix to be aligned.

        Returns:
            Aligned embedding matrix.
        """
        alignment_matrix = self._compute_alignment_matrix()
        aligned_embedding_matrix = embedding_matrix @ alignment_matrix
        return aligned_embedding_matrix
