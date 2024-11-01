from pathlib import Path

import numpy as np
import pytest

from langsfer.alignment import BilingualDictionaryAlignment
from langsfer.embeddings import FastTextEmbeddings


@pytest.fixture(scope="session")
def fasttext_embeddings_en() -> FastTextEmbeddings:
    return FastTextEmbeddings.from_model_name_or_path("en")


@pytest.fixture(scope="session")
def fasttext_embeddings_de() -> FastTextEmbeddings:
    return FastTextEmbeddings.from_model_name_or_path("de")


def test_bilingual_dictionary_alignment(
    fasttext_embeddings_en,
    fasttext_embeddings_de: FastTextEmbeddings,
    german_bilingual_dictionary_file: Path,
    german_bilingual_dictionary_alignment_matrix: np.ndarray,
):
    alignment_strategy = BilingualDictionaryAlignment(
        source_word_embeddings=fasttext_embeddings_en,
        target_word_embeddings=fasttext_embeddings_de,
        bilingual_dictionary_file=german_bilingual_dictionary_file,
    )
    alignment_matrix = alignment_strategy._compute_alignment_matrix()
    assert np.allclose(
        alignment_matrix, german_bilingual_dictionary_alignment_matrix, atol=10e-5
    )
