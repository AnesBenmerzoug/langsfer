"""This module contains high-level user functions
for well-known methods described in papers and publications.
These are meant to make it easier for users who don't necessarily
want or need to care about all the details of the package.
"""

import os

from transformers import PreTrainedTokenizerBase

from langsfer.initialization import WeightedAverageEmbeddingsInitialization
from langsfer.alignment import BilingualDictionaryAlignment, IdentityAlignment
from langsfer.embeddings import TransformersEmbeddings, FastTextEmbeddings
from langsfer.similarity import CosineSimilarity
from langsfer.weight import (
    IdentityWeights,
    SoftmaxWeights,
    TopKWeights,
    SparsemaxWeights,
)
from langsfer.token_overlap import (
    SpecialTokenOverlap,
    ExactMatchTokenOverlap,
    FuzzyMatchTokenOverlap,
)


__all__ = ["wechsel", "clp_transfer", "focus"]


def wechsel(
    source_embeddings: TransformersEmbeddings,
    target_tokenizer: PreTrainedTokenizerBase,
    target_auxiliary_embeddings: FastTextEmbeddings,
    source_auxiliary_embeddings: FastTextEmbeddings,
    bilingual_dictionary: dict[str, list[str]] | None = None,
    bilingual_dictionary_file: str | os.PathLike | None = None,
    *,
    temperature: float = 0.7,
    k: int = 10,
) -> WeightedAverageEmbeddingsInitialization:
    """WECHSEL method.

    Described in [WECHSEL: Effective initialization of subword embeddings for cross-lingual transfer of monolingual language models.](https://arxiv.org/abs/2112.06598) Minixhofer, Benjamin, Fabian Paischer, and Navid Rekabsaz. arXiv preprint arXiv:2112.06598 (2021)

    Args:
        source_embeddings:
        target_tokenizer: Target language tokenizer
        target_auxiliary_embeddings:
        source_auxiliary_embeddings:
        bilingual_dictionary: Dictionary mapping words in source language to words in target language.
        bilingual_dictionary_file: Path to a bilingual dictionary file
        temperature: Softmax temperature to apply for weight computation
        k: Number of closest / most similar tokens to consider for weight computation
    """
    embeddings_init = WeightedAverageEmbeddingsInitialization(
        source_embeddings,
        target_tokenizer,
        target_auxiliary_embeddings,
        source_auxiliary_embeddings,
        alignment_strategy=BilingualDictionaryAlignment(
            source_auxiliary_embeddings,
            target_auxiliary_embeddings,
            bilingual_dictionary=bilingual_dictionary,
            bilingual_dictionary_file=bilingual_dictionary_file,
        ),
        similarity_strategy=CosineSimilarity(),
        weights_strategy=TopKWeights(k=k).compose(
            SoftmaxWeights(temperature=temperature)
        ),
        token_overlap_strategy=SpecialTokenOverlap(),
    )
    return embeddings_init


def clp_transfer(
    source_embeddings: TransformersEmbeddings,
    target_tokenizer: PreTrainedTokenizerBase,
    target_auxiliary_embeddings: TransformersEmbeddings,
):
    """Cross-Lingual and Progressive (CLP) Transfer method.

    Described in [CLP-Transfer: Efficient language model training through cross-lingual and progressive transfer learning.](https://arxiv.org/abs/2301.09626) Ostendorff, Malte, and Georg Rehm. arXiv preprint arXiv:2301.09626 (2023).

    Args:
        source_embeddings:
        target_tokenizer: Target language tokenizer
        target_auxiliary_embeddings:
    """
    embeddings_init = WeightedAverageEmbeddingsInitialization(
        source_embeddings,
        target_tokenizer,
        target_auxiliary_embeddings,
        alignment_strategy=IdentityAlignment(),
        similarity_strategy=CosineSimilarity(),
        weights_strategy=IdentityWeights(),
        token_overlap_strategy=ExactMatchTokenOverlap(),
    )
    return embeddings_init


def focus(
    source_embeddings: TransformersEmbeddings,
    target_tokenizer: PreTrainedTokenizerBase,
    target_auxiliary_embeddings: FastTextEmbeddings,
    source_auxiliary_embeddings: FastTextEmbeddings,
) -> WeightedAverageEmbeddingsInitialization:
    """Fast Overlapping Token Combinations Using Sparsemax (FOCUS) method.

    Described in [FOCUS: Effective Embedding Initialization for Specializing Pretrained Multilingual Models on a Single Language.](https://arxiv.org/abs/2305.14481) Dobler, Konstantin, and Gerard de Melo. arXiv preprint arXiv:2305.14481 (2023).

    Args:
        source_embeddings:
        target_tokenizer: Target language tokenizer
        target_auxiliary_embeddings:
        source_auxiliary_embeddings:
    """
    embeddings_init = WeightedAverageEmbeddingsInitialization(
        source_embeddings,
        target_tokenizer,
        target_auxiliary_embeddings,
        source_auxiliary_embeddings=source_auxiliary_embeddings,
        alignment_strategy=IdentityAlignment(),
        similarity_strategy=CosineSimilarity(),
        weights_strategy=SparsemaxWeights(),
        token_overlap_strategy=FuzzyMatchTokenOverlap(),
    )
    return embeddings_init
