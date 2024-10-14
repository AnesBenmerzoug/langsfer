"""This module contains high-level user interfaces
to make it easier certain well known methods without necessarily
instantiating all the details of the different methods.
"""

import os

from transformers import PreTrainedTokenizerBase

from language_transfer.initialization import WeightedAverageEmbeddingsInitialization
from language_transfer.alignment import BilingualDictionaryAlignment, IdentityAlignment
from language_transfer.embeddings import TransformersEmbeddings, FastTextEmbeddings
from language_transfer.similarity import CosineSimilarity
from language_transfer.weight import IdentityWeights, SoftmaxWeights, TopKWeights
from language_transfer.token_overlap import SpecialTokenOverlap, ExactMatchTokenOverlap


__all__ = ["wechsel", "clp_transfer"]


def wechsel(
    source_embeddings: TransformersEmbeddings,
    target_tokenizer: PreTrainedTokenizerBase,
    target_auxiliary_embeddings: FastTextEmbeddings,
    source_auxiliary_embeddings: FastTextEmbeddings,
    bilingual_dictionary: dict[str, list[str]] | None,
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
    """Cross-Lingual and Progresseive (CLP) Transfer method.

    Described in [CLP-Transfer: Efficient language model training through cross-lingual and progressive transfer learning.](https://arxiv.org/abs/2301.09626) Ostendorff, Malte, and Georg Rehm. arXiv preprint arXiv:2301.09626 (2023).

    Args:
        source_embeddings:
        target_tokenizer:
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
