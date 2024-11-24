"""
This module provides high-level user functions for well-known methods
described in research papers and publications, facilitating their
application for cross-lingual transfer learning in language models.

These functions abstract away the complex details of the underlying methods
to offer an easy-to-use interface for users who want to implement language
model transfer techniques without needing to dive into the low-level
implementation details.

The module supports various strategies such as:
- WECHSEL: Cross-lingual transfer using pre-trained embeddings and a bilingual dictionary.
- CLP Transfer: A cross-lingual and progressive transfer method for efficient language model training.
- FOCUS: Specializing pre-trained multilingual models through efficient token combinations using Sparsemax.

Functions in this module are designed to work with tokenizers and pre-trained
embeddings from various models, including FastText and Transformers.
"""

import os

from numpy.typing import NDArray
from transformers import PreTrainedTokenizerBase

from langsfer.initialization import WeightedAverageEmbeddingsInitialization
from langsfer.alignment import BilingualDictionaryAlignment, IdentityAlignment
from langsfer.embeddings import TransformersEmbeddings, FastTextEmbeddings
from langsfer.similarity import CosineSimilarity
from langsfer.weights import (
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
    source_tokenizer: PreTrainedTokenizerBase,
    source_embeddings_matrix: NDArray,
    target_tokenizer: PreTrainedTokenizerBase,
    target_auxiliary_embeddings: FastTextEmbeddings,
    source_auxiliary_embeddings: FastTextEmbeddings,
    bilingual_dictionary: dict[str, list[str]] | None = None,
    bilingual_dictionary_file: str | os.PathLike | None = None,
    *,
    temperature: float = 0.1,
    k: int = 10,
    batch_size: int = 1024,
) -> WeightedAverageEmbeddingsInitialization:
    """WECHSEL cross-lingual language transfer method.

    Described in [WECHSEL: Effective initialization of subword embeddings for cross-lingual transfer of monolingual language models.](https://arxiv.org/abs/2112.06598) Minixhofer, Benjamin, Fabian Paischer, and Navid Rekabsaz. arXiv preprint arXiv:2112.06598 (2021).

    The WECHSEL method efficiently initializes the embedding parameters of a language model in a target language
    by leveraging the embedding parameters of a pre-trained model in a source language. This facilitates more efficient
    training in the target language by aligning and transferring knowledge from the source language.

    The method requires as input:

    - tokenizer of the source language model,
    - pre-trained language model as source,
    - tokenizer of the target language model,
    - 2 monolingual fastText embeddings for source and target languages respectively.
        They can be obtained in one of 2 ways:

        - using pre-trained fastText embeddings,
        - trainining fastText embeddings from scratch.

    Args:
        source_tokenizer: Tokenizer of the source language model.
        source_embeddings_matrix: 2D matrix containing the weights of the source model's embedding layer.
        target_tokenizer: Tokenizer of the target language model.
        target_auxiliary_embeddings: FastText auxiliary embeddings for the target language.
        source_auxiliary_embeddings: FastText auxiliary embeddings for the source language.
        bilingual_dictionary: Optional dictionary mapping source language words to target language words.
        bilingual_dictionary_file: Optional path to a file containing a bilingual dictionary.
        temperature: Softmax temperature used to adjust weight computation.
        k: Number of closest tokens to consider for weight computation.
        batch_size: Number of tokens to process in each batch for non-overlapping token computations.

    Returns:
        The embedding initializer object for the target model, based on WECHSEL.
    """
    embeddings_initializer = WeightedAverageEmbeddingsInitialization(
        source_tokenizer=source_tokenizer,
        source_embeddings_matrix=source_embeddings_matrix,
        target_tokenizer=target_tokenizer,
        target_auxiliary_embeddings=target_auxiliary_embeddings,
        source_auxiliary_embeddings=source_auxiliary_embeddings,
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
        batch_size=batch_size,
    )
    return embeddings_initializer


def clp_transfer(
    source_tokenizer: PreTrainedTokenizerBase,
    source_embeddings_matrix: NDArray,
    target_tokenizer: PreTrainedTokenizerBase,
    target_auxiliary_embeddings: TransformersEmbeddings,
    *,
    batch_size: int = 1024,
):
    """Cross-Lingual and Progressive (CLP) Transfer method.

    Described in [CLP-Transfer: Efficient language model training through cross-lingual and progressive transfer learning.](https://arxiv.org/abs/2301.09626) Ostendorff, Malte, and Georg Rehm. arXiv preprint arXiv:2301.09626 (2023).

    CLP Transfer is a technique that combines cross-lingual and progressive transfer learning for efficient training
    of language models. The method initializes the target embeddings by transferring knowledge from a source model
    through embeddings and auxiliary information.

    The method requires as input:

    - tokenizer of the source language model,
    - pre-trained language model as source,
    - tokenizer of the target language model,
    - helper pre-trained language model in the target language.

    Args:
        source_tokenizer: Tokenizer of the source language model.
        source_embeddings_matrix: 2D matrix containing the weights of the source model's embedding layer.
        target_tokenizer: Tokenizer of the target language model.
        target_auxiliary_embeddings: FastText auxiliary embeddings for the target language.
        batch_size: Number of tokens to process in each batch for non-overlapping token computations.

    Returns:
        The embedding initializer object for the target model, based on CLP-Transfer.
    """
    embeddings_initializer = WeightedAverageEmbeddingsInitialization(
        source_tokenizer=source_tokenizer,
        source_embeddings_matrix=source_embeddings_matrix,
        target_tokenizer=target_tokenizer,
        target_auxiliary_embeddings=target_auxiliary_embeddings,
        alignment_strategy=IdentityAlignment(),
        similarity_strategy=CosineSimilarity(),
        weights_strategy=IdentityWeights(),
        token_overlap_strategy=ExactMatchTokenOverlap(),
        batch_size=batch_size,
    )
    return embeddings_initializer


def focus(
    source_tokenizer: PreTrainedTokenizerBase,
    source_embeddings_matrix: NDArray,
    target_tokenizer: PreTrainedTokenizerBase,
    target_auxiliary_embeddings: FastTextEmbeddings,
    source_auxiliary_embeddings: FastTextEmbeddings,
    *,
    batch_size: int = 1024,
) -> WeightedAverageEmbeddingsInitialization:
    """Fast Overlapping Token Combinations Using Sparsemax (FOCUS) method.

    Described in [FOCUS: Effective Embedding Initialization for Specializing Pretrained Multilingual Models on a Single Language.](https://arxiv.org/abs/2305.14481) Dobler, Konstantin, and Gerard de Melo. arXiv preprint arXiv:2305.14481 (2023).

    The FOCUS method specializes pre-trained multilingual models by efficiently combining overlapping token embeddings
    using Sparsemax weights. It utilizes auxiliary embeddings and calculates the target language embeddings based
    on the source embeddings and the overlap between the source and target token sets.

    The method requires as input:

    - tokenizer of the source language model,
    - pre-trained language model as source,
    - tokenizer of the target language model,
    - 2 monolingual fastText embeddings for source and target languages respectively
        trained from scratch for both languages using pre-tokenized text with the respective language tokenizer.

    Args:
        source_tokenizer: Tokenizer of the source language model.
        source_embeddings_matrix: 2D matrix containing the weights of the source model's embedding layer.
        target_tokenizer: Tokenizer of the target language model.
        target_auxiliary_embeddings: FastText auxiliary embeddings for the target language.
        source_auxiliary_embeddings: FastText auxiliary embeddings for the source language.
        batch_size: Number of tokens to process in each batch for non-overlapping token computations.

    Returns:
        The embedding initializer object for the target model, based on FOCUS.
    """
    embeddings_initializer = WeightedAverageEmbeddingsInitialization(
        source_tokenizer=source_tokenizer,
        source_embeddings_matrix=source_embeddings_matrix,
        target_tokenizer=target_tokenizer,
        target_auxiliary_embeddings=target_auxiliary_embeddings,
        source_auxiliary_embeddings=source_auxiliary_embeddings,
        alignment_strategy=IdentityAlignment(),
        similarity_strategy=CosineSimilarity(),
        weights_strategy=SparsemaxWeights(),
        token_overlap_strategy=FuzzyMatchTokenOverlap(),
        batch_size=batch_size,
    )
    return embeddings_initializer
