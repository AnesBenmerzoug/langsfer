"""
This module contains strategies for finding token overlap between source and target tokenizer vocabularies.

It defines several strategies for detecting the intersection and difference of tokens in two tokenizers' vocabularies.
These strategies are used to analyze and compare token vocabularies, which is essential for cross-lingual transfer and token alignment.

Each strategy provides a method to apply the logic to two tokenizers and return the overlapping and non-overlapping tokens.
"""

import logging
from abc import ABC, abstractmethod

from tokenizers.models import BPE, WordPiece, Unigram
from transformers import PreTrainedTokenizerBase

__all__ = [
    "ExactMatchTokenOverlap",
    "NoTokenOverlap",
    "SpecialTokenOverlap",
    "FuzzyMatchTokenOverlap",
]

logger = logging.getLogger(__name__)


class TokenOverlapStrategy(ABC):
    """Abstract base class for strategies that compute token overlap between source and target tokenizers.

    This class provides an abstract method `apply()` which should be implemented by subclasses to define the strategy
    for finding overlapping and non-overlapping (missing) tokens between the vocabularies of two tokenizers.
    """

    @abstractmethod
    def apply(
        self,
        source_tokenizer: PreTrainedTokenizerBase,
        target_tokenizer: PreTrainedTokenizerBase,
    ) -> tuple[list[str], list[str]]: ...


class ExactMatchTokenOverlap(TokenOverlapStrategy):
    """Strategy to find overlapping and non-overlapping tokens that match exactly between source and target tokenizers.

    This class compares the vocabularies of the source and target tokenizers and finds the tokens that exactly match
    in both vocabularies. The `_get_source_vocab` and `_get_target_vocab` methods are used to retrieve the vocabularies
    of the source and target tokenizers respectively.
    """

    def apply(
        self,
        source_tokenizer: PreTrainedTokenizerBase,
        target_tokenizer: PreTrainedTokenizerBase,
    ) -> tuple[list[str], list[str]]:
        """Finds the overlapping and non-overlapping tokens that exactly match between source and target tokenizers.

        Args:
            source_tokenizer: Tokenizer for the source language model.
            target_tokenizer: Tkenizer for the target language model.

        Returns:
            Tuple containing:
                - overlapping_tokens: A sorted list of tokens that appear in both the source and target vocabularies.
                - non_overlapping_tokens: A sorted list of tokens that are in the target vocabulary but not in the source vocabulary.
        """
        overlapping_tokens: list[str] = []
        non_overlapping_tokens: list[str] = []

        source_vocab = self._get_source_vocab(source_tokenizer)
        target_vocab = self._get_target_vocab(target_tokenizer)

        overlapping_tokens = list(source_vocab.intersection(target_vocab))
        overlapping_tokens.sort()
        non_overlapping_tokens = list(target_vocab.difference(source_vocab))
        non_overlapping_tokens.sort()

        return overlapping_tokens, non_overlapping_tokens

    def _get_source_vocab(self, tokenizer: PreTrainedTokenizerBase) -> set[str]:
        vocabulary = set(tokenizer.get_vocab().keys())
        return vocabulary

    def _get_target_vocab(self, tokenizer: PreTrainedTokenizerBase) -> set[str]:
        vocabulary = set(tokenizer.get_vocab().keys())
        return vocabulary


class NoTokenOverlap(ExactMatchTokenOverlap):
    """Subclass of `ExactMatchTokenOverlap` that ensures no tokens overlap between source and target vocabularies.

    This class overrides the `_get_source_vocab` method to return an empty set, ensuring that no tokens from the source
    vocabulary will be considered overlapping with the target vocabulary.
    """

    def _get_source_vocab(self, tokenizer: PreTrainedTokenizerBase) -> set[str]:
        # Return empty set so that no overlap is possible
        return set()


class SpecialTokenOverlap(ExactMatchTokenOverlap):
    """Subclass of `ExactMatchTokenOverlap` that finds only special tokens that overlap between the source and target vocabularies.

    This class overrides the `_get_source_vocab` method to return only special tokens from the source tokenizer's vocabulary,
    ensuring that only special tokens are considered for overlap comparison.
    """

    def _get_source_vocab(self, tokenizer: PreTrainedTokenizerBase) -> set[str]:
        vocabulary: set[str] = set()
        for token in tokenizer.special_tokens_map.values():
            if isinstance(token, str):
                token = [token]
            for t in token:
                vocabulary.add(t)
        return vocabulary


class FuzzyMatchTokenOverlap(TokenOverlapStrategy):
    """Strategy to find overlapping and non-overlapping tokens between source and target tokenizers using fuzzy matching.

    This class uses a technique inspired by the FOCUS fuzzy token matcher to compare the canonical forms of tokens
    between the source and target tokenizers. The canonical form of a token is a lowercased version without any tokenizer
    prefixes (such as WordPiece's `##`, BPE's `Ġ`, or Unigram's `▁`).
    """

    BPE_TOKEN_PREFIX = "Ġ"
    UNIGRAM_TOKEN_PREFIX = "▁"
    WORDPIECE_TOKEN_PREFIX = "##"

    def apply(
        self,
        source_tokenizer: PreTrainedTokenizerBase,
        target_tokenizer: PreTrainedTokenizerBase,
    ) -> tuple[list[str], list[str]]:
        """Finds the overlapping and non-overlapping tokens between source and target tokenizers based on their canonicalized forms.

        Args:
            source_tokenizer: Tokenizer for the source language.
            target_tokenizer: Tokenizer for the target language.

        Returns:
            Tuple containing:
                - overlapping_tokens: A list of tokens from the target tokenizer that
                    match canonicalized tokens from the source tokenizer.
                - non_overlapping_tokens: A list of tokens from the target tokenizer that
                    do not match any canonicalized tokens from the source tokenizer.
        """
        canonical_source_vocab = self._canonicalize_vocab(source_tokenizer)
        canonical_target_vocab = self._canonicalize_vocab(target_tokenizer)
        canonical_source_tokens = set(x for x in canonical_source_vocab.values())

        overlapping_tokens: list[str] = []
        non_overlapping_tokens: list[str] = []

        for target_token, canonical_target_token in canonical_target_vocab.items():
            if canonical_target_token in canonical_source_tokens:
                overlapping_tokens.append(target_token)
            else:
                non_overlapping_tokens.append(target_token)
        return overlapping_tokens, non_overlapping_tokens

    def _canonicalize_vocab(self, tokenizer: PreTrainedTokenizerBase) -> dict[str, str]:
        """Canonicalizes the vocabulary of a tokenizer by converting each token to its canonical form.

        This method processes the tokens of the tokenizer's vocabulary by removing any tokenizer-specific prefixes
        and converting tokens to lowercase.

        Args:
            tokenizer: Tokenizer whose vocabulary is to be canonicalized.

        Returns:
            Dictionary mapping tokens to their canonicalized forms.
        """
        canonical_vocab: dict[str, str] = {}

        for token, token_idx in sorted(tokenizer.vocab.items(), key=lambda x: x[1]):
            canonical_form = self._canonicalize_token(tokenizer, token_idx)
            canonical_vocab[token] = canonical_form
        return canonical_vocab

    def _canonicalize_token(
        self, tokenizer: PreTrainedTokenizerBase, token_id: int
    ) -> str:
        """Converts a token to its canonical form by removing tokenizer-specific prefixes and converting to lowercase.

        Args:
            tokenizer: Tokenizer used to convert the token ID to its canonical form.
            token_id: ID of the token to canonicalize.

        Returns:
            Canonical form of the token.
        """
        # We use `convert_ids_to_tokens` instead of `decode`
        # because the former adds the beginning of word prefix to tokens
        # and because it doesn't outright remove tokens like '\u2028'
        # or badly convert tokens like 'Âł'
        canonical_token: str = tokenizer.convert_ids_to_tokens(token_id)

        if isinstance(tokenizer._tokenizer.model, WordPiece):
            token_prefix = self.WORDPIECE_TOKEN_PREFIX
        elif isinstance(tokenizer._tokenizer.model, Unigram):
            token_prefix = self.UNIGRAM_TOKEN_PREFIX
        elif isinstance(tokenizer._tokenizer.model, BPE):
            token_prefix = self.BPE_TOKEN_PREFIX
        else:
            raise ValueError(
                f"Unsupported tokenizer model {type(tokenizer._tokenizer.model).__name__}"
            )

        canonical_token = canonical_token.removeprefix(token_prefix)
        canonical_token = canonical_token.lower()
        return canonical_token
