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
    """Abstract strategy for finding the overlapping and non-overlapping (missing) tokens
    between a source tokenizer vocabulary and target tokenizer vocabulary.
    """

    @abstractmethod
    def apply(
        self,
        source_tokenizer: PreTrainedTokenizerBase,
        target_tokenizer: PreTrainedTokenizerBase,
    ) -> tuple[list[str], list[str]]: ...


class ExactMatchTokenOverlap(TokenOverlapStrategy):
    """Token overlap strategy for finding the overlapping and non-overlapping (missing) tokens
    that match exactly between a source tokenizer vocabulary and target tokenizer vocabulary.

    The source tokenizer's vocabulary to match is returned by the `_get_source_vocab` method
    and target tokenizer's vocabulary ot match is returned by the `_get_target_vocab` method.
    """

    def apply(
        self,
        source_tokenizer: PreTrainedTokenizerBase,
        target_tokenizer: PreTrainedTokenizerBase,
    ) -> tuple[list[str], list[str]]:
        """Applies the strategy to the source and target tokenizer vocabularies.

        Args:
            source_tokenizer: Source tokenizer.
            target_tokenizer: Target tokenizer.

        Returns:
            A tuple containing:
                overlapping_tokens: Sorted list of overlapping tokens.
                missing_tokens: Sorted list of missing tokens.
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
    """Subclass of ExactMatchTokenOverlap that returns an empty set as the source tokenizer's vocabulary
    to guarantee that no overlapping token is found.
    """

    def _get_source_vocab(self, tokenizer: PreTrainedTokenizerBase) -> set[str]:
        # Return empty set so that no overlap is possible
        return set()


class SpecialTokenOverlap(ExactMatchTokenOverlap):
    """Subclass of ExactMatchTokenOverlap that returns an only special tokens from the source tokenizer's vocabulary
    to guarantee that only overlapping special tokens are found.
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
    """Token overlap strategy for finding the overlapping and non-overlapping (missing) tokens
    between a source tokenizer vocabulary and target tokenizer vocabulary whose canonicalized form matches.

    Inspired by the fuzzy token matcher described and implemented in FOCUS.
    """

    BPE_TOKEN_PREFIX = "Ġ"
    UNIGRAM_TOKEN_PREFIX = "▁"
    WORDPIECE_TOKEN_PREFIX = "##"

    def apply(
        self,
        source_tokenizer: PreTrainedTokenizerBase,
        target_tokenizer: PreTrainedTokenizerBase,
    ) -> tuple[list[str], list[str]]:
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
        canonical_vocab: dict[str, str] = {}

        for token, token_idx in sorted(tokenizer.vocab.items(), key=lambda x: x[1]):
            canonical_form = self._canonicalize_token(tokenizer, token_idx)
            canonical_vocab[token] = canonical_form
        return canonical_vocab

    def _canonicalize_token(
        self, tokenizer: PreTrainedTokenizerBase, token_id: int
    ) -> str:
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
