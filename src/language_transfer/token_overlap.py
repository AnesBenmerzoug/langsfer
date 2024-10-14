import logging
from abc import ABC, abstractmethod

from transformers import PreTrainedTokenizerBase

__all__ = ["NoTokenOverlap", "SpecialTokenOverlap", "ExactMatchTokenOverlap"]

logger = logging.getLogger(__name__)


class TokenOverlapStrategy(ABC):
    """Abstract strategy for finding the overlapping and non-overlapping (missing) tokens
    between a source tokenizer vocabulary and target tokenizer vocabulary.
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

    @abstractmethod
    def _get_source_vocab(self, tokenizer: PreTrainedTokenizerBase) -> set[str]: ...

    @abstractmethod
    def _get_target_vocab(self, tokenizer: PreTrainedTokenizerBase) -> set[str]: ...


class NoTokenOverlap(TokenOverlapStrategy):
    def _get_source_vocab(self, tokenizer: PreTrainedTokenizerBase) -> set[str]:
        # Return empty set so that no overlap is possible
        return set()

    def _get_target_vocab(self, tokenizer: PreTrainedTokenizerBase) -> set[str]:
        vocabulary = set(tokenizer.get_vocab().keys())
        return vocabulary


class SpecialTokenOverlap(TokenOverlapStrategy):
    def _get_source_vocab(self, tokenizer: PreTrainedTokenizerBase) -> set[str]:
        vocabulary: set[str] = set()
        for token in tokenizer.special_tokens_map.values():
            if isinstance(token, str):
                token = [token]
            for t in token:
                vocabulary.add(t)
        return vocabulary

    def _get_target_vocab(self, tokenizer: PreTrainedTokenizerBase) -> set[str]:
        vocabulary = set(tokenizer.get_vocab().keys())
        return vocabulary


class ExactMatchTokenOverlap(TokenOverlapStrategy):
    def _get_source_vocab(self, tokenizer: PreTrainedTokenizerBase) -> set[str]:
        vocabulary = set(tokenizer.get_vocab().keys())
        return vocabulary

    def _get_target_vocab(self, tokenizer: PreTrainedTokenizerBase) -> set[str]:
        vocabulary = set(tokenizer.get_vocab().keys())
        return vocabulary
