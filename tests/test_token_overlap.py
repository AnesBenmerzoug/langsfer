import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from language_transfer.token_overlap import (
    NoTokenOverlap,
    SpecialTokenOverlap,
    ExactMatchTokenOverlap,
)


@pytest.fixture(scope="session")
def source_model_id() -> str:
    return "roberta-base"


@pytest.fixture(scope="session")
def target_model_id() -> str:
    return "benjamin/roberta-base-wechsel-german"


@pytest.fixture(scope="session")
def source_tokenizer(source_model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(source_model_id)
    return tokenizer


@pytest.fixture(scope="session")
def target_tokenizer(target_model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(target_model_id)
    return tokenizer


def test_no_token_overlap(
    source_tokenizer: PreTrainedTokenizerBase, target_tokenizer: PreTrainedTokenizerBase
):
    token_overlap_strategy = NoTokenOverlap()
    overlapping_tokens, non_overlapping_tokens = token_overlap_strategy.apply(
        source_tokenizer, target_tokenizer
    )
    assert len(overlapping_tokens) == 0
    assert len(non_overlapping_tokens) == len(target_tokenizer.get_vocab())
    assert len(overlapping_tokens) + len(non_overlapping_tokens) == len(
        target_tokenizer.get_vocab()
    )


def test_special_token_overlap(
    source_tokenizer: PreTrainedTokenizerBase, target_tokenizer: PreTrainedTokenizerBase
):
    token_overlap_strategy = SpecialTokenOverlap()
    overlapping_tokens, non_overlapping_tokens = token_overlap_strategy.apply(
        source_tokenizer, target_tokenizer
    )
    assert len(overlapping_tokens) > 0
    assert len(non_overlapping_tokens) < len(target_tokenizer.get_vocab())
    assert overlapping_tokens == sorted(["<s>", "</s>", "<mask>", "<pad>", "<unk>"])
    assert len(overlapping_tokens) + len(non_overlapping_tokens) == len(
        target_tokenizer.get_vocab()
    )


def test_exact_match_token_overlap(
    source_tokenizer: PreTrainedTokenizerBase, target_tokenizer: PreTrainedTokenizerBase
):
    token_overlap_strategy = ExactMatchTokenOverlap()
    overlapping_tokens, non_overlapping_tokens = token_overlap_strategy.apply(
        source_tokenizer, target_tokenizer
    )
    assert len(overlapping_tokens) > 0
    assert len(non_overlapping_tokens) < len(target_tokenizer.get_vocab())
    assert len(overlapping_tokens) + len(non_overlapping_tokens) == len(
        target_tokenizer.get_vocab()
    )
