from transformers import PreTrainedTokenizerBase, AutoTokenizer

from language_transfer.token_overlap import (
    NoTokenOverlap,
    SpecialTokenOverlap,
    ExactMatchTokenOverlap,
    FuzzyMatchTokenOverlap,
)


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


def test_focus_fuzzy_token_overlap(focus_fuzzy_token_overlap: dict):
    source_tokenizer = AutoTokenizer.from_pretrained(
        focus_fuzzy_token_overlap["source_model"]
    )
    target_tokenizer = AutoTokenizer.from_pretrained(
        focus_fuzzy_token_overlap["target_model"]
    )
    token_overlap_strategy = FuzzyMatchTokenOverlap()
    overlapping_tokens, non_overlapping_tokens = token_overlap_strategy.apply(
        source_tokenizer, target_tokenizer
    )
    # prepare variables because calling `get_vocab()` is slow
    source_tokenizer_vocab = source_tokenizer.get_vocab()
    target_tokenizer_vocab = target_tokenizer.get_vocab()
    # sanity checks
    assert len(overlapping_tokens) > 0
    assert len(non_overlapping_tokens) < len(target_tokenizer_vocab)
    assert len(overlapping_tokens) + len(non_overlapping_tokens) == len(
        target_tokenizer_vocab
    )
    # we use sets so that if there is an error, it's an easier to see in the error messages which tokens are missing
    overlapping_token_difference = set(overlapping_tokens).difference(
        set(focus_fuzzy_token_overlap["overlapping_tokens"].keys())
    )
    for token in overlapping_tokens:
        assert token in target_tokenizer_vocab
        assert (
            token in focus_fuzzy_token_overlap["overlapping_tokens"]
        ), overlapping_token_difference

    for token in non_overlapping_tokens:
        assert token in target_tokenizer_vocab
        assert token not in source_tokenizer_vocab
        assert token in focus_fuzzy_token_overlap["non_overlapping_tokens"]

    # Check the order as well
    assert overlapping_tokens == list(
        focus_fuzzy_token_overlap["overlapping_tokens"].keys()
    )
    assert non_overlapping_tokens == focus_fuzzy_token_overlap["non_overlapping_tokens"]
