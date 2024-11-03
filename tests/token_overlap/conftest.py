import pytest
from transformers import AutoTokenizer


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
