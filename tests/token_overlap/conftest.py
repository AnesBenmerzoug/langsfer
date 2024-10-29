import json
from pathlib import Path

import pytest
from pytest import Metafunc
from transformers import AutoTokenizer


def pytest_generate_tests(metafunc: Metafunc) -> None:
    # Parametrize test that uses `focus_fuzzy_token_overlap` fixture from json file
    if "focus_fuzzy_token_overlap" in metafunc.fixturenames:
        file = Path(__file__).parent / "focus_fuzzy_token_overlap.json"
        with file.open() as f:
            token_overlap_dicts = json.load(f)
        ids = [
            x["source_model"] + " -> " + x["target_model"] for x in token_overlap_dicts
        ]
        metafunc.parametrize("focus_fuzzy_token_overlap", token_overlap_dicts, ids=ids)


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
