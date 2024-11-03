import json
from pathlib import Path

import pytest
import numpy as np
from pytest import Config, Metafunc


def pytest_addoption(parser):
    parser.addoption(
        "--skip-heavy-tests",
        action="store_true",
        default=False,
        help="Skips heavy tests. Used in CI",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "heavy: mark test as heavy to run i.e. requiring large resources to run",
    )


def pytest_collection_modifyitems(config: Config, items):
    if config.getoption("--skip-heavy-tests") is None:
        # do not skip tests
        return
    skip_heavy = pytest.mark.skip(
        reason="skipping because --skip-heavy-tests was passed"
    )
    for item in items:
        if "heavy" in item.keywords:
            item.add_marker(skip_heavy)


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return Path(__file__).parent.parent / "data"


@pytest.fixture(scope="session")
def german_bilingual_dictionary_file(data_dir: Path) -> Path:
    return data_dir / "wechsel/german_bilingual_dictionary.txt"


@pytest.fixture(scope="session")
def german_bilingual_dictionary_alignment_matrix(data_dir: Path) -> np.ndarray:
    return np.load(data_dir / "wechsel/bilingual_dictionary_alignment_matrix.npy")


def pytest_generate_tests(metafunc: Metafunc) -> None:
    # Parametrize test that uses `focus_fuzzy_token_overlap` fixture from json file
    if "focus_fuzzy_token_overlap" in metafunc.fixturenames:
        file = Path(__file__).parent.parent / "data/focus/fuzzy_token_overlap.json"
        with file.open() as f:
            token_overlap_dicts = json.load(f)

        ids = []
        values = []
        for token_overlap_dict in token_overlap_dicts:
            source_model_name = token_overlap_dict["source_model"]
            target_model_name = token_overlap_dict["target_model"]
            ids.append(source_model_name + " -> " + target_model_name)
            if (
                source_model_name == "xlm-roberta-base"
                and target_model_name == "benjamin/gpt2-wechsel-french"
            ):
                values.append(
                    pytest.param(
                        token_overlap_dict,
                        marks=pytest.mark.xfail(
                            reason="Expected test failure that indicates a difference from the original implementation in FOCUS"
                        ),
                    )
                )
            else:
                values.append(token_overlap_dict)

        metafunc.parametrize("focus_fuzzy_token_overlap", values, ids=ids)
