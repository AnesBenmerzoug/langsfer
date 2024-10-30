import pytest

from langsfer.embeddings import FastTextEmbeddings


@pytest.mark.parametrize("language_id", ["en", pytest.param("123", marks=pytest.mark.xfail)])
def test_fasttext_from_language_id(language_id: str):
    FastTextEmbeddings.from_model_name_or_path(language_id)
