import functools
import os
import shutil
from pathlib import Path
from typing import Iterable

import requests
from gensim.models import FastText
from tqdm.auto import tqdm

__all__ = ["download_file", "train_fasttext_model"]


def download_file(
    url: str, destination_path: str | os.PathLike, *, verbose: bool = False
):
    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get("Content-Length", 0))

    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    r.raw.read = functools.partial(
        r.raw.read, decode_content=True
    )  # Decompress if needed

    with tqdm.wrapattr(
        r.raw,
        "read",
        total=file_size,
        disable=not verbose,
        desc=f"Downloading {url}",
    ) as r_raw:
        with destination_path.open("wb") as f:
            shutil.copyfileobj(r_raw, f)

    return destination_path


def train_fasttext_model(
    corpus_iterable: Iterable[list[str]],
    *,
    total_examples: int,
    vector_size: int = 300,
    window: int = 3,
    min_count: int = 10,
    epochs: int = 3,
) -> FastText:
    """Trains a FastText model.

    Args:
        corpus_iterable: Iterator of lists of tokens (tokenized sentences) used as training data.
        total_examples : Count of sentences.
        vector_size : Dimensionality of the word vectors.
        window : The maximum distance between the current and predicted word within a sentence.
        min_count : The model ignores all words with total frequency lower than this.
        epochs : Number of iterations (epochs) over the corpus.

    Returns:
        Trained FastText model.
    """
    model = FastText(vector_size=vector_size, window=window, min_count=min_count)
    model.build_vocab(corpus_iterable=corpus_iterable)
    model.train(
        corpus_iterable=corpus_iterable,
        total_examples=total_examples,
        epochs=epochs,
    )
    return model
