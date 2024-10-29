import functools
import os
import shutil
from pathlib import Path

import requests
from tqdm.auto import tqdm

__all__ = ["download_file"]


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
