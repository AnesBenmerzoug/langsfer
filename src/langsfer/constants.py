from platformdirs import user_cache_dir
from pathlib import Path


__all__ = ["MODEL_CACHE_DIR"]

CACHE_DIR = Path(user_cache_dir())
LANGSFER_CACHE_DIR = CACHE_DIR / "langsfer"
MODEL_CACHE_DIR = LANGSFER_CACHE_DIR / "models"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
