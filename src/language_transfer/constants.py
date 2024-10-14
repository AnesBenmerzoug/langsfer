from pathlib import Path


__all__ = ["MODEL_CACHE_DIR"]

HOME_DIR = Path.home()
CACHE_DIR = HOME_DIR / ".cache"
LANGUAGE_TRANSFER_CACHE_DIR = CACHE_DIR / "language_transfer"
MODEL_CACHE_DIR = LANGUAGE_TRANSFER_CACHE_DIR / "models"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
