"""
src/config.py
Central path resolution using env vars with sensible defaults.
All scripts import from here instead of duplicating path logic.

Usage:
    from src.config import DATA_DIR, MODELS_DIR, REPORTS_DIR, SPLITS_DIR
"""

import os
import pathlib

_REPO_ROOT = pathlib.Path(__file__).parent.parent

# Env-var overrides, then Kaggle/Colab auto-detect, then local repo defaults
def _resolve(env_var: str, kaggle_path: str, colab_path: str, local_rel: str) -> pathlib.Path:
    if env_var in os.environ:
        return pathlib.Path(os.environ[env_var])
    if pathlib.Path(kaggle_path).exists():
        return pathlib.Path(kaggle_path)
    if pathlib.Path(colab_path).exists():
        return pathlib.Path(colab_path)
    return _REPO_ROOT / local_rel


DATA_DIR = _resolve(
    "TEJALENS_DATA_DIR",
    "/kaggle/working/data",
    "/content/data",
    "data",
)

MODELS_DIR = _resolve(
    "TEJALENS_MODELS_DIR",
    "/kaggle/working/models",
    "/content/models",
    "models",
)

REPORTS_DIR = _resolve(
    "TEJALENS_REPORTS_DIR",
    "/kaggle/working/reports",
    "/content/reports",
    "reports",
)

SPLITS_DIR = DATA_DIR / "splits"
