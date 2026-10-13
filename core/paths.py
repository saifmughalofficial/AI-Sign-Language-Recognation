"""Every folder/file location used by the project, computed from this file's place.

Nothing here depends on the current working directory, so the app can be
started from anywhere (double-click, .bat file, terminal...).
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
CONFIG_DIR = ROOT / "config"
ASSETS_DIR = ROOT / "assets"
GESTURE_IMG_DIR = ASSETS_DIR / "gestures"

DATASET_DIR = ROOT / "dataset"
RAW_DIR = DATASET_DIR / "raw"
PROCESSED_DIR = DATASET_DIR / "processed"
CUSTOM_DIR = DATASET_DIR / "custom"          # gestures you record yourself

SETTINGS_FILE = CONFIG_DIR / "settings.json"
HISTORY_FILE = CONFIG_DIR / "history.json"

HAND_TASK_FILE = MODELS_DIR / "hand_landmarker.task"   # only for new MediaPipe versions


def ensure_dirs() -> None:
    """Create all folders the app expects (safe to call many times)."""
    for folder in (
        MODELS_DIR, CONFIG_DIR, GESTURE_IMG_DIR, RAW_DIR, CUSTOM_DIR,
        PROCESSED_DIR / "alphabet", PROCESSED_DIR / "numbers", PROCESSED_DIR / "words",
    ):
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass  # read-only install location: the app still runs
