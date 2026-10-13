"""Shared constants used all over the project.

Keeping them in one place means that adding a new mode, category or model file
only needs a change here.
"""
from __future__ import annotations

APP_NAME = "Sign Language Assistant"
APP_VERSION = "2.0.0"

# --- Recognition modes shown on the dashboard --------------------------------
MODES = ["ALPHABET", "NUMBERS", "WORDS", "SENTENCES", "AUTO"]

# Which gesture categories are searched in each mode (order = priority on a tie).
MODE_CATEGORIES = {
    "ALPHABET": ["alphabet"],
    "NUMBERS": ["number"],
    "WORDS": ["word", "custom"],
    "SENTENCES": ["sentence"],
    "AUTO": ["word", "custom", "number", "alphabet"],
}

CATEGORY_TITLES = {
    "alphabet": "Alphabet",
    "number": "Numbers",
    "word": "Words",
    "sentence": "Sentences",
    "custom": "Custom",
}

# --- Model files (inside ./models) ---------------------------------------------
MODEL_FILES = {
    "alphabet": "asl_alphabet_model.pkl",
    "number": "numbers_model.pkl",
    "word": "words_model.pkl",
    "custom": "custom_model.pkl",      # user-recorded static gestures
    "dynamic": "dynamic_model.pkl",    # user-recorded motion gestures (sequences)
}

# --- Folder names inside dataset/processed --------------------------------------
PROCESSED_DIRS = {"alphabet": "alphabet", "number": "numbers", "word": "words"}

# Accepted spellings on the command line -> internal category key
CATEGORY_ALIASES = {
    "alphabet": "alphabet", "letters": "alphabet",
    "number": "number", "numbers": "number", "digits": "number",
    "word": "word", "words": "word",
    "custom": "custom",
    "dynamic": "dynamic",
}

# ASL letters that need movement (cannot be recognised from a single frame)
DYNAMIC_LETTERS = ("J", "Z")

# Bump this whenever feature extraction changes; old models are then rejected.
FEATURE_VERSION = 1

# Number of frames a motion gesture is resampled to.
SEQUENCE_LENGTH = 16

# User-facing messages
MSG_NO_HAND = "No hand detected — position your hand inside the camera area."
MSG_LOW_CONF = "Gesture unclear — hold the pose steady, palm towards the camera."
MSG_PAUSED = "Detection paused."
MSG_CAMERA_OFF = "Camera is OFF (privacy). Press START CAMERA to begin."
MSG_DEMO = "Smart rule-based recognition is active — no training needed. Hold a pose until the ring fills."

PRIVACY_NOTICE = (
    "All camera processing and recognition is performed locally. "
    "No camera frames, microphone recordings or conversations are uploaded "
    "to a cloud service."
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
