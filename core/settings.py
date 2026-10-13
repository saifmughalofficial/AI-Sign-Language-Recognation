"""User settings: defaults, validation and saving to config/settings.json.

Invalid or missing values never crash the app - they are replaced by defaults
and a short human-readable note is returned so the UI can tell the user.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields, asdict
from pathlib import Path
from typing import Any

from core.paths import SETTINGS_FILE

RESOLUTIONS = ["320x240", "640x480", "800x600", "1280x720"]
THEMES = ["dark", "light"]
LANGUAGES = ["en"]           # add more here once data/vocabulary.json has them
MODES = ["ALPHABET", "NUMBERS", "WORDS", "SENTENCES", "AUTO"]


@dataclass
class Settings:
    # camera
    camera_index: int = 0
    resolution: str = "1280x720"
    mirror_camera: bool = True
    show_landmarks: bool = True
    process_every_n_frames: int = 1          # performance: 2 = run AI on every 2nd frame
    # recognition
    detection_confidence: float = 0.5        # MediaPipe: is there a hand?
    tracking_confidence: float = 0.5         # MediaPipe: keep following the hand
    recognition_confidence: float = 0.6      # our classifier: minimum accepted score
    gesture_hold_seconds: float = 0.7        # how long a gesture must stay stable
    release_seconds: float = 0.35            # how long the hand must change/leave before repeating
    experimental_dynamic: bool = True        # J / Z motion gestures
    mode: str = "ALPHABET"
    # speech output
    speech_enabled: bool = True
    speak_letters: bool = False
    speak_words: bool = True
    speak_sentences: bool = True
    voice_id: str = ""
    speech_rate: int = 170
    speech_volume: float = 1.0
    speak_voice_text: bool = False           # read Person B's words aloud too
    # interface
    theme: str = "dark"
    language: str = "en"
    auto_clear_sentence: bool = False        # clear sentence after SPEAK
    # personal
    user_name: str = ""
    save_history: bool = True

    # ---- helpers -------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def resolution_wh(self) -> tuple[int, int]:
        w, h = self.resolution.split("x")
        return int(w), int(h)


# name -> (kind, minimum, maximum, choices)
_RULES: dict[str, tuple[str, Any, Any, Any]] = {
    "camera_index": ("int", 0, 10, None),
    "resolution": ("choice", None, None, RESOLUTIONS),
    "mirror_camera": ("bool", None, None, None),
    "show_landmarks": ("bool", None, None, None),
    "process_every_n_frames": ("int", 1, 6, None),
    "detection_confidence": ("float", 0.1, 0.99, None),
    "tracking_confidence": ("float", 0.1, 0.99, None),
    "recognition_confidence": ("float", 0.1, 0.99, None),
    "gesture_hold_seconds": ("float", 0.2, 3.0, None),
    "release_seconds": ("float", 0.1, 2.0, None),
    "experimental_dynamic": ("bool", None, None, None),
    "mode": ("choice", None, None, MODES),
    "speech_enabled": ("bool", None, None, None),
    "speak_letters": ("bool", None, None, None),
    "speak_words": ("bool", None, None, None),
    "speak_sentences": ("bool", None, None, None),
    "voice_id": ("str", None, None, None),
    "speech_rate": ("int", 80, 300, None),
    "speech_volume": ("float", 0.0, 1.0, None),
    "speak_voice_text": ("bool", None, None, None),
    "theme": ("choice", None, None, THEMES),
    "language": ("choice", None, None, LANGUAGES),
    "auto_clear_sentence": ("bool", None, None, None),
    "user_name": ("str", None, None, None),
    "save_history": ("bool", None, None, None),
}


def _coerce(name: str, value: Any, default: Any) -> tuple[Any, str | None]:
    """Return (clean_value, note). `note` is None when the value was fine."""
    kind, lo, hi, choices = _RULES[name]
    try:
        if kind == "bool":
            if isinstance(value, bool):
                return value, None
            raise ValueError("expected true/false")
        if kind == "int":
            if isinstance(value, bool):
                raise ValueError("expected a whole number")
            num = int(value)
            clipped = max(lo, min(hi, num))
            return clipped, (None if clipped == num else f"'{name}' was out of range")
        if kind == "float":
            if isinstance(value, bool):
                raise ValueError("expected a number")
            num = float(value)
            clipped = max(lo, min(hi, num))
            return clipped, (None if clipped == num else f"'{name}' was out of range")
        if kind == "choice":
            if value in choices:
                return value, None
            raise ValueError(f"must be one of {choices}")
        if kind == "str":
            if isinstance(value, str):
                return value.strip(), None
            raise ValueError("expected text")
    except (TypeError, ValueError) as exc:
        return default, f"'{name}' was invalid ({exc}); using default"
    return default, None


def settings_from_dict(data: Any) -> tuple[Settings, list[str]]:
    """Build a valid Settings object from any dict. Returns (settings, notes)."""
    notes: list[str] = []
    defaults = Settings()
    if not isinstance(data, dict):
        return defaults, ["Settings file was not a JSON object; using defaults"]
    clean: dict[str, Any] = {}
    for f in fields(Settings):
        default = getattr(defaults, f.name)
        if f.name not in data:
            clean[f.name] = default
            continue
        value, note = _coerce(f.name, data[f.name], default)
        clean[f.name] = value
        if note:
            notes.append(note)
    return Settings(**clean), notes


class SettingsStore:
    """Loads and saves Settings as JSON."""

    def __init__(self, path: Path | str = SETTINGS_FILE):
        self.path = Path(path)
        self.notes: list[str] = []

    def load(self) -> Settings:
        self.notes = []
        if not self.path.exists():
            return Settings()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.notes.append(f"Could not read settings ({exc}); using defaults")
            return Settings()
        settings, notes = settings_from_dict(data)
        self.notes = notes
        return settings

    def save(self, settings: Settings) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")
            return True
        except OSError as exc:
            self.notes.append(f"Could not save settings: {exc}")
            return False
