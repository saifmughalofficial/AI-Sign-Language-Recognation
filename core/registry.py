"""Loads the JSON knowledge files (gestures, phrases, help, vocabulary).

All files live in ./data and are plain JSON, so they work 100% offline and can
be edited with any text editor. If a file is missing or broken the registry
records a friendly problem message and carries on with empty data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.paths import DATA_DIR

VALID_CATEGORIES = ("alphabet", "number", "word", "sentence", "custom")
VALID_TYPES = ("static", "dynamic")


def validate_gesture_entry(entry: Any) -> list[str]:
    """Return a list of problems (empty list = entry is valid)."""
    problems: list[str] = []
    if not isinstance(entry, dict):
        return ["entry is not an object"]
    name = entry.get("gesture")
    if not isinstance(name, str) or not name.strip():
        problems.append("'gesture' must be non-empty text")
    if entry.get("category") not in VALID_CATEGORIES:
        problems.append(f"'category' must be one of {VALID_CATEGORIES}")
    if entry.get("type", "static") not in VALID_TYPES:
        problems.append(f"'type' must be one of {VALID_TYPES}")
    if entry.get("hands", 1) not in (1, 2):
        problems.append("'hands' must be 1 or 2")
    examples = entry.get("sentence_examples", [])
    if not isinstance(examples, list):
        problems.append("'sentence_examples' must be a list")
    pose = entry.get("demo_pose")
    if pose is not None and not (isinstance(pose, str) and len(pose) == 5 and set(pose) <= {"0", "1"}):
        problems.append("'demo_pose' must be five 0/1 characters (thumb..pinky) or null")
    shape = entry.get("shape")
    if shape is not None and not (isinstance(shape, str) and shape.strip()):
        problems.append("'shape' must be a shape name (see vision/shape_library.py) or null")
    if entry.get("category") == "sentence" and not isinstance(entry.get("output"), str):
        problems.append("sentence gestures need an 'output' text")
    return problems


class GestureRegistry:
    def __init__(self, data_dir: Path | str = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.gestures: list[dict] = []
        self.phrases: dict = {}
        self.help: dict = {}
        self.vocabulary: dict = {}
        self.problems: list[str] = []
        self.load()

    # ---- loading -----------------------------------------------------------
    def _read(self, filename: str) -> Any:
        path = self.data_dir / filename
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self.problems.append(f"Missing data file: {filename}")
        except (OSError, json.JSONDecodeError) as exc:
            self.problems.append(f"Could not read {filename}: {exc}")
        return None

    def load(self) -> None:
        self.problems = []
        raw = self._read("gestures.json")
        items = raw.get("gestures", []) if isinstance(raw, dict) else (raw or [])
        self.gestures = []
        for entry in items:
            issues = validate_gesture_entry(entry)
            if issues:
                label = entry.get("gesture", "?") if isinstance(entry, dict) else "?"
                self.problems.append(f"Gesture {label!r} skipped: {'; '.join(issues)}")
                continue
            entry = dict(entry)
            entry["gesture"] = entry["gesture"].strip().upper()
            entry.setdefault("type", "static")
            entry.setdefault("hands", 1)
            entry.setdefault("status", "experimental")
            entry.setdefault("instructions", "")
            entry.setdefault("sentence_examples", [])
            self.gestures.append(entry)
        self.phrases = self._read("phrases.json") or {}
        self.help = self._read("help.json") or {}
        self.vocabulary = self._read("vocabulary.json") or {}

    # ---- queries -----------------------------------------------------------
    def get(self, name: str, category: str | None = None) -> dict | None:
        name = name.strip().upper()
        for g in self.gestures:
            if g["gesture"] == name and (category is None or g["category"] == category):
                return g
        return None

    def by_category(self, category: str) -> list[dict]:
        return [g for g in self.gestures if g["category"] == category]

    def labels(self, category: str) -> list[str]:
        return [g["gesture"] for g in self.by_category(category)]

    def is_dynamic(self, name: str) -> bool:
        g = self.get(name)
        return bool(g and g.get("type") == "dynamic")

    def demo_poses(self, category: str) -> dict[str, str]:
        """{'10000': 'A', ...} for the demo-mode finger-pattern recogniser."""
        return {g["demo_pose"]: g["gesture"] for g in self.by_category(category)
                if g.get("demo_pose")}

    def shape_map(self, category: str) -> dict[str, dict]:
        """{'W': <WATER entry>, ...}: which rule-based shape means which gesture."""
        return {g["shape"]: g for g in self.by_category(category) if g.get("shape")}

    def search(self, query: str, category: str | None = None) -> list[dict]:
        """Case-insensitive search over name, instructions and example sentences."""
        q = query.strip().lower()
        pool = self.by_category(category) if category else list(self.gestures)
        if not q:
            return pool
        scored: list[tuple[int, dict]] = []
        for g in pool:
            name = g["gesture"].lower()
            if name == q:
                score = 0
            elif name.startswith(q):
                score = 1
            elif q in name:
                score = 2
            else:
                blob = " ".join([g.get("instructions", ""), g["category"], g.get("output", ""),
                                 " ".join(g.get("sentence_examples", []))]).lower()
                if q not in blob:
                    continue
                score = 3
            scored.append((score, g))
        scored.sort(key=lambda item: (item[0], item[1]["gesture"]))
        return [g for _, g in scored]

    # ---- other data --------------------------------------------------------
    def vocabulary_words(self, language: str = "en") -> list[str]:
        return list(self.vocabulary.get("languages", {}).get(language, {}).get("words", []))

    def quick_phrases(self, language: str = "en") -> list[str]:
        return list(self.vocabulary.get("languages", {}).get(language, {}).get("quick_phrases", []))

    def help_sections(self) -> list[dict]:
        return list(self.help.get("sections", []))

    def phrase_library(self) -> list[dict]:
        return list(self.phrases.get("phrase_library", []))

    def privacy_notice(self) -> str:
        from core.constants import PRIVACY_NOTICE
        return self.help.get("privacy_notice", PRIVACY_NOTICE)

    def save_gestures(self) -> None:
        """Write the current gesture list back to data/gestures.json."""
        path = self.data_dir / "gestures.json"
        path.write_text(json.dumps({"version": 1, "gestures": self.gestures},
                                   indent=2, ensure_ascii=False), encoding="utf-8")
