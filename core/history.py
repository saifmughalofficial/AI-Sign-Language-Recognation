"""Communication history, stored ONLY on this computer (config/history.json)."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from core.paths import HISTORY_FILE


@dataclass
class HistoryEntry:
    time: str        # "09:30"
    source: str      # "SIGN" | "VOICE" | "SYSTEM"
    text: str

    def line(self) -> str:
        return f"{self.time}  {self.source} → \"{self.text}\""


class ConversationHistory:
    def __init__(self, path: Path | str = HISTORY_FILE, max_entries: int = 200,
                 persist: bool = True):
        self.path = Path(path)
        self.max_entries = max_entries
        self.persist = persist
        self.entries: list[HistoryEntry] = []
        self.last_error = ""

    def load(self) -> None:
        if not self.persist or not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.entries = [HistoryEntry(**item) for item in raw][-self.max_entries:]
        except (OSError, ValueError, TypeError) as exc:
            self.last_error = f"History could not be loaded: {exc}"
            self.entries = []

    def add(self, source: str, text: str, when: datetime | None = None) -> HistoryEntry | None:
        text = (text or "").strip()
        if not text:
            return None
        stamp = (when or datetime.now()).strftime("%H:%M")
        entry = HistoryEntry(stamp, source.upper(), text)
        self.entries.append(entry)
        self.entries = self.entries[-self.max_entries:]
        self._save()
        return entry

    def clear(self) -> None:
        self.entries = []
        self._save()

    def _save(self) -> None:
        if not self.persist:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps([asdict(e) for e in self.entries], indent=2, ensure_ascii=False),
                encoding="utf-8")
        except OSError as exc:
            self.last_error = f"History could not be saved: {exc}"
