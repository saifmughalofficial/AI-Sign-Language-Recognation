"""Gesture stabilisation ("debouncing").

Problem: a camera sees 30 frames per second, so holding an "A" for one second
would type AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA.

Solution, in plain words:
1. A gesture must stay the same for `hold_seconds` before it is ACCEPTED.
2. After it is accepted it is LOCKED. The same gesture is not accepted again
   until the hand has changed to something else (or left the picture) for
   at least `release_seconds`. Tiny flickers are ignored.

Tip for double letters (LL in HELLO): change your hand or lower it briefly
between the two L's.
"""
from __future__ import annotations

import time
from typing import Callable


class GestureStabilizer:
    def __init__(self, hold_seconds: float = 0.8, min_confidence: float = 0.6,
                 release_seconds: float = 0.35,
                 clock: Callable[[], float] = time.monotonic):
        self.hold_seconds = hold_seconds
        self.min_confidence = min_confidence
        self.release_seconds = release_seconds
        self._clock = clock
        self.reset()

    def reset(self) -> None:
        self._candidate: str | None = None
        self._since = 0.0
        self._locked: str | None = None
        self._away_since: float | None = None
        self.progress = 0.0

    def configure(self, hold_seconds: float, min_confidence: float,
                  release_seconds: float) -> None:
        self.hold_seconds = hold_seconds
        self.min_confidence = min_confidence
        self.release_seconds = release_seconds

    @property
    def locked_label(self) -> str | None:
        return self._locked

    def update(self, label: str | None, confidence: float = 1.0,
               now: float | None = None) -> str | None:
        """Feed one observation. Returns the label ONLY at the moment it is accepted."""
        now = self._clock() if now is None else now
        if label is not None and confidence < self.min_confidence:
            label = None                       # too unsure: treat as "no gesture"

        # ---- unlock after the hand has changed / left long enough -------------
        if self._locked is not None:
            if label == self._locked:
                self._away_since = None
            else:
                if self._away_since is None:
                    self._away_since = now
                if now - self._away_since >= self.release_seconds:
                    self._locked = None
                    self._away_since = None

        # ---- track how long the current candidate has been stable -------------
        if label is None:
            self._candidate = None
            self.progress = 0.0
            return None
        if label != self._candidate:
            self._candidate = label
            self._since = now

        if self._locked is not None:           # waiting for the hand to change
            self.progress = 1.0 if label == self._locked else 0.0
            return None

        held = now - self._since
        self.progress = min(1.0, held / self.hold_seconds) if self.hold_seconds > 0 else 1.0
        if held >= self.hold_seconds:
            self._locked = label
            self._away_since = None
            self.progress = 1.0
            return label
        return None
