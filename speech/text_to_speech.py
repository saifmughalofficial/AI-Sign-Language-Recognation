"""Offline text-to-speech using pyttsx3 (Windows SAPI5 voices - free, no internet).

pyttsx3 blocks while talking, so all speech happens on ONE background thread.
If pyttsx3 or a voice is missing, the app keeps running; `available` is False
and `error` explains why.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import Future
from typing import Callable

log = logging.getLogger(__name__)
_QUIT = object()


def _default_engine_factory():
    import pyttsx3
    return pyttsx3.init()


class TextToSpeech:
    def __init__(self, rate: int = 170, volume: float = 1.0, voice_id: str = "",
                 engine_factory: Callable[[], object] | None = None):
        self.rate = rate
        self.volume = volume
        self.voice_id = voice_id
        self._factory = engine_factory or _default_engine_factory
        self._queue: queue.Queue = queue.Queue()
        self._ready = threading.Event()
        self._stop_flag = threading.Event()
        self._engine = None
        self._available: bool | None = None       # None = still starting
        self.error = ""
        self.last_text = ""
        self._last_spoken: dict[str, float] = {}
        self._thread = threading.Thread(target=self._worker, daemon=True, name="tts")
        self._thread.start()

    # ----------------------------------------------------------------- status
    @property
    def available(self) -> bool:
        return self._available is True

    def wait_ready(self, timeout: float = 5.0) -> bool:
        self._ready.wait(timeout)
        return self.available

    @property
    def status_text(self) -> str:
        if self._available is None:
            return "starting…"
        return "ready" if self._available else f"unavailable ({self.error})"

    # ------------------------------------------------------------------ worker
    def _make_engine(self):
        engine = self._factory()
        self._apply(engine)
        return engine

    def _apply(self, engine) -> None:
        try:
            engine.setProperty("rate", int(self.rate))
            engine.setProperty("volume", float(self.volume))
            if self.voice_id:
                engine.setProperty("voice", self.voice_id)
        except Exception as exc:  # noqa: BLE001 - a bad voice id must not disable speech
            log.warning("Could not apply voice settings: %s", exc)

    def _worker(self) -> None:
        try:
            self._engine = self._make_engine()
            self._available = True
        except Exception as exc:  # noqa: BLE001 - missing pyttsx3, no voices, driver error...
            self._available = False
            self.error = f"{type(exc).__name__}: {exc}"
            log.warning("Text-to-speech unavailable: %s", self.error)
        self._ready.set()
        while True:
            item = self._queue.get()
            try:
                if item is _QUIT:
                    return
                kind, payload, future = item
                if not self.available:
                    if future:
                        future.set_result(None)
                    continue
                if kind == "speak":
                    self._stop_flag.clear()
                    self._engine.say(payload)
                    self._engine.runAndWait()
                    if self._stop_flag.is_set():         # stop() was pressed: fresh engine
                        try:
                            self._engine = self._make_engine()
                        except Exception as exc:  # noqa: BLE001
                            self._available, self.error = False, str(exc)
                elif kind == "call":
                    result = payload(self._engine)
                    if future:
                        future.set_result(result)
            except Exception as exc:  # noqa: BLE001
                log.warning("Speech error: %s", exc)
                self.error = str(exc)
                if item is not _QUIT and item[2] and not item[2].done():
                    item[2].set_result(None)
            finally:
                self._queue.task_done()

    # ----------------------------------------------------------------- actions
    def speak(self, text: str, dedupe_seconds: float = 0.0) -> bool:
        """Queue text to be spoken. `dedupe_seconds` ignores the same text if it was
        spoken very recently (cooldown to avoid repeating letters)."""
        text = (text or "").strip()
        if not text:
            return False
        now = time.monotonic()
        if dedupe_seconds and now - self._last_spoken.get(text, -1e9) < dedupe_seconds:
            return False
        self._last_spoken[text] = now
        self.last_text = text
        if self._available is False:
            return False
        self._queue.put(("speak", text, None))
        return True

    def repeat(self) -> bool:
        return self.speak(self.last_text) if self.last_text else False

    def stop(self) -> None:
        """Stop talking now and forget everything still queued."""
        while True:
            try:
                item = self._queue.get_nowait()
                self._queue.task_done()
                if item is not _QUIT and item[2]:
                    item[2].set_result(None)
            except queue.Empty:
                break
        self._stop_flag.set()
        try:
            if self._engine is not None:
                self._engine.stop()
        except Exception:  # noqa: BLE001
            pass

    def wait_idle(self, timeout: float = 3.0) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self._queue.unfinished_tasks == 0:
                return True
            time.sleep(0.01)
        return False

    def _call(self, func: Callable, timeout: float = 4.0):
        if not self.available:
            return None
        future: Future = Future()
        self._queue.put(("call", func, future))
        try:
            return future.result(timeout)
        except Exception:  # noqa: BLE001
            return None

    def list_voices(self) -> list[tuple[str, str]]:
        """[(voice_id, readable name), ...] of installed system voices."""
        result = self._call(lambda e: [(v.id, getattr(v, "name", v.id))
                                       for v in e.getProperty("voices")])
        return result or []

    def configure(self, rate: int | None = None, volume: float | None = None,
                  voice_id: str | None = None) -> None:
        if rate is not None:
            self.rate = rate
        if volume is not None:
            self.volume = volume
        if voice_id is not None:
            self.voice_id = voice_id
        self._call(lambda e: self._apply(e))

    def shutdown(self) -> None:
        self.stop()
        self._queue.put(_QUIT)
