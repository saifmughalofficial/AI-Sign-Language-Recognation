"""One object that wires all parts of the app together (and makes testing easy)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.history import ConversationHistory
from core.paths import ensure_dirs
from core.registry import GestureRegistry
from core.settings import Settings, SettingsStore
from sentence.phrase_engine import PhraseEngine
from speech.text_to_speech import TextToSpeech


@dataclass
class AppContext:
    store: SettingsStore
    settings: Settings
    registry: GestureRegistry
    history: ConversationHistory
    phrase_engine: PhraseEngine
    tts: TextToSpeech
    recognizer: Any            # vision.gesture_recognizer.GestureRecognizer
    pipeline: Any              # vision.pipeline.VisionPipeline
    startup_notes: list

    @classmethod
    def create(cls) -> "AppContext":
        from vision.gesture_recognizer import GestureRecognizer
        from vision.pipeline import VisionPipeline
        ensure_dirs()
        store = SettingsStore()
        settings = store.load()
        registry = GestureRegistry()
        history = ConversationHistory(persist=settings.save_history)
        history.load()
        recognizer = GestureRecognizer(registry)
        ctx = cls(
            store=store, settings=settings, registry=registry, history=history,
            phrase_engine=PhraseEngine(registry.phrases),
            tts=TextToSpeech(settings.speech_rate, settings.speech_volume, settings.voice_id),
            recognizer=recognizer, pipeline=None,
            startup_notes=list(store.notes) + list(registry.problems) + list(recognizer.load_errors),
        )
        ctx.pipeline = VisionPipeline(lambda: ctx.settings, recognizer)
        ctx.pipeline.mode = settings.mode
        return ctx
