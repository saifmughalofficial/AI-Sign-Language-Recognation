"""Offline speech recognition with Vosk (free, open source, no internet needed).

Setup (one time, needs internet only for the download):
    python setup_dataset.py --download-vosk
or download a model manually from https://alphacephei.com/vosk/models
(the small English model ~40 MB is enough) and unzip it so that the folder
looks like:   models/vosk-model-small-en-us-0.15/

If Vosk, sounddevice, the model or the microphone is missing, VoiceListener
reports a friendly message and the app keeps working (Person B can type).
"""
from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path
from typing import Callable

from core.paths import MODELS_DIR

log = logging.getLogger(__name__)

VOSK_SETUP_TEXT = (
    "Voice Mode needs a free offline speech model.\n"
    "1. Run:  python setup_dataset.py --download-vosk   (needs internet once)\n"
    "   or download 'vosk-model-small-en-us-0.15' from https://alphacephei.com/vosk/models\n"
    "2. Unzip it into the 'models' folder so you have models/vosk-model-small-en-us-0.15/\n"
    "3. Restart the app.  Until then, Person B can type in the box instead."
)


def find_vosk_model(models_dir: Path | str = MODELS_DIR) -> Path | None:
    """First folder inside `models` whose name starts with 'vosk' and looks like a model."""
    base = Path(models_dir)
    if not base.exists():
        return None
    for folder in sorted(base.iterdir()):
        if folder.is_dir() and folder.name.lower().startswith("vosk"):
            if (folder / "am").exists() or (folder / "conf").exists():
                return folder
            for sub in folder.iterdir():                # zip sometimes adds an extra folder level
                if sub.is_dir() and ((sub / "am").exists() or (sub / "conf").exists()):
                    return sub
    return None


def check_voice_support(models_dir: Path | str = MODELS_DIR) -> tuple[bool, str]:
    """(ready, explanation). Never raises."""
    try:
        import vosk  # noqa: F401
    except Exception:  # noqa: BLE001
        return False, "Vosk is not installed (pip install vosk)."
    try:
        import sounddevice  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - OSError when PortAudio is missing
        return False, f"sounddevice / microphone driver unavailable ({exc})."
    if find_vosk_model(models_dir) is None:
        return False, "No Vosk speech model found in the models folder."
    return True, "Voice recognition ready."


class VoiceListener:
    """Listens to the microphone in a background thread.

    Callbacks (called from the background thread - keep them quick / thread-safe):
        on_partial(text)  words still being spoken
        on_final(text)    a finished sentence
        on_status(text)   "listening", errors, ...
    """

    SAMPLE_RATE = 16000

    def __init__(self, on_partial: Callable[[str], None] | None = None,
                 on_final: Callable[[str], None] | None = None,
                 on_status: Callable[[str], None] | None = None,
                 models_dir: Path | str = MODELS_DIR, device: int | None = None):
        self.on_partial = on_partial or (lambda t: None)
        self.on_final = on_final or (lambda t: None)
        self.on_status = on_status or (lambda t: None)
        self.models_dir = Path(models_dir)
        self.device = device
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.listening = False
        self.error = ""

    def start(self) -> tuple[bool, str]:
        if self.listening:
            return True, "Already listening."
        ok, message = check_voice_support(self.models_dir)
        if not ok:
            self.error = message
            return False, message
        self._stop.clear()
        self.listening = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="voice")
        self._thread.start()
        return True, "Listening…"

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self.listening = False

    def _run(self) -> None:
        try:
            import sounddevice as sd
            from vosk import KaldiRecognizer, Model, SetLogLevel
            SetLogLevel(-1)
            model_path = find_vosk_model(self.models_dir)
            self.on_status("Loading speech model…")
            recognizer = KaldiRecognizer(Model(str(model_path)), self.SAMPLE_RATE)
            audio: queue.Queue = queue.Queue()

            def callback(indata, frames, time_info, status):  # runs in the audio thread
                audio.put(bytes(indata))

            with sd.RawInputStream(samplerate=self.SAMPLE_RATE, blocksize=4000, dtype="int16",
                                   channels=1, device=self.device, callback=callback):
                self.on_status("listening")
                while not self._stop.is_set():
                    try:
                        chunk = audio.get(timeout=0.3)
                    except queue.Empty:
                        continue
                    if recognizer.AcceptWaveform(chunk):
                        text = json.loads(recognizer.Result()).get("text", "").strip()
                        if text:
                            self.on_final(text)
                    else:
                        partial = json.loads(recognizer.PartialResult()).get("partial", "").strip()
                        if partial:
                            self.on_partial(partial)
        except Exception as exc:  # noqa: BLE001 - no microphone, permission denied, bad model...
            self.error = f"Microphone / speech error: {exc}"
            log.warning(self.error)
            self.on_status(self.error)
        finally:
            self.listening = False
            self.on_status("stopped")
