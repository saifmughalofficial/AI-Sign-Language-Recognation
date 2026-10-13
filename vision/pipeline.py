"""Background worker: camera -> hand detection -> gesture recognition -> preview picture.

Runs in its own thread so the window never freezes. The dashboard simply asks
for `latest()` a few times per second.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from core.constants import MSG_LOW_CONF, MSG_NO_HAND, MSG_PAUSED
from core.settings import Settings
from vision.camera import Camera
from vision.gesture_recognizer import GestureRecognizer, Prediction

try:
    import cv2
except ImportError:
    cv2 = None

log = logging.getLogger(__name__)


@dataclass
class FrameResult:
    seq: int = 0
    image_rgb: np.ndarray | None = None
    prediction: Prediction | None = None
    hands: int = 0
    fps: float = 0.0
    message: str = ""
    state: str = "stopped"           # stopped | starting | running | error
    camera_ok: bool = False
    hand_detected: bool = False
    recognition_active: bool = False
    paused: bool = False
    error: str = ""
    extra: dict = field(default_factory=dict)


class VisionPipeline:
    def __init__(self, get_settings: Callable[[], Settings], recognizer: GestureRecognizer):
        self._get_settings = get_settings
        self.recognizer = recognizer
        self.mode = "ALPHABET"
        self.paused = False
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = FrameResult()
        self._seq = 0
        self.engine = None                 # vision.live_engine.LiveEngine while the camera runs
        self.display_size = (960, 720)
        self.quality = "max"               # hand graphics: max | balanced | fast

    # ------------------------------------------------------------------ control
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._publish(FrameResult(state="starting", message="Starting camera…"))
        self._thread = threading.Thread(target=self._run, daemon=True, name="vision")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._thread = None
        self._publish(FrameResult(state="stopped"))

    def latest(self) -> FrameResult:
        with self._lock:
            return self._latest

    def _publish(self, result: FrameResult) -> None:
        with self._lock:
            self._seq += 1
            result.seq = self._seq
            self._latest = result

    # -------------------------------------------------------- UI <-> worker
    def set_display_size(self, width: int, height: int) -> None:
        """Size of the video area: frames are resized here, in the worker thread."""
        self.display_size = (max(1, width), max(1, height))

    def take_accepted(self) -> list[tuple[str, str, str]]:
        """Gestures the engine accepted since the last call: (label, category, output)."""
        engine = self.engine
        if engine is None:
            return []
        with engine.lock:
            items, engine.accepted = engine.accepted, []
        return items

    def engine_state(self) -> dict:
        return self.engine.snapshot(drain=False) if self.engine is not None else {}

    # ------------------------------------------------------------------- worker
    def _run(self) -> None:
        from vision.live_engine import LiveEngine

        settings = self._get_settings()
        width, height = settings.resolution_wh
        camera = Camera(settings.camera_index, width, height)
        ok, message = camera.open()
        if not ok:
            self._publish(FrameResult(state="error", error=message, message=message))
            return
        engine = LiveEngine(self.recognizer.registry, settings.user_name, recognizer=self.recognizer)
        self.recognizer.set_dynamic_enabled(settings.experimental_dynamic)
        self.engine = engine
        frame_id = -1
        try:
            while not self._stop.is_set():
                frame, frame_id_new = camera.read(frame_id)
                if frame is None:
                    if camera.failed:
                        self._publish(FrameResult(state="error", error=camera.error, message=camera.error))
                        return
                    time.sleep(0.003)
                    continue
                frame_id = frame_id_new
                settings = self._get_settings()
                with engine.lock:
                    engine.mode, engine.paused, engine.mirror = self.mode, self.paused, settings.mirror_camera
                    engine.hold, engine.min_conf = settings.gesture_hold_seconds, settings.recognition_confidence
                    engine.name = settings.user_name
                    engine.quality = self.quality if settings.show_landmarks else "fast"
                    engine.ai_max_hz = 30 / max(1, settings.process_every_n_frames)
                shown = engine.on_frame(frame)
                dw, dh = self.display_size
                scale = min(dw / shown.shape[1], dh / shown.shape[0])
                if scale > 0 and abs(scale - 1.0) > 0.02:
                    shown = cv2.resize(shown, (max(1, int(shown.shape[1] * scale)), max(1, int(shown.shape[0] * scale))),
                                       interpolation=cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA)
                snap = engine.snapshot(drain=False)
                pred = engine.pred
                if self.paused:
                    note = MSG_PAUSED
                elif not snap["hands"]:
                    note = MSG_NO_HAND
                elif pred is None:
                    note = self.recognizer.last_note or MSG_LOW_CONF
                elif pred.confidence < settings.recognition_confidence:
                    note = MSG_LOW_CONF
                else:
                    note = ""
                self._publish(FrameResult(
                    image_rgb=cv2.cvtColor(shown, cv2.COLOR_BGR2RGB), prediction=pred, hands=snap["hands"],
                    fps=snap["cam_fps"], message=note, state="running", camera_ok=True,
                    hand_detected=bool(snap["hands"]), recognition_active=not self.paused and not snap["error"],
                    paused=self.paused, error=snap["error"],
                    extra={"progress": snap["progress"], "locked": snap["locked"], "fingers": snap["fingers"],
                           "ai_fps": snap["ai_fps"], "camera": f"{camera.backend} {frame.shape[1]}x{frame.shape[0]}"}))
        except Exception as exc:  # noqa: BLE001
            log.exception("vision worker crashed")
            self._publish(FrameResult(state="error", error=f"Vision error: {exc}", message=f"Vision error: {exc}"))
        finally:
            camera.close()
            self.engine = None
