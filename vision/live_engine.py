"""Live webcam engine for the Streamlit edition.

Two threads, so the video never waits for the AI:
* on_frame()  - called for EVERY browser frame: mirror + draw the HandFX animations (~4 ms)
* AI thread   - MediaPipe + rule-based recognition + stabiliser on the NEWEST frame only
"""
from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

from core.registry import GestureRegistry
from ui.colors import THEMES
from vision.gesture_recognizer import GestureRecognizer
from vision.gesture_stabilizer import GestureStabilizer
from vision.hand_detector import HandDetector
from vision.hand_fx import HandFX
from vision.hand_shape import analyze

log = logging.getLogger(__name__)
T = THEMES["dark"]

class LiveEngine:
    """Shared by the video callback thread, the AI thread and the page."""

    def __init__(self, registry: GestureRegistry, name: str, recognizer: GestureRecognizer | None = None):
        self.lock = threading.Lock()
        self.new_frame = threading.Condition(self.lock)
        self.mode, self.name = "ALPHABET", name
        self.hold, self.min_conf, self.mirror, self.paused = 0.7, 0.6, True, False
        self.recognizer = recognizer or GestureRecognizer(registry)
        self.recognizer.set_dynamic_enabled(True)             # J and Z
        self.stabilizer = GestureStabilizer(0.7, 0.6, 0.35)
        self.quality = "max"          # HandFX graphics: max | balanced | fast
        self.accurate = True          # MediaPipe model: True = accurate, False = fastest
        self.ai_max_hz = 30           # never run the AI more often than this (leaves CPU for video)
        self.fx = HandFX(T, self.quality)
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._ai_thread: threading.Thread | None = None
        self._last_frame_time = 0.0
        self._dt = 0.0
        self._ai_dt = 0.0
        self._last_result = 0.0
        self._last_dynamic = 0.0
        # results (written by the AI thread)
        self.hands: list = []
        self.hands_seq = 0
        self.pred = None
        self.progress, self.locked = 0.0, False
        self.fingers = [0.0] * 5
        self.cam_fps = self.ai_fps = 0.0
        self.error = ""
        self.accepted: list[tuple[str, str, str]] = []
        self._celebrate: list[str] = []
        self.top: list[tuple[str, float]] = []      # "also looks like" candidates
        self.accept_seq, self.last_accepted = 0, ""
        self._drawn_seq = -1

    # ---- called for EVERY camera frame (must be fast) ---------------------------
    def on_frame(self, img: np.ndarray) -> np.ndarray:
        now = time.perf_counter()
        with self.lock:
            self._frame, self._frame_id = img, self._frame_id + 1
            self.new_frame.notify()
            mirror, paused = self.mirror, self.paused
            hands, seq, pred = self.hands, self.hands_seq, self.pred
            progress, locked = self.progress, self.locked
            celebrate, self._celebrate = self._celebrate, []
            if self._last_frame_time:
                dt = now - self._last_frame_time
                self._dt = 0.92 * self._dt + 0.08 * dt if self._dt else dt
                self.cam_fps = 1.0 / max(self._dt, 1e-3)
            self._last_frame_time = now
            name = self.name
            self.fx.quality = self.quality
        self._ensure_ai_thread()
        out = cv2.flip(img, 1) if mirror else img.copy()
        if seq != self._drawn_seq:
            self._drawn_seq = seq
            self.fx.update(hands, (out.shape[1], out.shape[0]), mirror)
        for text in celebrate:
            self.fx.celebrate(text)
        label = None
        if pred is not None and not paused:
            label = pred.output.replace("{name}", name or "…") if pred.output else pred.label
        self.fx.draw(out, label, progress, locked, searching=not paused)
        return out

    # ---- AI thread: MediaPipe + recognition on the newest frame -----------------
    def _ensure_ai_thread(self) -> None:
        if self._ai_thread is None or not self._ai_thread.is_alive():
            self._ai_thread = threading.Thread(target=self._ai_loop, daemon=True, name="sign-ai")
            self._ai_thread.start()

    def _ai_loop(self) -> None:
        accurate = self.accurate
        detector = HandDetector(2, 0.5, 0.5, model_complexity=1 if accurate else 0)
        self.error = detector.error
        last_id, idle_since = 0, time.monotonic()
        while True:
            if self.accurate != accurate:                 # accuracy setting changed
                detector.close()
                accurate = self.accurate
                detector = HandDetector(2, 0.5, 0.5, model_complexity=1 if accurate else 0)
            with self.lock:
                if self._frame_id == last_id:
                    self.new_frame.wait(timeout=0.5)
                if self._frame_id == last_id:
                    if time.monotonic() - idle_since > 30:      # camera stopped: end the thread
                        self._ai_thread = None
                        detector.close()
                        return
                    continue
                idle_since = time.monotonic()
                img, last_id = self._frame, self._frame_id
                mode, paused = self.mode, self.paused
                self.stabilizer.configure(self.hold, self.min_conf, 0.35)
            started = time.perf_counter()
            try:
                h, w = img.shape[:2]
                if w > 640:                               # HD camera: the AI only needs 640 px
                    img = cv2.resize(img, (640, int(640 * h / w)), interpolation=cv2.INTER_AREA)
                hands = [] if paused else detector.detect(img)
            except Exception:  # noqa: BLE001 - a bad frame must not stop recognition
                log.exception("hand detection failed")
                continue
            self._recognize(hands, mode)
            rest = 1.0 / self.ai_max_hz - (time.perf_counter() - started)
            if rest > 0:
                time.sleep(rest)

    # ---- hands found by the BROWSER (cloud: MediaPipe runs in the visitor's browser) ----
    def on_landmarks(self, hands: list, cam_fps: float = 0.0) -> None:
        with self.lock:
            mode, paused = self.mode, self.paused
            self.stabilizer.configure(self.hold, self.min_conf, 0.35)
            self.cam_fps = cam_fps
        self._recognize([] if paused else hands, mode)

    def _recognize(self, hands: list, mode: str) -> None:
        """Rule-based recognition + stabiliser for one set of hands; publishes the results."""
        try:
            pred = self.recognizer.recognize(hands, mode) if hands else None
            if not hands:
                self.recognizer.reset_motion()
            fingers = list(analyze(hands[0].landmarks, hands[0].aspect).ext) if hands else [0.0] * 5
            top = list(getattr(self.recognizer.rules, "top", [])) if hands else []
        except Exception:  # noqa: BLE001 - a bad frame must not stop recognition
            log.exception("recognition failed")
            return
        accepted = None
        if pred is None:
            self.stabilizer.update(None)
        elif pred.dynamic:
            if time.monotonic() - self._last_dynamic > 1.0:
                self._last_dynamic, accepted = time.monotonic(), pred
        elif self.stabilizer.update(pred.label, pred.confidence):
            accepted = pred
        now = time.perf_counter()
        interval = now - self._last_result if self._last_result else 0.0
        self._last_result = now
        with self.lock:
            self.hands, self.hands_seq, self.pred = hands, self.hands_seq + 1, pred
            self.progress = self.stabilizer.progress if pred else 0.0
            self.locked = bool(pred and self.stabilizer.locked_label == pred.label)
            self.fingers, self.top = fingers, top
            if interval:
                self._ai_dt = 0.85 * self._ai_dt + 0.15 * interval if self._ai_dt else interval
                self.ai_fps = 1.0 / max(self._ai_dt, 1e-3)
            if accepted:
                self.accepted.append((accepted.label, accepted.category, accepted.output))
                self._celebrate.append(f"+ {accepted.label}")
                self.accept_seq += 1
                self.last_accepted = accepted.label

    def snapshot(self, drain: bool = True) -> dict:
        with self.lock:
            p = self.pred
            data = dict(label=p.label if p else None, output=p.output if p else "", shape=p.shape if p else "",
                        conf=p.confidence if p else 0.0, dynamic=bool(p and p.dynamic),
                        progress=self.progress, locked=self.locked, hands=len(self.hands),
                        fingers=list(self.fingers), cam_fps=self.cam_fps, ai_fps=self.ai_fps, error=self.error,
                        top=list(self.top), accept_seq=self.accept_seq, last_accepted=self.last_accepted,
                        mode=self.mode, name=self.name, min_conf=self.min_conf)
            data["accepted"] = self.accepted
            if drain:
                self.accepted = []
        return data


def make_callback(engine: LiveEngine):
    import av                          # only needed for WebRTC streaming

    def callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        try:
            img = engine.on_frame(img)
        except Exception:  # noqa: BLE001 - never freeze the stream
            log.exception("frame failed")
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    return callback
