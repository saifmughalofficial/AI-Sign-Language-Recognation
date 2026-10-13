"""Webcam access with a background reader thread (keeps the newest frame only).

Never raises on a missing/busy camera: `open()` returns (ok, friendly_message).
"""
from __future__ import annotations

import os
import sys
import threading

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")   # fast webcam start on Windows
import time

import numpy as np

try:
    import cv2
    try:                     # driver probing prints pages of harmless warnings otherwise
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:  # noqa: BLE001
        pass
except ImportError:          # the app must still start without OpenCV
    cv2 = None


def opencv_available() -> bool:
    return cv2 is not None


class Camera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        self.index = index
        self.width = width
        self.height = height
        self._cap = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._failures = 0
        self.error = ""
        self.error_hint = ""
        self.fps = 0.0
        self.backend = ""

    # ---------------------------------------------------------------- opening
    @staticmethod
    def _backends() -> list:
        if sys.platform.startswith("win"):
            return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        return [cv2.CAP_ANY]

    def _modes(self) -> list[tuple]:
        """(backend, use MJPG, name) to try, fastest first. MJPG lets most webcams do 30 FPS."""
        if sys.platform.startswith("win"):
            return [(cv2.CAP_MSMF, False, "Media Foundation"), (cv2.CAP_DSHOW, True, "DirectShow MJPG"),
                    (cv2.CAP_DSHOW, False, "DirectShow")]
        return [(cv2.CAP_ANY, True, "default MJPG"), (cv2.CAP_ANY, False, "default")]

    def _try(self, index: int, backend, mjpg: bool, width: int, height: int):
        """Open one camera one way. Returns (cap or None, picture_is_not_black, measured fps)."""
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            return None, False, 0.0
        if mjpg:                                    # must be set BEFORE the size on DirectShow
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        got_frame = bright = False
        deadline = time.monotonic() + 2.5
        while time.monotonic() < deadline:          # some webcams need a moment to warm up
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                got_frame = True
                if float(frame.mean()) > 1.0:
                    bright = True
                    break
            time.sleep(0.02)
        if not got_frame:
            cap.release()
            return None, False, 0.0
        count, start = 0, time.monotonic()          # measure the REAL frame rate
        while time.monotonic() - start < 0.7:
            ok, _ = cap.read()
            count += bool(ok)
        return cap, bright, count / max(time.monotonic() - start, 1e-3)

    def open(self) -> tuple[bool, str]:
        if cv2 is None:
            return False, "OpenCV is not installed. Run: pip install opencv-python"
        try:
            best = None                             # (score, cap, index, fps, name, bright)
            indexes = [self.index] + [i for i in range(3) if i != self.index]
            for index in indexes:                   # chosen camera first, then the others
                sizes = [(self.width, self.height)]
                if (self.width, self.height) != (640, 480):
                    sizes.append((640, 480))        # HD too slow? retry at standard size
                for width, height in sizes:
                    for backend, mjpg, name in self._modes():
                        cap, bright, fps = self._try(index, backend, mjpg, width, height)
                        if cap is None:
                            continue
                        score = fps + (100 if bright else 0)
                        if best is None or score > best[0]:
                            if best is not None:
                                best[1].release()
                            best = (score, cap, index, fps, name, bright)
                        else:
                            cap.release()
                        if bright and fps >= 24:    # good enough: stop searching
                            break
                    if best is not None and best[5] and best[3] >= 20:
                        break
                if best is not None:
                    break
            if best is None:
                return False, (f"No working camera found (tried camera {self.index} and others). "
                               "Is it connected, blocked in Windows Privacy settings, "
                               "or being used by another app (Zoom, Teams, another browser tab)?")
            _, cap, self.index, self.fps, self.backend, bright = best
            if not bright:
                self.error_hint = "The camera picture is black — is the lens cover / privacy shutter closed?"
            elif self.fps < 15:
                self.error_hint = (f"The webcam only delivers {self.fps:.0f} FPS — add light to the room "
                                   "(webcams slow down in the dark) or close other apps using the camera.")
            self._cap = cap
        except Exception as exc:  # noqa: BLE001 - driver errors vary a lot
            return False, f"Camera error: {exc}"
        self._stop.clear()
        self._failures = 0
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        return True, "Camera started."

    def _reader(self) -> None:
        while not self._stop.is_set():
            try:
                ok, frame = self._cap.read()
            except Exception:  # noqa: BLE001
                ok, frame = False, None
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame
                    self._frame_id += 1
                self._failures = 0
            else:
                self._failures += 1
                if self._failures > 60:
                    self.error = "Camera stopped sending pictures (unplugged or in use?)."
                    return
                time.sleep(0.02)

    # ---------------------------------------------------------------- reading
    def read(self, last_id: int = -1) -> tuple[np.ndarray | None, int]:
        """Newest frame if it is newer than `last_id`, else (None, last_id)."""
        with self._lock:
            if self._frame is None or self._frame_id == last_id:
                return None, last_id
            return self._frame.copy(), self._frame_id

    @property
    def failed(self) -> bool:
        return bool(self.error)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:  # noqa: BLE001
                pass
        self._cap = None
        self._frame = None


def list_cameras(max_index: int = 4) -> list[int]:
    """Indexes of cameras that can be opened (slow: call only on request)."""
    found: list[int] = []
    if cv2 is None:
        return found
    for i in range(max_index):
        try:
            backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                found.append(i)
            cap.release()
        except Exception:  # noqa: BLE001
            continue
    return found
