"""Local webcam -> LiveEngine -> MJPEG video stream for the Streamlit page.

No WebRTC: Python opens the webcam itself (vision/camera.py), every frame goes
through LiveEngine.on_frame (animations), and the result is served as an MJPEG
stream on http://127.0.0.1:<port>/stream that the page shows in an <img> tag.
Only this computer can connect (it listens on 127.0.0.1).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from vision.camera import Camera
from vision.shape_library import SHAPE_HINTS

log = logging.getLogger(__name__)
BOUNDARY = b"frame"


class CameraStream:
    def __init__(self, engine, port: int = 8900):
        self.engine = engine
        self.port = port
        self.camera: Camera | None = None
        self.error = ""
        self._jpeg: bytes = b""
        self.size = (0, 0)
        self._seq = 0
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._server = ThreadingHTTPServer(("127.0.0.1", port), self._handler())
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True, name="mjpeg-http").start()

    # ---------------------------------------------------------------- control
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, index: int = 0, width: int = 640, height: int = 480) -> tuple[bool, str]:
        if self.running:
            return True, "Camera already running."
        cam = Camera(index, width, height)
        ok, message = cam.open()
        if not ok:
            self.error = message
            return False, message
        self.camera, self.error = cam, getattr(cam, "error_hint", "")
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="camera-loop")
        self._thread.start()
        return True, message

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._thread = None
        if self.camera is not None:
            self.camera.close()
        self.camera = None

    # ------------------------------------------------------------ camera loop
    def _loop(self) -> None:
        frame_id = -1
        while not self._stop.is_set():
            frame, new_id = self.camera.read(frame_id)
            if frame is None:
                if self.camera.failed:
                    self.error = self.camera.error
                    return
                time.sleep(0.003)
                continue
            frame_id = new_id
            try:
                out = self.engine.on_frame(frame)
                self.size = (out.shape[1], out.shape[0])
                ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
            except Exception:  # noqa: BLE001 - one bad frame must not stop the video
                log.exception("frame failed")
                continue
            if ok:
                with self._cond:
                    self._jpeg, self._seq = buf.tobytes(), self._seq + 1
                    self._cond.notify_all()

    # ------------------------------------------------------------------- http
    def _handler(self):
        stream = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):          # keep the console quiet
                pass

            def do_GET(self):
                if self.path.startswith("/state"):     # live numbers for the page's HUD widget
                    self._state()
                    return
                if not self.path.startswith("/stream"):
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache, no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                last = -1
                try:
                    while True:
                        with stream._cond:
                            stream._cond.wait_for(lambda: stream._seq != last, timeout=1.0)
                            jpeg, last = stream._jpeg, stream._seq
                        if not jpeg:
                            continue
                        self.wfile.write(b"--" + BOUNDARY + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    pass                            # the page was closed or reloaded

            def _state(self):
                snap = stream.engine.snapshot(drain=False)
                snap.pop("accepted", None)
                name = snap.get("name") or "…"
                snap["output"] = (snap.get("output") or "").replace("{name}", name)
                snap["hint_shape"] = SHAPE_HINTS.get(snap.get("shape") or "", "")
                cam = stream.camera
                snap.update(running=stream.running, size=list(stream.size),
                            backend=cam.backend if cam else "", hint=stream.error)
                body = json.dumps(snap).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except OSError:                     # the page was closed mid-reply
                    pass

        return Handler
