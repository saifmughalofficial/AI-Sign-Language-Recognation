"""MediaPipe hand-landmark detection (100% local).

Two MediaPipe generations exist and this wrapper supports both:

* "legacy"  mediapipe <= 0.10.21  -> mp.solutions.hands  (model is inside the pip package)
* "tasks"   newer versions        -> HandLandmarker, needs models/hand_landmarker.task
                                    (free one-time download: python setup_dataset.py --download-hand-model)

requirements.txt pins the legacy version so nothing extra is needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.paths import HAND_TASK_FILE

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import mediapipe as mp
except Exception:  # noqa: BLE001 - import can fail in many ways
    mp = None

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
    (17, 18), (18, 19), (19, 20),
]


@dataclass
class DetectedHand:
    landmarks: np.ndarray      # (21, 3) x,y in 0..1 of the UN-flipped frame, z relative
    handedness: str = "Right"  # MediaPipe's label (see feature_extractor note)
    score: float = 1.0
    aspect: float = 1.0        # image width / height (x is squashed by this much)


class HandDetector:
    def __init__(self, max_hands: int = 2, detection_confidence: float = 0.6,
                 tracking_confidence: float = 0.5, static_image_mode: bool = False,
                 model_complexity: int = 1):
        self.backend: str | None = None
        self.error = ""
        self.static_image_mode = static_image_mode
        self._detector = None
        self._timestamp_ms = 0
        if cv2 is None:
            self.error = "OpenCV is not installed (pip install opencv-python)."
            return
        if mp is None:
            self.error = "MediaPipe is not installed (pip install mediapipe==0.10.14)."
            return
        try:
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                self._detector = mp.solutions.hands.Hands(
                    static_image_mode=static_image_mode,
                    max_num_hands=max_hands,
                    model_complexity=model_complexity,   # 1 = accurate, 0 = fastest
                    min_detection_confidence=detection_confidence,
                    min_tracking_confidence=tracking_confidence)
                self.backend = "legacy"
            else:
                self._init_tasks(max_hands, detection_confidence, tracking_confidence)
        except Exception as exc:  # noqa: BLE001
            self.backend = None
            self._detector = None
            self.error = f"Hand detector could not start: {exc}"

    def _init_tasks(self, max_hands: int, det: float, trk: float) -> None:
        if not HAND_TASK_FILE.exists():
            self.error = ("This MediaPipe version needs models/hand_landmarker.task. "
                          "Run: python setup_dataset.py --download-hand-model  "
                          "(or install mediapipe==0.10.14).")
            return
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        mode = (mp_vision.RunningMode.IMAGE if self.static_image_mode
                else mp_vision.RunningMode.VIDEO)
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(HAND_TASK_FILE)),
            running_mode=mode, num_hands=max_hands,
            min_hand_detection_confidence=det,
            min_hand_presence_confidence=det,
            min_tracking_confidence=trk)
        self._detector = mp_vision.HandLandmarker.create_from_options(options)
        self.backend = "tasks"

    @property
    def available(self) -> bool:
        return self._detector is not None

    # -------------------------------------------------------------- detection
    def detect(self, frame_bgr: np.ndarray) -> list[DetectedHand]:
        """Find hands in an un-flipped BGR frame. Returns [] when none / on error."""
        if not self.available:
            return []
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            hands = self._detect_legacy(rgb) if self.backend == "legacy" else self._detect_tasks(rgb)
            h, w = frame_bgr.shape[:2]
            for hand in hands:
                hand.aspect = w / float(h) if h else 1.0
            return hands
        except Exception as exc:  # noqa: BLE001 - one bad frame must not stop the app
            self.error = f"Hand detection error: {exc}"
            return []

    def _detect_legacy(self, rgb: np.ndarray) -> list[DetectedHand]:
        rgb.flags.writeable = False
        result = self._detector.process(rgb)
        hands: list[DetectedHand] = []
        if not result.multi_hand_landmarks:
            return hands
        labels = result.multi_handedness or []
        for i, lm in enumerate(result.multi_hand_landmarks):
            pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
            label, score = "Right", 1.0
            if i < len(labels):
                cls = labels[i].classification[0]
                label, score = cls.label, float(cls.score)
            hands.append(DetectedHand(pts, label, score))
        return hands

    def _detect_tasks(self, rgb: np.ndarray) -> list[DetectedHand]:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if self.static_image_mode:
            result = self._detector.detect(image)
        else:
            self._timestamp_ms += 33
            result = self._detector.detect_for_video(image, self._timestamp_ms)
        hands: list[DetectedHand] = []
        for i, lm in enumerate(result.hand_landmarks or []):
            pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
            label, score = "Right", 1.0
            if result.handedness and i < len(result.handedness) and result.handedness[i]:
                cat = result.handedness[i][0]
                label, score = cat.category_name, float(cat.score)
            hands.append(DetectedHand(pts, label, score))
        return hands

    def close(self) -> None:
        try:
            if self._detector is not None and hasattr(self._detector, "close"):
                self._detector.close()
        except Exception:  # noqa: BLE001
            pass
        self._detector = None


# One colour per finger (BGR): thumb pink, index orange, middle yellow, ring green, pinky cyan
FINGER_COLORS = [(180, 105, 255), (60, 160, 255), (60, 220, 250), (120, 230, 100), (250, 210, 80)]
_PALM = {(0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)}


def _bone_color(a: int, b: int) -> tuple:
    if (a, b) in _PALM:
        return (235, 235, 235)
    return FINGER_COLORS[(max(a, b) - 1) // 4]


def hand_box(hand: DetectedHand, mirrored: bool) -> tuple[float, float, float, float]:
    """Bounding box (x1, y1, x2, y2) as fractions of the SHOWN picture."""
    xs = hand.landmarks[:, 0]
    if mirrored:
        xs = 1.0 - xs
    ys = hand.landmarks[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def draw_hands(display_bgr: np.ndarray, hands: list[DetectedHand], mirrored: bool,
               draw_box: bool = True, box_color: tuple = (250, 190, 56)) -> None:
    """Glowing, colour-coded skeleton + corner brackets, drawn on the frame that will be SHOWN."""
    if cv2 is None or not hands:
        return
    h, w = display_bgr.shape[:2]
    glow = display_bgr.copy()
    all_pts = []
    for hand in hands:
        xs = hand.landmarks[:, 0]
        ys = hand.landmarks[:, 1]
        if mirrored:
            xs = 1.0 - xs
        pts = [(int(x * w), int(y * h)) for x, y in zip(xs, ys)]
        all_pts.append(pts)
        for a, b in HAND_CONNECTIONS:            # soft wide halo, blended below
            cv2.line(glow, pts[a], pts[b], _bone_color(a, b), 9, cv2.LINE_AA)
    cv2.addWeighted(glow, 0.35, display_bgr, 0.65, 0, dst=display_bgr)

    for pts in all_pts:
        for a, b in HAND_CONNECTIONS:
            cv2.line(display_bgr, pts[a], pts[b], _bone_color(a, b), 2, cv2.LINE_AA)
        for i, p in enumerate(pts):
            if i in (4, 8, 12, 16, 20):
                color = FINGER_COLORS[i // 4 - 1]
                cv2.circle(display_bgr, p, 7, color, 2, cv2.LINE_AA)
                cv2.circle(display_bgr, p, 3, (255, 255, 255), -1, cv2.LINE_AA)
            else:
                cv2.circle(display_bgr, p, 3, (255, 255, 255), -1, cv2.LINE_AA)
        if draw_box:
            x1, y1 = max(min(p[0] for p in pts) - 18, 0), max(min(p[1] for p in pts) - 18, 0)
            x2, y2 = min(max(p[0] for p in pts) + 18, w - 1), min(max(p[1] for p in pts) + 18, h - 1)
            arm = max(12, (x2 - x1) // 6)
            color = box_color
            for (cx, cy), (dx, dy) in (((x1, y1), (1, 1)), ((x2, y1), (-1, 1)),
                                       ((x1, y2), (1, -1)), ((x2, y2), (-1, -1))):
                cv2.line(display_bgr, (cx, cy), (cx + dx * arm, cy), color, 3, cv2.LINE_AA)
                cv2.line(display_bgr, (cx, cy), (cx, cy + dy * arm), color, 3, cv2.LINE_AA)
