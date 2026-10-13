"""A small 3-D kinematic hand used to test the rule-based recogniser without a camera.

Fingers are chains of three bones bent by joint angles (degrees); the thumb tip is
placed at a target point and its joints follow a smooth curve. The finished hand
can be rotated, scaled, moved and jittered, and is returned in MediaPipe's
format: x, y as fractions of the image (y down), z relative depth (negative =
closer to the camera).

Model frame (palm units, wrist at 0): x from the index side (-) to the pinky
side (+), y up along the palm, z negative = palm side (towards the camera when
the palm faces it).
"""
from __future__ import annotations

import math

import numpy as np

from vision.hand_detector import DetectedHand

MCP = {1: (-0.34, 0.92), 2: (-0.10, 1.00), 3: (0.13, 0.95), 4: (0.34, 0.84)}
BONES = {1: (0.42, 0.25, 0.20), 2: (0.47, 0.28, 0.21), 3: (0.44, 0.27, 0.20), 4: (0.33, 0.20, 0.18)}
CMC = np.array([-0.20, 0.18, -0.05])

EXT = (0, 0, 0)
FOLD = (85, 105, 50)
HOOK = (5, 95, 70)
CURVE_C = (30, 55, 35)
CURVE_O = (45, 70, 45)
BENT_E = (15, 100, 75)
CLAW = (20, 60, 45)

NORMAL = {1: -3, 2: 0, 3: 3, 4: 6}
TOGETHER = {1: 2, 2: 0, 3: -2, 4: -5}
SPREAD = {1: -15, 2: 2, 3: 15, 4: 30}


def _finger(i: int, flex: tuple, abduct: float) -> list[np.ndarray]:
    base = np.array([MCP[i][0], MCP[i][1], 0.0])
    pts = [base]
    theta = 0.0
    a = math.radians(abduct)
    for bone, angle in zip(BONES[i], flex):
        theta += math.radians(angle)
        d = np.array([math.sin(a) * math.cos(theta), math.cos(a) * math.cos(theta), -math.sin(theta)])
        pts.append(pts[-1] + bone * d)
    return pts               # MCP, PIP, DIP, TIP


def build(fingers: dict, thumb, abduct: dict | None = None, bulge=(0.0, 0.0, -0.08)) -> np.ndarray:
    """fingers: {1..4: (mcp, pip, dip)}; thumb: tip target (x,y,z) or a function(points)->target."""
    abduct = {**NORMAL, **(abduct or {})}
    pts = np.zeros((21, 3))
    for i in (1, 2, 3, 4):
        chain = _finger(i, fingers.get(i, EXT), abduct[i])
        for k, p in enumerate(chain):
            pts[4 * i + 1 + k] = p
    tip = np.asarray(thumb(pts) if callable(thumb) else thumb, dtype=float)
    ctrl = (CMC + tip) / 2 + np.asarray(bulge)
    curve = lambda t: (1 - t) ** 2 * CMC + 2 * (1 - t) * t * ctrl + t ** 2 * tip  # noqa: E731
    pts[1], pts[2], pts[3], pts[4] = CMC, curve(0.38), curve(0.7), tip
    return pts


def touching(*finger_ids: int, offset=(-0.04, -0.02, -0.04)):
    """Thumb-tip target that touches the given fingertip(s)."""
    def target(pts):
        return np.mean([pts[4 * i + 4] for i in finger_ids], axis=0) + np.asarray(offset)
    return target


def _rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    r, p, y = (math.radians(v) for v in (roll, pitch, yaw))
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    return rz @ rx @ ry


def place(pts: np.ndarray, roll=0.0, pitch=0.0, yaw=0.0, scale=0.18, center=(0.5, 0.75),
          aspect=4 / 3, noise=0.0, rng=None, mirror=False) -> np.ndarray:
    """Rotate (roll = in the picture plane, + = counter-clockwise), scale and convert to MediaPipe format."""
    p = pts.copy()
    if mirror:                      # a left hand is the mirror image of a right hand
        p[:, 0] = -p[:, 0]
    p = p @ _rot(roll, pitch, yaw).T
    if noise and rng is not None:
        p = p + rng.normal(0, noise, p.shape) * np.array([1.0, 1.0, 2.0])
    out = np.zeros_like(p)
    out[:, 0] = (center[0] * aspect + scale * p[:, 0]) / aspect
    out[:, 1] = center[1] - scale * p[:, 1]
    out[:, 2] = scale * p[:, 2] / aspect
    return out.astype(np.float32)


def hand(pts: np.ndarray, aspect=4 / 3, handedness="Right", **kw) -> DetectedHand:
    h = DetectedHand(place(pts, aspect=aspect, **kw), handedness, 0.99)
    h.aspect = aspect
    return h


# ----------------------------------------------------------------- pose library
OUT = (-0.95, 0.5, -0.1)          # thumb sticking out sideways
ACROSS = (0.1, 0.5, -0.2)         # thumb folded across the palm
FIST = {1: FOLD, 2: FOLD, 3: FOLD, 4: FOLD}

POSES: dict[str, dict] = {
    # ---- letters
    "A": dict(fingers=FIST, thumb=(-0.5, 0.85, -0.28), bulge=(-0.05, 0, 0)),
    "B": dict(fingers={}, thumb=ACROSS, abduct=TOGETHER),
    "C": dict(fingers={i: CURVE_C for i in (1, 2, 3, 4)}, thumb=touching(1, offset=(-0.1, -0.5, -0.15)),
              abduct=TOGETHER, view=dict(yaw=-35)),
    "D": dict(fingers={2: CURVE_O, 3: CURVE_O, 4: CURVE_O}, thumb=touching(2)),
    "E": dict(fingers={i: BENT_E for i in (1, 2, 3, 4)}, thumb=(-0.02, 0.72, -0.3)),
    "F": dict(fingers={1: CURVE_O}, thumb=touching(1), abduct=SPREAD),
    "G": dict(fingers={2: FOLD, 3: FOLD, 4: FOLD}, thumb=(-0.6, 0.9, -0.15), view=dict(roll=90)),
    "H": dict(fingers={3: FOLD, 4: FOLD}, thumb=ACROSS, abduct=TOGETHER, view=dict(roll=90)),
    "I": dict(fingers={1: FOLD, 2: FOLD, 3: FOLD}, thumb=(0.0, 0.62, -0.5)),
    "K": dict(fingers={3: FOLD, 4: FOLD}, thumb=(-0.22, 1.12, -0.25), abduct=SPREAD),
    "L": dict(fingers={2: FOLD, 3: FOLD, 4: FOLD}, thumb=OUT),
    "M": dict(fingers=FIST, thumb=(0.22, 0.76, -0.32)),
    "N": dict(fingers=FIST, thumb=(0.0, 0.8, -0.34)),
    "O": dict(fingers={i: CURVE_O for i in (1, 2, 3, 4)}, thumb=touching(1), abduct=TOGETHER),
    "P": dict(fingers={2: (0, 30, 10), 3: FOLD, 4: FOLD}, thumb=(-0.22, 1.12, -0.25), abduct=SPREAD,
              view=dict(roll=160)),
    "Q": dict(fingers={2: FOLD, 3: FOLD, 4: FOLD}, thumb=(-0.6, 0.9, -0.15), view=dict(roll=170)),
    "R": dict(fingers={1: (8, 0, 0), 3: FOLD, 4: FOLD}, thumb=ACROSS, abduct={1: 16, 2: -6}),
    "S": dict(fingers=FIST, thumb=(-0.02, 0.62, -0.55)),
    "T": dict(fingers=FIST, thumb=(-0.22, 0.82, -0.36)),
    "U": dict(fingers={3: FOLD, 4: FOLD}, thumb=ACROSS, abduct=TOGETHER),
    "V": dict(fingers={3: FOLD, 4: FOLD}, thumb=ACROSS, abduct=SPREAD),
    "W": dict(fingers={4: FOLD}, thumb=touching(4), abduct=SPREAD),
    "X": dict(fingers={1: HOOK, 2: FOLD, 3: FOLD, 4: FOLD}, thumb=(0.0, 0.62, -0.5)),
    "Y": dict(fingers={1: FOLD, 2: FOLD, 3: FOLD}, thumb=OUT),
    # ---- numbers
    "ONE": dict(fingers={2: FOLD, 3: FOLD, 4: FOLD}, thumb=(0.0, 0.62, -0.5)),
    "TWO": dict(fingers={3: FOLD, 4: FOLD}, thumb=ACROSS, abduct=SPREAD),
    "THREE": dict(fingers={3: FOLD, 4: FOLD}, thumb=OUT, abduct=SPREAD),
    "FOUR": dict(fingers={}, thumb=ACROSS, abduct=SPREAD),
    "FIVE": dict(fingers={}, thumb=OUT, abduct=SPREAD),
    "SEVEN": dict(fingers={3: CURVE_O}, thumb=touching(3), abduct=SPREAD),
    "EIGHT": dict(fingers={2: CURVE_O}, thumb=touching(2), abduct=SPREAD),
    # ---- words
    "OPEN": dict(fingers={}, thumb=OUT, abduct=SPREAD),
    "PALM": dict(fingers={}, thumb=ACROSS, abduct=TOGETHER),
    "FLAT_THUMB": dict(fingers={}, thumb=OUT, abduct=TOGETHER, view=dict(roll=40)),
    "FLAT_SIDE": dict(fingers={}, thumb=ACROSS, abduct=TOGETHER, view=dict(roll=95)),
    "THUMBS_UP": dict(fingers=FIST, thumb=(-0.5, 1.65, -0.2)),
    "THUMBS_DOWN": dict(fingers=FIST, thumb=(-0.5, 1.65, -0.2), view=dict(roll=180)),
    "ILY": dict(fingers={2: FOLD, 3: FOLD}, thumb=OUT, abduct=SPREAD),
    "CLAW": dict(fingers={i: CLAW for i in (1, 2, 3, 4)}, thumb=(-0.85, 0.55, -0.35), abduct=SPREAD),
    "PINCH2": dict(fingers={1: (40, 20, 10), 2: (40, 20, 10), 3: FOLD, 4: FOLD},
                   thumb=touching(1, 2), abduct=TOGETHER),
    "POINT_SIDE": dict(fingers={2: FOLD, 3: FOLD, 4: FOLD}, thumb=(0.0, 0.62, -0.5), view=dict(roll=-90)),
}
# shapes that are only a re-use of another pose
POSES["2"] = POSES["TWO"]


def pose_points(name: str) -> tuple[np.ndarray, dict]:
    spec = POSES[name]
    pts = build(spec["fingers"], spec["thumb"], spec.get("abduct"), spec.get("bulge", (0.0, 0.0, -0.08)))
    return pts, dict(spec.get("view", {}))


def make(name: str, rng=None, jitter: bool = False, **overrides) -> DetectedHand:
    """A DetectedHand showing the named pose (optionally randomly tilted + noisy)."""
    pts, view = pose_points(name)
    if jitter and rng is not None:
        view["roll"] = view.get("roll", 0) + float(rng.uniform(-12, 12))
        view["pitch"] = view.get("pitch", 0) + float(rng.uniform(-15, 15))
        view["yaw"] = view.get("yaw", 0) + float(rng.uniform(-20, 20))
        view["noise"] = 0.012
        view["rng"] = rng
        view["scale"] = float(rng.uniform(0.12, 0.25))
    view.update(overrides)
    return hand(pts, **view)


def make_pair(left: str, right: str, gap: float = 0.30, **kw) -> list[DetectedHand]:
    """Two hands side by side (`gap` = distance between wrists as a fraction of the image width)."""
    a = make(left, center=(0.5 - gap / 2, 0.75), mirror=True, handedness="Left", **kw)
    b = make(right, center=(0.5 + gap / 2, 0.75), **kw)
    return [a, b]
