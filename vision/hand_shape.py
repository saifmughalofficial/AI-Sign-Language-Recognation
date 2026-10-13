"""Hand-shape geometry: turns 21 MediaPipe landmarks into human-meaningful measurements.

No training is needed. Everything is measured from the landmarks of ONE frame:

* how straight / folded / hooked / curved each finger is  (real 3-D joint angles)
* where the thumb is: sticking out, tucked across the palm, between two fingers,
  touching a fingertip, in front of the fist ...
* whether fingers are spread, held together or crossed
* where the index finger, the thumb and the whole hand point on the screen

All values are soft scores between 0 and 1 (or plain measurements) so that the
shape library (vision/shape_library.py) can combine them into a confidence.

Coordinates: MediaPipe gives x and y as fractions of the image width and height.
For a 640x480 camera that squashes x, so x (and z, which uses the same scale as x)
are multiplied by the aspect ratio before measuring anything.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

WRIST = 0
# finger number -> (MCP, PIP, DIP, TIP) landmark ids; 1 = index ... 4 = pinky
FINGERS = {1: (5, 6, 7, 8), 2: (9, 10, 11, 12), 3: (13, 14, 15, 16), 4: (17, 18, 19, 20)}
TIPS = {0: 4, 1: 8, 2: 12, 3: 16, 4: 20}


# ------------------------------------------------------------------ soft logic
def ramp(value: float, lo: float, hi: float) -> float:
    """0 at `lo`, 1 at `hi`, linear in between. lo > hi gives a falling ramp."""
    if hi == lo:
        return 1.0 if value >= hi else 0.0
    return float(min(1.0, max(0.0, (value - lo) / (hi - lo))))


def band(value: float, lo: float, hi: float, soft: float) -> float:
    """1 inside [lo, hi], fading to 0 within `soft` outside the band."""
    return min(ramp(value, lo - soft, lo), ramp(value, hi + soft, hi))


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v * 0.0


def _angle(u: np.ndarray, v: np.ndarray) -> float:
    """Angle between two vectors in degrees (0 = same direction)."""
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    c = float(np.dot(u, v)) / (nu * nv)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


@dataclass
class Direction:
    """Where a vector points on the SCREEN (y grows downward)."""
    up: float = 0.0
    down: float = 0.0
    side: float = 0.0
    forward: float = 0.0       # pointing at the camera (looks very short on screen)

    @classmethod
    def of(cls, vec2: np.ndarray, length3: float) -> "Direction":
        length2 = float(np.linalg.norm(vec2))
        if length2 < 1e-9:
            return cls(forward=1.0)
        u = vec2 / length2
        ratio = length2 / max(length3, 1e-9)
        return cls(up=ramp(-u[1], 0.35, 0.75), down=ramp(u[1], 0.35, 0.75),
                   side=ramp(abs(u[0]), 0.6, 0.85), forward=ramp(ratio, 0.7, 0.45))


@dataclass
class HandFeatures:
    pts: np.ndarray                 # (21, 3) wrist-centred, palm-size normalised, aspect-corrected
    origin: np.ndarray              # wrist position (aspect-corrected image units)
    palm: float                     # palm size (wrist -> middle knuckle) in image units
    ext: list = field(default_factory=list)     # [thumb, index, middle, ring, pinky] extended 0..1
    fold: list = field(default_factory=list)    # finger folded into the palm
    hook: list = field(default_factory=list)    # finger raised but bent at the middle joints (X)
    curve: list = field(default_factory=list)   # finger gently curved (C / O)
    bend: list = field(default_factory=list)    # PIP+DIP bend in degrees
    mcp: list = field(default_factory=list)     # knuckle bend in degrees
    touch: list = field(default_factory=list)   # thumb tip touching that fingertip 0..1
    gap: list = field(default_factory=list)     # thumb tip -> fingertip distance (palm units)
    spread: list = field(default_factory=list)  # angle between neighbours: [im, mr, rp] degrees
    thumb_out: float = 0.0          # thumb clearly away from the hand
    thumb_straight: float = 0.0
    thumb_reach: float = 0.0        # thumb tip -> index knuckle distance (palm units)
    thumb_tx: float = 0.0           # thumb tip across the knuckles: 0 = index side, 1 = pinky side
    thumb_ty: float = 0.0           # thumb tip height along the palm (0 wrist, 1 knuckles)
    thumb_depth: float = 0.0        # >0 thumb tip in front of the folded fingers, <0 tucked behind
    tips_over_thumb: float = 0.0    # mean height of index..ring tips minus thumb tip height
    tips_to_thumb: float = 0.0      # mean distance of the 4 fingertips to the thumb tip
    crossed: float = 0.0            # index crossed over the middle finger (R)
    thumb_index_angle: float = 0.0  # degrees between thumb and index finger
    index_dir: Direction = field(default_factory=Direction)
    hand_dir: Direction = field(default_factory=Direction)
    thumb_dir: Direction = field(default_factory=Direction)

    # handy groups -------------------------------------------------------------
    @property
    def together(self) -> float:
        """Index..pinky held together (not spread)."""
        return ramp(max(self.spread), 16.0, 8.0)

    @property
    def spread_out(self) -> float:
        return ramp(float(np.mean(self.spread)), 7.0, 15.0)

    @property
    def thumb_in(self) -> float:
        return 1.0 - self.thumb_out


def analyze(landmarks: np.ndarray, aspect: float = 1.0) -> HandFeatures:
    """Measure one hand. `landmarks` is (21, 3) straight from MediaPipe."""
    raw = np.asarray(landmarks, dtype=np.float64).reshape(21, 3).copy()
    raw[:, 0] *= aspect
    raw[:, 2] *= aspect
    origin = raw[0].copy()
    p = raw - origin
    palm = float(np.linalg.norm(p[9]))
    if palm < 1e-6:
        palm = max(float(np.max(np.linalg.norm(p, axis=1))), 1e-6)
    q = p / palm

    # ---- hand-local frame: y along the palm, x from index to pinky knuckle, n = palm normal
    y_ax = _unit(q[9])
    x_raw = q[17] - q[5]
    x_ax = _unit(x_raw - np.dot(x_raw, y_ax) * y_ax)
    if not np.any(x_ax):
        x_ax = np.array([1.0, 0.0, 0.0])
    n_ax = np.cross(x_ax, y_ax)
    local = np.stack([q @ x_ax, q @ y_ax, q @ n_ax], axis=1)

    f = HandFeatures(pts=q, origin=origin, palm=palm)

    # ---- the four fingers ----------------------------------------------------
    ext, fold, hook, curve, bend_l, mcp_l = [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]
    for i in (1, 2, 3, 4):
        m, pp, d, t = FINGERS[i]
        prox = q[pp] - q[m]
        flex = math.degrees(math.atan2(abs(float(np.dot(prox, n_ax))), float(np.dot(prox, y_ax))))
        bend = _angle(prox, q[d] - q[pp]) + _angle(q[d] - q[pp], q[t] - q[d])
        total = bend + 0.8 * flex
        ext.append(ramp(bend, 95.0, 45.0) * ramp(flex, 80.0, 50.0))
        fold.append(ramp(total, 125.0, 185.0))
        hook.append(ramp(bend, 75.0, 115.0) * ramp(flex, 60.0, 35.0))
        curve.append(band(bend + flex, 70.0, 175.0, 30.0) * ramp(bend, 30.0, 55.0))
        bend_l.append(bend)
        mcp_l.append(flex)

    # ---- thumb ---------------------------------------------------------------
    t_bend = _angle(q[2] - q[1], q[3] - q[2]) + _angle(q[3] - q[2], q[4] - q[3])
    reach = float(np.linalg.norm(q[4] - q[5]))
    f.thumb_reach = reach
    f.thumb_straight = ramp(t_bend, 75.0, 35.0)
    # "out" = far from the index knuckle AND either beside the hand (past the
    # index side) or high above the knuckles (thumbs up). A thumb lying across
    # the palm is never "out", however far it reaches.
    lateral = ramp(float(local[5, 0] - local[4, 0]), 0.12, 0.35)
    f.thumb_out = ramp(reach, 0.45, 0.7) * max(lateral, ramp(local[4, 1], 1.15, 1.35))
    ext[0] = f.thumb_out
    span = local[17, 0] - local[5, 0]
    f.thumb_tx = float((local[4, 0] - local[5, 0]) / span) if abs(span) > 1e-6 else 0.0
    f.thumb_ty = float(local[4, 1])

    # palm side = the side the folded fingers' middle knuckles stick out to
    side = sum((local[FINGERS[i][1], 2] - local[FINGERS[i][0], 2]) * fold[i] for i in (1, 2, 3, 4))
    sign = 1.0 if side > 0.05 else (-1.0 if side < -0.05 else 0.0)
    f.thumb_depth = sign * float(local[4, 2] - (local[6, 2] + local[10, 2]) / 2.0)

    tips_y = [local[TIPS[i], 1] for i in (1, 2, 3)]
    f.tips_over_thumb = float(np.mean(tips_y) - local[4, 1])

    touch, gap = [0.0], [0.0]
    for i in (1, 2, 3, 4):
        g = float(np.linalg.norm(q[4] - q[TIPS[i]]))
        gap.append(g)
        touch.append(ramp(g, 0.42, 0.22))
    f.tips_to_thumb = float(np.mean(gap[1:]))

    # spread is measured in the palm plane, so it survives curled (claw) fingers
    dirs = {i: local[TIPS[i], :2] - local[FINGERS[i][0], :2] for i in (1, 2, 3, 4)}
    f.spread = [_angle(dirs[1], dirs[2]), _angle(dirs[2], dirs[3]), _angle(dirs[3], dirs[4])]
    f.crossed = ramp(float(local[8, 0] - local[12, 0]), -0.02, 0.06)
    f.thumb_index_angle = _angle(q[4] - q[2], q[8] - q[5])

    f.index_dir = Direction.of(q[8, :2] - q[5, :2], float(np.linalg.norm(q[8] - q[5])))
    f.hand_dir = Direction.of(q[9, :2], float(np.linalg.norm(q[9])))
    f.thumb_dir = Direction.of(q[4, :2] - q[2, :2], float(np.linalg.norm(q[4] - q[2])))

    f.ext, f.fold, f.hook, f.curve = ext, fold, hook, curve
    f.bend, f.mcp, f.touch, f.gap = bend_l, mcp_l, touch, gap
    return f
