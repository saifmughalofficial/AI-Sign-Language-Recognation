"""Library of hand shapes recognised WITHOUT training (rule-based, from MediaPipe landmarks).

Each shape is a list of soft conditions (0..1). They are combined with a weighted
geometric mean, so ONE clearly wrong finger pulls the score down a lot, while
small wobbles only cost a little. The score doubles as the confidence shown in
the app.

data/gestures.json decides which shape means what in each mode, e.g.

    {"gesture": "WATER", "category": "word",     "shape": "W"}
    {"gesture": "6",     "category": "number",   "shape": "W"}
    {"gesture": "NEED WATER", "category": "sentence", "shape": "W", "output": "I need water."}

so you can re-map gestures without touching this file.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

from vision.hand_shape import HandFeatures, analyze, band, ramp



def gm(*conds) -> float:
    """Weighted geometric mean. Each condition is a value or (value, weight)."""
    total = weights = 0.0
    for c in conds:
        value, weight = c if isinstance(c, tuple) else (c, 1.0)
        total += weight * math.log(max(float(value), 0.02))
        weights += weight
    return math.exp(total / weights) if weights else 0.0


def _folded(h: HandFeatures, *fingers: int) -> list:
    return [h.fold[i] for i in fingers]


def _straight(h: HandFeatures, *fingers: int) -> list:
    return [h.ext[i] for i in fingers]


def _bent_not_folded(h: HandFeatures, i: int) -> float:
    return max(h.curve[i], h.hook[i]) * (1.0 - h.ext[i])


def _pip_bent(h: HandFeatures, i: int) -> float:
    """Bent at the middle joints while the knuckle stays fairly straight (E)."""
    return ramp(h.bend[i], 75.0, 115.0) * ramp(h.mcp[i], 75.0, 45.0)


def _thumb_along_index(h: HandFeatures) -> float:
    """Thumb straight and parallel to the index finger, on its outer side (G, Q)."""
    return gm(h.thumb_straight, ramp(h.thumb_tx, 0.2, -0.05), ramp(h.thumb_index_angle, 55, 25))


# ============================================================ one-hand shapes
def _fist(h):                       # all four fingers folded
    return _folded(h, 1, 2, 3, 4)


def shape_A(h):
    return gm(*_fist(h), ramp(h.thumb_tx, 0.25, 0.0), (ramp(h.thumb_reach, 0.85, 0.6), 0.8),
              (ramp(h.tips_over_thumb, 0.1, -0.05), 0.5))


def shape_B(h):
    return gm(*_straight(h, 1, 2, 3, 4), h.together, h.thumb_in, ramp(h.thumb_tx, -0.15, 0.1),
              (h.hand_dir.up, 0.5))


def shape_C(h):
    return gm(*[_bent_not_folded(h, i) for i in (1, 2, 3, 4)], 1.0 - h.touch[1],
              ramp(h.gap[1], 0.3, 0.45), ramp(h.gap[1], 1.3, 1.0), (1.0 - h.fold[1], 0.7),
              (h.together, 0.7), (h.thumb_in, 0.6))


def shape_D(h):
    rest = [max(h.fold[i], h.curve[i]) for i in (2, 3, 4)]
    return gm((h.ext[1], 1.2), *rest, (h.touch[2], 1.2), (1.0 - h.thumb_out, 0.5))


def shape_E(h):
    rest = [_pip_bent(h, i) for i in (1, 2, 3, 4)]
    return gm(*rest, ramp(h.tips_over_thumb, -0.02, 0.12), ramp(h.thumb_tx, -0.1, 0.15),
              h.thumb_in, ramp(h.tips_to_thumb, 0.65, 0.4), (1.0 - h.ext[1], 0.5))


def shape_F(h):
    return gm((h.touch[1], 1.5), *_straight(h, 2, 3, 4), 1.0 - h.ext[1])


def shape_G(h):
    return gm(h.ext[1], *_folded(h, 2, 3, 4), h.index_dir.side, (_thumb_along_index(h), 1.2))


def shape_H(h):
    return gm(*_straight(h, 1, 2), ramp(h.spread[0], 20, 10), h.index_dir.side,
              *_folded(h, 3, 4), (1.0 - h.crossed, 0.5))


def shape_I(h):
    return gm(h.ext[4], *_folded(h, 1, 2, 3), (h.thumb_in, 0.7))


def _thumb_between_index_middle(h):
    return band(h.thumb_tx, -0.05, 0.45, 0.15) * ramp(h.thumb_ty, 0.75, 1.0)


def shape_K(h):
    return gm(*_straight(h, 1, 2), *_folded(h, 3, 4), (_thumb_between_index_middle(h), 1.3),
              (h.index_dir.up, 0.6), (1.0 - h.crossed, 0.5))


def shape_L(h):
    return gm(h.ext[1], h.thumb_out, *_folded(h, 2, 3, 4), ramp(h.thumb_index_angle, 35, 60),
              (h.index_dir.up, 0.5))


def _tucked(h, lo, hi):
    return gm(*_fist(h), (band(h.thumb_tx, lo, hi, 0.12), 1.2), (ramp(h.thumb_depth, 0.08, -0.04), 0.7),
              (h.thumb_in, 0.7))


def shape_M(h):
    return _tucked(h, 0.58, 0.95)


def shape_N(h):
    return _tucked(h, 0.3, 0.58)


def shape_T(h):
    near_index = ramp(float(np.linalg.norm(h.pts[4] - h.pts[6])), 0.5, 0.28)
    return gm(*_fist(h), (band(h.thumb_tx, -0.12, 0.3, 0.12), 1.2), near_index,
              (ramp(h.thumb_depth, 0.08, -0.04), 0.5))


def shape_O(h):
    rest = [max(h.curve[i], 0.7 * h.hook[i]) * (1.0 - h.ext[i]) for i in (1, 2, 3, 4)]
    return gm(*rest, (h.touch[1], 1.5), ramp(h.tips_to_thumb, 0.7, 0.4))


def shape_P(h):
    pointing = max(h.index_dir.down, 0.8 * h.index_dir.forward)
    return gm(h.ext[1], max(h.ext[2], h.curve[2]), *_folded(h, 3, 4), (pointing, 1.3),
              (_thumb_between_index_middle(h), 0.4))


def shape_Q(h):
    return gm(h.ext[1], *_folded(h, 2, 3, 4), (h.index_dir.down, 1.3), (_thumb_along_index(h), 1.2))


def shape_R(h):
    return gm(*_straight(h, 1, 2), (h.crossed, 1.5), *_folded(h, 3, 4))


def shape_S(h):
    return gm(*_fist(h), band(h.thumb_tx, 0.1, 0.7, 0.15), (ramp(h.thumb_depth, -0.12, 0.02), 0.8),
              ramp(h.tips_over_thumb, 0.1, -0.04), (ramp(h.thumb_reach, 0.9, 0.65), 0.6))


def shape_U(h):
    return gm(*_straight(h, 1, 2), *_folded(h, 3, 4), ramp(h.spread[0], 14, 7),
              1.0 - h.crossed, (h.index_dir.up, 0.8), (h.thumb_in, 0.7),
              (1.0 - _thumb_between_index_middle(h), 0.8))


def shape_V(h):
    return gm(*_straight(h, 1, 2), *_folded(h, 3, 4), ramp(h.spread[0], 9, 17),
              1.0 - h.crossed, (h.index_dir.up, 0.6), (h.thumb_in, 0.7),
              (1.0 - _thumb_between_index_middle(h), 0.8))


def shape_W(h):
    return gm(*_straight(h, 1, 2, 3), h.fold[4], (h.thumb_in, 0.7))


def shape_X(h):
    return gm((h.hook[1], 1.5), 1.0 - h.ext[1], *_folded(h, 2, 3, 4), (h.thumb_in, 0.5),
              (1.0 - min(_pip_bent(h, 2), _pip_bent(h, 3)), 1.2))


def shape_Y(h):
    return gm(h.thumb_out, h.ext[4], *_folded(h, 1, 2, 3))


# ---- numbers ------------------------------------------------------------------
def shape_ONE(h):
    return gm(h.ext[1], *_folded(h, 2, 3, 4), h.thumb_in, (h.index_dir.up, 0.6), (1.0 - h.touch[2], 0.4))


def shape_TWO(h):
    return gm(*_straight(h, 1, 2), *_folded(h, 3, 4), h.thumb_in)


def shape_THREE(h):
    return gm(h.thumb_out, *_straight(h, 1, 2), *_folded(h, 3, 4))


def shape_FOUR(h):
    return gm(*_straight(h, 1, 2, 3, 4), h.thumb_in)


def shape_FIVE(h):
    return gm(h.thumb_out, *_straight(h, 1, 2, 3, 4), (h.spread_out, 0.5))


def shape_SEVEN(h):
    return gm(*_straight(h, 1, 2, 4), (h.touch[3], 1.3), 1.0 - h.ext[3])


def shape_EIGHT(h):
    return gm(*_straight(h, 1, 3, 4), (h.touch[2], 1.3), 1.0 - h.ext[2])


# ---- words / everyday gestures ---------------------------------------------------
def shape_OPEN(h):                  # ✋ open hand, fingers spread (hello)
    return gm(h.thumb_out, *_straight(h, 1, 2, 3, 4), h.spread_out)


def shape_PALM(h):                  # ✋ flat palm, fingers together, upright (stop)
    return gm(*_straight(h, 1, 2, 3, 4), h.together, (h.hand_dir.up, 1.2), (h.thumb_in, 0.8))


def shape_FLAT_THUMB(h):            # flat hand, fingers together, thumb out (thank you)
    return gm(*_straight(h, 1, 2, 3, 4), h.together, (h.thumb_out, 1.2), (1.0 - h.spread_out, 0.6))


def shape_FLAT_SIDE(h):             # flat hand lying sideways, like a hand on the chest (my)
    return gm(*_straight(h, 1, 2, 3, 4), h.together, (h.hand_dir.side, 1.3))


def shape_THUMBS_UP(h):
    return gm(*_fist(h), h.thumb_straight, (h.thumb_dir.up, 1.3), (h.thumb_out, 1.2))


def shape_THUMBS_DOWN(h):
    return gm(*_fist(h), h.thumb_straight, (h.thumb_dir.down, 1.3), (h.thumb_out, 1.2))


def shape_ILY(h):                   # 🤟 thumb + index + pinky
    return gm(h.thumb_out, h.ext[1], h.ext[4], *_folded(h, 2, 3))


def shape_CLAW(h):                  # all fingers bent like claws (want)
    return gm(*[_bent_not_folded(h, i) for i in (1, 2, 3, 4)], (1.0 - h.touch[1], 0.8),
              (ramp(h.gap[1], 0.45, 0.7), 0.8), (h.spread_out, 0.8), (h.thumb_out, 0.6))


def shape_PINCH2(h):                # index + middle tips touch the thumb (no)
    return gm(h.touch[1], h.touch[2], *_folded(h, 3, 4), (1.0 - h.fold[1], 0.6))


def shape_POINT_SIDE(h):            # 👉 index points sideways / at the other person (you)
    return gm(h.ext[1], *_folded(h, 2, 3, 4), max(h.index_dir.side, h.index_dir.forward))


ONE_HAND: dict[str, Callable[[HandFeatures], float]] = {
    name[len("shape_"):]: fn for name, fn in globals().items() if name.startswith("shape_")
}


# ============================================================ two-hand shapes
def _hand_distance(a: HandFeatures, b: HandFeatures, ia: int = 0, ib: int = 0) -> float:
    pa = a.origin + a.pts[ia] * a.palm
    pb = b.origin + b.pts[ib] * b.palm
    return float(np.linalg.norm(pa[:2] - pb[:2])) / ((a.palm + b.palm) / 2.0)


def _open4(h):
    return gm(*_straight(h, 1, 2, 3, 4))


def two_WAIT(a, b):                 # 🙌 both hands open
    return gm(_open4(a), _open4(b), ramp(_hand_distance(a, b), 1.0, 1.6))


def two_PRAY(a, b):                 # 🙏 flat hands pressed together (please)
    return gm(_open4(a), _open4(b), (ramp(_hand_distance(a, b, 12, 12), 0.8, 0.4), 1.3),
              ramp(_hand_distance(a, b), 1.5, 0.9))


def two_HELP(a, b):                 # 👍 on an open palm
    def one_way(fist, flat):
        return gm(max(shape_THUMBS_UP(fist), shape_A(fist)), _open4(flat),
                  (ramp(_hand_distance(fist, flat), 3.5, 2.0), 0.5))
    return max(one_way(a, b), one_way(b, a))


TWO_HAND: dict[str, Callable[[HandFeatures, HandFeatures], float]] = {
    name[len("two_"):] + "2": fn for name, fn in globals().items() if name.startswith("two_")
}
# -> "WAIT2", "PRAY2", "HELP2"

ALL_SHAPES = set(ONE_HAND) | set(TWO_HAND)

# Short "how to" used by the tips ribbon and the sentence guide.
SHAPE_HINTS = {
    "OPEN": "✋ open hand, fingers spread", "PALM": "✋ flat palm facing camera, fingers together",
    "FLAT_THUMB": "flat hand, fingers together, thumb out", "FLAT_SIDE": "flat hand held sideways",
    "THUMBS_UP": "👍 thumbs up", "THUMBS_DOWN": "👎 thumbs down", "ILY": "🤟 thumb + index + pinky",
    "CLAW": "fingers bent like claws", "PINCH2": "index + middle tips touch the thumb",
    "POINT_SIDE": "👉 point sideways / at the person", "ONE": "☝ index finger up",
    "TWO": "✌ index + middle up", "THREE": "thumb + index + middle", "FOUR": "four fingers up, thumb in",
    "FIVE": "✋ all five fingers", "SEVEN": "thumb touches ring finger", "EIGHT": "thumb touches middle finger",
    "W": "three middle fingers up (W)", "H": "index + middle together, pointing sideways",
    "O": "fingertips touch the thumb (O / flat-O)", "Y": "🤙 thumb + pinky out",
    "S": "✊ fist, thumb across the front", "A": "fist, thumb along the side", "C": "hand curved like a C",
    "X": "index finger hooked", "I": "pinky up", "WAIT2": "🙌 both hands open",
    "PRAY2": "🙏 both flat hands pressed together", "HELP2": "👍 thumbs up resting on the other open palm",
}


# ================================================================== scoring
def analyze_hands(hands: Sequence) -> list[HandFeatures]:
    """Features for the (up to) two most confident hands."""
    best = sorted(hands, key=lambda h: -float(getattr(h, "score", 1.0)))[:2]
    return [analyze(h.landmarks, float(getattr(h, "aspect", 1.0) or 1.0)) for h in best]


TWO_HAND_VISIBLE_PENALTY = 0.88   # a 2-hand sign beats the same 1-hand shape


def score_shapes(feats: list[HandFeatures], wanted) -> dict[str, float]:
    """{shape: score 0..1} for every wanted shape that can be scored."""
    scores: dict[str, float] = {}
    two = len(feats) >= 2
    for name in wanted:
        if name in TWO_HAND:
            if two:
                scores[name] = TWO_HAND[name](feats[0], feats[1])
            continue
        fn = ONE_HAND.get(name)
        if fn is None or not feats:
            continue
        value = max(fn(f) for f in feats)
        scores[name] = value * (TWO_HAND_VISIBLE_PENALTY if two else 1.0)
    return scores
