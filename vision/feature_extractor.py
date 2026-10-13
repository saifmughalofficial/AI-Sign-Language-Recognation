"""Turns raw hand landmarks into numbers a classifier can learn from.

MediaPipe gives 21 landmarks per hand, each with x, y, z. We

1. move the wrist to (0, 0, 0)          -> position in the picture no longer matters
2. divide by the hand size              -> distance from the camera no longer matters
3. mirror left hands (by MediaPipe's label) so every hand looks like the same hand
4. add the distances between fingertips -> helps tell similar shapes apart

IMPORTANT: training images and live camera frames must both be given to
MediaPipe *un-flipped*, so its left/right label means the same thing in both.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from core.constants import SEQUENCE_LENGTH

TIPS = (4, 8, 12, 16, 20)
FINGER_TIPS = (8, 12, 16, 20)
FINGER_PIPS = (6, 10, 14, 18)
HAND_FEATURE_LEN = 63 + 10        # 21*3 coordinates + 10 tip-to-tip distances


def normalize_landmarks(points: np.ndarray, mirror: bool = False) -> np.ndarray:
    """Wrist-centred, size-normalised copy of a (21, 3) array."""
    pts = np.asarray(points, dtype=np.float32).reshape(21, 3).copy()
    pts -= pts[0]
    scale = float(np.max(np.linalg.norm(pts[:, :2], axis=1)))
    if scale < 1e-6:
        scale = 1.0
    pts /= scale
    if mirror:
        pts[:, 0] = -pts[:, 0]
    return pts


def hand_features(points: np.ndarray, handedness: str = "Right",
                  mirror_left: bool = True) -> np.ndarray:
    norm = normalize_landmarks(points, mirror=mirror_left and handedness == "Left")
    tips = norm[list(TIPS)]
    dists = [float(np.linalg.norm(tips[i] - tips[j]))
             for i in range(5) for j in range(i + 1, 5)]
    return np.concatenate([norm.reshape(-1), np.asarray(dists, dtype=np.float32)]).astype(np.float32)


def _sorted_hands(hands: Sequence) -> list:
    return sorted(hands, key=lambda h: float(h.landmarks[0][0]))


def extract_features(hands: Sequence, two_hand: bool = False) -> np.ndarray | None:
    """Feature vector for the hands in one frame (None if there are no hands).

    one-hand models  -> 73 numbers (the most confident hand)
    two-hand models  -> 146 numbers (left-to-right order, zero-padded)
    """
    if not hands:
        return None
    if not two_hand:
        best = max(hands, key=lambda h: getattr(h, "score", 1.0))
        return hand_features(best.landmarks, getattr(best, "handedness", "Right"))
    ordered = _sorted_hands(hands)[:2]
    feats = [hand_features(h.landmarks, getattr(h, "handedness", "Right")) for h in ordered]
    while len(feats) < 2:
        feats.append(np.zeros(HAND_FEATURE_LEN, dtype=np.float32))
    return np.concatenate(feats)


# ------------------------------------------------------------- finger states
def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def finger_states(points: np.ndarray) -> tuple[bool, bool, bool, bool, bool]:
    """(thumb, index, middle, ring, pinky) -> True when the finger is extended.

    Uses only distances, so it works for any hand rotation. This powers the
    Demo Mode and the motion-gesture pose checks - it is NOT a trained model.
    """
    pts = normalize_landmarks(points)
    palm = max(_dist(pts[0], pts[9]), 1e-6)
    states = []
    # thumb: tip far from the base of the index finger
    states.append(_dist(pts[4], pts[5]) > 0.6 * palm and _dist(pts[4], pts[17]) > _dist(pts[3], pts[17]))
    for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
        states.append(_dist(pts[tip], pts[0]) > 1.08 * _dist(pts[pip], pts[0]))
    return tuple(states)  # type: ignore[return-value]


def pose_string(states: Sequence[bool]) -> str:
    return "".join("1" if s else "0" for s in states)


# ------------------------------------------------------------------ sequences
def resample_indices(n_frames: int, length: int = SEQUENCE_LENGTH) -> np.ndarray:
    return np.linspace(0, n_frames - 1, length).round().astype(int)


def sequence_features(frames: Sequence[np.ndarray], handedness: str = "Right",
                      length: int = SEQUENCE_LENGTH) -> np.ndarray | None:
    """Features for a MOTION gesture: a fixed number of frames, each with the hand
    shape plus how far the wrist has moved since the first frame."""
    if len(frames) < 4:
        return None
    frames = [np.asarray(f, dtype=np.float32).reshape(21, 3) for f in frames]
    idx = resample_indices(len(frames), length)
    origin = frames[0][0][:2]
    size = max(_dist(frames[0][0], frames[0][9]), 1e-3)
    parts = []
    for i in idx:
        shape = normalize_landmarks(frames[i], mirror=(handedness == "Left")).reshape(-1)
        shift = (frames[i][0][:2] - origin) / size
        if handedness == "Left":
            shift = shift * np.array([-1.0, 1.0], dtype=np.float32)
        parts.append(np.concatenate([shape, shift]))
    return np.concatenate(parts).astype(np.float32)
