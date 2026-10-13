"""Shared test helpers: build fake (synthetic) hands without a camera."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision.hand_detector import DetectedHand  # noqa: E402


def make_hand_points(pattern: str, scale: float = 0.2, origin=(0.5, 0.7)) -> np.ndarray:
    """21 fake landmarks for a hand pointing up.

    pattern = five characters (thumb, index, middle, ring, pinky);
    '1' = finger extended, '0' = finger folded.
    """
    k = scale / 0.2
    pts = np.zeros((21, 3), dtype=np.float32)
    wx, wy = origin
    pts[0] = [wx, wy, 0]
    for m, dx in {5: -0.06, 9: -0.02, 13: 0.02, 17: 0.06}.items():
        pts[m] = [wx + dx * k, wy - 0.5 * scale, 0]
    pts[1] = [wx - 0.03 * k, wy - 0.10 * k, 0]
    pts[2] = [wx - 0.06 * k, wy - 0.17 * k, 0]
    if pattern[0] == "1":
        pts[3] = [wx - 0.11 * k, wy - 0.22 * k, 0]
        pts[4] = [wx - 0.16 * k, wy - 0.26 * k, 0]
    else:
        pts[3] = [wx - 0.055 * k, wy - 0.11 * k, 0]
        pts[4] = [wx - 0.045 * k, wy - 0.08 * k, 0]
    fingers = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
    for i, (mcp, pip, dip, tip) in enumerate(fingers):
        m = pts[mcp]
        if pattern[i + 1] == "1":
            pts[pip] = [m[0], m[1] - 0.06 * k, 0]
            pts[dip] = [m[0], m[1] - 0.11 * k, 0]
            pts[tip] = [m[0], m[1] - 0.16 * k, 0]
        else:
            pts[pip] = [m[0], m[1] - 0.05 * k, 0]
            pts[dip] = [m[0], m[1] - 0.01 * k, 0]
            pts[tip] = [m[0], m[1] + 0.03 * k, 0]
    return pts


def make_hand(pattern: str, handedness: str = "Right", score: float = 0.99, **kw) -> DetectedHand:
    return DetectedHand(make_hand_points(pattern, **kw), handedness, score)
