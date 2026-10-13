"""Animated hand-tracking graphics, drawn with OpenCV on every camera frame.

Quality levels (`HandFX.quality`):
  "max"       neon bloom glow, hologram fill + grid, depth-sized joints, fingertip sparks,
              pinch beam, HUD tag, orbiting ticks, motion trail, scan line, everything below
  "balanced"  soft glow, motion trail, scan line, lock-on brackets, hold ring, effects
  "fast"      plain skeleton, brackets, hold ring, effects (for slow PCs)

Always:
* smooth tracking   - landmarks glide towards each new detection (no jitter, no jumps)
* appear            - "target lock" brackets zoom in and a ripple leaves the palm
* holding a pose    - progress ring around the palm (+ label pill above the hand)
* accepted          - shockwave + particle burst + "+ WORD" toast rising from the hand
* no hand           - radar sweep with "Show your hand here"

Detection may run slower than the camera: `update()` is called whenever new
landmarks arrive, `draw()` on EVERY camera frame, so the picture stays fluid.
"""
from __future__ import annotations

import math
import random
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vision.hand_detector import FINGER_COLORS, HAND_CONNECTIONS, _bone_color

TIPS = (4, 8, 12, 16, 20)
PALM = (0, 5, 9, 13, 17)
QUALITIES = ("max", "balanced", "fast")


def bgr(hex_color: str) -> tuple[int, int, int]:
    return int(hex_color[5:7], 16), int(hex_color[3:5], 16), int(hex_color[1:3], 16)


def _ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


_FONTS = ["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"]


def _font(size: int):
    for name in _FONTS:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


@dataclass
class Track:
    pts: np.ndarray                    # (21, 2) smoothed pixel positions
    target: np.ndarray                 # latest detection
    born: float
    seen: float
    z: np.ndarray = field(default_factory=lambda: np.zeros(21, np.float32))
    side: str = "Right"
    score: float = 1.0
    trail: deque = field(default_factory=lambda: deque(maxlen=18))
    prev_tips: np.ndarray | None = None


class HandFX:
    def __init__(self, theme: dict | None = None, quality: str = "max"):
        self.set_theme(theme or {"accent": "#38BDF8", "ok": "#4ADE80", "warn": "#FBBF24"})
        self.quality = quality if quality in QUALITIES else "max"
        self.tracks: list[Track] = []
        self.ripples: list[tuple[float, float, float]] = []
        self.bursts: list[tuple[float, np.ndarray, np.ndarray, list]] = []
        self.waves: list[tuple[float, float, float]] = []
        self.toasts: list[tuple[float, str, float, float]] = []
        self.sparks: list[list] = []        # [t0, x, y, vx, vy, color]
        self._patches: dict = {}
        self._size = (640, 480)

    def set_theme(self, theme: dict) -> None:
        self.accent, self.ok = bgr(theme["accent"]), bgr(theme["ok"])
        self.warn = bgr(theme.get("warn", "#FBBF24"))
        self._patches = {}

    # ------------------------------------------------------------ new landmarks
    def update(self, hands, size: tuple[int, int], mirrored: bool) -> None:
        """hands: DetectedHand list (normalised, NOT mirrored). size = (W, H) of the drawn picture."""
        now = time.monotonic()
        W, H = size
        self._size = size
        tracks, free = [], list(self.tracks)
        for h in hands:
            xs = 1.0 - h.landmarks[:, 0] if mirrored else h.landmarks[:, 0]
            tgt = np.stack([xs * W, h.landmarks[:, 1] * H], axis=1).astype(np.float32)
            scale = float(np.linalg.norm(tgt[0] - tgt[9])) + 1.0
            best = min(free, key=lambda tr: float(np.linalg.norm(tr.pts[0] - tgt[0])), default=None)
            if best is not None and np.linalg.norm(best.pts[0] - tgt[0]) < 2.5 * scale:
                free.remove(best)
                tr = best
                tr.target, tr.seen = tgt, now
            else:                                           # a new hand appeared
                tr = Track(tgt.copy(), tgt, now, now)
                c = tgt[list(PALM)].mean(axis=0)
                self.ripples.append((now, float(c[0]), float(c[1])))
            tr.z = h.landmarks[:, 2].astype(np.float32)
            # MediaPipe names the hand as if the picture were mirrored
            side = getattr(h, "handedness", "Right")
            tr.side = side if mirrored else ("Left" if side == "Right" else "Right")
            tr.score = float(getattr(h, "score", 1.0))
            tracks.append(tr)
        tracks += [tr for tr in free if now - tr.seen < 0.3]   # short fade-out
        self.tracks = tracks

    def celebrate(self, text: str) -> None:
        """A gesture was accepted: shockwave, particles and a rising toast."""
        now = time.monotonic()
        if self.tracks:
            c = self.tracks[0].pts[list(PALM)].mean(axis=0)
            x, y = float(c[0]), float(c[1])
        else:
            x, y = self._size[0] / 2, self._size[1] / 2
        n = 44 if self.quality == "max" else 26
        ang = np.random.uniform(0, 2 * math.pi, n)
        spd = np.random.uniform(140, 380, n)
        vel = np.stack([np.cos(ang) * spd, np.sin(ang) * spd - 120], axis=1)
        cols = [random.choice(FINGER_COLORS + [self.ok]) for _ in range(n)]
        self.bursts.append((now, np.array([x, y], np.float32), vel.astype(np.float32), cols))
        self.waves.append((now, x, y))
        self.toasts.append((now, text, x, y))
        self.toasts = self.toasts[-3:]

    # -------------------------------------------------------------------- draw
    def draw(self, frame: np.ndarray, label: str | None = None, progress: float = 0.0,
             locked: bool = False, searching: bool = True) -> None:
        now = time.monotonic()
        H, W = frame.shape[:2]
        self._size = (W, H)
        for tr in self.tracks:                              # glide towards the detection
            diff = tr.target - tr.pts
            follow = min(0.85, 0.4 + float(np.abs(diff).mean()) / 35.0)
            tr.pts += diff * follow
            tr.trail.append(tr.pts[8].copy())
        live = [tr for tr in self.tracks if now - tr.seen < 0.12]
        if not live and searching:
            self._radar(frame, now)
        for tr in self.tracks:
            self._hand(frame, tr, now, label is not None, progress, locked)
        if live and label:
            self._label(frame, live, label, locked, now)
        self._effects(frame, now)

    # ---------------------------------------------------------------- the hand
    def _hand(self, frame, tr: Track, now, holding, progress, locked) -> None:
        H, W = frame.shape[:2]
        fade = 1.0 - min(1.0, max(0.0, (now - tr.seen - 0.1) / 0.2))
        if fade <= 0:
            return
        q = self.quality
        pts = tr.pts.astype(np.int32)
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        size = max(x2 - x1, y2 - y1, 1)
        pad = int(max(14, 0.12 * size))
        bx1, by1, bx2, by2 = max(x1 - pad, 0), max(y1 - pad, 0), min(x2 + pad, W - 1), min(y2 + pad, H - 1)
        if bx2 <= bx1 or by2 <= by1:
            return
        palm = tr.pts[list(PALM)].mean(axis=0)
        center = (int(palm[0]), int(palm[1]))
        ring_r = int(0.62 * size / 2 + 10)
        color = self.ok if locked else self.accent

        # ---- glow / hologram layers, limited to the area around the hand -------------
        if q != "fast":
            gx1, gy1 = max(bx1 - 40, 0), max(by1 - 40, 0)
            gx2, gy2 = min(bx2 + 40, W), min(by2 + 40, H)
            roi = frame[gy1:gy2, gx1:gx2]
            lp = pts - np.array([gx1, gy1])
            if q == "max":
                fill = roi.copy()                           # hologram: hull fill + grid
                cv2.fillConvexPoly(fill, cv2.convexHull(lp), color, cv2.LINE_AA)
                ox, oy = bx1 - gx1, by1 - gy1
                for gy in range(oy, by2 - gy1, 16):
                    cv2.line(fill, (ox, gy), (bx2 - gx1, gy), color, 1)
                for gx in range(ox, bx2 - gx1, 16):
                    cv2.line(fill, (gx, oy), (gx, by2 - gy1), color, 1)
                if holding and not locked:
                    sy = int(by1 + (by2 - by1) * ((now * 1.1) % 1.0)) - gy1
                    cv2.line(fill, (ox, sy), (bx2 - gx1, sy), (255, 255, 255), 3, cv2.LINE_AA)
                cv2.addWeighted(fill, 0.16 * fade, roi, 1 - 0.16 * fade, 0, dst=roi)
                bloom = np.zeros_like(roi)                  # neon bloom
                for a, b in HAND_CONNECTIONS:
                    cv2.line(bloom, tuple(lp[a]), tuple(lp[b]), _bone_color(a, b), 9, cv2.LINE_AA)
                for i in TIPS:
                    cv2.circle(bloom, tuple(lp[i]), 12, FINGER_COLORS[i // 4 - 1], -1, cv2.LINE_AA)
                bloom = cv2.GaussianBlur(bloom, (0, 0), 7)
                cv2.addWeighted(bloom, 0.9 * fade, roi, 1.0, 0, dst=roi)
            else:
                layer = roi.copy()
                for a, b in HAND_CONNECTIONS:
                    cv2.line(layer, tuple(lp[a]), tuple(lp[b]), _bone_color(a, b), 11, cv2.LINE_AA)
                if holding and not locked:
                    sy = int(by1 + (by2 - by1) * ((now * 1.1) % 1.0)) - gy1
                    cv2.line(layer, (bx1 - gx1, sy), (bx2 - gx1, sy), self.accent, 6, cv2.LINE_AA)
                cv2.addWeighted(layer, 0.30 * fade, roi, 1 - 0.30 * fade, 0, dst=roi)

            trail = list(tr.trail)                          # index-finger motion trail
            for i in range(1, len(trail)):
                t = i / len(trail)
                cv2.line(frame, tuple(trail[i - 1].astype(int)), tuple(trail[i].astype(int)),
                         FINGER_COLORS[1], max(1, int(5 * t)), cv2.LINE_AA)

        # ---- bones, depth-sized joints, pulsing fingertips ---------------------------------
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), _bone_color(a, b), 2, cv2.LINE_AA)
        depth = np.clip(-(tr.z - tr.z.mean()) * 60.0, -1.5, 3.5) if q == "max" else np.zeros(21)
        for i, p in enumerate(pts):
            if i in TIPS:
                pr = int(6 + depth[i] + 2.5 * (0.5 + 0.5 * math.sin(now * 7 + i)))
                cv2.circle(frame, tuple(p), pr, FINGER_COLORS[i // 4 - 1], 2, cv2.LINE_AA)
            cv2.circle(frame, tuple(p), max(2, int(3 + depth[i])), (255, 255, 255), -1, cv2.LINE_AA)

        if q == "max":
            self._sparks_from(tr, now, size)
            self._pinch(frame, tr, size, now)

        # ---- target-lock brackets (zoom in when the hand appears) ---------------------------
        k = _ease((now - tr.born) / 0.4)
        cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
        s = 1.0 + 0.7 * (1 - k)
        hw, hh = (bx2 - bx1) / 2 * s, (by2 - by1) / 2 * s
        X1, Y1, X2, Y2 = int(cx - hw), int(cy - hh), int(cx + hw), int(cy + hh)
        arm = max(12, int(min(X2 - X1, Y2 - Y1) / 5))
        th = 3 if k >= 1 else 2
        for (px, py), (dx, dy) in (((X1, Y1), (1, 1)), ((X2, Y1), (-1, 1)), ((X1, Y2), (1, -1)), ((X2, Y2), (-1, -1))):
            cv2.line(frame, (px, py), (px + dx * arm, py), color, th, cv2.LINE_AA)
            cv2.line(frame, (px, py), (px, py + dy * arm), color, th, cv2.LINE_AA)
        if q == "max" and k >= 1:                           # HUD tag on the box corner
            tag = self._patch(f"{tr.side.upper()} HAND  {int(round(tr.score * 20) * 5)}%", 13, (20, 24, 32), fg=color)
            self._blit(frame, tag, X1, Y2 + 6, 0.9 * fade)

        # ---- hold progress ring around the palm --------------------------------------------
        if holding:
            cv2.circle(frame, center, ring_r, (90, 90, 90), 1, cv2.LINE_AA)
            sweep = 360 * (1.0 if locked else min(progress, 1.0))
            if sweep > 1:
                width = 4 + (int(2 * math.sin(now * 10)) if locked else 0)
                cv2.ellipse(frame, center, (ring_r, ring_r), -90, 0, sweep, color, max(2, width), cv2.LINE_AA)
                end = math.radians(-90 + sweep)
                dot = (int(center[0] + ring_r * math.cos(end)), int(center[1] + ring_r * math.sin(end)))
                cv2.circle(frame, dot, 6, (255, 255, 255), -1, cv2.LINE_AA)
                if q == "max":                              # orbiting ticks
                    for j in range(3):
                        a = math.radians(now * 160 + j * 120)
                        o = (int(center[0] + (ring_r + 9) * math.cos(a)), int(center[1] + (ring_r + 9) * math.sin(a)))
                        cv2.circle(frame, o, 3, color, -1, cv2.LINE_AA)

    def _sparks_from(self, tr: Track, now: float, size: int) -> None:
        tips = tr.pts[list(TIPS)]
        if tr.prev_tips is not None and len(self.sparks) < 140:
            speed = np.linalg.norm(tips - tr.prev_tips, axis=1)
            for i in np.nonzero(speed > 0.04 * size)[0][:3]:
                vx, vy = (tips[i] - tr.prev_tips[i]) * 4 + np.random.uniform(-40, 40, 2)
                self.sparks.append([now, float(tips[i][0]), float(tips[i][1]), float(vx), float(vy),
                                    FINGER_COLORS[i]])
        tr.prev_tips = tips.copy()

    def _pinch(self, frame, tr: Track, size: int, now: float) -> None:
        a, b = tr.pts[4], tr.pts[8]
        d = float(np.linalg.norm(a - b))
        if d < 0.28 * size:
            k = 1 - d / (0.28 * size)
            pa, pb = tuple(a.astype(int)), tuple(b.astype(int))
            cv2.line(frame, pa, pb, self.warn, int(2 + 4 * k), cv2.LINE_AA)
            cv2.line(frame, pa, pb, (255, 255, 255), 1, cv2.LINE_AA)
            m = tuple(((a + b) / 2).astype(int))
            cv2.circle(frame, m, int(5 + 6 * k * (0.5 + 0.5 * math.sin(now * 14))), self.warn, 2, cv2.LINE_AA)

    # -------------------------------------------------------------- overlays
    def _label(self, frame, tracks, label, locked, now) -> None:
        pts = np.concatenate([tr.pts for tr in tracks])
        x1, y1 = pts.min(axis=0)
        x2 = pts.max(axis=0)[0]
        text = label if len(label) <= 26 else label[:25] + "…"
        patch = self._patch(text, 22, self.ok if locked else self.accent)
        ph, pw = patch.shape[:2]
        bob = int(3 * math.sin(now * 3))
        self._blit(frame, patch, int((x1 + x2) / 2 - pw / 2), int(y1 - ph - 30 + bob))

    def _radar(self, frame, now) -> None:
        H, W = frame.shape[:2]
        c = (W // 2, int(H * 0.47))
        R = int(min(W, H) * 0.24)
        x0, y0 = max(c[0] - R - 4, 0), max(c[1] - R - 4, 0)
        roi = frame[y0:min(c[1] + R + 4, H), x0:min(c[0] + R + 4, W)]
        layer = roi.copy()
        a = (now * 140) % 360
        cv2.ellipse(layer, (c[0] - x0, c[1] - y0), (R, R), 0, a - 50, a, self.accent, -1, cv2.LINE_AA)
        cv2.addWeighted(layer, 0.22, roi, 0.78, 0, dst=roi)
        for r in (R, int(R * 0.66), int(R * 0.33)):
            cv2.circle(frame, c, r, self.accent, 1, cv2.LINE_AA)
        end = (int(c[0] + R * math.cos(math.radians(a))), int(c[1] + R * math.sin(math.radians(a))))
        cv2.line(frame, c, end, self.accent, 2, cv2.LINE_AA)
        pulse = 0.5 + 0.5 * math.sin(now * 4)
        arm = int(R * (0.3 + 0.06 * pulse))
        for (px, py), (dx, dy) in (((c[0] - R, c[1] - R), (1, 1)), ((c[0] + R, c[1] - R), (-1, 1)),
                                   ((c[0] - R, c[1] + R), (1, -1)), ((c[0] + R, c[1] + R), (-1, -1))):
            cv2.line(frame, (px, py), (px + dx * arm, py), self.accent, 3, cv2.LINE_AA)
            cv2.line(frame, (px, py), (px, py + dy * arm), self.accent, 3, cv2.LINE_AA)
        patch = self._patch("Show your hand here", 18, (20, 20, 20), fg=(255, 255, 255))
        self._blit(frame, patch, c[0] - patch.shape[1] // 2, c[1] + R + 14, 0.65 + 0.35 * pulse)

    def _effects(self, frame, now) -> None:
        self.ripples = [r for r in self.ripples if now - r[0] < 0.8]
        for t0, x, y in self.ripples:
            t = (now - t0) / 0.8
            for k in (0.0, 0.25):
                tt = t - k
                if 0 < tt < 1:
                    cv2.circle(frame, (int(x), int(y)), int(15 + 110 * _ease(tt)), self.accent,
                               max(1, int(3 * (1 - tt))), cv2.LINE_AA)
        self.waves = [w for w in self.waves if now - w[0] < 0.6]
        for t0, x, y in self.waves:
            t = (now - t0) / 0.6
            cv2.circle(frame, (int(x), int(y)), int(25 + 170 * _ease(t)), self.ok, max(1, int(5 * (1 - t))), cv2.LINE_AA)
        self.bursts = [b for b in self.bursts if now - b[0] < 0.9]
        for t0, p0, vel, cols in self.bursts:
            t = now - t0
            pos = p0 + vel * t + np.array([0, 420], np.float32) * (t * t / 2)
            r = max(1, int(5 * (1 - t / 0.9)))
            for (px, py), col in zip(pos.astype(int), cols):
                cv2.circle(frame, (int(px), int(py)), r, col, -1, cv2.LINE_AA)
        self.sparks = [s for s in self.sparks if now - s[0] < 0.5]
        for t0, x, y, vx, vy, col in self.sparks:
            t = now - t0
            cv2.circle(frame, (int(x + vx * t), int(y + vy * t + 300 * t * t)), max(1, int(3 * (1 - t / 0.5))),
                       col, -1, cv2.LINE_AA)
        self.toasts = [x for x in self.toasts if now - x[0] < 1.3]
        for t0, text, x, y in self.toasts:
            t = (now - t0) / 1.3
            patch = self._patch(text, 26, self.ok)
            alpha = 1.0 if t < 0.5 else max(0.0, 1 - (t - 0.5) / 0.5)
            self._blit(frame, patch, int(x - patch.shape[1] / 2), int(y - 60 - 90 * _ease(t)), alpha)

    # ------------------------------------------------------------ text patches
    def _patch(self, text: str, size: int, bg, fg=(10, 18, 32)) -> np.ndarray:
        """Rounded 'pill' with text as a BGRA array (cached: drawing text is the slow part)."""
        key = (text, size, bg, fg)
        if key not in self._patches:
            if len(self._patches) > 120:
                self._patches.clear()
            f = _font(size)
            tw = int(ImageDraw.Draw(Image.new("RGBA", (1, 1))).textlength(text, font=f))
            w, h = tw + 30, size + 18
            img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            d.rounded_rectangle((0, 0, w - 1, h - 1), radius=h // 2, fill=(bg[2], bg[1], bg[0], 235))
            d.text((15, 6), text, font=f, fill=(fg[2], fg[1], fg[0], 255))
            arr = np.asarray(img)
            self._patches[key] = arr[:, :, [2, 1, 0, 3]].copy()
        return self._patches[key]

    @staticmethod
    def _blit(frame, patch, x, y, alpha: float = 1.0) -> None:
        H, W = frame.shape[:2]
        h, w = patch.shape[:2]
        x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
        if x2 <= x1 or y2 <= y1 or alpha <= 0:
            return
        p = patch[y1 - y:y2 - y, x1 - x:x2 - x]
        a = (p[:, :, 3:4].astype(np.float32) / 255.0) * alpha
        dst = frame[y1:y2, x1:x2]
        dst[:] = (p[:, :, :3] * a + dst * (1 - a)).astype(np.uint8)
