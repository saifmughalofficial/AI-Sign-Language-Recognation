"""Animated drawings on top of the camera preview (Pillow only, redrawn every frame).

* a pulsing "show your hand here" frame while no hand is visible
* a label pill + circular hold-progress ring above each detected hand
* floating "+ WATER" toasts that rise and fade when a gesture is accepted
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

from ui.theme import ease_out_cubic, pulse

_FONT_FILES = ["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"]
_font_cache: dict[int, ImageFont.ImageFont] = {}


def _font(size: int):
    if size not in _font_cache:
        for name in _FONT_FILES:
            try:
                _font_cache[size] = ImageFont.truetype(name, size)
                break
            except OSError:
                continue
        else:
            try:
                _font_cache[size] = ImageFont.load_default(size)
            except TypeError:
                _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def _rgb(hex_color: str, alpha: int = 255) -> tuple:
    return tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5)) + (alpha,)


@dataclass
class Toast:
    text: str
    color: str
    born: float
    x: float = 0.5        # fraction of the picture
    y: float = 0.4


class VideoOverlay:
    TOAST_SECONDS = 1.3

    def __init__(self, theme: dict):
        self.theme = theme
        self.toasts: list[Toast] = []

    def add_toast(self, text: str, color: str, box: tuple | None = None) -> None:
        x, y = 0.5, 0.35
        if box:
            x, y = (box[0] + box[2]) / 2, max(0.08, box[1] - 0.04)
        self.toasts.append(Toast(text, color, time.monotonic(), x, y))
        self.toasts = self.toasts[-4:]

    # ---------------------------------------------------------------- drawing
    def draw(self, img: Image.Image, boxes: list, label: str | None, progress: float,
             locked: bool, searching: bool) -> Image.Image:
        base = img.convert("RGBA")
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        W, H = base.size
        if searching and not boxes:
            self._guide(d, W, H)
        if label and boxes:                 # one label over all hands (two-hand signs too)
            union = (min(b[0] for b in boxes), min(b[1] for b in boxes),
                     max(b[2] for b in boxes), max(b[3] for b in boxes))
            self._hand_label(d, W, H, union, label, progress, locked)
        self._draw_toasts(d, W, H)
        return Image.alpha_composite(base, layer).convert("RGB")

    def _guide(self, d: ImageDraw.ImageDraw, W: int, H: int) -> None:
        k = pulse(1.6)
        w, h = int(W * 0.40), int(H * 0.62)
        x1, y1 = (W - w) // 2, int(H * 0.16)
        x2, y2 = x1 + w, y1 + h
        color = _rgb(self.theme["accent"], int(110 + 120 * k))
        arm = int(min(w, h) * (0.16 + 0.05 * k))
        for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            d.line([(cx, cy), (cx + dx * arm, cy)], fill=color, width=4)
            d.line([(cx, cy), (cx, cy + dy * arm)], fill=color, width=4)
        text = "✋  Show your hand here"
        f = _font(max(14, W // 34))
        tw = d.textlength(text, font=f)
        ty = y2 + 10
        d.rounded_rectangle((W / 2 - tw / 2 - 14, ty, W / 2 + tw / 2 + 14, ty + f.size + 14),
                            radius=12, fill=(0, 0, 0, 150))
        d.text((W / 2 - tw / 2, ty + 6), text, font=f, fill=(255, 255, 255, int(170 + 85 * k)))

    def _hand_label(self, d, W, H, box, label, progress, locked) -> None:
        x1, y1, x2, _ = box[0] * W, box[1] * H, box[2] * W, box[3] * H
        f = _font(max(16, W // 26))
        text = label if len(label) <= 22 else label[:21] + "…"
        tw = d.textlength(text, font=f)
        ring = f.size + 10
        pill_w = tw + 28 + ring
        cx = (x1 + x2) / 2
        px1 = max(4, min(W - pill_w - 4, cx - pill_w / 2))
        py1 = max(4, y1 - f.size - 34)
        py2 = py1 + f.size + 18
        fill = self.theme["ok"] if locked else self.theme["accent"]
        d.rounded_rectangle((px1, py1, px1 + pill_w, py2), radius=(py2 - py1) // 2,
                            fill=_rgb(fill, 225))
        d.text((px1 + 14, py1 + 7), text, font=f, fill=(10, 18, 32, 255))
        # hold ring
        rx1 = px1 + pill_w - ring - 6
        ry1 = py1 + (py2 - py1 - ring) / 2
        bbox = (rx1, ry1, rx1 + ring, ry1 + ring)
        d.ellipse(bbox, outline=(10, 18, 32, 90), width=4)
        if progress > 0.01:
            d.arc(bbox, -90, -90 + 360 * min(progress, 1.0), fill=(10, 18, 32, 255), width=4)
        if locked:
            d.text((rx1 + ring * 0.24, ry1 - 1), "✓", font=_font(int(ring * 0.8)), fill=(10, 18, 32, 255))

    def _draw_toasts(self, d, W, H) -> None:
        now = time.monotonic()
        self.toasts = [t for t in self.toasts if now - t.born < self.TOAST_SECONDS]
        for t in self.toasts:
            age = (now - t.born) / self.TOAST_SECONDS
            rise = ease_out_cubic(min(1.0, age * 1.4))
            alpha = int(255 * (1.0 - max(0.0, age - 0.45) / 0.55))
            f = _font(max(18, int(W / 20 * (1.0 + 0.25 * (1 - rise)))))
            tw = d.textlength(t.text, font=f)
            x = t.x * W - tw / 2
            y = t.y * H - 70 * rise - f.size
            d.rounded_rectangle((x - 14, y - 6, x + tw + 14, y + f.size + 12), radius=14,
                                fill=_rgb(t.color, int(alpha * 0.9)))
            d.text((x, y), t.text, font=f, fill=(10, 18, 32, alpha))
