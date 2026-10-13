"""Draws a simple *schematic* picture of a hand shape with Pillow (no downloads).

These pictures show which fingers are up (tall), folded (short) or curved, the
direction the hand points and whether the sign uses movement or two hands.
They are illustrations, NOT photographs of official ASL signs. To use real
photos, drop a PNG named like the gesture (e.g. assets/gestures/WATER.png) and
the Help Center will show it instead.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from core.paths import GESTURE_IMG_DIR

_FONT_CANDIDATES = ["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf"]


def _font(size: int):
    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:                       # very old Pillow
        return ImageFont.load_default()


def draw_gesture(entry: dict, size: int = 240) -> Image.Image:
    """Return a size x size RGB picture for one gesture entry."""
    S = 480                                    # draw big, shrink at the end (smooth edges)
    bg, hand, outline, ink = "#E2E8F0", "#94A3B8", "#334155", "#0F172A"
    img = Image.new("RGB", (S, S), bg)
    hand_layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(hand_layer)

    fingers = str(entry.get("image_fingers") or "00000").ljust(5, "0")[:5]
    thumb, rest = fingers[0], fingers[1:]

    # palm
    palm = (150, 270, 340, 430)
    d.rounded_rectangle(palm, radius=38, fill=hand, outline=outline, width=6)
    # four fingers (index, middle, ring, pinky)
    xs = [162, 205, 248, 291]
    full = [170, 190, 170, 130]                 # finger lengths when extended
    for x, state, length in zip(xs, rest, full):
        if state == "1":                        # extended
            d.rounded_rectangle((x, 270 - length, x + 38, 290), radius=19, fill=hand, outline=outline, width=6)
        elif state == "2":                      # curved / bent
            d.rounded_rectangle((x, 270 - length // 2, x + 38, 290), radius=19, fill=hand, outline=outline, width=6)
            d.ellipse((x - 10, 270 - length // 2 - 8, x + 34, 270 - length // 2 + 30), fill=hand, outline=outline, width=6)
        else:                                   # folded: only a knuckle bump
            d.rounded_rectangle((x, 246, x + 38, 296), radius=19, fill=hand, outline=outline, width=6)
    # thumb
    if thumb == "1":
        d.line([(160, 385), (95, 315), (70, 255)], fill=outline, width=52, joint="curve")
        d.line([(160, 385), (95, 315), (70, 255)], fill=hand, width=40, joint="curve")
    elif thumb == "2":
        d.ellipse((120, 300, 215, 380), fill=hand, outline=outline, width=6)
    else:
        d.rounded_rectangle((165, 330, 270, 372), radius=21, fill=hand, outline=outline, width=6)

    orientation = entry.get("orientation", "up")
    angle = {"up": 0, "side": -90, "down": 180}.get(orientation, 0)
    hand_layer = hand_layer.rotate(angle, resample=Image.BICUBIC, center=(S // 2, S // 2))
    img.paste(hand_layer, (0, 0), hand_layer)

    d2 = ImageDraw.Draw(img)
    label = str(entry.get("gesture", "?"))
    d2.text((26, 14), label, fill=ink, font=_font(72 if len(label) <= 2 else 46))
    badges = []
    if entry.get("type") == "dynamic" or entry.get("asl_motion"):
        badges.append("MOTION")
    if entry.get("hands", 1) == 2:
        badges.append("2 HANDS")
    for i, text in enumerate(badges):
        w = 20 + 17 * len(text)
        y = 30 + i * 46
        d2.rounded_rectangle((S - w - 22, y, S - 22, y + 38), radius=12, fill=ink)
        d2.text((S - w - 12, y + 6), text, fill="#F8FAFC", font=_font(24))
    if entry.get("type") == "dynamic":                      # dashed motion arrow
        for i in range(6):
            d2.line([(380, 120 + i * 24), (380, 132 + i * 24)], fill="#EF4444", width=6)
        d2.polygon([(366, 262), (394, 262), (380, 290)], fill="#EF4444")
    d2.text((26, S - 34), "Schematic illustration", fill="#475569", font=_font(20))
    return img.resize((size, size), Image.LANCZOS)


def load_gesture_image(entry: dict, size: int = 160, folder: Path = GESTURE_IMG_DIR) -> Image.Image:
    """Use assets/gestures/<image> if it exists, otherwise draw one."""
    name = entry.get("image") or (str(entry.get("category", "x")) + "_" + str(entry.get("gesture", "x")).replace(" ", "_") + ".png")
    path = Path(folder) / name
    try:
        if path.exists():
            return Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    except OSError:
        pass
    return draw_gesture(entry, size)


def generate_all(registry, folder: Path = GESTURE_IMG_DIR, size: int = 240,
                 overwrite: bool = False) -> int:
    """Write a PNG for every gesture. Returns how many files were created."""
    folder.mkdir(parents=True, exist_ok=True)
    created = 0
    for entry in registry.gestures:
        name = entry.get("image") or (entry["category"] + "_" + entry["gesture"].replace(" ", "_") + ".png")
        path = folder / name
        if path.exists() and not overwrite:
            continue
        draw_gesture(entry, size).save(path)
        created += 1
    return created
