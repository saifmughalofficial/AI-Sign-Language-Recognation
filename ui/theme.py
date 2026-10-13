"""Colours, fonts and small reusable widgets (rounded cards, buttons, meters).

Everything visual lives here so the look of the whole app can be changed in one
place. Two palettes are provided: dark and light. Both are high contrast.
"""
from __future__ import annotations

import math
import time
import tkinter as tk
from tkinter import ttk

from ui.colors import THEMES  # noqa: F401 - palettes live in a tkinter-free module

FONT_FAMILY = "Segoe UI"     # Windows default; other systems fall back automatically


def font(size: int, weight: str = "normal") -> tuple:
    return (FONT_FAMILY, size, weight)


# Icons are plain text so no image files are needed. Change them here.
ICONS = {
    "start": "▶", "stop": "■", "pause": "‖", "resume": "▶", "clear": "✕", "speak": "🔊",
    "repeat": "↻", "voice": "🎤", "help": "?", "settings": "⚙", "lock": "🔒",
    "dot": "●", "offline": "📴", "space": "␣", "delete": "⌫", "guide": "📖", "spark": "✨",
}


def get_theme(name: str) -> dict:
    return THEMES.get(name, THEMES["dark"])


# ================================================================ animation
def mix(c1: str, c2: str, t: float) -> str:
    """Blend two #RRGGBB colours (t = 0 -> c1, t = 1 -> c2)."""
    t = max(0.0, min(1.0, t))
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(a, b))


def ease_out_cubic(t: float) -> float:
    return 1 - (1 - t) ** 3


def ease_out_back(t: float) -> float:
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


def pulse(period: float = 1.2) -> float:
    """0..1..0 wave that follows the clock (for blinking / breathing effects)."""
    return 0.5 + 0.5 * math.sin(2 * math.pi * time.monotonic() / period)


def animate(widget: tk.Misc, key: str, duration_ms: int, step, done=None, ease=ease_out_cubic) -> None:
    """Call step(progress 0..1) about 60x per second for `duration_ms`.

    Starting a new animation with the same `key` on the same widget cancels the
    old one, so rapid updates never pile up.
    """
    tokens = widget.__dict__.setdefault("_anim_tokens", {})
    token = object()
    tokens[key] = token
    start = time.perf_counter()

    def tick():
        try:
            if tokens.get(key) is not token or not widget.winfo_exists():
                return
            t = min(1.0, (time.perf_counter() - start) * 1000.0 / max(duration_ms, 1))
            step(ease(t))
            if t < 1.0:
                widget.after(16, tick)
            elif done:
                done()
        except tk.TclError:                     # widget destroyed mid-animation
            pass

    tick()


class RoundedCard(tk.Canvas):
    """A rounded rectangle you can put widgets into via `.body`."""

    def __init__(self, parent, theme: dict, radius: int = 18, padding: int = 14,
                 fill: str | None = None, **kw):
        super().__init__(parent, bg=theme["bg"], highlightthickness=0, bd=0,
                         width=kw.pop("width", 10), height=kw.pop("height", 10), **kw)
        self.theme = theme
        self.radius = radius
        self.padding = padding
        self.fill = fill or theme["card"]
        self.body = tk.Frame(self, bg=self.fill)
        self._window = self.create_window(padding, padding, anchor="nw", window=self.body)
        self.bind("<Configure>", self._on_resize)
        self._outline = theme["border"]
        self._outline_width = 1

    def flash(self, color: str, duration_ms: int = 700) -> None:
        """Briefly light the border up in `color`, then fade back."""
        def step(t):
            self._outline = mix(color, self.theme["border"], t)
            self._outline_width = 3 if t < 0.6 else 1
            self.itemconfigure("bg", outline=self._outline, width=self._outline_width)
        animate(self, "flash", duration_ms, step, ease=lambda t: t)

    def _round_rect(self, x1, y1, x2, y2, r, **kw):
        pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
               x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
        return self.create_polygon(pts, smooth=True, **kw)

    def _on_resize(self, event) -> None:
        self.delete("bg")
        self._round_rect(2, 2, max(event.width - 2, 10), max(event.height - 2, 10), self.radius,
                         fill=self.fill, outline=self._outline, width=self._outline_width, tags="bg")
        self.tag_lower("bg")
        self.itemconfigure(self._window, width=max(event.width - 2 * self.padding, 10),
                           height=max(event.height - 2 * self.padding, 10))


class FlatButton(tk.Button):
    """Large, flat, high-contrast button with hover feedback."""

    VARIANTS = ("primary", "secondary", "danger", "success")

    def __init__(self, parent, theme: dict, text: str, command=None, variant: str = "secondary",
                 size: int = 11, **kw):
        self.theme = theme
        self.variant = variant
        bg, fg = self._colors(variant)
        super().__init__(parent, text=text, command=command, bg=bg, fg=fg, activebackground=bg,
                         activeforeground=fg, relief="flat", bd=0, cursor="hand2",
                         font=font(size, "bold"), padx=kw.pop("padx", 10), pady=kw.pop("pady", 8),
                         highlightthickness=2, highlightbackground=theme["border"],
                         highlightcolor=theme["accent"], takefocus=True, **kw)
        self._bg, self._fg = bg, fg
        self.bind("<Enter>", lambda e: self._hover(True))
        self.bind("<Leave>", lambda e: self._hover(False))

    def _colors(self, variant: str) -> tuple[str, str]:
        t = self.theme
        return {
            "primary": (t["accent"], t["accent_text"]),
            "danger": (t["danger"], t["danger_text"]),
            "success": (t["ok"], t["accent_text"] if t["ok"].startswith("#4") else "#FFFFFF"),
        }.get(variant, (t["card_alt"], t["text"]))

    def set_variant(self, variant: str, text: str | None = None) -> None:
        self.variant = variant
        self._bg, self._fg = self._colors(variant)
        self.configure(bg=self._bg, fg=self._fg, activebackground=self._bg, activeforeground=self._fg)
        if text is not None:
            self.configure(text=text)

    def _hover(self, inside: bool) -> None:
        if str(self["state"]) == "disabled":
            return
        self.configure(highlightbackground=self.theme["accent"] if inside else self.theme["border"])
        lit = mix(self._bg, "#FFFFFF", 0.14)
        start, end = (self._bg, lit) if inside else (lit, self._bg)
        animate(self, "hover", 140, lambda t: self.configure(bg=mix(start, end, t),
                                                              activebackground=mix(start, end, t)))


class Meter(tk.Canvas):
    """A thin, rounded progress bar that glides smoothly to each new value."""

    def __init__(self, parent, theme: dict, height: int = 12, bg: str | None = None):
        super().__init__(parent, height=height, highlightthickness=0, bd=0,
                         bg=bg or theme["card"])
        self.theme = theme
        self.value = 0.0            # target
        self.shown = 0.0            # what is drawn right now
        self.color = theme["accent"]
        self._running = False
        self.bind("<Configure>", lambda e: self._draw())

    def set(self, value: float, color: str | None = None, smooth: bool = True) -> None:
        self.value = max(0.0, min(1.0, value))
        if color:
            self.color = color
        if not smooth:
            self.shown = self.value
            self._draw()
        elif not self._running:
            self._running = True
            self._glide()

    def _glide(self) -> None:
        try:
            if not self.winfo_exists():
                return
            diff = self.value - self.shown
            if abs(diff) < 0.004:
                self.shown = self.value
                self._running = False
            else:
                self.shown += diff * 0.28
            self._draw()
            if self._running:
                self.after(16, self._glide)
        except tk.TclError:
            self._running = False

    def _draw(self) -> None:
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 4:
            return
        r = h / 2
        self._pill(0, 0, w, h, r, self.theme["card_alt"])
        if self.shown > 0.003:
            x2 = max(h, w * self.shown)
            self._pill(0, 0, x2, h, r, self.color)
            if self.shown >= 0.999:            # full: soft highlight stripe
                self._pill(2, 2, x2 - 2, max(3, h / 2 - 1), r / 2, mix(self.color, "#FFFFFF", 0.35))

    def _pill(self, x1, y1, x2, y2, r, color) -> None:
        if x2 - x1 < 2 * r:
            self.create_oval(x1, y1, x1 + 2 * r, y2, fill=color, outline="")
            return
        self.create_oval(x1, y1, x1 + 2 * r, y2, fill=color, outline="")
        self.create_oval(x2 - 2 * r, y1, x2, y2, fill=color, outline="")
        self.create_rectangle(x1 + r, y1, x2 - r, y2, fill=color, outline="")


class FingerMeter(tk.Canvas):
    """Five small live bars (thumb .. pinky): how far each finger is extended."""

    COLORS = ["#FF69B4", "#FFA03C", "#FADC3C", "#64E678", "#50D2FA"]
    NAMES = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]

    def __init__(self, parent, theme: dict, height: int = 40):
        super().__init__(parent, height=height, highlightthickness=0, bd=0, bg=theme["card"])
        self.theme = theme
        self.values = [0.0] * 5
        self.shown = [0.0] * 5
        self._running = False
        self.bind("<Configure>", lambda e: self._draw())

    def set(self, values) -> None:
        self.values = [max(0.0, min(1.0, float(v))) for v in list(values)[:5]] + [0.0] * (5 - len(values))
        if not self._running:
            self._running = True
            self._glide()

    def _glide(self) -> None:
        try:
            done = True
            for i in range(5):
                d = self.values[i] - self.shown[i]
                if abs(d) > 0.01:
                    self.shown[i] += d * 0.35
                    done = False
                else:
                    self.shown[i] = self.values[i]
            self._draw()
            if done:
                self._running = False
            else:
                self.after(16, self._glide)
        except tk.TclError:
            self._running = False

    def _draw(self) -> None:
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 20:
            return
        gap, label_h = 6, 12
        bw = (w - gap * 4) / 5
        for i in range(5):
            x1 = i * (bw + gap)
            top, bottom = 2, h - label_h - 2
            self.create_rectangle(x1, top, x1 + bw, bottom, fill=self.theme["card_alt"], outline="")
            fill_top = bottom - (bottom - top) * self.shown[i]
            if self.shown[i] > 0.02:
                self.create_rectangle(x1, fill_top, x1 + bw, bottom, fill=self.COLORS[i], outline="")
            self.create_text(x1 + bw / 2, h - label_h / 2, text=self.NAMES[i], fill=self.theme["muted"],
                             font=font(7, "bold"))


class Segmented(tk.Frame):
    """Row of toggle buttons where exactly one is selected (used for the mode)."""

    def __init__(self, parent, theme: dict, options: list[str], command, initial: str,
                 bg: str | None = None):
        super().__init__(parent, bg=bg or theme["card"])
        self.theme = theme
        self.command = command
        self.buttons: dict[str, tk.Button] = {}
        for i, option in enumerate(options):
            btn = tk.Button(self, text=option, font=font(10, "bold"), relief="flat", bd=0,
                            cursor="hand2", padx=8, pady=7, takefocus=True,
                            command=lambda o=option: self._pick(o))
            btn.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 4, 0))
            self.columnconfigure(i, weight=1, uniform="seg")
            self.buttons[option] = btn
        self.value = initial
        self._paint()

    def _pick(self, option: str) -> None:
        self.set(option)
        self.command(option)

    def set(self, option: str) -> None:
        old = self.value
        self.value = option
        self._paint()
        if old != option and option in self.buttons:      # selected button fades in
            t = self.theme
            btn = self.buttons[option]
            animate(self, "seg", 260, lambda k: btn.configure(
                bg=mix(t["card_alt"], t["accent"], k), activebackground=mix(t["card_alt"], t["accent"], k)))

    def _paint(self) -> None:
        t = self.theme
        for option, btn in self.buttons.items():
            if option == self.value:
                btn.configure(bg=t["accent"], fg=t["accent_text"], activebackground=t["accent"])
            else:
                btn.configure(bg=t["card_alt"], fg=t["text"], activebackground=t["card_alt"])


def style_ttk(root: tk.Misc, theme: dict) -> None:
    """Make combobox/scrollbar match the theme."""
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure("TCombobox", fieldbackground=theme["entry_bg"], background=theme["card_alt"],
                    foreground=theme["text"], arrowcolor=theme["text"], padding=4)
    style.map("TCombobox", fieldbackground=[("readonly", theme["entry_bg"])],
              foreground=[("readonly", theme["text"])])
    style.configure("Vertical.TScrollbar", background=theme["card_alt"], troughcolor=theme["card"],
                    arrowcolor=theme["text"])
    root.option_add("*TCombobox*Listbox.background", theme["entry_bg"])
    root.option_add("*TCombobox*Listbox.foreground", theme["text"])
    root.option_add("*TCombobox*Listbox.selectBackground", theme["accent"])
    root.option_add("*TCombobox*Listbox.selectForeground", theme["accent_text"])
