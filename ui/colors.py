"""Colour palettes shared by the desktop app and the web edition.

No tkinter here: servers such as Streamlit Community Cloud have no Tk.
"""

THEMES = {
    "dark": {
        "bg": "#0B1220", "card": "#172033", "card_alt": "#22304A", "border": "#3B4C6B",
        "text": "#F8FAFC", "muted": "#B6C2D6", "accent": "#38BDF8", "accent_text": "#06121F",
        "ok": "#4ADE80", "warn": "#FBBF24", "danger": "#F87171", "danger_text": "#1F0A0A",
        "video_bg": "#000000", "entry_bg": "#0F182A",
    },
    "light": {
        "bg": "#EEF2F7", "card": "#FFFFFF", "card_alt": "#E3EAF3", "border": "#B8C4D6",
        "text": "#0B1220", "muted": "#3F4E66", "accent": "#0369A1", "accent_text": "#FFFFFF",
        "ok": "#15803D", "warn": "#B45309", "danger": "#B91C1C", "danger_text": "#FFFFFF",
        "video_bg": "#111827", "entry_bg": "#FFFFFF",
    },
}
