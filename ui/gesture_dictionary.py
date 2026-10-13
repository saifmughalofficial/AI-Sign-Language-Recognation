"""Scrollable, searchable visual gesture dictionary (used by the Help Center)."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from PIL import ImageTk

from ui.gesture_art import load_gesture_image
from ui.theme import font

STATUS_TEXT = {
    "supported": ("CAMERA ✓", "ok"),
    "experimental": ("EXPERIMENTAL", "warn"),
    "reference": ("TYPE-ONLY (not camera yet)", "muted"),
}


class GestureDictionary(tk.Frame):
    """Shows gesture cards for one category, or search results across all of them."""

    COLUMNS = 2

    def __init__(self, parent, theme: dict, registry, category: str | None = None,
                 query: str = "", show_search: bool = True):
        super().__init__(parent, bg=theme["bg"])
        self.theme = theme
        self.registry = registry
        self.category = category
        self._images: dict[str, ImageTk.PhotoImage] = {}
        self._search_var = tk.StringVar(value=query)

        if show_search:
            bar = tk.Frame(self, bg=theme["bg"])
            bar.pack(fill="x", pady=(0, 8))
            tk.Label(bar, text="🔍", bg=theme["bg"], fg=theme["muted"], font=font(14)).pack(side="left")
            entry = tk.Entry(bar, textvariable=self._search_var, font=font(13), bg=theme["entry_bg"],
                             fg=theme["text"], insertbackground=theme["text"], relief="flat",
                             highlightthickness=2, highlightbackground=theme["border"],
                             highlightcolor=theme["accent"])
            entry.pack(side="left", fill="x", expand=True, padx=8, ipady=5)
            self._search_var.trace_add("write", lambda *_: self.refresh())
            self.hint = tk.Label(bar, text="Search gestures… (e.g. WATER)", bg=theme["bg"],
                                 fg=theme["muted"], font=font(10))
            self.hint.pack(side="left")

        self.canvas = tk.Canvas(self, bg=theme["bg"], highlightthickness=0)
        scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas, bg=theme["bg"])
        self._window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=scroll.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._window, width=e.width))
        self.canvas.bind("<Enter>", lambda e: self._bind_wheel(True))
        self.canvas.bind("<Leave>", lambda e: self._bind_wheel(False))
        self.refresh()

    # -- mouse wheel only while the pointer is over the list
    def _bind_wheel(self, on: bool) -> None:
        if on:
            self.canvas.bind_all("<MouseWheel>", self._wheel)
        else:
            self.canvas.unbind_all("<MouseWheel>")

    def _wheel(self, event) -> None:
        self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def set_query(self, text: str) -> None:
        self._search_var.set(text)

    def refresh(self) -> None:
        for child in self.inner.winfo_children():
            child.destroy()
        query = self._search_var.get()
        items = self.registry.search(query, None if query.strip() else self.category)
        if not items:
            tk.Label(self.inner, text=f"No gesture found for “{query.strip()}”.\n"
                     "Try a letter, number, or word such as WATER.",
                     bg=self.theme["bg"], fg=self.theme["muted"], font=font(13), justify="left"
                     ).pack(padx=20, pady=30, anchor="w")
            return
        for c in range(self.COLUMNS):
            self.inner.columnconfigure(c, weight=1, uniform="cards")
        for i, entry in enumerate(items):
            self._card(entry).grid(row=i // self.COLUMNS, column=i % self.COLUMNS,
                                   sticky="nsew", padx=6, pady=6)
        self.canvas.yview_moveto(0)

    def _card(self, entry: dict) -> tk.Frame:
        t = self.theme
        card = tk.Frame(self.inner, bg=t["card"], highlightthickness=1, highlightbackground=t["border"])
        key = entry["gesture"] + entry["category"]
        if key not in self._images:
            self._images[key] = ImageTk.PhotoImage(load_gesture_image(entry, 150))
        tk.Label(card, image=self._images[key], bg=t["card"]).grid(row=0, column=0, rowspan=4,
                                                                    padx=10, pady=10, sticky="n")
        tk.Label(card, text=entry["gesture"], bg=t["card"], fg=t["text"],
                 font=font(20, "bold")).grid(row=0, column=1, sticky="w", pady=(10, 0))
        badge, tone = STATUS_TEXT.get(entry.get("status", ""), ("", "muted"))
        hands = "2 hands" if entry.get("hands", 1) == 2 else "1 hand"
        kind = "motion" if entry.get("type") == "dynamic" else "still pose"
        tk.Label(card, text=f"{hands} · {kind} · {badge}", bg=t["card"], fg=t[tone],
                 font=font(9, "bold")).grid(row=1, column=1, sticky="w")
        tk.Label(card, text=entry.get("instructions", ""), bg=t["card"], fg=t["text"], font=font(10),
                 wraplength=260, justify="left").grid(row=2, column=1, sticky="nw", pady=(4, 0))
        examples = entry.get("sentence_examples", [])
        if entry.get("output"):
            tk.Label(card, text=f"Says: “{entry['output']}”", bg=t["card"], fg=t["accent"],
                     font=font(11, "bold"), wraplength=260, justify="left"
                     ).grid(row=3, column=1, sticky="nw", pady=(4, 10))
        elif examples:
            tk.Label(card, text=f"Example: “{examples[0]}”", bg=t["card"], fg=t["muted"],
                     font=font(10, "italic"), wraplength=260, justify="left"
                     ).grid(row=3, column=1, sticky="nw", pady=(4, 10))
        card.columnconfigure(1, weight=1)
        return card
