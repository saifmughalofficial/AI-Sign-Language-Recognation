"""Help Center (with pictures) and OFFLINE HELP (plain text, works even if pictures fail).

Both windows read only local JSON files from ./data - no internet is ever used.
"""
from __future__ import annotations

import tkinter as tk

from ui.gesture_dictionary import GestureDictionary
from ui.theme import FlatButton, font

FALLBACK_HELP = """OFFLINE HELP (built-in copy)

1. Press START CAMERA (or C). Show your hand about an arm's length from the camera.
2. Hold a gesture steady until the bar fills - it is then added to the sentence.
3. Press SPEAK (or S) to hear the sentence. VOICE MODE (V) shows what the other person says.
4. Keyboard: C camera, S speak, H help, V voice, R reset sentence, Esc close.
5. If something fails: check the camera is not used by another app, improve the lighting,
   and read the Troubleshooting section in README.md.

The full help files (data/help.json) could not be loaded, so this short guide is shown instead.
"""

# Sections that show the gesture dictionary instead of text
GESTURE_SECTIONS = {"alphabet": ("Alphabet", "alphabet"), "numbers": ("Numbers", "number"),
                    "words": ("Words", "word"), "sentences": ("Sentences", "sentence")}


class HelpWindow(tk.Toplevel):
    """Help Center: navigation on the left, content on the right."""

    def __init__(self, parent, ctx, theme: dict, section: str | None = None):
        super().__init__(parent)
        self.ctx = ctx
        self.theme = theme
        self.title("Help Center — works offline")
        self.geometry("1080x720")
        self.minsize(900, 600)
        self.configure(bg=theme["bg"])
        self.bind("<Escape>", lambda e: self.destroy())
        self.transient(parent)

        header = tk.Frame(self, bg=theme["bg"])
        header.pack(fill="x", padx=16, pady=(14, 6))
        tk.Label(header, text="Help Center", bg=theme["bg"], fg=theme["text"],
                 font=font(22, "bold")).pack(side="left")
        tk.Label(header, text="  📴 Works without internet", bg=theme["bg"], fg=theme["ok"],
                 font=font(11, "bold")).pack(side="left")
        FlatButton(header, theme, "CLOSE (Esc)", self.destroy).pack(side="right")

        body = tk.Frame(self, bg=theme["bg"])
        body.pack(fill="both", expand=True, padx=16, pady=(0, 14))
        nav = tk.Frame(body, bg=theme["card"], width=230)
        nav.pack(side="left", fill="y", padx=(0, 12))
        nav.pack_propagate(False)
        self.content = tk.Frame(body, bg=theme["bg"])
        self.content.pack(side="left", fill="both", expand=True)

        # search box that searches ALL gestures
        tk.Label(nav, text="Search all gestures", bg=theme["card"], fg=theme["muted"],
                 font=font(9, "bold")).pack(anchor="w", padx=12, pady=(12, 2))
        self._suppress = False
        self.search_var = tk.StringVar()
        entry = tk.Entry(nav, textvariable=self.search_var, font=font(12), bg=theme["entry_bg"],
                         fg=theme["text"], insertbackground=theme["text"], relief="flat",
                         highlightthickness=2, highlightbackground=theme["border"],
                         highlightcolor=theme["accent"])
        entry.pack(fill="x", padx=12, pady=(0, 10), ipady=4)
        self.search_var.trace_add("write", lambda *_: self._on_search())

        self.sections: list[tuple[str, str]] = [
            ("alphabet", "Alphabet"), ("numbers", "Numbers"), ("words", "Words"),
            ("sentences", "Sentences (1 gesture)"), ("phrases", "Phrases")]
        for sec in ctx.registry.help_sections():
            self.sections.append((sec["id"], sec["title"]))
        self.sections.append(("support", "What is supported?"))
        self.sections.append(("shortcuts", "Keyboard shortcuts"))
        self.nav_buttons: dict[str, tk.Button] = {}
        for sec_id, title in self.sections:
            btn = tk.Button(nav, text=title, anchor="w", font=font(11, "bold"), relief="flat", bd=0,
                            padx=12, pady=6, cursor="hand2", command=lambda s=sec_id: self.show(s))
            btn.pack(fill="x", padx=6, pady=1)
            self.nav_buttons[sec_id] = btn
        self.current = ""
        self.show(section if section in self.nav_buttons else "alphabet")
        self.after(50, entry.focus_set)

    # ------------------------------------------------------------------ helpers
    def _clear(self) -> None:
        for child in self.content.winfo_children():
            child.destroy()

    def _on_search(self) -> None:
        if self._suppress:
            return
        text = self.search_var.get().strip()
        if not text:
            self.show(self.current or "alphabet")
            return
        self._clear()
        self._paint_nav("")
        tk.Label(self.content, text=f"Search results for “{text}”", bg=self.theme["bg"],
                 fg=self.theme["text"], font=font(16, "bold")).pack(anchor="w", pady=(0, 6))
        GestureDictionary(self.content, self.theme, self.ctx.registry, None, text,
                          show_search=False).pack(fill="both", expand=True)

    def _paint_nav(self, active: str) -> None:
        t = self.theme
        for sec_id, btn in self.nav_buttons.items():
            if sec_id == active:
                btn.configure(bg=t["accent"], fg=t["accent_text"], activebackground=t["accent"])
            else:
                btn.configure(bg=t["card"], fg=t["text"], activebackground=t["card_alt"])

    def show(self, section: str) -> None:
        self.current = section
        if self.search_var.get():
            self._suppress = True
            self.search_var.set("")
            self._suppress = False
        self._clear()
        self._paint_nav(section)
        t = self.theme
        title = dict(self.sections).get(section, section)
        tk.Label(self.content, text=title, bg=t["bg"], fg=t["text"],
                 font=font(18, "bold")).pack(anchor="w", pady=(0, 6))
        if section in GESTURE_SECTIONS:
            _, category = GESTURE_SECTIONS[section]
            GestureDictionary(self.content, t, self.ctx.registry, category).pack(fill="both", expand=True)
        elif section == "phrases":
            self._text(self._phrases_text())
        elif section == "support":
            self._text(self._support_text())
        elif section == "shortcuts":
            rows = self.ctx.registry.help.get("shortcuts", [])
            self._text("\n".join(f"  {r['key']:<6}  {r['action']}" for r in rows) or "No shortcuts listed.")
        else:
            data = next((s for s in self.ctx.registry.help_sections() if s["id"] == section), None)
            self._text("\n\n".join(data["body"]) if data else "This section is not available.")

    def _text(self, content: str) -> None:
        t = self.theme
        frame = tk.Frame(self.content, bg=t["card"], highlightthickness=1, highlightbackground=t["border"])
        frame.pack(fill="both", expand=True)
        text = tk.Text(frame, wrap="word", font=font(13), bg=t["card"], fg=t["text"], relief="flat",
                       padx=18, pady=14, spacing1=2, spacing3=4, cursor="arrow")
        scroll = tk.Scrollbar(frame, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        text.pack(side="left", fill="both", expand=True)
        text.insert("1.0", content)
        text.configure(state="disabled")

    def _phrases_text(self) -> str:
        lines = ["Ready-made sentences and the signs that build them:\n"]
        for item in self.ctx.registry.phrase_library():
            lines.append(f"“{item['phrase']}”")
            lines.append("    Signs:  " + "  +  ".join(item.get("signs", [])))
            if item.get("note"):
                lines.append(f"    Note:  {item['note']}")
            lines.append("")
        return "\n".join(lines)

    def _support_text(self) -> str:
        levels = self.ctx.registry.help.get("support_levels", {})
        out = ["This is a learning / demo prototype. It does NOT understand arbitrary ASL.\n"]
        for key, title in (("supported", "SUPPORTED"), ("experimental", "EXPERIMENTAL / CUSTOM"),
                           ("unsupported", "UNSUPPORTED")):
            out.append(f"{title}\n{levels.get(key, '')}\n")
        return "\n".join(out)


class OfflineHelpWindow(tk.Toplevel):
    """Plain-text guide + gesture list. Needs NO images and NO Pillow features."""

    def __init__(self, parent, ctx, theme: dict):
        super().__init__(parent)
        self.ctx = ctx
        self.theme = theme
        self.title("Offline Help")
        self.geometry("760x640")
        self.configure(bg=theme["bg"])
        self.bind("<Escape>", lambda e: self.destroy())
        self.transient(parent)
        top = tk.Frame(self, bg=theme["bg"])
        top.pack(fill="x", padx=14, pady=(12, 4))
        tk.Label(top, text="📴 Offline Help", bg=theme["bg"], fg=theme["text"],
                 font=font(18, "bold")).pack(side="left")
        FlatButton(top, theme, "CLOSE (Esc)", self.destroy).pack(side="right")
        self.var = tk.StringVar()
        entry = tk.Entry(self, textvariable=self.var, font=font(13), bg=theme["entry_bg"],
                         fg=theme["text"], insertbackground=theme["text"], relief="flat",
                         highlightthickness=2, highlightbackground=theme["border"],
                         highlightcolor=theme["accent"])
        entry.pack(fill="x", padx=14, pady=6, ipady=5)
        self.var.trace_add("write", lambda *_: self._fill())
        frame = tk.Frame(self, bg=theme["card"])
        frame.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        self.text = tk.Text(frame, wrap="word", font=("Consolas", 12), bg=theme["card"], fg=theme["text"],
                            relief="flat", padx=14, pady=10)
        scroll = tk.Scrollbar(frame, command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.text.pack(fill="both", expand=True)
        self._fill()
        entry.focus_set()

    def _guide_text(self) -> str:
        sections = self.ctx.registry.help_sections()
        if not sections:
            return FALLBACK_HELP
        parts = ["OFFLINE HELP — everything here is stored on this computer.\n"]
        for sec in sections:
            parts.append(f"## {sec['title']}\n" + "\n".join(sec.get("body", [])) + "\n")
        return "\n".join(parts)

    def _gesture_lines(self, query: str) -> list[str]:
        lines = []
        for g in self.ctx.registry.search(query):
            hands = "2 hands" if g.get("hands", 1) == 2 else "1 hand"
            lines.append(f"{g['gesture']:<10} [{g['category']}, {hands}, {g.get('status', '')}]")
            lines.append(f"    {g.get('instructions', '')}")
            if g.get("sentence_examples"):
                lines.append(f"    Example: \"{g['sentence_examples'][0]}\"")
        return lines

    def _fill(self) -> None:
        query = self.var.get().strip()
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        if query:
            lines = self._gesture_lines(query)
            self.text.insert("1.0", "\n".join(lines) if lines else f"No gesture found for “{query}”.")
        else:
            self.text.insert("1.0", self._guide_text() + "\n## GESTURE LIST\n" +
                             "\n".join(self._gesture_lines("")))
        self.text.configure(state="disabled")
