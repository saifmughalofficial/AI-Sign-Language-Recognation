"""Settings dialog. Values are validated by core.settings before they are saved."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox

from core.settings import LANGUAGES, RESOLUTIONS, THEMES, Settings, settings_from_dict
from ui.theme import FlatButton, font


class SettingsWindow(tk.Toplevel):
    def __init__(self, parent, ctx, theme: dict, on_saved):
        super().__init__(parent)
        self.ctx = ctx
        self.theme = theme
        self.on_saved = on_saved
        self.title("Settings")
        self.geometry("640x760")
        self.minsize(560, 560)
        self.configure(bg=theme["bg"])
        self.transient(parent)
        self.bind("<Escape>", lambda e: self.destroy())
        self.vars: dict[str, tk.Variable] = {}
        self._voices: list[tuple[str, str]] = []

        tk.Label(self, text="Settings", bg=theme["bg"], fg=theme["text"],
                 font=font(20, "bold")).pack(anchor="w", padx=18, pady=(14, 4))

        # scrollable form
        holder = tk.Frame(self, bg=theme["bg"])
        holder.pack(fill="both", expand=True, padx=12)
        canvas = tk.Canvas(holder, bg=theme["bg"], highlightthickness=0)
        scroll = ttk.Scrollbar(holder, orient="vertical", command=canvas.yview)
        self.form = tk.Frame(canvas, bg=theme["bg"])
        win = canvas.create_window((0, 0), window=self.form, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.form.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>",
                    lambda ev: canvas.yview_scroll(int(-ev.delta / 120), "units")))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        s = ctx.settings
        self._group("Camera")
        self._row("Camera number", self._combo("camera_index", [str(i) for i in range(4)], str(s.camera_index)),
                  "0 = built-in webcam. Restart camera after changing.")
        self._row("Resolution", self._combo("resolution", RESOLUTIONS, s.resolution),
                  "Lower = faster on slow laptops.")
        self._check("mirror_camera", "Mirror preview (like a selfie camera)", s.mirror_camera)
        self._check("show_landmarks", "Show hand landmarks on the preview", s.show_landmarks)
        self._row("Process every N frames", self._scale("process_every_n_frames", 1, 6, 1, s.process_every_n_frames),
                  "Higher = lighter on the CPU.")

        self._group("Recognition")
        self._row("Detection confidence", self._scale("detection_confidence", 0.1, 0.95, 0.05, s.detection_confidence),
                  "How sure MediaPipe must be that a hand is there.")
        self._row("Tracking confidence", self._scale("tracking_confidence", 0.1, 0.95, 0.05, s.tracking_confidence),
                  "How sure it must be to keep following the hand.")
        self._row("Recognition confidence", self._scale("recognition_confidence", 0.1, 0.95, 0.05, s.recognition_confidence),
                  "Minimum score before a gesture counts.")
        self._row("Gesture stabilization (sec)", self._scale("gesture_hold_seconds", 0.2, 3.0, 0.1, s.gesture_hold_seconds),
                  "How long a gesture must stay steady to be accepted.")
        self._row("Repeat delay (sec)", self._scale("release_seconds", 0.1, 2.0, 0.05, s.release_seconds),
                  "Hand must change/leave this long before the same gesture repeats.")
        self._check("experimental_dynamic", "Experimental: motion letters J and Z", s.experimental_dynamic)

        self._group("Speech")
        self._check("speech_enabled", "Speech enabled", s.speech_enabled)
        self._check("speak_letters", "Speak individual letters", s.speak_letters)
        self._check("speak_words", "Speak completed words", s.speak_words)
        self._check("speak_sentences", "Speak completed sentences", s.speak_sentences)
        self._check("speak_voice_text", "Read Person B's voice text aloud", s.speak_voice_text)
        self.voice_combo = self._combo("voice_id", ["(default voice)"], "(default voice)", key_only=False)
        self._row("Voice", self.voice_combo, "Installed system voices.")
        self._row("Speed", self._scale("speech_rate", 80, 300, 10, s.speech_rate), "Words per minute.")
        self._row("Volume", self._scale("speech_volume", 0.0, 1.0, 0.1, s.speech_volume), "")
        FlatButton(self.form, theme, "🔊 TEST VOICE", self._test_voice, size=10).pack(anchor="w", padx=18, pady=4)
        self.after(100, self._load_voices)

        self._group("Interface")
        self._row("Theme", self._combo("theme", THEMES, s.theme), "Dark or light (applied on save).")
        self._row("Language", self._combo("language", LANGUAGES, s.language), "More languages: see README.")
        self._check("auto_clear_sentence", "Clear the sentence after SPEAK", s.auto_clear_sentence)
        self._check("save_history", "Save conversation history on this computer", s.save_history)

        bar = tk.Frame(self, bg=theme["bg"])
        bar.pack(fill="x", padx=16, pady=12)
        FlatButton(bar, theme, "SAVE", self._save, "primary").pack(side="right", padx=(8, 0))
        FlatButton(bar, theme, "CANCEL (Esc)", self.destroy).pack(side="right", padx=(8, 0))
        FlatButton(bar, theme, "RESET TO DEFAULTS", self._reset, size=10).pack(side="left")

    # ------------------------------------------------------------- form helpers
    def _group(self, title: str) -> None:
        tk.Label(self.form, text=title.upper(), bg=self.theme["bg"], fg=self.theme["accent"],
                 font=font(11, "bold")).pack(anchor="w", padx=8, pady=(14, 4))

    def _row(self, label: str, widget: tk.Widget, hint: str) -> None:
        row = tk.Frame(self.form, bg=self.theme["card"], highlightthickness=1,
                       highlightbackground=self.theme["border"])
        row.pack(fill="x", padx=6, pady=3)
        left = tk.Frame(row, bg=self.theme["card"])
        left.pack(side="left", fill="x", expand=True, padx=12, pady=6)
        tk.Label(left, text=label, bg=self.theme["card"], fg=self.theme["text"],
                 font=font(11, "bold")).pack(anchor="w")
        if hint:
            tk.Label(left, text=hint, bg=self.theme["card"], fg=self.theme["muted"],
                     font=font(9)).pack(anchor="w")
        widget.pack(in_=row, side="right", padx=12, pady=6)

    def _combo(self, name: str, values: list[str], current: str, key_only: bool = True) -> ttk.Combobox:
        var = tk.StringVar(value=current)
        self.vars[name] = var
        return ttk.Combobox(self.form, textvariable=var, values=values, state="readonly", width=18,
                            font=font(11))

    def _scale(self, name: str, lo: float, hi: float, step: float, current: float) -> tk.Scale:
        var = tk.DoubleVar(value=current)
        self.vars[name] = var
        return tk.Scale(self.form, variable=var, from_=lo, to=hi, resolution=step, orient="horizontal",
                        length=200, bg=self.theme["card"], fg=self.theme["text"], troughcolor=self.theme["card_alt"],
                        highlightthickness=0, font=font(10), activebackground=self.theme["accent"])

    def _check(self, name: str, text: str, current: bool) -> None:
        var = tk.BooleanVar(value=current)
        self.vars[name] = var
        tk.Checkbutton(self.form, text=text, variable=var, bg=self.theme["bg"], fg=self.theme["text"],
                       selectcolor=self.theme["card_alt"], activebackground=self.theme["bg"],
                       activeforeground=self.theme["text"], font=font(11), anchor="w"
                       ).pack(fill="x", padx=14, pady=2)

    # ----------------------------------------------------------------- actions
    def _load_voices(self) -> None:
        self._voices = self.ctx.tts.list_voices()
        names = ["(default voice)"] + [name for _, name in self._voices]
        self.voice_combo.configure(values=names)
        current = self.ctx.settings.voice_id
        match = next((n for i, n in self._voices if i == current), "(default voice)")
        self.vars["voice_id"].set(match)
        if not self._voices and not self.ctx.tts.available:
            self.voice_combo.configure(values=["(speech unavailable)"])
            self.vars["voice_id"].set("(speech unavailable)")

    def _collect(self) -> dict:
        data = dict(self.ctx.settings.to_dict())
        for name, var in self.vars.items():
            value = var.get()
            if name == "camera_index":
                value = int(value)
            elif name == "voice_id":
                value = next((i for i, n in self._voices if n == value), "")
            elif name in ("process_every_n_frames", "speech_rate"):
                value = int(float(value))
            data[name] = value
        return data

    def _test_voice(self) -> None:
        if not self.ctx.tts.available:
            messagebox.showinfo("Speech", f"Text-to-speech is unavailable: {self.ctx.tts.error or 'unknown reason'}",
                                parent=self)
            return
        data = self._collect()
        self.ctx.tts.configure(int(data["speech_rate"]), float(data["speech_volume"]), data["voice_id"])
        self.ctx.tts.speak("Hello. This is how I will sound.")

    def _reset(self) -> None:
        defaults = Settings().to_dict()
        for name, var in self.vars.items():
            if name == "voice_id":
                var.set("(default voice)")
            else:
                var.set(defaults[name] if not isinstance(var, tk.StringVar) else str(defaults[name]))

    def _save(self) -> None:
        try:
            new, notes = settings_from_dict(self._collect())
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Settings", f"Some values are invalid: {exc}", parent=self)
            return
        old = self.ctx.settings
        new.user_name = old.user_name
        new.mode = old.mode
        if notes:
            messagebox.showinfo("Settings adjusted", "\n".join(notes), parent=self)
        self.ctx.settings = new
        if not self.ctx.store.save(new):
            messagebox.showwarning("Settings", "\n".join(self.ctx.store.notes), parent=self)
        self.on_saved(old, new)
        self.destroy()
