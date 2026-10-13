"""The main window: camera, detection, sentence builder, two-way communication, history."""
from __future__ import annotations

import logging
import queue
import time
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

from core.constants import MODE_CATEGORIES, MODES, MSG_CAMERA_OFF, MSG_DEMO, PRIVACY_NOTICE
from sentence.sentence_builder import SentenceBuilder
from speech.speech_to_text import VOSK_SETUP_TEXT, VoiceListener, check_voice_support
from ui.help_window import HelpWindow, OfflineHelpWindow
from ui.overlay import VideoOverlay
from ui.settings_window import SettingsWindow
from ui.theme import (FingerMeter, FlatButton, ICONS, Meter, RoundedCard, Segmented, animate, ease_out_back,
                      font, get_theme, mix, pulse, style_ttk)
from vision.gesture_recognizer import Prediction
from vision.gesture_stabilizer import GestureStabilizer
from vision.shape_library import SHAPE_HINTS

log = logging.getLogger(__name__)

# Scripted example for RUN DEMO (works without a camera)
DEMO_STEPS = [
    ("mode", "WORDS"), ("clear", None),
    ("gesture", ("I", "word")), ("gesture", ("NEED", "word")), ("gesture", ("WATER", "word")),
    ("speak", None), ("pause", None), ("clear", None),
    ("mode", "SENTENCES"),
    ("gesture", ("GREETING", "sentence")), ("gesture", ("MY NAME IS", "sentence")),
    ("speak", None), ("pause", None), ("clear", None),
    ("mode", "ALPHABET"),
    ("gesture", ("A", "alphabet")), ("gesture", ("L", "alphabet")), ("gesture", ("I", "alphabet")),
    ("use_name", None), ("speak", None), ("done", None),
]

# Help Center section for each mode (GESTURE GUIDE button)
GUIDE_SECTIONS = {"ALPHABET": "alphabet", "NUMBERS": "numbers", "WORDS": "words",
                  "SENTENCES": "sentences", "AUTO": "words"}
GESTURE_LABEL_SIZE = 34


class Dashboard:
    def __init__(self, root: tk.Tk, ctx):
        self.root = root
        self.ctx = ctx
        s = ctx.settings
        self.t = get_theme(s.theme)
        self.mode = s.mode
        self.builder = SentenceBuilder(ctx.phrase_engine, lambda: self.ctx.settings.user_name)
        self.stabilizer = GestureStabilizer(s.gesture_hold_seconds, s.recognition_confidence,
                                            s.release_seconds)
        self.camera_on = False
        self.last_seq = -1
        self._photo = None
        self._camera_error_shown = False
        self._last_dynamic = 0.0
        self.demo_running = False
        self._demo_index = 0
        self.voice: VoiceListener | None = None
        self.voice_on = False
        self.voice_events: queue.Queue = queue.Queue()
        self._voice_help_shown = False
        self._dialogs: dict[str, tk.Toplevel] = {}
        self.person_b_text = ""
        self.overlay = VideoOverlay(self.t)
        self._last_boxes: list = []
        self._shown_label = ""
        self._sentence_shown = ""
        self._tip_index = 0
        self._tip_changed = 0.0
        self._idle_frame = 0

        root.title("Sign Language Assistant")
        root.geometry("1280x860")
        root.minsize(1100, 680)
        try:
            root.state("zoomed")               # use the whole screen on Windows
        except tk.TclError:
            pass
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._build_ui()
        self._bind_shortcuts()
        self._poll()

    # =================================================================== UI build
    def _build_ui(self) -> None:
        for child in self.root.winfo_children():
            if not isinstance(child, tk.Toplevel):
                child.destroy()
        self.t = get_theme(self.ctx.settings.theme)
        t = self.t
        self.overlay.theme = t
        style_ttk(self.root, t)
        self.root.configure(bg=t["bg"])
        self.root.columnconfigure(0, weight=1)
        for row, weight in enumerate((0, 0, 1, 0, 0)):        # camera area gets all spare room
            self.root.rowconfigure(row, weight=weight)
        self.compact = self.root.winfo_screenheight() < 820       # small laptop screen
        self.show_conversation = getattr(self, "show_conversation", not self.compact)
        self.root.rowconfigure(4, minsize=150 if self.show_conversation else 0)

        self._build_header()
        self._build_banner()
        main = tk.Frame(self.root, bg=t["bg"])
        main.grid(row=2, column=0, sticky="nsew", padx=14, pady=6)
        main.columnconfigure(0, weight=3, uniform="main")
        main.columnconfigure(1, weight=2, uniform="main")
        main.rowconfigure(0, weight=1)
        self._build_camera_card(main)
        right = tk.Frame(main, bg=t["bg"])
        right.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1, uniform="r")
        right.rowconfigure(1, weight=1, uniform="r")
        self._build_detection_card(right)
        self._build_sentence_card(right)
        self._build_controls()
        self._build_bottom()
        self._refresh_all()

    # ------------------------------------------------------------------ header
    def _build_header(self) -> None:
        t = self.t
        bar = tk.Frame(self.root, bg=t["bg"])
        bar.grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 4))
        tk.Label(bar, text="SIGN LANGUAGE ASSISTANT", bg=t["bg"], fg=t["text"],
                 font=font(20, "bold")).pack(side="left")
        tk.Label(bar, text=f"  {ICONS['lock']} Offline · Private", bg=t["bg"], fg=t["ok"],
                 font=font(11, "bold")).pack(side="left")
        FlatButton(bar, t, f"{ICONS['settings']} SETTINGS", self.open_settings, size=10).pack(side="right", padx=(8, 0))
        FlatButton(bar, t, f"{ICONS['help']}  HELP", self.open_help, "primary", size=12,
                   padx=18).pack(side="right", padx=(8, 0))
        FlatButton(bar, t, f"{ICONS['offline']} OFFLINE HELP", self.open_offline_help, size=10).pack(side="right")
        FlatButton(bar, t, f"{ICONS['guide']} GESTURE GUIDE", self.open_guide, "success", size=10).pack(
            side="right", padx=(0, 8))

    def _build_banner(self) -> None:
        t = self.t
        self.banner = tk.Frame(self.root, bg=t["card_alt"], highlightthickness=1,
                               highlightbackground=t["accent"])
        if self.ctx.recognizer.demo_mode:
            self.banner.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 4))
            self.banner_label = tk.Label(self.banner, text=f"{ICONS['spark']}  {MSG_DEMO}", bg=t["card_alt"],
                                         fg=t["accent"], font=font(11, "bold"))
            self.banner_label.pack(side="left", padx=12, pady=6)
            for text, cmd in (("TRAIN YOUR OWN MODEL (optional)", self.open_setup_help),
                              ("▶ RUN DEMO", self.run_demo)):
                FlatButton(self.banner, t, text, cmd, "primary" if "DEMO" in text else "secondary",
                           size=9, pady=3).pack(side="right", padx=6, pady=5)
            animate(self.banner_label, "intro", 900,
                    lambda k: self.banner_label.configure(fg=mix(t["card_alt"], t["accent"], k)))
            self.root.after(12000, self._collapse_banner)

    def _collapse_banner(self) -> None:
        """The info banner slides away after a while to give the camera more room."""
        banner = self.banner
        if not banner.winfo_exists() or not banner.winfo_ismapped():
            return
        full = banner.winfo_height()
        banner.grid_propagate(False)
        banner.pack_propagate(False)
        banner.configure(height=full)

        def step(k):
            banner.configure(height=max(1, int(full * (1 - k))))
        animate(banner, "collapse", 450, step, done=banner.grid_remove)

    # ------------------------------------------------------------------ camera
    def _build_camera_card(self, parent) -> None:
        t = self.t
        card = RoundedCard(parent, t)
        card.grid(row=0, column=0, sticky="nsew")
        body = card.body
        top = tk.Frame(body, bg=t["card"])
        top.pack(fill="x")
        tk.Label(top, text="CAMERA", bg=t["card"], fg=t["muted"], font=font(10, "bold")).pack(side="left")
        self.live_label = tk.Label(top, text="", bg=t["card"], fg=t["danger"], font=font(10, "bold"))
        self.live_label.pack(side="left", padx=10)
        self.fps_label = tk.Label(top, text="", bg=t["card"], fg=t["muted"], font=font(10))
        self.fps_label.pack(side="right")

        self.video_box = tk.Frame(body, bg=t["video_bg"])
        self.video_box.pack(fill="both", expand=True, pady=6)
        self.video_box.pack_propagate(False)
        self.video_label = tk.Label(self.video_box, text=MSG_CAMERA_OFF, bg=t["video_bg"], fg="#D1D5DB",
                                    font=font(14), wraplength=460, justify="center")
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")

        chips = tk.Frame(body, bg=t["card"])
        chips.pack(fill="x")
        self.chip_camera = tk.Label(chips, bg=t["card"], font=font(11, "bold"))
        self.chip_hand = tk.Label(chips, bg=t["card"], font=font(11, "bold"))
        self.chip_recog = tk.Label(chips, bg=t["card"], font=font(11, "bold"))
        for chip in (self.chip_camera, self.chip_hand, self.chip_recog):
            chip.pack(side="left", padx=(0, 16))
        modes = tk.Frame(body, bg=t["card"])
        modes.pack(fill="x", pady=(8, 0))
        tk.Label(modes, text="MODE", bg=t["card"], fg=t["muted"], font=font(10, "bold")).pack(side="left", padx=(0, 10))
        self.mode_seg = Segmented(modes, t, MODES, self.set_mode, self.mode)
        self.mode_seg.pack(side="left", fill="x", expand=True)
        self.tip_label = tk.Label(body, text="", bg=t["card"], fg=t["muted"], font=font(10), anchor="w",
                                  justify="left", cursor="hand2")
        self.tip_label.pack(fill="x", pady=(6, 0))
        self.tip_label.bind("<Button-1>", lambda e: self.open_guide())
        body.bind("<Configure>", lambda e: self.tip_label.configure(wraplength=max(e.width - 10, 100)), add="+")

    # ---------------------------------------------------------------- detection
    def _build_detection_card(self, parent) -> None:
        t = self.t
        card = RoundedCard(parent, t)
        card.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        body = card.body
        top = tk.Frame(body, bg=t["card"])
        top.pack(fill="x")
        tk.Label(top, text="DETECTION", bg=t["card"], fg=t["muted"], font=font(10, "bold")).pack(side="left")
        self.tag_label = tk.Label(top, text="", bg=t["card"], fg=t["warn"], font=font(10, "bold"))
        self.tag_label.pack(side="right")
        row = tk.Frame(body, bg=t["card"])
        row.pack(fill="x")

        self.gesture_label = tk.Label(row, text="—", bg=t["card"], fg=t["text"],
                                      font=font(GESTURE_LABEL_SIZE, "bold"), height=1)
        self.gesture_label.pack(side="left", padx=12)
        self.sub_label = tk.Label(body, text="", bg=t["card"], fg=t["accent"], font=font(12, "bold italic"),
                                  anchor="w", justify="left")
        self.sub_label.pack(fill="x")
        body.bind("<Configure>", lambda e: self.sub_label.configure(wraplength=max(e.width - 10, 100)), add="+")
        meters = tk.Frame(body, bg=t["card"])
        meters.pack(fill="x", pady=(2, 2))
        meters.columnconfigure(1, weight=1)
        self.conf_label = tk.Label(meters, text="Confidence: —", bg=t["card"], fg=t["text"], font=font(10, "bold"),
                                   anchor="w", width=16)
        self.conf_label.grid(row=0, column=0, sticky="w")
        self.conf_meter = Meter(meters, t, height=10)
        self.conf_meter.grid(row=0, column=1, sticky="ew")
        tk.Label(meters, text="Hold to accept", bg=t["card"], fg=t["muted"], font=font(9), anchor="w",
                 width=16).grid(row=1, column=0, sticky="w", pady=(3, 0))
        self.hold_meter = Meter(meters, t, height=10)
        self.hold_meter.grid(row=1, column=1, sticky="ew", pady=(3, 0))
        self.finger_meter = FingerMeter(body, t)
        self.finger_meter.pack(fill="x", pady=(4, 0))
        self.hint_label = tk.Label(body, text=MSG_CAMERA_OFF, bg=t["card"], fg=t["warn"], font=font(10),
                                   anchor="w", justify="left")
        self.hint_label.pack(fill="x")
        body.bind("<Configure>", lambda e: self.hint_label.configure(wraplength=max(e.width - 10, 100)))

    # ----------------------------------------------------------------- sentence
    def _build_sentence_card(self, parent) -> None:
        t = self.t
        s = self.ctx.settings
        card = RoundedCard(parent, t)
        card.grid(row=1, column=0, sticky="nsew")
        self.sentence_card = card
        body = card.body
        top = tk.Frame(body, bg=t["card"])
        top.pack(fill="x")
        tk.Label(top, text="SENTENCE", bg=t["card"], fg=t["muted"], font=font(10, "bold")).pack(side="left")
        self.raw_label = tk.Label(top, text="", bg=t["card"], fg=t["muted"], font=font(10))
        self.raw_label.pack(side="right")
        compact = self.root.winfo_screenheight() < 820
        self.sentence_label = tk.Label(body, text="", bg=t["card"], fg=t["text"], font=font(15 if compact else 20, "bold"),
                                       anchor="nw", justify="left", height=2)
        self.sentence_label.pack(fill="x", pady=(2, 0))
        body.bind("<Configure>", lambda e: self.sentence_label.configure(wraplength=max(e.width - 10, 100)), add="+")
        self.pending_label = tk.Label(top, text="", bg=t["card"], fg=t["accent"], font=font(10, "bold"), anchor="w")
        self.pending_label.pack(side="left", padx=10)

        tools = tk.Frame(body, bg=t["card"])
        tools.pack(fill="x", pady=4)
        buttons = [(f"{ICONS['space']} SPACE", self.add_space), (f"{ICONS['delete']} DELETE", self.delete_last),
                   ("Aa CAPITALIZE", self.capitalize_last), ("★ USE AS NAME", self.use_as_name)]
        for i, (text, cmd) in enumerate(buttons):
            FlatButton(tools, t, text, cmd, size=9, pady=5, padx=4).grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 4, 0))
            tools.columnconfigure(i, weight=1, uniform="tools")

        row_a = tk.Frame(body, bg=t["card"])
        row_a.pack(fill="x", pady=2)
        tk.Label(row_a, text="Name:", bg=t["card"], fg=t["text"], font=font(11)).pack(side="left")
        self.name_var = tk.StringVar(value=s.user_name)
        name_entry = tk.Entry(row_a, textvariable=self.name_var, width=12, font=font(12), bg=t["entry_bg"],
                              fg=t["text"], insertbackground=t["text"], relief="flat", highlightthickness=2,
                              highlightbackground=t["border"], highlightcolor=t["accent"])
        name_entry.pack(side="left", padx=6, ipady=3)
        name_entry.bind("<FocusOut>", lambda e: self._save_name())
        name_entry.bind("<Return>", lambda e: self._save_name())
        FlatButton(row_a, t, "MY NAME", self.insert_my_name, size=9, pady=4).pack(side="left", padx=2)
        tk.Label(row_a, text="  Quick phrase:", bg=t["card"], fg=t["text"], font=font(11)).pack(side="left")
        self.quick_var = tk.StringVar()
        quick = ttk.Combobox(row_a, textvariable=self.quick_var, state="readonly", width=16, font=font(10),
                             values=self.ctx.registry.quick_phrases(s.language))
        quick.pack(side="left", padx=4)
        quick.bind("<<ComboboxSelected>>", lambda e: self.load_quick_phrase())

        row_b = tk.Frame(body, bg=t["card"])
        row_b.pack(fill="x", pady=2)
        tk.Label(row_b, text="Add word:", bg=t["card"], fg=t["text"], font=font(11)).pack(side="left")
        self.word_var = tk.StringVar()
        word_box = ttk.Combobox(row_b, textvariable=self.word_var, width=16, font=font(11),
                                values=self.ctx.registry.vocabulary_words(s.language))
        word_box.pack(side="left", padx=6)
        word_box.bind("<Return>", lambda e: self.add_typed_word())
        FlatButton(row_b, t, "ADD WORD", self.add_typed_word, size=9, pady=4).pack(side="left")

        row_c = tk.Frame(body, bg=t["card"])
        row_c.pack(fill="x", pady=(4, 0))
        tk.Label(row_c, text="Speak:", bg=t["card"], fg=t["muted"], font=font(10, "bold")).pack(side="left")
        self.speech_vars = {}
        for key, text in (("speak_letters", "letters"), ("speak_words", "words"), ("speak_sentences", "sentences")):
            var = tk.BooleanVar(value=getattr(s, key))
            self.speech_vars[key] = var
            tk.Checkbutton(row_c, text=text, variable=var, command=self._speech_option_changed,
                           bg=t["card"], fg=t["text"], selectcolor=t["card_alt"], activebackground=t["card"],
                           activeforeground=t["text"], font=font(10)).pack(side="left", padx=4)

    # ----------------------------------------------------------------- controls
    def _build_controls(self) -> None:
        t = self.t
        card = RoundedCard(self.root, t, radius=16, padding=8, height=64)
        card.grid(row=3, column=0, sticky="ew", padx=14, pady=4)
        body = card.body
        specs = [
            ("btn_camera", f"{ICONS['start']} START CAMERA", self.toggle_camera, "success"),
            ("btn_pause", f"{ICONS['pause']} PAUSE DETECTION", self.toggle_pause, "secondary"),
            ("btn_clear", f"{ICONS['clear']} CLEAR SENTENCE", self.clear_sentence, "secondary"),
            ("btn_speak", f"{ICONS['speak']} SPEAK", self.speak_sentence, "primary"),
            ("btn_stop_speech", f"{ICONS['stop']} STOP SPEECH", self.stop_speech, "secondary"),
            ("btn_repeat", f"{ICONS['repeat']} REPEAT", self.repeat_speech, "secondary"),
            ("btn_voice", f"{ICONS['voice']} VOICE MODE", self.toggle_voice, "secondary"),
            ("btn_talk", "💬 CONVERSATION", self.toggle_conversation, "secondary"),
            ("btn_settings", f"{ICONS['settings']} SETTINGS", self.open_settings, "secondary"),
        ]
        for i, (attr, text, cmd, variant) in enumerate(specs):
            btn = FlatButton(body, t, text, cmd, variant, size=10, pady=9, padx=2)
            btn.grid(row=0, column=i, sticky="nsew", padx=3)
            body.columnconfigure(i, weight=1, uniform="ctl")
            setattr(self, attr, btn)
        body.rowconfigure(0, weight=1)
        self._refresh_buttons()

    # ------------------------------------------------------------------- bottom
    def _build_bottom(self) -> None:
        t = self.t
        bottom = tk.Frame(self.root, bg=t["bg"])
        bottom.grid(row=4, column=0, sticky="nsew", padx=14, pady=(4, 12))
        self.bottom = bottom
        if not self.show_conversation:          # more room for the camera on small screens
            bottom.grid_remove()
        bottom.rowconfigure(0, weight=1)
        bottom.columnconfigure(0, weight=3, uniform="bot")
        bottom.columnconfigure(1, weight=2, uniform="bot")

        comm = RoundedCard(bottom, t)
        comm.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        cb = comm.body
        cb.columnconfigure(0, weight=1, uniform="p")
        cb.columnconfigure(1, weight=1, uniform="p")
        cb.rowconfigure(0, weight=1)
        # Person A
        a = tk.Frame(cb, bg=t["card"])
        a.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        tk.Label(a, text="PERSON A — SIGN LANGUAGE", bg=t["card"], fg=t["muted"], font=font(10, "bold")).pack(anchor="w")
        tk.Label(a, text="Detected:", bg=t["card"], fg=t["muted"], font=font(10)).pack(anchor="w")
        self.person_a_label = tk.Label(a, text="—", bg=t["card"], fg=t["text"], font=font(13, "bold"),
                                       anchor="w", justify="left", wraplength=300)
        self.person_a_label.pack(fill="x")
        FlatButton(a, t, f"{ICONS['speak']} SPEAK", self.speak_sentence, "primary", size=10, pady=4).pack(anchor="w", pady=4)
        # Person B
        b = tk.Frame(cb, bg=t["card"])
        b.grid(row=0, column=1, sticky="nsew")
        tk.Label(b, text="PERSON B — VOICE", bg=t["card"], fg=t["muted"], font=font(10, "bold")).pack(anchor="w")
        tk.Label(b, text="You said:", bg=t["card"], fg=t["muted"], font=font(10)).pack(anchor="w")
        self.person_b_label = tk.Label(b, text="—", bg=t["card"], fg=t["text"], font=font(13, "bold"),
                                       anchor="w", justify="left", wraplength=300)
        self.person_b_label.pack(fill="x")
        self.voice_status = tk.Label(b, text="Voice mode is off. You can also type below.", bg=t["card"],
                                     fg=t["muted"], font=font(9), anchor="w")
        self.voice_status.pack(fill="x")
        typed = tk.Frame(b, bg=t["card"])
        typed.pack(fill="x", pady=2)
        self.typed_var = tk.StringVar()
        entry = tk.Entry(typed, textvariable=self.typed_var, font=font(11), bg=t["entry_bg"], fg=t["text"],
                         insertbackground=t["text"], relief="flat", highlightthickness=2,
                         highlightbackground=t["border"], highlightcolor=t["accent"])
        entry.pack(side="left", fill="x", expand=True, ipady=3)
        entry.bind("<Return>", lambda e: self.submit_person_b())
        FlatButton(typed, t, "SEND", self.submit_person_b, size=9, pady=3).pack(side="left", padx=(6, 0))

        hist = RoundedCard(bottom, t)
        hist.grid(row=0, column=1, sticky="nsew")
        hb = hist.body
        head = tk.Frame(hb, bg=t["card"])
        head.pack(fill="x")
        tk.Label(head, text="COMMUNICATION HISTORY", bg=t["card"], fg=t["muted"], font=font(10, "bold")).pack(side="left")
        FlatButton(head, t, "CLEAR HISTORY", self.clear_history, size=8, pady=2, padx=6).pack(side="right")
        box = tk.Frame(hb, bg=t["card"])
        box.pack(fill="both", expand=True, pady=(4, 0))
        self.history_text = tk.Text(box, height=3, wrap="word", font=font(10), bg=t["entry_bg"], fg=t["text"],
                                    relief="flat", padx=8, pady=4, state="disabled", cursor="arrow")
        scroll = ttk.Scrollbar(box, orient="vertical", command=self.history_text.yview)
        self.history_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.history_text.pack(side="left", fill="both", expand=True)
        tk.Label(hb, text=PRIVACY_NOTICE, bg=t["card"], fg=t["muted"], font=font(8), anchor="w",
                 justify="left", wraplength=420).pack(fill="x", pady=(3, 0))

    # =============================================================== shortcuts
    def _bind_shortcuts(self) -> None:
        actions = {"c": self.toggle_camera, "s": self.speak_sentence, "h": self.open_help,
                   "v": self.toggle_voice, "r": self.clear_sentence}
        actions["g"] = self.open_guide
        for key, fn in actions.items():
            for k in (key, key.upper()):
                self.root.bind(f"<{k}>", lambda e, f=fn: self._shortcut(e, f))
        for number, mode in enumerate(MODES, start=1):          # 1..5 switch mode
            self.root.bind(f"<Key-{number}>", lambda e, m=mode: self._shortcut(e, lambda: self.set_mode(m)))

    def _shortcut(self, event, fn) -> None:
        if isinstance(event.widget, (tk.Entry, ttk.Entry, ttk.Combobox, tk.Text)):
            return                      # the user is typing
        fn()

    # =================================================================== refresh
    def _refresh_all(self) -> None:
        self._refresh_sentence()
        self._refresh_history()
        self._refresh_chips(None)
        self._refresh_buttons()
        self.person_b_label.configure(text=self.person_b_text or "—")

    def _refresh_sentence(self, typewriter: bool = False) -> None:
        sentence = self.builder.sentence()
        old = self._sentence_shown
        self._sentence_shown = sentence
        # Only the NEW part types itself out; trailing punctuation may change, so compare without it.
        stem = old.rstrip(".!?")
        if typewriter and sentence and stem and sentence.startswith(stem) and len(sentence) > len(stem):
            start = len(stem)

            def step(k, text=sentence, start=start):
                self.sentence_label.configure(text=text[:start + round((len(text) - start) * k)])
            animate(self.sentence_label, "type", min(900, 28 * (len(sentence) - start)), step, ease=lambda k: k)
        elif typewriter and sentence:
            animate(self.sentence_label, "type", min(900, 28 * len(sentence)),
                    lambda k, text=sentence: self.sentence_label.configure(text=text[:round(len(text) * k)] or "…"),
                    ease=lambda k: k)
        else:
            animate(self.sentence_label, "type", 1, lambda k: None)       # cancel a running one
            self.sentence_label.configure(text=sentence or "…")
        self.raw_label.configure(text=self.builder.raw_text())
        self.person_a_label.configure(text=self.builder.raw_text().upper() or "—")
        self.pending_label.configure(
            text=(f"Spelling: {self.builder.pending}  (SPACE finishes the word)" if self.builder.pending else ""))

    def _refresh_history(self) -> None:
        box = self.history_text
        box.configure(state="normal")
        box.delete("1.0", "end")
        entries = self.ctx.history.entries[-60:]
        if not entries:
            box.insert("end", "Nothing yet. Sentences you speak and words you hear appear here.\nStored only on this computer.")
        for e in entries:
            box.insert("end", f"{e.time}\n", ())
            box.insert("end", f"{e.source} → \"{e.text}\"\n")
        box.configure(state="disabled")
        box.see("end")

    def _refresh_buttons(self) -> None:
        if self.camera_on:
            self.btn_camera.set_variant("danger", f"{ICONS['stop']} STOP CAMERA")
        else:
            self.btn_camera.set_variant("success", f"{ICONS['start']} START CAMERA")
        paused = self.ctx.pipeline.paused
        self.btn_pause.set_variant("primary" if paused else "secondary",
                                   f"{ICONS['resume']} RESUME" if paused else f"{ICONS['pause']} PAUSE DETECTION")
        self.btn_voice.set_variant("danger" if self.voice_on else "secondary",
                                   f"{ICONS['voice']} STOP VOICE" if self.voice_on else f"{ICONS['voice']} VOICE MODE")

    def _refresh_chips(self, res) -> None:
        t = self.t
        cam = bool(res and res.camera_ok and self.camera_on)
        hand = bool(res and res.hand_detected)
        if res and res.state == "starting":
            cam_text, cam_col = f"{ICONS['dot']} Camera: Starting…", t["warn"]
        else:
            cam_text, cam_col = (f"{ICONS['dot']} Camera: Active", t["ok"]) if cam else (f"{ICONS['dot']} Camera: Off", t["muted"])
        self.chip_camera.configure(text=cam_text, fg=cam_col)
        self.chip_hand.configure(text=f"{ICONS['dot']} Hand: {'Detected' if hand else 'Not detected'}",
                                 fg=t["ok"] if hand else t["muted"])
        if res and res.paused:
            rec, col = "Paused", t["warn"]
        elif res and res.recognition_active:
            rec, col = "Active", t["ok"]
        else:
            rec, col = ("Unavailable", t["danger"]) if (res and res.camera_ok) else ("Off", t["muted"])
        self.chip_recog.configure(text=f"{ICONS['dot']} Recognition: {rec}", fg=col)

    # ================================================================== polling
    def _poll(self) -> None:
        try:
            self._handle_frame()
            self._handle_voice_events()
            self._heartbeat()
        except Exception:  # noqa: BLE001 - never let the UI loop die
            log.exception("UI update failed")
        self.root.after(15, self._poll)

    def _heartbeat(self) -> None:
        """Small always-running animations: LIVE dot, tips ribbon, idle camera picture."""
        t = self.t
        now = time.monotonic()
        if self.camera_on:
            self.live_label.configure(text=f"{ICONS['dot']} LIVE", fg=mix(t["card"], t["danger"], 0.35 + 0.65 * pulse(1.4)))
        else:
            self.live_label.configure(text="")
            self._idle_frame += 1
            if self._idle_frame % 18 == 0 and self._photo is None and not str(self.video_label.cget("text")).startswith("⚠"):
                hand = "👋" if (self._idle_frame // 18) % 2 else "✋"
                self.video_label.configure(text=f"{hand}\n\n{MSG_CAMERA_OFF}", font=font(14))
        if now - self._tip_changed > 5.0:
            self._next_tip()

    def _mode_tips(self) -> list[str]:
        tips = []
        for cat in MODE_CATEGORIES.get(self.mode, []):
            for g in self.ctx.registry.by_category(cat):
                shape = g.get("shape")
                if not shape:
                    continue
                how = SHAPE_HINTS.get(shape, g.get("instructions", ""))
                said = f" → “{g['output'].replace('{name}', self.ctx.settings.user_name or '…')}”" if g.get("output") else ""
                tips.append(f"{ICONS['spark']} Try {g['gesture']}: {how}{said}")
        return tips or [f"{ICONS['spark']} Press GESTURE GUIDE (G) to see every pose."]

    def _next_tip(self) -> None:
        tips = self._mode_tips()
        self._tip_index = (self._tip_index + 1) % len(tips)
        text = tips[self._tip_index] + "   ·   click for the full guide"
        self._tip_changed = time.monotonic()
        t = self.t
        self.tip_label.configure(text=text)
        animate(self.tip_label, "fade", 500, lambda k: self.tip_label.configure(fg=mix(t["card"], t["muted"], k)))

    def _handle_frame(self) -> None:
        if not self.camera_on:
            return
        self.ctx.pipeline.set_display_size(self.video_box.winfo_width(), self.video_box.winfo_height())
        res = self.ctx.pipeline.latest()
        if res.seq == self.last_seq:
            return
        self.last_seq = res.seq
        self._refresh_chips(res)
        if res.state == "error":
            self._camera_failed(res.error or res.message)
            return
        if res.state != "running":
            self.hint_label.configure(text=res.message or "Starting camera…", fg=self.t["warn"])
            return
        ai = res.extra.get("ai_fps", 0.0)
        self.fps_label.configure(text=f"{res.fps:.0f} FPS · AI {ai:.0f}/s · {res.extra.get('camera', '')}")
        if not self.demo_running:
            for label, category, output in self.ctx.pipeline.take_accepted():
                entry = self.ctx.registry.get(label, category) or {}
                self._accept(Prediction(label, 1.0, category, source="rules", shape=entry.get("shape") or "",
                                        output=output))
            self._show_prediction(res)
        self.finger_meter.set(res.extra.get("fingers", [0.0] * 5))
        if res.image_rgb is not None:
            self._show_frame(res)

    def _show_frame(self, res) -> None:
        """The worker already resized and decorated the frame: just hand it to Tk."""
        img = Image.fromarray(res.image_rgb)
        if self._photo is not None and (self._photo.width(), self._photo.height()) == img.size:
            self._photo.paste(img)             # reuse the Tk image: much faster than a new one
        else:
            self._photo = ImageTk.PhotoImage(img)   # keep a reference or Tk shows nothing
            self.video_label.configure(image=self._photo, text="")

    def _pop_label(self, text: str, category: str) -> None:
        """New gesture: the big label pops in (overshoots a little) and flashes the accent colour."""
        t = self.t
        base = GESTURE_LABEL_SIZE if len(text) <= 3 else (26 if len(text) <= 9 else 18)
        self.gesture_label.configure(text=text)

        def step(k):
            self.gesture_label.configure(font=font(max(8, int(base * (0.55 + 0.45 * k))), "bold"),
                                         fg=mix(t["accent"], t["text"], min(1.0, k)))
        animate(self.gesture_label, "pop", 320, step, ease=ease_out_back)

    def _clear_prediction(self, message: str) -> None:
        t = self.t
        if self._shown_label:
            self._shown_label = ""
            self.gesture_label.configure(text="—", font=font(GESTURE_LABEL_SIZE, "bold"), fg=t["muted"])
        self.sub_label.configure(text="")
        self.conf_label.configure(text="Confidence: —")
        self.conf_meter.set(0)
        self.tag_label.configure(text="")
        self.hint_label.configure(text=message, fg=t["warn"])

    def _show_prediction(self, res) -> None:
        t = self.t
        pred: Prediction | None = res.prediction
        s = self.ctx.settings
        if res.paused or pred is None:
            self._clear_prediction(res.message)
            self.stabilizer.update(None)
            self.hold_meter.set(0)
            return
        if pred.label != self._shown_label:
            self._shown_label = pred.label
            self._pop_label(pred.label, pred.category)
        if pred.output:
            self.sub_label.configure(text=f"“{pred.output.replace('{name}', s.user_name or '…')}”")
        else:
            hint = SHAPE_HINTS.get(pred.shape, "")
            self.sub_label.configure(text=hint)
        pct = int(pred.confidence * 100)
        self.conf_label.configure(text=f"Confidence: {pct}%")
        color = t["ok"] if pred.confidence >= 0.8 else (t["warn"] if pred.confidence >= s.recognition_confidence else t["danger"])
        self.conf_meter.set(pred.confidence, color)
        if pred.dynamic:
            self.tag_label.configure(text="MOTION GESTURE", fg=t["warn"])
        elif pred.is_rule_based:
            self.tag_label.configure(text="SMART RULES · NO TRAINING", fg=t["accent"])
        else:
            self.tag_label.configure(text="TRAINED MODEL", fg=t["ok"])
        self.hint_label.configure(text=res.message or "Hold the pose until the ring fills…",
                                  fg=t["warn"] if res.message else t["muted"])

        progress = res.extra.get("progress", 0.0)      # the engine holds / accepts gestures
        self.hold_meter.set(progress, t["ok"] if res.extra.get("locked") else t["accent"])

    def _accept(self, pred: Prediction) -> None:
        """A gesture has been held steady long enough: add it to the sentence."""
        s = self.ctx.settings
        t = self.t
        output = pred.output
        if pred.category == "sentence" and not output:
            entry = self.ctx.registry.get(pred.label, "sentence")
            output = entry.get("output", "") if entry else ""
        kind = self.builder.add_gesture(pred.label, pred.category, output)
        spoken = output.replace("{name}", s.user_name or "") if output else ""
        if s.speech_enabled:
            if kind in ("letter", "digit") and s.speak_letters:
                self.ctx.tts.speak(pred.label, dedupe_seconds=1.0)
            elif kind == "word" and s.speak_words:
                self.ctx.tts.speak(pred.label.lower(), dedupe_seconds=1.0)
            elif kind == "sentence" and s.speak_sentences:
                self.ctx.tts.speak(spoken, dedupe_seconds=1.5)
        if kind == "sentence" and "{name}" in output and not s.user_name:
            self.hint_label.configure(text="Type your name in the Name box (or spell it and press USE AS NAME).",
                                      fg=t["warn"])
        else:
            self.hint_label.configure(text=f"✓ Added: {spoken or pred.label}", fg=t["ok"])
        self.sentence_card.flash(t["ok"])
        self._refresh_sentence(typewriter=True)

    def _camera_failed(self, message: str) -> None:
        self.camera_on = False
        self.ctx.pipeline.stop()
        self.video_label.configure(image="", text=f"⚠ {message}")
        self._photo = None
        self.hint_label.configure(text=message, fg=self.t["danger"])
        self._refresh_chips(None)
        self._refresh_buttons()
        if not self._camera_error_shown:
            self._camera_error_shown = True
            messagebox.showwarning("Camera", f"{message}\n\nThe rest of the app (typing, speech, help, demo) still works.")

    # ================================================================== camera
    def toggle_camera(self) -> None:
        if self.camera_on:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self) -> None:
        self._camera_error_shown = False
        self.ctx.pipeline.mode = self.mode
        self.ctx.pipeline.paused = False
        self.stabilizer.reset()
        self.camera_on = True
        self.last_seq = -1
        self.ctx.pipeline.start()
        self.video_label.configure(image="", text="Starting camera…")
        self.hint_label.configure(text="Starting camera…", fg=self.t["warn"])
        self._refresh_buttons()

    def stop_camera(self) -> None:
        self.camera_on = False
        self.ctx.pipeline.stop()
        self.stabilizer.reset()
        self._photo = None
        self.video_label.configure(image="", text=MSG_CAMERA_OFF)
        self.fps_label.configure(text="")
        self._last_boxes = []
        self._clear_prediction(MSG_CAMERA_OFF)
        self.hold_meter.set(0)
        self._refresh_chips(None)
        self._refresh_buttons()

    def toggle_conversation(self) -> None:
        """Show / hide the Person A / Person B / history panel (gives the camera more room)."""
        self.show_conversation = not self.show_conversation
        if self.show_conversation:
            self.bottom.grid()
            self.root.rowconfigure(4, minsize=150)
        else:
            self.bottom.grid_remove()
            self.root.rowconfigure(4, minsize=0)
        self.btn_talk.set_variant("primary" if self.show_conversation else "secondary")

    def toggle_pause(self) -> None:
        self.ctx.pipeline.paused = not self.ctx.pipeline.paused
        self.stabilizer.reset()
        self._refresh_buttons()

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.ctx.pipeline.mode = mode
        self.ctx.settings.mode = mode
        self.ctx.store.save(self.ctx.settings)
        self.stabilizer.reset()
        self.mode_seg.set(mode)
        self._tip_index = -1
        self._next_tip()
        self.hint_label.configure(text=f"Mode: {mode.title()} — hold a pose until the ring fills.", fg=self.t["muted"])

    # ============================================================== sentence
    def add_space(self) -> None:
        word = self.builder.space()
        s = self.ctx.settings
        if word and s.speech_enabled and s.speak_words:
            self.ctx.tts.speak(word.lower(), dedupe_seconds=0.5)
        self._refresh_sentence()

    def delete_last(self) -> None:
        self.builder.delete()
        self._refresh_sentence()

    def capitalize_last(self) -> None:
        self.builder.capitalize_last()
        self._refresh_sentence()

    def use_as_name(self) -> None:
        name = self.builder.use_as_name()
        if name:
            self.name_var.set(name)
            self._save_name()
            self.hint_label.configure(text=f"Name set to {name}.", fg=self.t["ok"])
        else:
            self.hint_label.configure(text="Spell your name with letters first (for example A, L, I).", fg=self.t["warn"])
        self._refresh_sentence()

    def insert_my_name(self) -> None:
        self._save_name()
        if not self.builder.insert_name_phrase(self.ctx.settings.user_name):
            self.hint_label.configure(text="Type your name in the Name box first.", fg=self.t["warn"])
        self._refresh_sentence()

    def add_typed_word(self) -> None:
        text = self.word_var.get().strip()
        if text:
            self.builder.add_word(text, "typed")
            self.word_var.set("")
            self._refresh_sentence()

    def load_quick_phrase(self) -> None:
        phrase = self.quick_var.get()
        if not phrase:
            return
        self.builder.load_phrase(phrase)
        self._save_name()
        if phrase.lower().endswith("my name is") and self.ctx.settings.user_name:
            self.builder.add_word(self.ctx.settings.user_name, "name")
        self._refresh_sentence()
        if self.ctx.settings.speak_sentences:
            self.speak_sentence()

    def clear_sentence(self) -> None:
        self.builder.clear()
        self.stabilizer.reset()
        self._refresh_sentence()

    def _save_name(self) -> None:
        name = self.name_var.get().strip()
        if name != self.ctx.settings.user_name:
            self.ctx.settings.user_name = name
            self.ctx.store.save(self.ctx.settings)
            self._refresh_sentence()

    def _speech_option_changed(self) -> None:
        for key, var in self.speech_vars.items():
            setattr(self.ctx.settings, key, bool(var.get()))
        self.ctx.store.save(self.ctx.settings)

    # ================================================================= speech
    def speak_sentence(self) -> None:
        text = self.builder.sentence()
        s = self.ctx.settings
        if not text:
            self.hint_label.configure(text="Nothing to speak yet — build a sentence first.", fg=self.t["warn"])
            return
        self.ctx.history.add("SIGN", text)
        self._refresh_history()
        if not s.speech_enabled:
            self.hint_label.configure(text="Speech is turned off in Settings (the sentence is saved in history).", fg=self.t["warn"])
        elif not self.ctx.tts.available:
            self.hint_label.configure(text=f"Text-to-speech unavailable: {self.ctx.tts.error or self.ctx.tts.status_text}. "
                                           "The text is still shown on screen.", fg=self.t["danger"])
        else:
            self.ctx.tts.speak(text)
            self.hint_label.configure(text=f"Speaking: {text}", fg=self.t["ok"])
        if s.auto_clear_sentence:
            self.builder.clear()
            self._refresh_sentence()

    def stop_speech(self) -> None:
        self.ctx.tts.stop()

    def repeat_speech(self) -> None:
        if not self.ctx.tts.repeat():
            self.hint_label.configure(text="Nothing to repeat yet.", fg=self.t["warn"])

    # ============================================================ voice / person B
    def toggle_voice(self) -> None:
        if self.voice_on:
            self._stop_voice()
            return
        ok, message = check_voice_support()
        if not ok:
            self.voice_status.configure(text=f"Voice unavailable: {message} Type below instead.", fg=self.t["warn"])
            if not self._voice_help_shown:
                self._voice_help_shown = True
                messagebox.showinfo("Voice Mode setup", f"{message}\n\n{VOSK_SETUP_TEXT}")
            return
        self.voice = VoiceListener(
            on_partial=lambda text: self.voice_events.put(("partial", text)),
            on_final=lambda text: self.voice_events.put(("final", text)),
            on_status=lambda text: self.voice_events.put(("status", text)))
        started, message = self.voice.start()
        if not started:
            self.voice_status.configure(text=message, fg=self.t["danger"])
            return
        self.voice_on = True
        self.voice_status.configure(text=f"{ICONS['dot']} Microphone: starting…", fg=self.t["warn"])
        self._refresh_buttons()

    def _stop_voice(self) -> None:
        if self.voice:
            self.voice.stop()
        self.voice_on = False
        self.voice_status.configure(text="Voice mode is off. You can also type below.", fg=self.t["muted"])
        self._refresh_buttons()

    def _handle_voice_events(self) -> None:
        while True:
            try:
                kind, text = self.voice_events.get_nowait()
            except queue.Empty:
                return
            if kind == "partial" and self.voice_on:
                self.person_b_label.configure(text=text.upper() + " …")
            elif kind == "final" and self.voice_on:
                self._person_b_says(text)
            elif kind == "status":
                if text == "listening":
                    self.voice_status.configure(text=f"{ICONS['dot']} Microphone: listening", fg=self.t["ok"])
                elif text == "stopped" and self.voice_on:
                    self.voice_on = False
                    self.voice_status.configure(text="Voice mode stopped.", fg=self.t["muted"])
                    self._refresh_buttons()
                elif text not in ("stopped",):
                    self.voice_status.configure(text=text, fg=self.t["warn"])

    def _person_b_says(self, text: str) -> None:
        self.person_b_text = text.upper()
        self.person_b_label.configure(text=self.person_b_text)
        self.ctx.history.add("VOICE", text)
        self._refresh_history()
        s = self.ctx.settings
        if s.speech_enabled and s.speak_voice_text and self.ctx.tts.available:
            self.ctx.tts.speak(text)

    def submit_person_b(self) -> None:
        text = self.typed_var.get().strip()
        if text:
            self.typed_var.set("")
            self._person_b_says(text)

    def clear_history(self) -> None:
        self.ctx.history.clear()
        self._refresh_history()

    # ==================================================================== demo
    def run_demo(self) -> None:
        if self.demo_running:
            return
        self.demo_running = True
        self._demo_index = 0
        self.builder.clear()
        self.stabilizer.reset()
        self.tag_label.configure(text="DEMO PREDICTION")
        self.hint_label.configure(text="Running scripted demo — no camera needed.", fg=self.t["warn"])
        self._demo_step()

    def _demo_step(self) -> None:
        if not self.demo_running:
            return
        kind, value = DEMO_STEPS[self._demo_index]
        self._demo_index += 1
        delay = 1300
        if kind == "mode":
            self.set_mode(value)
        elif kind == "clear":
            self.builder.clear()
            self.gesture_label.configure(text="—")
        elif kind == "gesture":
            label, category = value
            entry = self.ctx.registry.get(label, category) or {}
            self._shown_label = label
            self._pop_label(label, category)
            self.sub_label.configure(text=f"“{entry['output'].replace('{name}', self.ctx.settings.user_name or '…')}”"
                                     if entry.get("output") else SHAPE_HINTS.get(entry.get("shape") or "", ""))
            self.conf_label.configure(text="Demo (scripted)")
            self.hold_meter.set(1.0, self.t["ok"])
            self._accept(Prediction(label, 1.0, category, source="demo", shape=entry.get("shape") or "",
                                    output=entry.get("output", "")))
        elif kind == "use_name":
            self.use_as_name()
        elif kind == "speak":
            self.speak_sentence()
            delay = 2600
        elif kind == "done":
            self.demo_running = False
            self.tag_label.configure(text="")
            self.hold_meter.set(0)
            self.hint_label.configure(text="Demo finished. Press START CAMERA to try your own hand.", fg=self.t["ok"])
            return
        self._refresh_sentence()
        self.root.after(delay, self._demo_step)

    # ================================================================= dialogs
    def _open_dialog(self, key: str, factory) -> None:
        existing = self._dialogs.get(key)
        if existing is not None and existing.winfo_exists():
            existing.lift()
            existing.focus_force()
            return
        win = factory()
        self._dialogs[key] = win

    def open_help(self, section: str | None = None) -> None:
        self._open_dialog("help", lambda: HelpWindow(self.root, self.ctx, self.t, section if isinstance(section, str) else None))

    def open_guide(self) -> None:
        """Gesture Guide = the Help Center opened on the current mode's gestures."""
        existing = self._dialogs.get("help")
        if existing is not None and existing.winfo_exists():
            existing.show(GUIDE_SECTIONS.get(self.mode, "alphabet"))
            existing.lift()
            return
        self.open_help(GUIDE_SECTIONS.get(self.mode, "alphabet"))

    def open_setup_help(self) -> None:
        existing = self._dialogs.get("help")
        if existing is not None and existing.winfo_exists():
            existing.destroy()
        self.open_help("setup")

    def open_offline_help(self) -> None:
        self._open_dialog("offline", lambda: OfflineHelpWindow(self.root, self.ctx, self.t))

    def open_settings(self) -> None:
        self._open_dialog("settings", lambda: SettingsWindow(self.root, self.ctx, self.t, self._settings_saved))

    def _settings_saved(self, old, new) -> None:
        tts = self.ctx.tts
        tts.configure(new.speech_rate, new.speech_volume, new.voice_id)
        self.stabilizer.configure(new.gesture_hold_seconds, new.recognition_confidence, new.release_seconds)
        self.ctx.recognizer.set_dynamic_enabled(new.experimental_dynamic)
        self.ctx.history.persist = new.save_history
        camera_changed = any(getattr(old, k) != getattr(new, k) for k in
                             ("camera_index", "resolution", "detection_confidence", "tracking_confidence"))
        if self.camera_on and camera_changed:
            self.ctx.pipeline.stop()
            self.ctx.pipeline.start()
        if old.theme != new.theme or old.language != new.language:
            self._build_ui()

    # ==================================================================== exit
    def on_close(self) -> None:
        try:
            self.demo_running = False
            self.ctx.pipeline.stop()
            if self.voice:
                self.voice.stop()
            self.ctx.tts.shutdown()
            self.ctx.store.save(self.ctx.settings)
        finally:
            self.root.destroy()
