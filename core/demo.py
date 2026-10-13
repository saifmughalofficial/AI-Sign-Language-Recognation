"""Scripted walk-through used by the RUN DEMO button (works with NO camera).

Each step is (kind, value, pause_in_milliseconds_after_the_step).
The dashboard plays the steps as if a real person had signed them, so you can
see the complete flow: gesture -> sentence builder -> speech -> history.
"""
from __future__ import annotations

DEMO_SCRIPT: list[tuple[str, object, int]] = [
    ("note", "Demo: signing  I → NEED → WATER", 1200),
    ("clear", None, 300),
    ("mode", "WORDS", 500),
    ("gesture", ("I", "word"), 1300),
    ("gesture", ("NEED", "word"), 1300),
    ("gesture", ("WATER", "word"), 1500),
    ("speak", None, 2500),
    ("clear", None, 300),
    ("note", "Demo: spelling a name  A → L → I, then USE AS NAME", 1200),
    ("mode", "ALPHABET", 500),
    ("gesture", ("A", "alphabet"), 1100),
    ("gesture", ("L", "alphabet"), 1100),
    ("gesture", ("I", "alphabet"), 1300),
    ("use_name", None, 1500),
    ("speak", None, 2500),
    ("clear", None, 300),
    ("note", "Demo: a question  WHERE → HOSPITAL", 1200),
    ("mode", "WORDS", 500),
    ("gesture", ("WHERE", "word"), 1300),
    ("gesture", ("HOSPITAL", "word"), 1500),
    ("speak", None, 2500),
    ("note", "Demo finished. These were DEMO predictions, not real ASL recognition.", 0),
]
