"""Sign Language Assistant - start here:   python app.py

A free, offline demo/prototype that helps a person who signs communicate with a
person who does not. Everything runs on this computer.
"""
from __future__ import annotations

import logging
import math
import sys
import time
import traceback
import os
from pathlib import Path

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")   # webcam opens in ~2 s, not ~20 s

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.constants import APP_NAME, APP_VERSION, PRIVACY_NOTICE  # noqa: E402


def _splash(root, tk):
    """Small borderless welcome window shown while the app loads."""
    win = tk.Toplevel(root)
    win.overrideredirect(True)
    w, h = 520, 260
    x = (win.winfo_screenwidth() - w) // 2
    y = (win.winfo_screenheight() - h) // 3
    win.geometry(f"{w}x{h}+{x}+{y}")
    win.configure(bg="#0B1220", highlightthickness=2, highlightbackground="#38BDF8")
    icon = tk.Label(win, text="🤟", bg="#0B1220", fg="#38BDF8", font=("Segoe UI Emoji", 40))
    icon.pack(pady=(18, 0))
    tk.Label(win, text=APP_NAME.upper(), bg="#0B1220", fg="#F8FAFC", font=("Segoe UI", 20, "bold")).pack()
    tk.Label(win, text=f"Version {APP_VERSION} · A–Z · 0–9 · words · sentences · offline", bg="#0B1220",
             fg="#B6C2D6", font=("Segoe UI", 10)).pack(pady=(2, 8))
    status = tk.Label(win, text="Starting…", bg="#0B1220", fg="#FBBF24", font=("Segoe UI", 11, "bold"))
    status.pack()
    bar = tk.Canvas(win, width=420, height=6, bg="#172033", highlightthickness=0)
    bar.pack(pady=(8, 0))
    tk.Label(win, text=PRIVACY_NOTICE, bg="#0B1220", fg="#B6C2D6", font=("Segoe UI", 8),
             wraplength=470, justify="center").pack(pady=8)
    try:
        win.attributes("-alpha", 0.0)
    except tk.TclError:
        pass
    start = time.monotonic()

    def animate():                     # fade in, bobbing hand, shimmering loading bar
        if not win.winfo_exists():
            return
        t = time.monotonic() - start
        try:
            win.attributes("-alpha", min(1.0, t / 0.35))
        except tk.TclError:
            pass
        icon.pack_configure(pady=(18 + int(4 * math.sin(t * 6)), 0))
        bar.delete("all")
        x = (t * 320) % 560 - 140
        bar.create_rectangle(x, 0, x + 140, 6, fill="#38BDF8", outline="")
        win.after(16, animate)

    animate()
    win.update()
    return win, status


def _fade_in(root, step: int = 0) -> None:
    """Main window fades in instead of popping up."""
    try:
        root.attributes("-alpha", min(1.0, step / 12))
    except Exception:  # noqa: BLE001 - some window managers do not support alpha
        return
    if step < 12:
        root.after(16, lambda: _fade_in(root, step + 1))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        import tkinter as tk
        from tkinter import messagebox
    except ImportError:
        print("Tkinter is missing. Reinstall Python from python.org and tick 'tcl/tk and IDLE'.")
        return 1

    root = tk.Tk()
    root.withdraw()

    def report(exc_type, exc, tb):          # any unexpected UI error is logged, not fatal
        logging.error("Unhandled UI error:\n%s", "".join(traceback.format_exception(exc_type, exc, tb)))
    root.report_callback_exception = report

    splash, status = _splash(root, tk)
    started = time.monotonic()
    try:
        status.configure(text="Loading settings, gestures and models…")
        splash.update()
        from core.context import AppContext
        ctx = AppContext.create()
        status.configure(text="Opening dashboard…")
        splash.update()
        from ui.dashboard import Dashboard
    except Exception as exc:  # noqa: BLE001
        splash.destroy()
        messagebox.showerror(APP_NAME, f"The app could not start:\n\n{exc}\n\nSee README > Troubleshooting.")
        logging.exception("startup failed")
        return 1

    def open_main():
        splash.destroy()
        try:
            root.attributes("-alpha", 0.0)
        except tk.TclError:
            pass
        root.deiconify()
        Dashboard(root, ctx)
        _fade_in(root)
        if ctx.startup_notes:
            root.after(400, lambda: messagebox.showwarning(
                APP_NAME, "Some files need attention (the app still works):\n\n• " + "\n• ".join(ctx.startup_notes[:6])))

    root.after(max(0, 1200 - int((time.monotonic() - started) * 1000)), open_main)
    root.mainloop()
    return 0


def _under_streamlit() -> bool:
    """True when started with `streamlit run app.py` (e.g. on Streamlit Community Cloud)."""
    try:
        from streamlit.runtime import exists
    except ImportError:
        return False
    return exists()


if __name__ == "__main__":
    if _under_streamlit():                  # web edition - Tkinter has no screen on a server
        import runpy
        runpy.run_path(str(Path(__file__).with_name("streamlit_app.py")), run_name="__main__")
    else:
        sys.exit(main())
