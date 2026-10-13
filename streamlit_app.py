"""Sign Language Assistant - Streamlit web edition (same look as the desktop app).

Start:   streamlit run streamlit_app.py        (or double-click run_streamlit.bat)
Opens:   http://localhost:8899                 (port is set in .streamlit/config.toml)

Smooth video: Python opens the webcam itself and streams the animated picture to the
page as MJPEG (http://127.0.0.1:8900/stream). Drawing takes a few ms per frame;
MediaPipe + recognition run in their own background thread on the newest frame, so
the picture never waits for the AI. No WebRTC, no STUN/TURN - nothing leaves the PC.
"""
from __future__ import annotations

import html
import json
import random
import sys
import time
import os
from pathlib import Path

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")   # webcam opens in ~2 s, not ~20 s

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from core.constants import MODE_CATEGORIES, MODES, PRIVACY_NOTICE  # noqa: E402
from core.registry import GestureRegistry  # noqa: E402
from core.settings import SettingsStore  # noqa: E402
from sentence.phrase_engine import PhraseEngine  # noqa: E402
from sentence.sentence_builder import SentenceBuilder  # noqa: E402
from ui.theme import THEMES  # noqa: E402
from ui.web_hud import hud_html  # noqa: E402
from vision.live_engine import LiveEngine  # noqa: E402
from vision.mjpeg_stream import CameraStream  # noqa: E402
from vision.shape_library import SHAPE_HINTS  # noqa: E402

T = THEMES["dark"]                       # the original desktop palette (navy + sky blue)
FINGER_HEX = ["#FF69B4", "#FFA03C", "#FADC3C", "#64E678", "#50D2FA"]

st.set_page_config(page_title="Sign Language Assistant", page_icon="🤟", layout="wide",
                   initial_sidebar_state="collapsed")


@st.cache_resource
def load_registry() -> GestureRegistry:
    return GestureRegistry()


STREAM_PORT = 8900


@st.cache_resource
def load_live() -> tuple[LiveEngine, CameraStream]:
    """One engine + one webcam for the whole server (a webcam can only be opened once)."""
    engine = LiveEngine(load_registry(), SettingsStore().load().user_name)
    return engine, CameraStream(engine, STREAM_PORT)


# ===================================================================== session
registry = load_registry()
ss = st.session_state
if "history" not in ss:              # first run of this browser session
    ss.history = []                 # (time, who, text)
    ss.say = (0, "")                # (id, text) for the browser voice
    ss.mark = (0, 0.0)              # (word count before the last add, time) -> typing animation
    ss.flip = (None, 0)             # (last label, parity) -> pop animation
    ss.flash = 0                    # sentence card flash counter
    ss.reply_text = ""
    ss.practice = dict(target="", score=0, streak=0, best=0, tries=0, event="", n=0, last="")
engine, stream = load_live()
if "builder" not in ss:
    ss.builder = SentenceBuilder(PhraseEngine(registry.phrases), lambda: engine.name)
builder: SentenceBuilder = ss.builder


def _iframe(page: str, height: int) -> None:
    """Our own (trusted) HTML in an iframe; st.iframe on new Streamlit, components.html on old."""
    if hasattr(st, "iframe"):
        st.iframe(page, height=height)
    else:
        components.html(page, height=height)


def say(text: str) -> None:
    if text:
        ss.say = (ss.say[0] + 1, text)


# ===================================================================== styles
st.markdown(f"""
<style>
:root {{ --bg:{T['bg']}; --card:{T['card']}; --alt:{T['card_alt']}; --border:{T['border']}; --text:{T['text']};
        --muted:{T['muted']}; --accent:{T['accent']}; --ok:{T['ok']}; --warn:{T['warn']}; --danger:{T['danger']}; }}
.stApp {{ background:var(--bg); }}
.block-container {{ padding-top:1rem; padding-bottom:1rem; max-width:1500px; }}
header[data-testid="stHeader"] {{ background:transparent; }}
footer {{ visibility:hidden; }}
.top {{ display:flex; align-items:center; gap:14px; margin-bottom:6px; }}
.top h1 {{ font-size:1.7rem; font-weight:800; letter-spacing:.5px; color:var(--text); margin:0; }}
.top .priv {{ color:var(--ok); font-weight:700; font-size:.95rem; }}
.banner {{ background:var(--alt); border:1px solid var(--accent); border-radius:10px; padding:8px 14px;
   color:var(--accent); font-weight:700; font-size:.92rem; margin-bottom:10px; animation: fadeIn .9s ease both; }}
@keyframes fadeIn {{ from {{ opacity:0; transform:translateY(-6px); }} to {{ opacity:1; transform:none; }} }}
.card {{ background:var(--card); border:1px solid var(--border); border-radius:18px; padding:14px 16px; margin-bottom:12px; }}
.card h4 {{ margin:0 0 6px 0; font-size:.78rem; letter-spacing:1.2px; color:var(--muted); font-weight:800; }}
.card h4 .r {{ float:right; color:var(--accent); }}
.live {{ color:var(--danger); margin-left:10px; animation: blink 1.4s ease-in-out infinite; }}
@keyframes blink {{ 50% {{ opacity:.25; }} }}
.flashA {{ animation: flashA .8s ease; }} .flashB {{ animation: flashB .8s ease; }}
@keyframes flashA {{ 0%,40% {{ border-color:var(--ok); box-shadow:0 0 0 3px rgba(74,222,128,.35); }} 100% {{ border-color:var(--border); }} }}
@keyframes flashB {{ 0%,40% {{ border-color:var(--ok); box-shadow:0 0 0 3px rgba(74,222,128,.35); }} 100% {{ border-color:var(--border); }} }}
/* detection */
.glabel {{ font-size:2.5rem; font-weight:800; color:var(--text); line-height:1.1; min-height:1.15em; }}
.glabel.long {{ font-size:1.6rem; }} .glabel.idle {{ color:var(--muted); }}
.popA {{ animation: popA .32s cubic-bezier(.3,1.7,.5,1) both; }} .popB {{ animation: popB .32s cubic-bezier(.3,1.7,.5,1) both; }}
@keyframes popA {{ from {{ transform:scale(.55); color:var(--accent); }} to {{ transform:none; }} }}
@keyframes popB {{ from {{ transform:scale(.55); color:var(--accent); }} to {{ transform:none; }} }}
.sub {{ color:var(--accent); font-weight:700; font-style:italic; min-height:1.3em; }}
.mrow {{ display:flex; align-items:center; gap:10px; margin-top:6px; font-size:.82rem; color:var(--text); font-weight:700; }}
.mrow span {{ width:120px; flex:none; }} .mrow span.m {{ color:var(--muted); font-weight:400; }}
.meter {{ flex:1; height:10px; border-radius:99px; background:var(--alt); overflow:hidden; }}
.meter > div {{ height:100%; border-radius:99px; transition: width .3s ease, background .3s; }}
.fingers {{ display:flex; gap:8px; align-items:flex-end; height:62px; margin-top:10px; }}
.finger {{ flex:1; text-align:center; font-size:.7rem; color:var(--muted); font-weight:700; }}
.finger .bar {{ height:44px; background:var(--alt); border-radius:7px; position:relative; overflow:hidden; margin-bottom:3px; }}
.finger .bar > div {{ position:absolute; bottom:0; left:0; right:0; border-radius:7px; transition: height .25s ease; }}
.hint {{ color:var(--warn); font-size:.86rem; margin-top:6px; }} .hint.okk {{ color:var(--muted); }}
.chips {{ margin-top:8px; font-weight:700; font-size:.88rem; }}
.chips span {{ margin-right:16px; }} .on {{ color:var(--ok); }} .off {{ color:var(--muted); }}
.idlecam {{ height:360px; border-radius:12px; background:#000; display:flex; flex-direction:column;
   align-items:center; justify-content:center; color:#D1D5DB; font-size:1.05rem; text-align:center; }}
.idlecam .wave {{ font-size:3rem; margin-bottom:10px; transform-origin:70% 80%; animation: wave 2.2s ease-in-out infinite; }}
@keyframes wave {{ 0%,60%,100% {{ transform:rotate(0); }} 15%,35% {{ transform:rotate(16deg); }} 25% {{ transform:rotate(-10deg); }} }}
.tip {{ color:var(--muted); font-size:.85rem; margin-top:6px; animation: fadeIn .6s ease both; }}
/* sentence */
.sentence {{ font-size:1.45rem; font-weight:800; color:var(--text); min-height:2.6em; line-height:1.3; }}
.sentence .w {{ display:inline-block; margin-right:.3em; }}
.typeA {{ animation: typeA .45s ease both; }} .typeB {{ animation: typeB .45s ease both; }}
@keyframes typeA {{ from {{ opacity:0; transform:translateY(8px); color:var(--ok); }} to {{ opacity:1; transform:none; }} }}
@keyframes typeB {{ from {{ opacity:0; transform:translateY(8px); color:var(--ok); }} to {{ opacity:1; transform:none; }} }}
.caret {{ display:inline-block; width:3px; height:1em; background:var(--accent); vertical-align:-2px; animation: blink 1s steps(2) infinite; }}
.raw {{ color:var(--muted); font-size:.82rem; }}
.person {{ font-size:1.15rem; font-weight:800; color:var(--text); min-height:1.6em; }}
.hist {{ font-size:.86rem; color:var(--text); border-bottom:1px solid var(--alt); padding:4px 0; animation: fadeIn .4s ease both; }}
.hist small {{ color:var(--muted); }}
/* widgets: flat, high-contrast buttons like the desktop app */
.stButton > button {{ background:var(--alt); color:var(--text); border:2px solid var(--border); border-radius:10px;
   font-weight:800; letter-spacing:.3px; transition: border-color .15s, background .15s, transform .1s; }}
.stButton > button:hover {{ border-color:var(--accent); background:#2A3B5A; color:var(--text); }}
.stButton > button:active {{ transform:scale(.97); }}
.stButton > button[kind="primary"] {{ background:var(--accent); color:{T['accent_text']}; border-color:var(--accent); }}
div[data-testid="stSegmentedControl"] button {{ font-weight:800; }}
video {{ border-radius:12px; background:#000; }}
/* camera: animated neon frame */
@property --ang {{ syntax:'<angle>'; initial-value:0deg; inherits:false; }}
.camwrap {{ position:relative; padding:3px; border-radius:15px; animation: spin 7s linear infinite;
   background:conic-gradient(from var(--ang), var(--accent), #A78BFA, #22D3EE, var(--ok), var(--accent));
   box-shadow:0 0 34px rgba(56,189,248,.28); }}
.camwrap img {{ width:100%; display:block; border-radius:12px; background:#000; }}
@keyframes spin {{ to {{ --ang:360deg; }} }}
.card {{ transition: transform .2s, box-shadow .2s; }}
.card:hover {{ box-shadow:0 6px 24px rgba(0,0,0,.35); }}
iframe {{ border:none; }}
/* practice */
.practice .tgt {{ display:flex; align-items:baseline; gap:14px; }}
.practice .tgt b {{ font-size:3rem; font-weight:900; color:var(--accent); text-shadow:0 0 22px rgba(56,189,248,.45);
   animation: breathe 2.4s ease-in-out infinite; display:inline-block; }}
@keyframes breathe {{ 50% {{ transform:scale(1.07); }} }}
.practice .stat {{ display:flex; gap:10px; margin-top:8px; }}
.practice .stat div {{ flex:1; background:var(--alt); border-radius:10px; padding:6px 8px; text-align:center;
   font-size:.72rem; color:var(--muted); font-weight:800; letter-spacing:.8px; }}
.practice .stat div b {{ display:block; font-size:1.25rem; color:var(--text); letter-spacing:0; }}
.pmsg {{ margin-top:8px; font-weight:800; min-height:1.3em; }}
.ok0, .ok1 {{ animation: okpulse .9s ease; }} .ok0 .pmsg, .ok1 .pmsg {{ color:var(--ok); }}
@keyframes okpulse {{ 0%,40% {{ border-color:var(--ok); box-shadow:0 0 0 4px rgba(74,222,128,.4), 0 0 40px rgba(74,222,128,.3); }} }}
.miss0, .miss1 {{ animation: shake .45s ease; }} .miss0 .pmsg, .miss1 .pmsg {{ color:var(--danger); }}
@keyframes shake {{ 20%,60% {{ transform:translateX(-7px); }} 40%,80% {{ transform:translateX(7px); }} }}
/* gesture guide */
.ggrid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(190px, 1fr)); gap:10px; }}
.gcard {{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:10px 12px;
   transition: transform .15s, border-color .15s; animation: fadeIn .4s ease both; }}
.gcard:hover {{ transform:translateY(-3px); border-color:var(--accent); }}
.gcard .gn {{ font-size:1.5rem; font-weight:900; color:var(--accent); }}
.gcard .gh {{ font-size:.82rem; color:var(--text); }} .gcard .gs {{ font-size:.8rem; color:var(--ok); font-style:italic; margin-top:3px; }}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="top"><h1>SIGN LANGUAGE ASSISTANT</h1><span class="priv">🔒 Offline · Private</span></div>'
            '<div class="banner">✨ Smart rule-based recognition is active — no training needed. '
            'Hold a pose until the ring around your hand fills.</div>', unsafe_allow_html=True)


# ===================================================================== settings row
with st.expander("⚙ SETTINGS", expanded=False):
    s1, s2, s3, s4, s5 = st.columns(5)
    name = s1.text_input("Name", value=engine.name)
    hold = s2.slider("Hold time (s)", 0.3, 2.0, engine.hold, 0.1)
    min_conf = s3.slider("Minimum confidence", 0.3, 0.95, engine.min_conf, 0.05)
    with s4:
        mirror = st.toggle("Mirror camera", value=engine.mirror)
        speak_words = st.toggle("Speak words", value=True)
        speak_sentences = st.toggle("Speak sentences", value=True)
    with s5:
        cam_index = int(st.number_input("Camera number", 0, 5, 0, help="0 = built-in webcam, 1 = USB camera"))
    g1, g2, g3 = st.columns(3)
    quality = g1.radio("Graphics", ["max", "balanced", "fast"], horizontal=True,
                       index=["max", "balanced", "fast"].index(engine.quality),
                       format_func=lambda q: {"max": "🔥 Max", "balanced": "⚖ Balanced", "fast": "⚡ Fast"}[q],
                       help="Max = neon glow, hologram, sparks. Fast = for slow PCs.")
    resolution = g2.radio("Camera resolution", ["1280x720", "640x480"], horizontal=True,
                          format_func=lambda r: "HD 720p" if r.startswith("1280") else "SD 480p",
                          help="Used the next time the camera starts.")
    accurate = g3.toggle("Accurate hand model", value=engine.accurate,
                         help="Off = faster MediaPipe model for slow PCs")
    st.caption(PRIVACY_NOTICE)


# ===================================================================== main area
left, right = st.columns([3, 2], gap="medium")

with left:
    st.markdown('<div class="card" style="padding:10px 16px;margin-bottom:6px"><h4 style="margin:0">CAMERA'
                '<span class="live">● LIVE</span><span class="r">the camera is read directly by Python - no browser permission needed</span></h4></div>',
                unsafe_allow_html=True)
    if stream.running:
        st.markdown(f'<div class="camwrap"><img src="http://localhost:{STREAM_PORT}/stream?s={id(stream.camera)}" '
                    'alt="camera"></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="idlecam"><div class="wave">✋</div>Camera is OFF (privacy).<br>'
                    'Press START CAMERA to begin.</div>', unsafe_allow_html=True)
        if stream.error:
            st.warning(stream.error)
    c_start, c_stop, _ = st.columns([1, 1, 2])
    if c_start.button("▶ START CAMERA", type="primary", use_container_width=True, disabled=stream.running):
        with st.spinner("Opening the webcam…"):
            w, h = (int(v) for v in resolution.split("x"))
            ok, msg = stream.start(cam_index, w, h)
        if not ok:
            st.error(msg)
        st.rerun()
    if c_stop.button("■ STOP CAMERA", use_container_width=True, disabled=not stream.running):
        stream.stop()
        st.rerun()

    mode = st.segmented_control("MODE", MODES, default=engine.mode if engine.mode in MODES else "ALPHABET",
                                key="mode_ctl") or "ALPHABET"
    practice = st.toggle("🎯 PRACTICE MODE — the app shows a sign, you make it, it scores you", key="practice_on")

with engine.lock:
    if engine.mode != mode:
        engine.recognizer.reset_motion()
        engine.stabilizer.reset()
    engine.mode, engine.name, engine.hold, engine.min_conf, engine.mirror = mode, name.strip(), hold, min_conf, mirror
    engine.quality, engine.accurate = quality, accurate

TIPS, GUIDE = [], []
for cat in MODE_CATEGORIES.get(mode, []):
    for g in registry.by_category(cat):
        if g.get("shape"):
            how = SHAPE_HINTS.get(g["shape"]) or g.get("instructions", "").split(".")[0]
            output = g["output"].replace("{name}", engine.name or "…") if g.get("output") else ""
            GUIDE.append(dict(gesture=g["gesture"], how=how, said=output))
            TIPS.append(f"✨ Try {g['gesture']}: {how}" + (f" → “{output}”" if output else ""))
GUIDE_BY_NAME = {g["gesture"]: g for g in GUIDE}


def _new_target() -> None:
    pr = ss.practice
    pool = [g["gesture"] for g in GUIDE if g["gesture"] != pr["target"]] or [g["gesture"] for g in GUIDE]
    pr["target"] = random.choice(pool) if pool else ""


if ss.practice["target"] not in GUIDE_BY_NAME:      # first start, or the mode changed
    _new_target()


def _sentence_html(sentence: str) -> str:
    before, at = ss.mark
    fresh = time.monotonic() - at < 1.2
    cls = "typeA" if before % 2 == 0 else "typeB"
    words = []
    for i, w in enumerate(sentence.split()):
        anim = f' {cls}" style="animation-delay:{(i - before) * 0.06:.2f}s' if fresh and i >= before else ""
        words.append(f'<span class="w{anim}">{html.escape(w)}</span>')
    return "".join(words) + '<span class="caret"></span>'


def _meter(label: str, value: float, color: str, muted: bool = False) -> str:
    return (f'<div class="mrow"><span class="{"m" if muted else ""}">{label}</span>'
            f'<div class="meter"><div style="width:{value * 100:.0f}%;background:{color}"></div></div></div>')


def _practice_card() -> None:
    pr = ss.practice
    g = GUIDE_BY_NAME.get(pr["target"], {})
    fresh = time.monotonic() - pr.get("at", 0.0) < 1.2
    anim = f'{pr["event"]}{pr["n"] % 2}' if fresh and pr["event"] else ""
    if fresh and pr["event"] == "ok":
        msg = f"✓ Correct — {html.escape(pr['last'])}! Next one:"
    elif fresh and pr["event"] == "miss":
        msg = f"✗ That was {html.escape(pr['last'])} — try {html.escape(pr['target'])} again"
    else:
        msg = "Make this sign and hold it until the ring fills."
    acc = f"{pr['score'] / pr['tries'] * 100:.0f}%" if pr["tries"] else "—"
    said = f'<div class="sub">→ “{html.escape(g["said"])}”</div>' if g.get("said") else ""
    st.markdown(f"""<div class="card practice {anim}"><h4>🎯 PRACTICE<span class="r">{html.escape(mode)}</span></h4>
      <div class="tgt"><span class="raw">SHOW&nbsp;ME</span><b>{html.escape(pr['target'] or '—')}</b></div>
      <div class="raw" style="color:var(--text)">{html.escape(g.get('how', ''))}</div>{said}
      <div class="pmsg">{msg}</div>
      <div class="stat"><div>SCORE<b>{pr['score']}</b></div><div>STREAK<b>🔥 {pr['streak']}</b></div>
      <div>BEST<b>{pr['best']}</b></div><div>ACCURACY<b>{acc}</b></div></div></div>""", unsafe_allow_html=True)
    p1, p2 = st.columns(2)
    if p1.button("⏭ SKIP SIGN", use_container_width=True):
        pr["streak"] = 0
        _new_target()
    if p2.button("↺ RESET SCORE", use_container_width=True):
        pr.update(score=0, streak=0, tries=0, event="")


def _practice_result(label: str) -> None:
    pr = ss.practice
    pr["tries"], pr["last"], pr["n"], pr["at"] = pr["tries"] + 1, label, pr["n"] + 1, time.monotonic()
    if label == pr["target"]:
        pr["score"], pr["streak"] = pr["score"] + 1, pr["streak"] + 1
        pr["best"], pr["event"] = max(pr["best"], pr["streak"]), "ok"
        if speak_words:
            say("Correct!")
        _new_target()
    else:
        pr["streak"], pr["event"] = 0, "miss"


@st.fragment(run_every=0.25)
def live_panel() -> None:
    snap = engine.snapshot()

    for label, category, output in snap["accepted"]:       # accepted by the AI thread
        if practice:                                       # practice: score it, don't type it
            _practice_result(label)
            continue
        before = len(builder.sentence().split())
        kind = builder.add_gesture(label, category, output)
        ss.mark, ss.flash = (before, time.monotonic()), ss.flash + 1
        if kind == "sentence" and speak_sentences:
            say(output.replace("{name}", engine.name))
        elif kind == "word" and speak_words:
            say(label.lower())

    if practice:
        _practice_card()

    # ---- SENTENCE --------------------------------------------------------------
    sentence = builder.sentence()
    pending = f" · Spelling: <b>{html.escape(builder.pending)}</b> (SPACE finishes the word)" if builder.pending else ""
    flash = ("flashA" if ss.flash % 2 else "flashB") if time.monotonic() - ss.mark[1] < 1.0 else ""
    st.markdown(f"""<div class="card {flash}"><h4>SENTENCE<span class="r" style="color:var(--muted)">
      {html.escape(builder.raw_text())}</span></h4>
      <div class="sentence">{_sentence_html(sentence) if sentence else '<span class="caret"></span>'}</div>
      <div class="raw">{pending or "Gestures you hold steady are added here."}</div></div>""", unsafe_allow_html=True)

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("🔊 SPEAK", use_container_width=True, type="primary"):
        if sentence:
            say(sentence)
            ss.history.append((time.strftime("%H:%M"), "SIGN", sentence))
    if b2.button("␣ SPACE", use_container_width=True):
        word = builder.space()
        if word and speak_words:
            say(word.lower())
    if b3.button("⌫ DELETE", use_container_width=True):
        builder.delete()
    if b4.button("✕ CLEAR", use_container_width=True):
        builder.clear()
    b5, b6 = st.columns(2)
    if b5.button("★ USE AS NAME", use_container_width=True):
        new = builder.use_as_name()
        if new:
            with engine.lock:
                engine.name = new
    if b6.button("MY NAME", use_container_width=True) and engine.name:
        builder.insert_name_phrase(engine.name)

    if snap["error"]:
        st.error(snap["error"])

    sid, text = ss.say                                     # browser voice (Web Speech API)
    if sid:
        _iframe(f"""<script>/* {sid} */ const u = new SpeechSynthesisUtterance({json.dumps(text)});
          speechSynthesis.cancel(); speechSynthesis.speak(u);</script>""", 1)


@st.fragment(run_every=0.5)
def status_chips() -> None:
    snap = engine.snapshot(drain=False)
    running = stream.running
    on = lambda ok: "on" if ok else "off"  # noqa: E731
    stats = (f'<span class="{on(running)}">● Camera: {"Active" if running else "Off"}</span>'
             f'<span class="{on(snap["hands"])}">● Hand: {"Detected" if snap["hands"] else "Not detected"}</span>'
             f'<span class="{on(running)}">● Recognition: {"Active" if running else "Off"}</span>')
    if running:
        cam = stream.camera
        info = f" · {cam.backend} {stream.size[0]}x{stream.size[1]}" if cam else ""
        stats += f'<span class="off">{snap["cam_fps"]:.0f} FPS · AI {snap["ai_fps"]:.0f}/s{info}</span>'
    tip = TIPS[int(time.time() / 5) % len(TIPS)] if TIPS else ""
    st.markdown(f'<div class="chips">{stats}</div><div class="tip">{html.escape(tip)}</div>', unsafe_allow_html=True)


with left:
    status_chips()

with right:
    _iframe(hud_html(T, STREAM_PORT), 372)   # live DETECTION card (browser-side, 20 updates/s)
    live_panel()

# ===================================================================== bottom
pa, hist = st.columns([3, 2], gap="medium")
with pa:
    a, b = st.columns(2)
    a.markdown(f'<div class="card"><h4>PERSON A — SIGN LANGUAGE</h4><div class="raw">Detected:</div>'
               f'<div class="person">{html.escape(builder.raw_text().upper()) or "—"}</div></div>', unsafe_allow_html=True)
    with b:
        st.markdown(f'<div class="card"><h4>PERSON B — VOICE / TEXT</h4><div class="raw">You said:</div>'
                    f'<div class="person">{html.escape(ss.reply_text) or "—"}</div></div>', unsafe_allow_html=True)
        with st.form("person_b", clear_on_submit=True, border=False):
            typed = st.text_input("Type a reply", label_visibility="collapsed", placeholder="Type a reply…")
            if st.form_submit_button("SEND", use_container_width=True) and typed.strip():
                ss.reply_text = typed.strip().upper()
                ss.history.append((time.strftime("%H:%M"), "TEXT", typed.strip()))
                st.rerun()
        quick = st.columns(4)
        for col, text in zip(quick, ("👍 YES", "👎 NO", "🙏 THANK YOU", "⏳ PLEASE WAIT")):
            if col.button(text, use_container_width=True, key=f"quick_{text}"):
                reply = text.split(" ", 1)[1]
                ss.reply_text = reply
                ss.history.append((time.strftime("%H:%M"), "TEXT", reply.capitalize()))
                say(reply.lower())
                st.rerun()
with hist:
    rows = "".join(f'<div class="hist"><small>{t} · {who}</small><br>“{html.escape(s)}”</div>'
                   for t, who, s in reversed(ss.history[-8:])) or '<div class="raw">Nothing yet. Spoken sentences appear here.</div>'
    st.markdown(f'<div class="card"><h4>COMMUNICATION HISTORY<span class="r">{len(ss.history)}</span></h4>{rows}</div>',
                unsafe_allow_html=True)
    if ss.history:
        h1, h2 = st.columns(2)
        h1.download_button("⬇ SAVE CONVERSATION", "\n".join(f"[{t}] {who}: {s}" for t, who, s in ss.history),
                           file_name=f"conversation_{time.strftime('%Y%m%d_%H%M')}.txt", use_container_width=True)
        if h2.button("🗑 CLEAR HISTORY", use_container_width=True):
            ss.history = []
            st.rerun()

with st.expander(f"📖 GESTURE GUIDE — all {len(GUIDE)} signs for {mode} mode"):
    cards = "".join(
        f'<div class="gcard" style="animation-delay:{min(i, 30) * 0.02:.2f}s"><div class="gn">{html.escape(g["gesture"])}</div>'
        f'<div class="gh">{html.escape(g["how"])}</div>'
        + (f'<div class="gs">“{html.escape(g["said"])}”</div>' if g["said"] else "") + "</div>"
        for i, g in enumerate(GUIDE))
    st.markdown(f'<div class="ggrid">{cards}</div>' if cards else "No gestures for this mode.", unsafe_allow_html=True)
