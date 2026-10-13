"""Live DETECTION card for the Streamlit page.

Runs entirely in the browser: it polls http://localhost:<port>/state (served by
vision/mjpeg_stream.py) 20x per second and animates every value with
requestAnimationFrame, so the card feels as fast as the video - no page reruns.
"""
from __future__ import annotations

import json

FINGER_HEX = ["#FF69B4", "#FFA03C", "#FADC3C", "#64E678", "#50D2FA"]

_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<style>
:root { --bg:%(bg)s; --card:%(card)s; --alt:%(card_alt)s; --border:%(border)s; --text:%(text)s; --muted:%(muted)s;
        --accent:%(accent)s; --ok:%(ok)s; --warn:%(warn)s; --danger:%(danger)s; }
* { box-sizing:border-box; }
html, body { margin:0; background:transparent; color:var(--text); font-family:"Segoe UI", system-ui, sans-serif;
             overflow:hidden; }
.card { position:relative; background:var(--card); border:1px solid var(--border); border-radius:18px;
        padding:14px 16px; height:%(height)dpx; overflow:hidden; transition:border-color .3s, box-shadow .3s; }
.card.hit { border-color:var(--ok); box-shadow:0 0 0 3px rgba(74,222,128,.35), 0 0 28px rgba(74,222,128,.25); }
.card::before { content:""; position:absolute; inset:-40%%; pointer-events:none; opacity:.0; transition:opacity .4s;
        background:radial-gradient(circle at 50%% 50%%, rgba(56,189,248,.18), transparent 55%%); }
.card.live::before { opacity:1; animation:drift 6s ease-in-out infinite alternate; }
@keyframes drift { from { transform:translate(-12%%,-8%%); } to { transform:translate(12%%,8%%); } }
h4 { margin:0 0 8px; font-size:12px; letter-spacing:1.3px; color:var(--muted); font-weight:800; position:relative; }
h4 .r { float:right; color:var(--accent); }
h4 .dot { display:inline-block; width:8px; height:8px; border-radius:50%%; background:var(--muted); margin-right:6px;
          vertical-align:1px; }
.live h4 .dot { background:var(--danger); animation:blink 1.2s ease-in-out infinite; }
@keyframes blink { 50%% { opacity:.25; } }
.row { display:flex; gap:16px; align-items:center; position:relative; }
.ring { width:128px; height:128px; flex:none; position:relative; }
.ring svg { width:100%%; height:100%%; transform:rotate(-90deg); overflow:visible; }
.ring .pct { position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center;
             font-weight:800; font-size:24px; }
.ring .pct small { font-size:10px; letter-spacing:1px; color:var(--muted); font-weight:700; }
.ring.spin .orbit { animation:orbit 1.6s linear infinite; transform-origin:64px 64px; }
@keyframes orbit { to { transform:rotate(360deg); } }
.info { flex:1; min-width:0; }
#label { font-size:46px; font-weight:800; line-height:1.05; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
         min-height:50px; }
#label.long { font-size:28px; } #label.idle { color:var(--muted); }
#label.pop { animation:pop .34s cubic-bezier(.3,1.7,.5,1) both; }
@keyframes pop { from { transform:scale(.5); color:var(--accent); filter:blur(3px); } to { transform:none; filter:none; } }
#sub { color:var(--accent); font-weight:700; font-style:italic; min-height:20px; font-size:15px;
       white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.mrow { display:flex; align-items:center; gap:10px; margin-top:8px; font-size:12.5px; font-weight:700; }
.mrow span { width:118px; flex:none; }
.meter { flex:1; height:10px; border-radius:99px; background:var(--alt); overflow:hidden; position:relative; }
.meter > div { height:100%%; border-radius:99px; width:0; }
.meter > div::after { content:""; position:absolute; inset:0; border-radius:99px;
        background:linear-gradient(90deg, transparent, rgba(255,255,255,.35), transparent); background-size:40%% 100%%;
        background-repeat:no-repeat; animation:shine 1.8s linear infinite; }
@keyframes shine { from { background-position:-40%% 0; } to { background-position:140%% 0; } }
.alts { display:flex; gap:8px; margin-top:10px; min-height:34px; }
.alt { flex:1; background:var(--alt); border-radius:9px; padding:4px 8px; font-size:12px; font-weight:700;
       position:relative; overflow:hidden; transition:opacity .3s; }
.alt i { position:absolute; left:0; top:0; bottom:0; background:rgba(56,189,248,.22); width:0; }
.alt b { position:relative; } .alt em { position:relative; float:right; color:var(--muted); font-style:normal; }
.alt.first i { background:rgba(74,222,128,.28); }
.fingers { display:flex; gap:8px; align-items:flex-end; margin-top:12px; }
.finger { flex:1; text-align:center; font-size:10.5px; color:var(--muted); font-weight:700; letter-spacing:.5px; }
.finger .bar { height:46px; background:var(--alt); border-radius:8px; position:relative; overflow:hidden; margin-bottom:3px; }
.finger .bar > div { position:absolute; bottom:0; left:0; right:0; border-radius:8px; height:0; }
#hint { margin-top:10px; font-size:13.5px; color:var(--warn); min-height:18px; transition:color .3s; }
#hint.okk { color:var(--muted); } #hint.good { color:var(--ok); font-weight:700; }
#stats { position:absolute; right:16px; bottom:10px; font-size:11px; color:var(--muted); }
#toast { position:absolute; left:50%%; top:46%%; transform:translate(-50%%,-50%%) scale(.6); opacity:0; pointer-events:none;
         background:var(--ok); color:#06210F; font-weight:900; font-size:22px; padding:8px 20px; border-radius:99px;
         box-shadow:0 8px 30px rgba(74,222,128,.45); white-space:nowrap; }
#toast.show { animation:toast 1.1s cubic-bezier(.2,1.4,.4,1) both; }
@keyframes toast { 0%% { opacity:0; transform:translate(-50%%,-30%%) scale(.6); } 18%% { opacity:1; transform:translate(-50%%,-50%%) scale(1.08); }
                   70%% { opacity:1; transform:translate(-50%%,-50%%) scale(1); } 100%% { opacity:0; transform:translate(-50%%,-90%%) scale(.95); } }
canvas { position:absolute; inset:0; pointer-events:none; }
</style></head><body>
<div class="card" id="card">
  <canvas id="fx"></canvas>
  <h4><span class="dot"></span>DETECTION<span class="r" id="tag"></span></h4>
  <div class="row">
    <div class="ring" id="ring">
      <svg viewBox="0 0 128 128">
        <defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stop-color="%(accent)s"/><stop offset="1" stop-color="#A78BFA"/></linearGradient></defs>
        <circle cx="64" cy="64" r="54" fill="none" stroke="%(card_alt)s" stroke-width="11"/>
        <circle id="arc" cx="64" cy="64" r="54" fill="none" stroke="url(#g)" stroke-width="11" stroke-linecap="round"
                stroke-dasharray="339.3" stroke-dashoffset="339.3"/>
        <g class="orbit"><circle cx="64" cy="4" r="4" fill="%(accent)s" opacity=".9"/></g>
      </svg>
      <div class="pct"><span id="pct">0%%</span><small id="ringlbl">HOLD</small></div>
    </div>
    <div class="info">
      <div id="label" class="idle">—</div>
      <div id="sub"></div>
      <div class="mrow"><span id="conflbl">Confidence: —</span><div class="meter"><div id="conf"></div></div></div>
      <div class="alts" id="alts"></div>
    </div>
  </div>
  <div class="fingers" id="fingers"></div>
  <div id="hint">Connecting to the camera…</div>
  <div id="stats"></div>
  <div id="toast"></div>
</div>
<script>
const URL_STATE = "http://localhost:%(port)d/state";
const C = %(colors)s, FCOL = %(fingers)s, NAMES = ["THUMB","INDEX","MIDDLE","RING","PINKY"];
const $ = id => document.getElementById(id);
$("fingers").innerHTML = NAMES.map((n, i) =>
  `<div class="finger"><div class="bar"><div id="f${i}" style="background:${FCOL[i]};box-shadow:0 0 12px ${FCOL[i]}88"></div></div>${n}</div>`).join("");
let S = null, lastLabel = undefined, lastSeq = null, failures = 0;
const shown = { conf:0, prog:0, f:[0,0,0,0,0], alts:[0,0,0] };
const lerp = (a, b, k) => a + (b - a) * k;
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

async function poll() {
  try {
    const r = await fetch(URL_STATE, { cache:"no-store" });
    S = await r.json(); failures = 0; apply();
  } catch (e) { if (++failures > 6) { S = null; apply(); } }
  setTimeout(poll, 50);
}

function apply() {
  const card = $("card"), running = S && S.running;
  card.classList.toggle("live", !!running);
  const label = running && S.label ? S.label : null;
  if (label !== lastLabel) {
    const el = $("label");
    el.textContent = label || "—";
    el.className = (label ? "" : "idle") + (label && label.length > 6 ? " long" : "");
    if (label) { void el.offsetWidth; el.classList.add("pop"); }
    lastLabel = label;
  }
  $("sub").textContent = label ? (S.output ? "“" + S.output + "”" : (S.hint_shape || "")) : "";
  $("tag").textContent = label ? (S.dynamic ? "MOTION GESTURE" : "SMART RULES · NO TRAINING") : (S ? S.mode : "");
  $("conflbl").textContent = label ? `Confidence: ${Math.round(S.conf * 100)}%%` : "Confidence: —";
  // "looks like" candidates
  const top = running ? (S.top || []) : [];
  $("alts").innerHTML = top.map((t, i) =>
    `<div class="alt${i === 0 ? " first" : ""}"><i id="a${i}" style="width:${shown.alts[i] * 100}%%"></i>` +
    `<b>${esc(t[0])}</b><em>${Math.round(t[1] * 100)}%%</em></div>`).join("") ||
    `<div class="alt" style="opacity:.45"><b>Looks like…</b><em>—</em></div>`;
  // hint line
  const hint = $("hint");
  if (!S) { hint.textContent = "Camera server not reachable — press ▶ START CAMERA."; hint.className = ""; }
  else if (!running) { hint.textContent = "Camera is OFF (privacy). Press ▶ START CAMERA to begin."; hint.className = ""; }
  else if (S.locked) { hint.textContent = "✓ Accepted — change the pose for the next sign"; hint.className = "good"; }
  else if (label) { hint.textContent = "Hold the pose steady until the ring fills…"; hint.className = "okk"; }
  else if (S.hands) { hint.textContent = "Hand found — this pose is not a gesture in " + S.mode + " mode."; hint.className = ""; }
  else { hint.textContent = "No hand detected — show your hand inside the camera picture."; hint.className = ""; }
  $("stats").textContent = running ? `${Math.round(S.cam_fps)} FPS · AI ${Math.round(S.ai_fps)}/s · ${S.hands} hand${S.hands === 1 ? "" : "s"}` : "";
  $("ring").classList.toggle("spin", !!(running && S.hands && !S.locked));
  // accepted -> celebration
  if (S && lastSeq !== null && S.accept_seq > lastSeq) celebrate(S.last_accepted);
  if (S) lastSeq = S.accept_seq;
}

function frame() {
  const running = S && S.running, label = running && S.label;
  shown.conf = lerp(shown.conf, label ? S.conf : 0, .25);
  shown.prog = lerp(shown.prog, label ? S.progress : 0, .3);
  const fingers = running ? S.fingers : [0,0,0,0,0];
  for (let i = 0; i < 5; i++) {
    shown.f[i] = lerp(shown.f[i], fingers[i] || 0, .3);
    $("f" + i).style.height = (shown.f[i] * 100).toFixed(1) + "%%";
  }
  const top = running ? (S.top || []) : [];
  for (let i = 0; i < 3; i++) {
    shown.alts[i] = lerp(shown.alts[i], top[i] ? top[i][1] : 0, .3);
    const a = $("a" + i); if (a) a.style.width = (shown.alts[i] * 100).toFixed(1) + "%%";
  }
  const conf = $("conf"), minc = S ? S.min_conf : .6;
  conf.style.width = (shown.conf * 100).toFixed(1) + "%%";
  conf.style.background = shown.conf >= .8 ? C.ok : shown.conf >= minc ? C.warn : C.danger;
  const locked = running && S.locked;
  $("arc").style.strokeDashoffset = (339.3 * (1 - (locked ? 1 : shown.prog))).toFixed(1);
  $("arc").setAttribute("stroke", locked ? C.ok : "url(#g)");
  $("pct").textContent = locked ? "✓" : Math.round(shown.prog * 100) + "%%";
  $("pct").style.color = locked ? C.ok : C.text;
  $("ringlbl").textContent = locked ? "ADDED" : "HOLD";
  drawParticles();
  requestAnimationFrame(frame);
}

// ---- particles ---------------------------------------------------------------------------
const cv = $("fx"), ctx = cv.getContext("2d"); let parts = [];
function resize() { cv.width = cv.offsetWidth * devicePixelRatio; cv.height = cv.offsetHeight * devicePixelRatio; }
addEventListener("resize", resize); resize();
function celebrate(text) {
  const card = $("card"); card.classList.remove("hit"); void card.offsetWidth; card.classList.add("hit");
  setTimeout(() => card.classList.remove("hit"), 900);
  const t = $("toast"); t.textContent = "✓ " + text; t.classList.remove("show"); void t.offsetWidth; t.classList.add("show");
  const r = $("ring").getBoundingClientRect(), cx = (r.left + r.width / 2) * devicePixelRatio,
        cy = (r.top + r.height / 2) * devicePixelRatio;
  const cols = [C.ok, C.accent, "#A78BFA", "#FADC3C", "#FF69B4"];
  for (let i = 0; i < 90; i++) {
    const a = Math.random() * Math.PI * 2, s = (2 + Math.random() * 7) * devicePixelRatio;
    parts.push({ x:cx, y:cy, vx:Math.cos(a) * s, vy:Math.sin(a) * s - 2 * devicePixelRatio, life:1,
                 r:(2 + Math.random() * 3.5) * devicePixelRatio, c:cols[i %% cols.length], rot:Math.random() * 6 });
  }
}
function drawParticles() {
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!parts.length) return;
  parts = parts.filter(p => p.life > 0);
  for (const p of parts) {
    p.x += p.vx; p.y += p.vy; p.vy += .18 * devicePixelRatio; p.vx *= .985; p.life -= .016; p.rot += .2;
    ctx.globalAlpha = Math.max(p.life, 0); ctx.fillStyle = p.c;
    ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(p.rot); ctx.fillRect(-p.r, -p.r / 2, p.r * 2, p.r); ctx.restore();
  }
  ctx.globalAlpha = 1;
}
poll(); requestAnimationFrame(frame);
</script></body></html>"""


def hud_html(theme: dict, port: int, height: int = 372) -> str:
    keys = ("bg", "card", "card_alt", "border", "text", "muted", "accent", "ok", "warn", "danger")
    values = {k: theme[k] for k in keys}
    return _TEMPLATE % dict(values, port=port, height=height - 4, colors=json.dumps(values),
                            fingers=json.dumps(FINGER_HEX))
