"""Browser camera for the cloud edition - no WebRTC, no STUN/TURN.

The visitor's browser opens the webcam, runs MediaPipe hand detection itself
(MediaPipe Tasks for the web) and draws the video. Only the 21 hand points per hand
are sent to Python (~8 times a second), where the normal rule-based recognizer runs.
Works on phones and on networks that block video calls, because no video is uploaded.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from vision.hand_detector import DetectedHand

MP_VERSION = "0.10.14"     # @mediapipe/tasks-vision on jsDelivr

_HTML = """
<div class="sc">
  <canvas class="sc-view" width="640" height="480"></canvas>
  <div class="sc-bar">
    <button class="sc-btn" type="button">▶ START CAMERA</button>
    <span class="sc-status">Camera is OFF (privacy). Press START CAMERA.</span>
  </div>
</div>
"""

_CSS = """
.sc { font-family: "Source Sans Pro", "Segoe UI", system-ui, sans-serif; }
.sc-view { width: 100%; aspect-ratio: 4 / 3; display: block; border-radius: 14px; background: #000;
           box-shadow: 0 0 0 3px #38BDF8, 0 0 34px rgba(56,189,248,.28); }
.sc-bar { display: flex; align-items: center; gap: 12px; margin-top: 10px; flex-wrap: wrap; }
.sc-btn { background: #38BDF8; color: #06121F; border: 0; border-radius: 10px; padding: 10px 18px;
          font-weight: 800; font-size: 15px; letter-spacing: .3px; cursor: pointer; }
.sc-btn.on { background: #22304A; color: #F8FAFC; border: 2px solid #3B4C6B; }
.sc-status { color: #B6C2D6; font-size: 14px; }
"""

_JS = r"""
const MP = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@%(ver)s";
const MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const BONES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],
               [14,15],[15,16],[13,17],[0,17],[17,18],[18,19],[19,20]];
const FCOL = ["#FF69B4", "#FFA03C", "#FADC3C", "#64E678", "#50D2FA"];
const SEND_MS = 120;

function el(S, cls) { return S.parent ? S.parent.querySelector("." + cls) : null; }
function status(S, text) { S.statusText = text; const s = el(S, "sc-status"); if (s) s.textContent = text; }
function paintButton(S) {
  const b = el(S, "sc-btn"); if (!b) return;
  b.textContent = S.running ? "■ STOP CAMERA" : (S.busy ? "… starting" : "▶ START CAMERA");
  b.classList.toggle("on", S.running);
}

async function makeDetector(S) {
  const vision = await import(MP + "/vision_bundle.mjs");
  const files = await vision.FilesetResolver.forVisionTasks(MP + "/wasm");
  const opts = (delegate) => ({ baseOptions: { modelAssetPath: MODEL, delegate }, runningMode: "VIDEO",
                                numHands: 2, minHandDetectionConfidence: 0.5, minHandPresenceConfidence: 0.5,
                                minTrackingConfidence: 0.5 });
  try { const d = await vision.HandLandmarker.createFromOptions(files, opts("GPU")); S.delegate = "GPU"; return d; }
  catch (e) { S.delegate = "CPU"; return await vision.HandLandmarker.createFromOptions(files, opts("CPU")); }
}

// Runs inside a Web Worker: the hand AI never blocks the video thread (smooth video on phones).
function workerMain() {
  let det = null;
  self.onmessage = async (ev) => {
    const m = ev.data;
    if (m.type === "init") {
      try {
        const v = await import(m.mp + "/vision_bundle.mjs");
        const files = await v.FilesetResolver.forVisionTasks(m.mp + "/wasm");
        const opts = (d) => ({ baseOptions: { modelAssetPath: m.model, delegate: d }, runningMode: "VIDEO",
                               numHands: 2, minHandDetectionConfidence: 0.5, minHandPresenceConfidence: 0.5,
                               minTrackingConfidence: 0.5 });
        let delegate = "GPU";
        try { det = await v.HandLandmarker.createFromOptions(files, opts("GPU")); }
        catch (e) { delegate = "CPU"; det = await v.HandLandmarker.createFromOptions(files, opts("CPU")); }
        try { det.detectForVideo(new ImageData(320, 240), 1); } catch (e) { /* warm-up only */ }
        self.postMessage({ type: "ready", delegate });
      } catch (e) { self.postMessage({ type: "error", message: String((e && e.message) || e) }); }
    } else if (m.type === "frame") {
      const out = { type: "result", hands: [] };
      try {
        const r = det.detectForVideo(m.bitmap, m.ts);
        const hd = r.handedness || r.handednesses || [];
        out.hands = (r.landmarks || []).map((lm, i) => ({ lm: lm.map((p) => ({ x: p.x, y: p.y, z: p.z })),
          side: (hd[i] && hd[i][0] && hd[i][0].categoryName) || "Right",
          score: (hd[i] && hd[i][0] && hd[i][0].score) || 1 }));
      } catch (e) { out.error = String(e); }
      if (m.bitmap && m.bitmap.close) m.bitmap.close();
      self.postMessage(out);
    }
  };
}

function makeWorker(S) {           // resolves to a ready Worker, or null -> use the main-thread detector
  return new Promise((resolve) => {
    let w = null;
    const fail = () => { clearTimeout(timer); if (w) w.terminate(); resolve(null); };
    const timer = setTimeout(fail, 90000);
    try {
      const url = URL.createObjectURL(new Blob(["(" + workerMain.toString() + ")()"], { type: "text/javascript" }));
      w = new Worker(url);
    } catch (e) { fail(); return; }
    w.onerror = fail;
    w.onmessage = (ev) => {
      if (ev.data.type !== "ready") { fail(); return; }
      clearTimeout(timer);
      S.delegate = ev.data.delegate + " worker";
      w.onmessage = (e2) => onWorkerResult(S, e2.data);
      w.onerror = () => { S.worker = null; S.workerBusy = false; };   // crashed later: main-thread fallback
      resolve(w);
    };
    w.postMessage({ type: "init", mp: MP, model: MODEL });
  });
}

function onWorkerResult(S, d) {
  S.workerBusy = false;
  if (!S.running || d.type !== "result") return;
  S.hands = d.hands;
  if (S.shown.length !== S.hands.length) S.shown = [];
  S.aiN++;
  sendHands(S);
}

// Python writes its latest results into a hidden <div class="sign-state"> (see streamlit_app.py)
function readState(S) {
  const e = document.querySelector(".sign-state");
  if (!e || e.textContent === S.stateText) return;
  S.stateText = e.textContent;
  try { S.data = Object.assign({}, S.data, JSON.parse(e.textContent)); } catch (err) { /* half-rendered */ }
}

function sendHands(S) {
  const now = performance.now(), v = S.video;
  if (now - S.lastSend <= SEND_MS) return;
  // pacing: wait until the server confirmed the last message (its rerun writes `ack`), so reruns
  // never pile up and interrupt each other; resend after 1.5 s in case a message was lost
  readState(S);
  if ((S.data.ack || 0) < (S.seq || 0) && now - S.lastSend < 1500) return;
  S.lastSend = now;
  send(S, S.hands.map((hand) => ({ side: hand.side, score: hand.score,
    lm: hand.lm.map((p) => [+p.x.toFixed(4), +p.y.toFixed(4), +p.z.toFixed(4)]) })), v.videoWidth, v.videoHeight);
}

async function start(S) {
  if (S.busy || S.running) return;
  S.busy = true; paintButton(S);
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia)
      throw { name: "NotSupported", message: "This browser cannot open a camera here (the page must be https)." };
    status(S, "Opening the camera… allow access if the browser asks.");
    const want = { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } };
    try {   // ask for 60 FPS; phones that can't do it fall back to their best rate
      S.stream = await navigator.mediaDevices.getUserMedia({ video: { ...want, frameRate: { ideal: 60 } }, audio: false });
    } catch (e) {
      if (e && (e.name === "NotAllowedError" || e.name === "NotReadableError")) throw e;
      S.stream = await navigator.mediaDevices.getUserMedia({ video: want, audio: false });
    }
    if (!S.video) {
      const v = document.createElement("video");
      v.muted = true; v.playsInline = true; v.setAttribute("playsinline", ""); v.setAttribute("muted", "");
      v.style.cssText = "position:fixed;left:0;top:0;width:2px;height:2px;opacity:0;pointer-events:none;";
      document.body.appendChild(v); S.video = v;
    }
    S.video.srcObject = S.stream;
    await S.video.play();
    if (!S.worker && !S.detector) {
      status(S, "Loading the hand model (first time ~5-20 s)…");
      S.worker = await makeWorker(S);
    }
    if (!S.worker && !S.detector) {                       // no worker support: run the AI on this thread
      status(S, "Loading the hand model…");
      S.detector = await makeDetector(S);
      status(S, "Warming up the hand AI…");               // the first run compiles GPU shaders (slow):
      await new Promise((r) => setTimeout(r, 30));        // do it now, not while the video is playing
      const warm = document.createElement("canvas"); warm.width = 320; warm.height = 240;
      try { S.detector.detectForVideo(warm, performance.now()); } catch (e) { /* ignore */ }
    }
    S.running = true; S.lastSend = 0; S.hands = []; S.shown = [];
    S.vidN = 0; S.aiN = 0; S.statT = performance.now(); S.fps = 0; S.aiFps = 0;
    status(S, "Camera ON — show your hand.");
    requestAnimationFrame(() => renderLoop(S));
    setTimeout(() => detectLoop(S), 0);
  } catch (e) {
    stopTracks(S);
    const hint = e && e.name === "NotAllowedError" ? "Camera permission was denied — allow it in the address bar and press START again."
               : e && e.name === "NotReadableError" ? "The camera is used by another app (Zoom, WhatsApp, another tab…). Close it and press START again."
               : e && e.name === "NotFoundError" ? "No camera was found on this device."
               : "Could not start: " + ((e && (e.message || e.name)) || e);
    status(S, hint);
  } finally { S.busy = false; paintButton(S); }
}

function stopTracks(S) {
  if (S.stream) S.stream.getTracks().forEach((t) => t.stop());
  S.stream = null;
}

function stop(S) {
  S.running = false; stopTracks(S); paintButton(S);
  send(S, []);
  status(S, "Camera is OFF (privacy). Press START CAMERA.");
  drawIdle(S);
}

function send(S, hands, w, h) {
  if (!S.trigger) return;
  S.seq = (S.seq || 0) + 1;
  S.trigger("frame", { seq: S.seq, running: S.running, w: w || 640, h: h || 480, fps: Math.round(S.fps || 0), hands });
}

function drawIdle(S) {
  const c = el(S, "sc-view"); if (!c) return;
  const g = c.getContext("2d"); g.fillStyle = "#000"; g.fillRect(0, 0, c.width, c.height);
  g.fillStyle = "#D1D5DB"; g.font = "bold 22px sans-serif"; g.textAlign = "center";
  g.fillText("✋  Camera is OFF", c.width / 2, c.height / 2 - 8);
  g.font = "16px sans-serif"; g.fillText("Press START CAMERA to begin", c.width / 2, c.height / 2 + 22);
}

// ---- 1) VIDEO: drawn on every screen refresh (up to 60 FPS), never waits for the AI -------------
function renderLoop(S) {
  if (!S.running) return;
  const v = S.video, c = el(S, "sc-view");
  if (c && v.readyState >= 2 && v.videoWidth) {
    const w = v.videoWidth, h = v.videoHeight;
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    if (v.currentTime !== S.lastDrawnTime) { S.lastDrawnTime = v.currentTime; S.vidN++; }
    // skeleton glides towards the newest detection -> smooth even when the AI runs slower than the video
    S.shown = S.hands.map((hand, i) => {
      const prev = S.shown[i];
      return hand.lm.map((p, j) => prev ? { x: prev[j].x + (p.x - prev[j].x) * 0.55, y: prev[j].y + (p.y - prev[j].y) * 0.55 }
                                        : { x: p.x, y: p.y });
    });
    readState(S);
    draw(S, c, v, S.shown, w, h);
  }
  const now = performance.now();
  if (now - S.statT >= 1000) {
    S.fps = S.vidN * 1000 / (now - S.statT); S.aiFps = S.aiN * 1000 / (now - S.statT);
    S.vidN = 0; S.aiN = 0; S.statT = now;
    status(S, `Camera ON · video ${Math.round(S.fps)} FPS · hand AI ${Math.round(S.aiFps)}/s (${S.delegate || "…"})`);
  }
  requestAnimationFrame(() => renderLoop(S));
}

// ---- 2) HAND AI: on a small 320 px copy, paced so the video always gets most of the time --------
function smallFrame(S, v) {
  const sw = 320, sh = Math.round(320 * v.videoHeight / v.videoWidth);
  if (!S.small) S.small = document.createElement("canvas");
  if (S.small.width !== sw || S.small.height !== sh) { S.small.width = sw; S.small.height = sh; }
  S.small.getContext("2d").drawImage(v, 0, 0, sw, sh);
  return S.small;
}

function detectLoop(S) {
  if (!S.running) return;
  const v = S.video, t0 = performance.now();
  if (S.worker) {                                     // worker: hand the frame over and keep drawing
    if (S.workerBusy && t0 - S.workerSentAt > 3000) S.workerBusy = false;     // lost frame guard
    if (!S.workerBusy && v.readyState >= 2 && v.videoWidth && v.currentTime !== S.lastAiTime) {
      S.lastAiTime = v.currentTime; S.workerBusy = true; S.workerSentAt = t0;
      createImageBitmap(smallFrame(S, v)).then((bmp) => {
        if (!S.worker) { bmp.close(); S.workerBusy = false; return; }
        S.worker.postMessage({ type: "frame", bitmap: bmp, ts: performance.now() }, [bmp]);
      }).catch(() => { S.workerBusy = false; });
    }
    setTimeout(() => detectLoop(S), 30);
    return;
  }
  if (!S.detector) {                                  // the worker crashed: switch to this thread once
    if (!S.loadingDetector) {
      S.loadingDetector = true;
      makeDetector(S).then((d) => { S.detector = d; }).finally(() => { S.loadingDetector = false; });
    }
    setTimeout(() => detectLoop(S), 200);
    return;
  }
  if (v.readyState >= 2 && v.videoWidth && v.currentTime !== S.lastAiTime) {
    S.lastAiTime = v.currentTime;
    smallFrame(S, v);
    try {
      const res = S.detector.detectForVideo(S.small, t0);
      const hd = res.handedness || res.handednesses || [];
      S.hands = (res.landmarks || []).map((lm, i) => ({ lm,
        side: (hd[i] && hd[i][0] && hd[i][0].categoryName) || "Right",
        score: (hd[i] && hd[i][0] && hd[i][0].score) || 1 }));
      if (S.shown.length !== S.hands.length) S.shown = [];
    } catch (e) { /* one bad frame: keep the last hands */ }
    S.aiN++;
    sendHands(S);
  }
  const took = performance.now() - t0;
  // at most ~30 checks/s; if the device is slow, rest at least as long as the AI took
  setTimeout(() => detectLoop(S), Math.max(33 - took, took, 4));
}

function draw(S, c, v, hands, w, h) {
  const g = c.getContext("2d"), D = S.data || {}, mirror = D.mirror !== false;
  g.save();
  if (mirror) { g.translate(w, 0); g.scale(-1, 1); }
  g.drawImage(v, 0, 0, w, h);
  g.lineCap = "round";
  for (const lm of hands) {
    const P = lm.map((p) => [p.x * w, p.y * h]);
    for (const [a, b] of BONES) {
      g.strokeStyle = a === 0 || (a %% 4 === 1 && b %% 4 === 1) ? "#EBEBEB" : FCOL[Math.floor((Math.max(a, b) - 1) / 4)];
      g.lineWidth = 9; g.globalAlpha = .3; g.beginPath(); g.moveTo(...P[a]); g.lineTo(...P[b]); g.stroke();
      g.lineWidth = 3; g.globalAlpha = 1; g.beginPath(); g.moveTo(...P[a]); g.lineTo(...P[b]); g.stroke();
    }
    g.fillStyle = "#fff";
    P.forEach((p, i) => { g.beginPath(); g.arc(p[0], p[1], i %% 4 === 0 && i ? 6 : 3.5, 0, 7); g.fill(); });
  }
  g.restore();
  // label + hold bar from Python's recognizer
  if (D.label) {
    const txt = D.label, prog = D.locked ? 1 : (D.progress || 0);
    g.font = "bold 34px sans-serif"; const tw = g.measureText(txt).width;
    g.fillStyle = "rgba(11,18,32,.78)"; g.fillRect(14, 14, tw + 32, 56);
    g.fillStyle = D.locked ? "#4ADE80" : "#F8FAFC"; g.textAlign = "left"; g.fillText(txt, 30, 56);
    g.fillStyle = "rgba(11,18,32,.78)"; g.fillRect(14, h - 26, w - 28, 12);
    g.fillStyle = D.locked ? "#4ADE80" : "#38BDF8"; g.fillRect(14, h - 26, (w - 28) * prog, 12);
  }
}

export default function (component) {
  const { parentElement, data, setStateValue } = component;
  const S = (window.__signCam = window.__signCam || {});
  S.parent = parentElement; S.data = data || {}; S.trigger = setStateValue;
  const btn = el(S, "sc-btn");
  if (btn && !btn.__wired) { btn.__wired = true; btn.addEventListener("click", () => (S.running ? stop(S) : start(S))); }
  paintButton(S);
  if (S.statusText) status(S, S.statusText);
  if (!S.running) drawIdle(S);
}
""" % {"ver": MP_VERSION}

_COMPONENT = st.components.v2.component("sign_browser_cam", html=_HTML, css=_CSS, js=_JS)


def browser_camera(data: dict) -> tuple[dict | None, list[DetectedHand]]:
    """Show the camera; returns (the newest message from the browser or None, hands in it)."""
    result = _COMPONENT(key="sign_browser_cam", data=data, on_frame_change=lambda: None)
    frame = result.get("frame") if hasattr(result, "get") else getattr(result, "frame", None)
    if not frame:
        return None, []
    w, h = frame.get("w") or 640, frame.get("h") or 480
    hands = []
    for hand in frame.get("hands") or []:
        pts = np.asarray(hand.get("lm") or [], dtype=np.float32)
        if pts.shape == (21, 3):
            hands.append(DetectedHand(pts, hand.get("side") or "Right", float(hand.get("score") or 1.0), w / h))
    return frame, hands
