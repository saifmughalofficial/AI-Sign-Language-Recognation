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

async function makeDetector() {
  const vision = await import(MP + "/vision_bundle.mjs");
  const files = await vision.FilesetResolver.forVisionTasks(MP + "/wasm");
  const opts = (delegate) => ({ baseOptions: { modelAssetPath: MODEL, delegate }, runningMode: "VIDEO",
                                numHands: 2, minHandDetectionConfidence: 0.5, minHandPresenceConfidence: 0.5,
                                minTrackingConfidence: 0.5 });
  try { return await vision.HandLandmarker.createFromOptions(files, opts("GPU")); }
  catch (e) { return await vision.HandLandmarker.createFromOptions(files, opts("CPU")); }
}

async function start(S) {
  if (S.busy || S.running) return;
  S.busy = true; paintButton(S);
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia)
      throw { name: "NotSupported", message: "This browser cannot open a camera here (the page must be https)." };
    status(S, "Opening the camera… allow access if the browser asks.");
    S.stream = await navigator.mediaDevices.getUserMedia(
      { video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
    if (!S.video) {
      const v = document.createElement("video");
      v.muted = true; v.playsInline = true; v.setAttribute("playsinline", ""); v.setAttribute("muted", "");
      v.style.cssText = "position:fixed;left:0;top:0;width:2px;height:2px;opacity:0;pointer-events:none;";
      document.body.appendChild(v); S.video = v;
    }
    S.video.srcObject = S.stream;
    await S.video.play();
    if (!S.detector) { status(S, "Loading the hand model (first time ~5 s)…"); S.detector = await makeDetector(); }
    S.running = true; S.lastSend = 0; S.frames = 0; S.fpsT = performance.now(); S.fps = 0;
    status(S, "Camera ON — show your hand.");
    requestAnimationFrame(() => loop(S));
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

function loop(S) {
  if (!S.running) return;
  const v = S.video, c = el(S, "sc-view");
  if (c && v.readyState >= 2 && v.videoWidth) {
    const w = v.videoWidth, h = v.videoHeight;
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    const now = performance.now();
    let res = null;
    if (v.currentTime !== S.lastVideoTime) {
      S.lastVideoTime = v.currentTime;
      try { res = S.detector.detectForVideo(v, now); S.lastRes = res; } catch (e) { res = S.lastRes; }
      S.frames++;
      if (now - S.fpsT > 1000) { S.fps = S.frames * 1000 / (now - S.fpsT); S.frames = 0; S.fpsT = now; }
    } else res = S.lastRes;
    draw(S, c, v, res, w, h);
    if (now - S.lastSend > SEND_MS) {
      S.lastSend = now;
      const lms = (res && res.landmarks) || [], hd = (res && (res.handedness || res.handednesses)) || [];
      send(S, lms.map((lm, i) => ({
        lm: lm.map((p) => [+p.x.toFixed(4), +p.y.toFixed(4), +p.z.toFixed(4)]),
        side: (hd[i] && hd[i][0] && hd[i][0].categoryName) || "Right",
        score: (hd[i] && hd[i][0] && hd[i][0].score) || 1 })), w, h);
    }
  }
  requestAnimationFrame(() => loop(S));
}

function draw(S, c, v, res, w, h) {
  const g = c.getContext("2d"), D = S.data || {}, mirror = D.mirror !== false;
  g.save();
  if (mirror) { g.translate(w, 0); g.scale(-1, 1); }
  g.drawImage(v, 0, 0, w, h);
  for (const lm of (res && res.landmarks) || []) {
    const P = lm.map((p) => [p.x * w, p.y * h]);
    g.lineCap = "round";
    for (const [a, b] of BONES) {
      g.strokeStyle = a === 0 || (a %% 4 === 1 && b %% 4 === 1) ? "#EBEBEB" : FCOL[Math.floor((Math.max(a, b) - 1) / 4)];
      g.lineWidth = 9; g.globalAlpha = .3; g.beginPath(); g.moveTo(...P[a]); g.lineTo(...P[b]); g.stroke();
      g.lineWidth = 3; g.globalAlpha = 1; g.beginPath(); g.moveTo(...P[a]); g.lineTo(...P[b]); g.stroke();
    }
    P.forEach((p, i) => {
      g.fillStyle = "#fff"; g.beginPath(); g.arc(p[0], p[1], i %% 4 === 0 && i ? 6 : 3.5, 0, 7); g.fill();
    });
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
