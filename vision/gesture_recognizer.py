"""Gesture recognition: trained models, rule-based shapes and motion gestures.

Three recognisers live here:

* ModelBundle / GestureRecognizer - real, trained scikit-learn models (./models/*.pkl)
* RuleRecognizer                  - NO training: measures the MediaPipe landmarks
                                    (joint angles, fingertip contact, spread, direction)
                                    and scores the shapes in vision/shape_library.py.
                                    Covers A-Z, 0-9, words and sentences out of the box.
* DynamicGestureRecognizer        - J and Z (and your own recorded motions) from a
                                    sequence of frames.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.constants import (DYNAMIC_LETTERS, FEATURE_VERSION, MODE_CATEGORIES,
                            MODEL_FILES, SEQUENCE_LENGTH)
from core.paths import MODELS_DIR
from vision.feature_extractor import (extract_features, finger_states, normalize_landmarks,
                                      pose_string, sequence_features)
from vision.shape_library import analyze_hands, score_shapes

log = logging.getLogger(__name__)


@dataclass
class Prediction:
    label: str
    confidence: float          # 0..1
    category: str              # alphabet | number | word | sentence | custom
    source: str = "model"      # model | rules | demo | dynamic
    dynamic: bool = False
    shape: str = ""            # rule-based shape name (e.g. "W")
    output: str = ""           # sentence gestures: the sentence to add

    @property
    def is_rule_based(self) -> bool:
        """True when no trained model was involved."""
        return self.source in ("rules", "demo")

    # kept for older code: "demo" used to mean "not a trained model"
    is_demo = is_rule_based


# =============================================================== trained models
@dataclass
class ModelBundle:
    category: str
    model: object              # sklearn Pipeline(StandardScaler + classifier)
    labels: list
    two_hand: bool
    n_features: int
    path: Path
    meta: dict


def load_bundle(path: Path, category: str) -> tuple[ModelBundle | None, str]:
    """Load one .pkl file. Returns (bundle, error_message)."""
    try:
        import joblib
        data = joblib.load(path)
        if data.get("feature_version") != FEATURE_VERSION:
            return None, f"{path.name}: made with a different feature version - please retrain."
        model = data["model"]
        if not hasattr(model, "predict_proba"):
            return None, f"{path.name}: model cannot give probabilities."
        return ModelBundle(category, model, list(data["labels"]), bool(data.get("two_hand", False)),
                           int(data["n_features"]), path, data), ""
    except Exception as exc:  # noqa: BLE001 - corrupt file, version mismatch, missing joblib...
        return None, f"{path.name} could not be loaded ({exc})."


# =============================================================== rule-based
class RuleRecognizer:
    """Scores every hand shape that data/gestures.json uses in the wanted categories."""

    MIN_SCORE = 0.5

    def __init__(self, registry):
        self.maps = {cat: registry.shape_map(cat)
                     for cat in ("alphabet", "number", "word", "sentence", "custom")}
        self.top: list[tuple[str, float]] = []     # best 3 (gesture, score) of the last frame

    def recognize(self, hands, categories: list[str]) -> Prediction | None:
        self.top = []
        if not hands:
            return None
        wanted = {shape for cat in categories for shape in self.maps.get(cat, {})}
        if not wanted:
            return None
        scores = score_shapes(analyze_hands(hands), wanted)
        best = None
        ranked: dict[str, float] = {}
        for cat in categories:                      # earlier category wins a tie
            for shape, entry in self.maps.get(cat, {}).items():
                score = scores.get(shape)
                if score is not None:
                    ranked[entry["gesture"]] = max(ranked.get(entry["gesture"], 0.0), float(score))
                if score is not None and (best is None or score > best[0] + 1e-6):
                    best = (score, entry, cat, shape)
        self.top = sorted(ranked.items(), key=lambda kv: -kv[1])[:3]   # for the "looks like" bars
        if best is None or best[0] < self.MIN_SCORE:
            return None
        score, entry, cat, shape = best
        return Prediction(entry["gesture"], float(score), cat, source="rules",
                          shape=shape, output=entry.get("output", ""))


DemoRecognizer = RuleRecognizer        # old name


class Smoother:
    """Majority vote over the last few frames: kills single-frame flicker and makes
    the confidence drop smoothly while the hand changes shape."""

    def __init__(self, size: int = 5):
        self.window: deque = deque(maxlen=size)

    def reset(self) -> None:
        self.window.clear()

    def update(self, pred: Prediction | None) -> Prediction | None:
        self.window.append(pred)
        totals: dict[tuple, float] = {}
        latest: dict[tuple, Prediction] = {}
        for p in self.window:
            if p is not None:
                key = (p.label, p.category)
                totals[key] = totals.get(key, 0.0) + p.confidence
                latest[key] = p
        if not totals:
            return None
        key = max(totals, key=totals.get)
        if pred is None or (pred.label, pred.category) != key:
            if len(self.window) > 1 and totals[key] / len(self.window) < 0.35:
                return None
        best = latest[key]
        return Prediction(best.label, totals[key] / len(self.window), best.category, best.source,
                          best.dynamic, best.shape, best.output)


# ================================================================ motion gestures
def _resample_path(path: np.ndarray, n: int = 24) -> np.ndarray:
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < 1e-9:
        return np.repeat(path[:1], n, axis=0)
    targets = np.linspace(0, cum[-1], n)
    return np.stack([np.interp(targets, cum, path[:, 0]), np.interp(targets, cum, path[:, 1])], axis=1)


def _normalize_path(path: np.ndarray) -> np.ndarray:
    path = path - path.min(axis=0)
    size = max(float(path.max()), 1e-6)
    return path / size


def _polyline(points: list[tuple[float, float]]) -> np.ndarray:
    return _normalize_path(_resample_path(np.asarray(points, dtype=np.float32)))


# Templates drawn as the viewer sees the (mirrored) preview; y grows downward.
_TEMPLATES = {
    "Z": {"tip": 8, "pose": "x1000",
          "shape": _polyline([(0, 0), (1, 0), (0, 1), (1, 1)])},
    "J": {"tip": 20, "pose": "x0001",
          "shape": _polyline([(0.75, 0), (0.75, 0.75), (0.55, 1.0), (0.25, 0.9), (0.15, 0.65)])},
}


def _pose_matches(states_str: str, wanted: str) -> bool:
    return all(w == "x" or w == s for s, w in zip(states_str, wanted))


class DynamicGestureRecognizer:
    """EXPERIMENTAL. Watches the last ~2 seconds of hand positions.

    * built-in path templates for J (pinky) and Z (index finger)
    * OR your own trained sequence model (models/dynamic_model.pkl) recorded with
      training/record_custom_gesture.py
    """

    WINDOW_SECONDS = 2.0
    MIN_FRAMES = 12
    COOLDOWN_SECONDS = 1.2
    MIN_TRAVEL = 0.9           # path must be at least this many "hand sizes" long
    ACCEPT = 0.55

    def __init__(self, bundle: ModelBundle | None = None):
        self.bundle = bundle
        self.frames: deque = deque()          # (time, landmarks(21,3), handedness)
        self._cooldown_until = 0.0

    def reset(self) -> None:
        self.frames.clear()

    def _hand_size(self, pts: np.ndarray) -> float:
        return max(float(np.linalg.norm(pts[0][:2] - pts[9][:2])), 1e-3)

    def is_moving(self) -> bool:
        """True when the hand has recently travelled a lot (so static output is suppressed)."""
        if len(self.frames) < 6:
            return False
        recent = [f for f in self.frames if f[0] >= self.frames[-1][0] - 0.5]
        if len(recent) < 4:
            return False
        wrist = np.array([f[1][0][:2] for f in recent])
        travel = float(np.linalg.norm(wrist.max(axis=0) - wrist.min(axis=0)))
        tip_paths = np.array([f[1][8][:2] for f in recent])
        tip_travel = float(np.linalg.norm(tip_paths.max(axis=0) - tip_paths.min(axis=0)))
        size = self._hand_size(recent[-1][1])
        return max(travel, tip_travel) / size > 0.85      # ignore the normal small wobble of a held hand

    def update(self, hands, now: float | None = None) -> Prediction | None:
        now = time.monotonic() if now is None else now
        if not hands:
            self.reset()
            return None
        hand = max(hands, key=lambda h: h.score)
        self.frames.append((now, hand.landmarks.copy(), hand.handedness))
        while self.frames and now - self.frames[0][0] > self.WINDOW_SECONDS:
            self.frames.popleft()
        if now < self._cooldown_until or len(self.frames) < self.MIN_FRAMES:
            return None
        pred = self._classify_with_model() if self.bundle else self._classify_templates()
        if pred:
            self._cooldown_until = now + self.COOLDOWN_SECONDS
            self.reset()
        return pred

    # -- built-in templates ----------------------------------------------------
    def _classify_templates(self) -> Prediction | None:
        frames = list(self.frames)
        size = float(np.mean([self._hand_size(f[1]) for f in frames]))
        best: tuple[float, str] | None = None
        for label, spec in _TEMPLATES.items():
            # hand shape must match for most of the frames
            ok = sum(_pose_matches(pose_string(finger_states(f[1])), spec["pose"]) for f in frames)
            if ok < 0.6 * len(frames):
                continue
            path = np.array([[1.0 - f[1][spec["tip"]][0], f[1][spec["tip"]][1]] for f in frames],
                            dtype=np.float32)          # x mirrored: same as the preview
            extent = float(np.linalg.norm(path.max(axis=0) - path.min(axis=0)))
            if extent / size < self.MIN_TRAVEL:
                continue
            probe = _normalize_path(_resample_path(path))
            mirrored_template = spec["shape"].copy()
            mirrored_template[:, 0] = 1.0 - mirrored_template[:, 0]
            dist = min(float(np.mean(np.linalg.norm(probe - t, axis=1)))
                       for t in (spec["shape"], mirrored_template))
            conf = max(0.0, 1.0 - dist / 0.30)
            if conf >= self.ACCEPT and (best is None or conf > best[0]):
                best = (conf, label)
        if best:
            return Prediction(best[1], best[0], "alphabet", source="dynamic", dynamic=True)
        return None

    # -- trained sequence model ------------------------------------------------
    def _classify_with_model(self) -> Prediction | None:
        frames = [f[1] for f in self.frames]
        feats = sequence_features(frames, self.frames[-1][2], SEQUENCE_LENGTH)
        if feats is None or len(feats) != self.bundle.n_features:
            return None
        if not self.is_moving():
            return None
        proba = self.bundle.model.predict_proba(feats.reshape(1, -1))[0]
        i = int(np.argmax(proba))
        label = str(self.bundle.labels[i])
        if label.upper() in ("NONE", "NOTHING", "IDLE") or proba[i] < 0.6:
            return None
        category = "alphabet" if label.upper() in DYNAMIC_LETTERS else "custom"
        return Prediction(label.upper(), float(proba[i]), category, source="dynamic", dynamic=True)


# ============================================================== main recogniser
class GestureRecognizer:
    def __init__(self, registry, models_dir: Path | str = MODELS_DIR):
        self.registry = registry
        self.models_dir = Path(models_dir)
        self.bundles: dict[str, ModelBundle] = {}
        self.load_errors: list[str] = []
        self.rules = RuleRecognizer(registry)
        self.smoother = Smoother()
        self._last_mode = ""
        self.dynamic = DynamicGestureRecognizer()
        self.dynamic_enabled = False
        self.last_note = ""
        self.reload()

    @property
    def demo(self) -> RuleRecognizer:          # old name
        return self.rules

    # ---------------------------------------------------------------- loading
    def reload(self) -> None:
        self.bundles = {}
        self.load_errors = []
        for category, filename in MODEL_FILES.items():
            path = self.models_dir / filename
            if not path.exists():
                continue
            bundle, error = load_bundle(path, category)
            if bundle:
                self.bundles[category] = bundle
            else:
                self.load_errors.append(error)
                log.warning(error)
        self.dynamic = DynamicGestureRecognizer(self.bundles.get("dynamic"))

    @property
    def static_categories_loaded(self) -> list[str]:
        return [c for c in ("alphabet", "number", "word", "custom") if c in self.bundles]

    @property
    def demo_mode(self) -> bool:
        """True while NO trained static model exists at all."""
        return not self.static_categories_loaded

    def status_text(self) -> str:
        if self.demo_mode:
            return "Rule-based recognition — works without training (MediaPipe landmarks)."
        names = ", ".join(self.static_categories_loaded)
        missing = [c for c in ("alphabet", "number", "word") if c not in self.bundles]
        text = f"Models loaded: {names}."
        if missing:
            text += f" Rule-based shapes used for: {', '.join(missing)}."
        return text

    def set_dynamic_enabled(self, enabled: bool) -> None:
        self.dynamic_enabled = enabled
        if not enabled:
            self.dynamic.reset()

    def reset_motion(self) -> None:
        self.dynamic.reset()
        self.smoother.reset()

    # ------------------------------------------------------------- recognising
    def recognize(self, hands, mode: str = "AUTO") -> Prediction | None:
        """Best prediction for the hands in this frame, or None. Never raises."""
        self.last_note = ""
        if mode != self._last_mode:
            self._last_mode = mode
            self.smoother.reset()
        if not hands:
            self.smoother.reset()
            return None
        categories = MODE_CATEGORIES.get(mode, MODE_CATEGORIES["AUTO"])
        try:
            return self._recognize(hands, mode, categories)
        except Exception as exc:  # noqa: BLE001 - a bad frame must not crash the app
            log.exception("recognition failed")
            self.last_note = f"Recognition error: {exc}"
            return None

    def _recognize(self, hands, mode: str, categories: list[str]) -> Prediction | None:
        # motion gestures (J, Z, recorded motions) first
        # J / Z only exist in ALPHABET (and AUTO); WORDS only needs it for your own recorded motions
        motion_modes = ("ALPHABET", "AUTO", "WORDS") if self.dynamic.bundle else ("ALPHABET", "AUTO")
        if self.dynamic_enabled and mode in motion_modes:
            moved = self.dynamic.update(hands)
            if moved:
                self.smoother.reset()
                return moved
            if self.dynamic.is_moving():
                self.last_note = "Motion detected — finish drawing the letter…"
                return None

        best: Prediction | None = None
        missing: list[str] = []
        for cat in categories:
            bundle = self.bundles.get(cat)
            if bundle is None:
                missing.append(cat)
                continue
            pred = self._predict(bundle, hands, cat)
            if pred and (best is None or pred.confidence > best.confidence):
                best = pred
        # Categories without a trained model use the rule-based shapes.
        rule_cats = [c for c in missing if c != "custom"]
        if rule_cats:
            pred = self.rules.recognize(hands, rule_cats)
            if pred and (best is None or pred.confidence > best.confidence):
                best = pred
        best = self.smoother.update(best)
        if best is None and rule_cats:
            self.last_note = (f"Hand found, but this pose is not a {mode.lower()} gesture yet — "
                              "open the Gesture Guide (Help) to see the poses.")
        return best

    def _predict(self, bundle: ModelBundle, hands, category: str) -> Prediction | None:
        feats = extract_features(hands, two_hand=bundle.two_hand)
        if feats is None or len(feats) != bundle.n_features:
            return None
        proba = bundle.model.predict_proba(feats.reshape(1, -1))[0]
        i = int(np.argmax(proba))
        label = str(bundle.labels[i]).upper()
        if category in ("alphabet", "number", "word"):
            actual = category
        else:
            actual = "custom"
        return Prediction(label, float(proba[i]), actual, source="model")
