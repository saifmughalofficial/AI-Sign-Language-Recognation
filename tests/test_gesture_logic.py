"""Stabilisation, duplicate prevention, features, demo recognition, missing models, training."""
import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np

import helpers
from helpers import make_hand, make_hand_points
from core.registry import GestureRegistry
from vision.feature_extractor import (HAND_FEATURE_LEN, extract_features, finger_states,
                                      pose_string, sequence_features)
from vision.gesture_recognizer import DynamicGestureRecognizer, GestureRecognizer
from vision.gesture_stabilizer import GestureStabilizer


class StabilizerTests(unittest.TestCase):
    def feed(self, stab, label, seconds, start=0.0, conf=0.9, fps=30):
        accepted = []
        n = int(seconds * fps)
        for i in range(n + 1):
            r = stab.update(label, conf, now=start + i / fps)
            if r:
                accepted.append(r)
        return accepted, start + seconds

    def test_needs_hold_time(self):
        s = GestureStabilizer(0.8, 0.6, 0.3)
        accepted, _ = self.feed(s, "A", 0.5)
        self.assertEqual(accepted, [])
        s.reset()
        accepted, _ = self.feed(s, "A", 1.0)
        self.assertEqual(accepted, ["A"])

    def test_no_repeat_while_holding(self):
        s = GestureStabilizer(0.5, 0.6, 0.3)
        accepted, _ = self.feed(s, "A", 5.0)
        self.assertEqual(accepted, ["A"])           # NOT AAAAAAA

    def test_repeat_after_change(self):
        s = GestureStabilizer(0.5, 0.6, 0.3)
        first, t = self.feed(s, "A", 1.0)
        second, t = self.feed(s, "B", 1.0, start=t + 0.033)
        third, _ = self.feed(s, "A", 1.0, start=t + 0.033)
        self.assertEqual(first + second + third, ["A", "B", "A"])

    def test_short_flicker_does_not_duplicate(self):
        s = GestureStabilizer(0.5, 0.6, 0.35)
        out = []
        t = 0.0
        for i in range(60):                         # hold A for 2 s with one bad frame
            label = "B" if i == 40 else "A"
            r = s.update(label, 0.9, now=t)
            if r:
                out.append(r)
            t += 1 / 30
        self.assertEqual(out, ["A"])

    def test_hand_leaving_allows_repeat(self):
        s = GestureStabilizer(0.5, 0.6, 0.3)
        a, t = self.feed(s, "A", 1.0)
        _, t = self.feed(s, None, 0.6, start=t + 0.033)
        b, _ = self.feed(s, "A", 1.0, start=t + 0.033)
        self.assertEqual(a + b, ["A", "A"])

    def test_low_confidence_ignored(self):
        s = GestureStabilizer(0.5, 0.6, 0.3)
        accepted, _ = self.feed(s, "A", 2.0, conf=0.3)
        self.assertEqual(accepted, [])

    def test_changing_gestures_never_accepted_quickly(self):
        s = GestureStabilizer(0.5, 0.6, 0.3)
        out = []
        for i, label in enumerate(itertools.islice(itertools.cycle("ABC"), 60)):
            r = s.update(label, 0.9, now=i / 30)
            if r:
                out.append(r)
        self.assertEqual(out, [])

    def test_progress_bar_value(self):
        s = GestureStabilizer(1.0, 0.6, 0.3)
        s.update("A", 0.9, now=0.0)
        s.update("A", 0.9, now=0.5)
        self.assertAlmostEqual(s.progress, 0.5, places=2)


class FeatureTests(unittest.TestCase):
    def test_all_32_finger_patterns(self):
        for bits in itertools.product("01", repeat=5):
            pattern = "".join(bits)
            got = pose_string(finger_states(make_hand_points(pattern)))
            self.assertEqual(got, pattern)

    def test_features_ignore_position_and_size(self):
        a = extract_features([make_hand("01100", scale=0.2, origin=(0.5, 0.7))])
        b = extract_features([make_hand("01100", scale=0.1, origin=(0.3, 0.6))])
        self.assertEqual(len(a), HAND_FEATURE_LEN)
        np.testing.assert_allclose(a, b, atol=1e-4)

    def test_two_hand_vector_is_padded(self):
        one = extract_features([make_hand("11111")], two_hand=True)
        self.assertEqual(len(one), 2 * HAND_FEATURE_LEN)
        self.assertTrue(np.all(one[HAND_FEATURE_LEN:] == 0))

    def test_no_hands_gives_none(self):
        self.assertIsNone(extract_features([]))

    def test_sequence_features_shape(self):
        frames = [make_hand_points("01000") for _ in range(20)]
        feats = sequence_features(frames)
        self.assertEqual(len(feats), 16 * 65)
        self.assertIsNone(sequence_features(frames[:2]))


class RecognizerTests(unittest.TestCase):
    def setUp(self):
        self.registry = GestureRegistry()
        self.empty = tempfile.TemporaryDirectory()
        self.rec = GestureRecognizer(self.registry, models_dir=self.empty.name)

    def tearDown(self):
        self.empty.cleanup()

    def test_missing_models_means_demo_mode(self):
        self.assertTrue(self.rec.demo_mode)
        self.assertIn("Rule-based", self.rec.status_text())

    def test_demo_prediction_is_labelled_demo(self):
        pred = self.rec.recognize([make_hand("11000")], "ALPHABET")
        self.assertEqual(pred.label, "L")
        self.assertTrue(pred.is_demo)

    def test_demo_numbers_and_words(self):
        self.assertEqual(self.rec.recognize([make_hand("01110")], "WORDS").label, "WATER")
        self.assertEqual(self.rec.recognize([make_hand("01100")], "NUMBERS").label, "2")

    def test_unsupported_pose_returns_none_with_note(self):
        self.assertIsNone(self.rec.recognize([make_hand("10110")], "NUMBERS"))
        self.assertTrue(self.rec.last_note)

    def test_no_hand_returns_none(self):
        self.assertIsNone(self.rec.recognize([], "AUTO"))

    def test_corrupt_model_is_reported_not_fatal(self):
        Path(self.empty.name, "asl_alphabet_model.pkl").write_bytes(b"not a model")
        rec = GestureRecognizer(self.registry, models_dir=self.empty.name)
        self.assertTrue(rec.demo_mode)
        self.assertEqual(len(rec.load_errors), 1)

    def test_nonexistent_models_folder(self):
        rec = GestureRecognizer(self.registry, models_dir=Path(self.empty.name) / "nope")
        self.assertTrue(rec.demo_mode)

    def test_unknown_mode_falls_back_to_auto(self):
        self.assertIsNotNone(self.rec.recognize([make_hand("11111")], "SOMETHING"))


class TrainingPipelineTests(unittest.TestCase):
    """Train a tiny model on synthetic hands, then let the recogniser use it."""

    def test_train_and_recognise(self):
        from training.train_model import train_category
        registry = GestureRegistry()
        rng = np.random.default_rng(0)
        classes = {"A": "10000", "B": "01111", "L": "11000", "V": "01100"}
        X, y = [], []
        for label, pattern in classes.items():
            for _ in range(30):
                hand = make_hand(pattern, scale=float(rng.uniform(0.12, 0.25)),
                                 origin=(float(rng.uniform(0.3, 0.7)), float(rng.uniform(0.6, 0.9))))
                hand.landmarks[:, :2] += rng.normal(0, 0.003, (21, 2)).astype(np.float32)
                X.append(extract_features([hand]))
                y.append(label)
        with tempfile.TemporaryDirectory() as tmp:
            processed, models = Path(tmp, "processed"), Path(tmp, "models")
            processed.mkdir()
            np.savez_compressed(processed / "features_alphabet.npz", X=np.array(X), y=np.array(y),
                                two_hand=np.array(False))
            bundle = train_category("alphabet", "rf", 0.25, processed, Path(tmp, "custom"), models, verbose=False)
            self.assertGreater(bundle["accuracy"], 0.9)
            self.assertTrue((models / "asl_alphabet_model.pkl").exists())
            rec = GestureRecognizer(registry, models_dir=models)
            self.assertFalse(rec.demo_mode)
            pred = rec.recognize([make_hand("11000", scale=0.18)], "ALPHABET")
            self.assertEqual(pred.label, "L")
            self.assertFalse(pred.is_demo)
            self.assertGreater(pred.confidence, 0.5)


class DynamicTests(unittest.TestCase):
    def _draw(self, rec, path, hand_pattern, tip):
        pred = None
        for i, (x, y) in enumerate(path):
            pts = make_hand_points(hand_pattern, origin=(0.5, 0.75))
            delta = np.array([x, y]) - pts[tip][:2]
            pts[:, :2] += delta.astype(np.float32)
            hand = helpers.DetectedHand(pts, "Right", 0.99)
            pred = rec.update([hand], now=i * 0.05) or pred
        return pred

    def test_z_gesture_detected(self):
        rec = DynamicGestureRecognizer()
        # display-space Z (index tip), x mirrored to raw coordinates
        pts = [(0, 0), (1, 0), (0, 1), (1, 1)]
        path = []
        for a, b in zip(pts, pts[1:]):
            for t in np.linspace(0, 1, 8, endpoint=False):
                path.append((0.3 + 0.4 * (1 - (a[0] + (b[0] - a[0]) * t)), 0.3 + 0.4 * (a[1] + (b[1] - a[1]) * t)))
        pred = self._draw(rec, path, "01000", 8)
        self.assertIsNotNone(pred)
        self.assertEqual(pred.label, "Z")
        self.assertTrue(pred.dynamic)

    def test_still_hand_is_not_a_motion_gesture(self):
        rec = DynamicGestureRecognizer()
        path = [(0.5, 0.5)] * 40
        self.assertIsNone(self._draw(rec, path, "01000", 8))


if __name__ == "__main__":
    unittest.main()
