"""Rule-based recognition (no trained model) on synthetic 3-D hands."""
import tempfile
import unittest

import numpy as np

import helpers  # noqa: F401  (sets sys.path)
import hand_model as hm
from core.registry import GestureRegistry
from vision.gesture_recognizer import GestureRecognizer, Prediction, Smoother
from vision.shape_library import ALL_SHAPES

LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY"            # J and Z are motion letters
NUMBERS = {"0": "O", "1": "ONE", "2": "TWO", "3": "THREE", "4": "FOUR", "5": "FIVE",
           "6": "W", "7": "SEVEN", "8": "EIGHT", "9": "F"}
WORDS = {"HELLO": "OPEN", "STOP": "PALM", "WATER": "W", "YES": "S", "NO": "PINCH2", "I": "I",
         "YOU": "POINT_SIDE", "NEED": "X", "WANT": "CLAW", "GOOD": "THUMBS_UP", "BAD": "THUMBS_DOWN",
         "THANK YOU": "FLAT_THUMB", "LOVE": "ILY", "NAME": "H", "MY": "FLAT_SIDE", "FOOD": "O",
         "WHERE": "ONE", "SORRY": "A", "HUNGRY": "C", "CALL": "Y"}
SENTENCES = {"GREETING": "OPEN", "NEED WATER": "W", "HUNGRY": "O", "MY NAME IS": "H",
             "PLEASE STOP": "PALM", "I AM FINE": "THUMBS_UP", "CALL DOCTOR": "Y", "I LOVE YOU": "ILY"}


class RuleRecognizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = GestureRegistry()
        cls.empty = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.empty.cleanup()

    def setUp(self):
        self.rec = GestureRecognizer(self.registry, models_dir=self.empty.name)

    def check(self, mode, expected_label, hands, min_conf=0.6):
        self.rec.reset_motion()
        pred = self.rec.recognize(hands, mode)
        self.assertIsNotNone(pred, f"{mode}: nothing recognised, expected {expected_label}")
        self.assertEqual(pred.label, expected_label, f"{mode}: got {pred.label} ({pred.confidence:.2f})")
        self.assertGreaterEqual(pred.confidence, min_conf)
        self.assertTrue(pred.is_rule_based)
        return pred

    # ---- every gesture, straight on ------------------------------------------
    def test_all_static_letters(self):
        for letter in LETTERS:
            with self.subTest(letter=letter):
                self.check("ALPHABET", letter, [hm.make(letter)])

    def test_all_numbers(self):
        for label, shape in NUMBERS.items():
            with self.subTest(number=label):
                self.check("NUMBERS", label, [hm.make(shape)])

    def test_words(self):
        for label, shape in WORDS.items():
            with self.subTest(word=label):
                self.check("WORDS", label, [hm.make(shape)])

    def test_one_hand_sentences_carry_their_text(self):
        for label, shape in SENTENCES.items():
            with self.subTest(sentence=label):
                pred = self.check("SENTENCES", label, [hm.make(shape)])
                self.assertEqual(pred.category, "sentence")
                self.assertTrue(pred.output.endswith((".", "!", "?")))

    # ---- two-hand signs ---------------------------------------------------------
    def test_two_hand_wait_beats_one_hand_hello(self):
        self.check("WORDS", "WAIT", hm.make_pair("OPEN", "OPEN", gap=0.35))
        self.check("SENTENCES", "PLEASE WAIT", hm.make_pair("OPEN", "OPEN", gap=0.35))

    def test_two_hand_please_and_help(self):
        self.check("WORDS", "PLEASE", hm.make_pair("PALM", "PALM", gap=0.04))
        self.check("WORDS", "HELP", hm.make_pair("THUMBS_UP", "OPEN", gap=0.2))
        self.check("SENTENCES", "HELP ME", hm.make_pair("THUMBS_UP", "OPEN", gap=0.2))

    # ---- robustness -------------------------------------------------------------
    def test_left_hand_and_other_aspect_ratios(self):
        for letter in "ABLVWY":
            with self.subTest(letter=letter):
                self.check("ALPHABET", letter, [hm.make(letter, mirror=True, handedness="Left")])
                self.check("ALPHABET", letter, [hm.make(letter, aspect=16 / 9)])

    def test_tilted_noisy_hands_mostly_right(self):
        rng = np.random.default_rng(7)
        wrong = []
        for letter in LETTERS:
            for _ in range(8):
                self.rec.reset_motion()
                pred = self.rec.recognize([hm.make(letter, rng=rng, jitter=True)], "ALPHABET")
                if pred is None or pred.label != letter:
                    wrong.append((letter, pred.label if pred else None))
        self.assertLessEqual(len(wrong), len(LETTERS) * 8 * 0.04, wrong)   # >= 96 % right

    def test_unknown_pose_gives_none_and_a_hint(self):
        # thumb + middle + ring up is not a number
        self.assertIsNone(self.rec.recognize([helpers.make_hand("10110")], "NUMBERS"))
        self.assertIn("Gesture Guide", self.rec.last_note)

    # ---- data consistency ------------------------------------------------------
    def test_every_shape_in_gestures_json_exists(self):
        for g in self.registry.gestures:
            if g.get("shape"):
                self.assertIn(g["shape"], ALL_SHAPES, g["gesture"])

    def test_shapes_unique_within_each_mode(self):
        for cat in ("alphabet", "number", "word", "sentence"):
            shapes = [g["shape"] for g in self.registry.by_category(cat) if g.get("shape")]
            self.assertEqual(len(shapes), len(set(shapes)), cat)

    def test_ten_sentences_including_name_and_water(self):
        outputs = [g["output"] for g in self.registry.by_category("sentence")]
        self.assertEqual(len(outputs), 10)
        self.assertIn("I need water.", outputs)
        self.assertIn("My name is {name}.", outputs)

    def test_stop_wait_water_are_recognisable_words(self):
        for word in ("STOP", "WAIT", "WATER"):
            self.assertTrue(self.registry.get(word, "word")["shape"], word)


class SmootherTests(unittest.TestCase):
    def test_single_frame_flicker_is_ignored(self):
        s = Smoother(5)
        a = Prediction("A", 0.9, "alphabet", "rules")
        b = Prediction("S", 0.9, "alphabet", "rules")
        for p in (a, a, a, b, a):
            out = s.update(p)
            self.assertEqual(out.label, "A")

    def test_confidence_drops_when_hand_changes(self):
        s = Smoother(5)
        for _ in range(5):
            s.update(Prediction("A", 0.9, "alphabet", "rules"))
        out = s.update(None)
        self.assertLess(out.confidence, 0.9)


if __name__ == "__main__":
    unittest.main()
