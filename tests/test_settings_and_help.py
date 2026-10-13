"""Settings (saving, invalid values), history and the Help dictionary."""
import json
import tempfile
import unittest
from pathlib import Path

import helpers  # noqa: F401
from core.history import ConversationHistory
from core.registry import GestureRegistry, validate_gesture_entry
from core.settings import Settings, SettingsStore, settings_from_dict


class SettingsTests(unittest.TestCase):
    def test_defaults_when_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(SettingsStore(Path(tmp) / "s.json").load(), Settings())

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SettingsStore(Path(tmp) / "config" / "s.json")
            s = Settings(theme="light", user_name="Ali", gesture_hold_seconds=1.2)
            self.assertTrue(store.save(s))
            self.assertEqual(store.load(), s)

    def test_invalid_values_fall_back_with_notes(self):
        s, notes = settings_from_dict({"theme": "neon", "camera_index": "abc",
                                       "detection_confidence": 7, "speech_enabled": "yes"})
        self.assertEqual(s.theme, "dark")
        self.assertEqual(s.camera_index, 0)
        self.assertEqual(s.detection_confidence, 0.99)      # clamped
        self.assertTrue(s.speech_enabled)                   # default kept
        self.assertGreaterEqual(len(notes), 3)

    def test_corrupt_file_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.json"
            path.write_text("{ not json", encoding="utf-8")
            store = SettingsStore(path)
            self.assertEqual(store.load(), Settings())
            self.assertTrue(store.notes)

    def test_not_an_object(self):
        s, notes = settings_from_dict([1, 2, 3])
        self.assertEqual(s, Settings())
        self.assertTrue(notes)

    def test_shipped_settings_file_is_valid(self):
        data = json.loads((Path(helpers.__file__).parent.parent / "config" / "settings.json").read_text())
        _, notes = settings_from_dict(data)
        self.assertEqual(notes, [])

    def test_resolution_parsing(self):
        self.assertEqual(Settings(resolution="1280x720").resolution_wh, (1280, 720))


class HistoryTests(unittest.TestCase):
    def test_add_clear_and_persist(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "h.json"
            h = ConversationHistory(path)
            h.add("SIGN", "I need water.")
            h.add("VOICE", "Sure.")
            self.assertEqual(len(h.entries), 2)
            again = ConversationHistory(path)
            again.load()
            self.assertEqual([e.source for e in again.entries], ["SIGN", "VOICE"])
            h.clear()
            again.load()
            self.assertEqual(ConversationHistory(path).entries, [])

    def test_not_saved_when_persistence_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "h.json"
            ConversationHistory(path, persist=False).add("SIGN", "secret")
            self.assertFalse(path.exists())


class HelpDictionaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg = GestureRegistry()

    def test_data_files_load_without_problems(self):
        self.assertEqual(self.reg.problems, [])

    def test_all_letters_and_digits_present(self):
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            self.assertIsNotNone(self.reg.get(ch, "alphabet"), ch)
        for d in "0123456789":
            self.assertIsNotNone(self.reg.get(d, "number"), d)

    def test_starter_words_present(self):
        for w in ("HELLO", "THANK YOU", "WATER", "HOSPITAL", "LOVE", "WANT", "NEED"):
            self.assertIsNotNone(self.reg.get(w, "word"), w)

    def test_search_water(self):
        hit = self.reg.search("water")[0]
        self.assertEqual(hit["gesture"], "WATER")
        self.assertIn("I need water.", hit["sentence_examples"])
        self.assertTrue(hit["instructions"])

    def test_search_no_result_and_empty(self):
        self.assertEqual(self.reg.search("zzzzqq"), [])
        self.assertEqual(len(self.reg.search("")), len(self.reg.gestures))

    def test_j_and_z_are_dynamic_and_experimental(self):
        for ch in "JZ":
            g = self.reg.get(ch)
            self.assertEqual(g["type"], "dynamic")
            self.assertEqual(g["status"], "experimental")

    def test_every_gesture_has_instructions_and_hands(self):
        for g in self.reg.gestures:
            self.assertTrue(g["instructions"], g["gesture"])
            self.assertIn(g["hands"], (1, 2))

    def test_help_sections_exist(self):
        titles = {s["title"] for s in self.reg.help_sections()}
        for needed in ("How to Use Camera", "How to Position Hand", "Voice Mode", "Sentence Builder", "Troubleshooting"):
            self.assertIn(needed, titles)

    def test_help_images_can_be_created(self):
        from ui.gesture_art import draw_gesture
        img = draw_gesture(self.reg.get("WATER"), 120)
        self.assertEqual(img.size, (120, 120))

    def test_missing_data_dir_is_handled(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = GestureRegistry(tmp)
            self.assertEqual(reg.gestures, [])
            self.assertGreaterEqual(len(reg.problems), 3)
            self.assertEqual(reg.search("water"), [])

    def test_invalid_gesture_entry_is_skipped(self):
        self.assertTrue(validate_gesture_entry({"gesture": "", "category": "x"}))
        self.assertEqual(validate_gesture_entry({"gesture": "WATER", "category": "word", "type": "static"}), [])
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "gestures.json").write_text(json.dumps({"gestures": [
                {"gesture": "OK", "category": "word"}, {"gesture": "BAD", "category": "nope"}]}))
            reg = GestureRegistry(tmp)
            self.assertEqual([g["gesture"] for g in reg.gestures], ["OK"])
            self.assertTrue(any("BAD" in p for p in reg.problems))

    def test_demo_poses_are_unique_per_category(self):
        for cat in ("alphabet", "number", "word"):
            poses = list(self.reg.demo_poses(cat))
            self.assertEqual(len(poses), len(set(poses)))


if __name__ == "__main__":
    unittest.main()
