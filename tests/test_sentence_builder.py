"""Sentence construction, delete, clear, names and phrase generation."""
import unittest

import helpers  # noqa: F401  (adds project root to sys.path)
from core.registry import GestureRegistry
from sentence.phrase_engine import PhraseEngine
from sentence.sentence_builder import SentenceBuilder


class SentenceBuilderTests(unittest.TestCase):
    def setUp(self):
        self.registry = GestureRegistry()
        self.name = ""
        self.b = SentenceBuilder(PhraseEngine(self.registry.phrases), lambda: self.name)

    def words(self, *ws):
        for w in ws:
            self.b.add_word(w)
        return self.b.sentence()

    # --- construction --------------------------------------------------------
    def test_simple_sentence(self):
        self.assertEqual(self.words("I", "NEED", "WATER"), "I need water.")

    def test_my_name_is_ali(self):
        self.assertEqual(self.words("MY", "NAME", "IS", "ALI"), "My name is Ali.")

    def test_letters_become_word(self):
        for ch in "ALI":
            self.b.add_letter(ch)
        self.assertEqual(self.b.pending, "ALI")
        self.b.space()
        self.assertEqual(self.b.pending, "")
        self.assertEqual(self.b.sentence(), "Ali.")

    def test_use_as_name(self):
        for ch in "ALI":
            self.b.add_letter(ch)
        self.assertEqual(self.b.use_as_name(), "Ali")
        self.assertEqual(self.b.sentence(), "My name is Ali.")

    def test_configured_name(self):
        self.name = "ali"
        self.assertEqual(self.words("MY", "NAME"), "My name is Ali.")

    def test_insert_name_phrase(self):
        self.assertTrue(self.b.insert_name_phrase("Sara"))
        self.assertEqual(self.b.sentence(), "My name is Sara.")
        self.assertFalse(SentenceBuilder().insert_name_phrase("  "))

    # --- phrase generation ---------------------------------------------------
    def test_required_phrases(self):
        cases = {
            ("I", "NEED", "FOOD"): "I need food.",
            ("I", "WANT", "WATER"): "I want water.",
            ("I", "NEED", "HELP"): "I need help.",
            ("PLEASE", "HELP", "ME"): "Please help me.",
            ("THANK YOU",): "Thank you.",
            ("WHERE", "HOSPITAL"): "Where is the hospital?",
            ("I", "HUNGRY"): "I am hungry.",
            ("I", "THIRSTY"): "I am thirsty.",
            ("NEED", "WATER"): "I need water.",
            ("DO", "YOU", "NEED", "WATER"): "Do you need water?",
        }
        for tokens, expected in cases.items():
            with self.subTest(tokens=tokens):
                self.b.clear()
                self.assertEqual(self.words(*tokens), expected)

    def test_capitalize_toggle(self):
        self.words("I", "WANT", "FOOD")
        self.b.capitalize_last()
        self.assertEqual(self.b.sentence(), "I want Food.")
        self.b.capitalize_last()
        self.assertEqual(self.b.sentence(), "I want food.")

    def test_numbers(self):
        self.b.add_digit("1")
        self.b.add_digit("2")
        self.b.space()
        self.assertEqual(self.b.sentence(), "12.")

    def test_routing_by_category(self):
        self.assertEqual(self.b.add_gesture("A", "alphabet"), "letter")
        self.assertEqual(self.b.add_gesture("5", "number"), "digit")
        self.assertEqual(self.b.add_gesture("WATER", "word"), "word")

    # --- delete / clear ------------------------------------------------------
    def test_delete_letter_then_word(self):
        self.words("I", "NEED")
        self.b.add_letter("A")
        self.b.add_letter("B")
        self.b.delete()
        self.assertEqual(self.b.pending, "A")
        self.b.delete()
        self.b.delete()                       # removes NEED
        self.assertEqual(self.b.sentence(), "I.")

    def test_delete_on_empty_is_safe(self):
        self.assertFalse(self.b.delete())

    def test_clear(self):
        self.words("I", "NEED", "WATER")
        self.b.add_letter("X")
        self.b.clear()
        self.assertTrue(self.b.is_empty)
        self.assertEqual(self.b.sentence(), "")

    def test_load_phrase(self):
        self.b.load_phrase("Where is the hospital?")
        self.assertEqual(self.b.sentence(), "Where is the hospital?")


if __name__ == "__main__":
    unittest.main()
