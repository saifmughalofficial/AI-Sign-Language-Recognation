"""Text-to-speech (with a fake engine) and voice-input fallbacks."""
import tempfile
import time
import unittest

import helpers  # noqa: F401
from speech.speech_to_text import VoiceListener, check_voice_support, find_vosk_model
from speech.text_to_speech import TextToSpeech


class FakeVoice:
    def __init__(self, i, name):
        self.id, self.name = i, name


class FakeEngine:
    def __init__(self):
        self.said, self.props, self.stopped = [], {}, 0
        self._pending = []

    def setProperty(self, k, v):
        self.props[k] = v

    def getProperty(self, k):
        return [FakeVoice("v1", "Voice One"), FakeVoice("v2", "Voice Two")] if k == "voices" else None

    def say(self, text):
        self._pending.append(text)

    def runAndWait(self):
        self.said += self._pending
        self._pending = []

    def stop(self):
        self.stopped += 1


class TTSTests(unittest.TestCase):
    def make(self, **kw):
        self.engine = FakeEngine()
        tts = TextToSpeech(engine_factory=lambda: self.engine, **kw)
        self.assertTrue(tts.wait_ready(3))
        self.addCleanup(tts.shutdown)
        return tts

    def test_speaks_text(self):
        tts = self.make()
        self.assertTrue(tts.speak("I need water."))
        tts.wait_idle()
        self.assertEqual(self.engine.said, ["I need water."])

    def test_empty_text_ignored(self):
        tts = self.make()
        self.assertFalse(tts.speak("   "))

    def test_cooldown_prevents_repeated_letters(self):
        tts = self.make()
        self.assertTrue(tts.speak("A", dedupe_seconds=1.0))
        self.assertFalse(tts.speak("A", dedupe_seconds=1.0))
        self.assertTrue(tts.speak("B", dedupe_seconds=1.0))
        tts.wait_idle()
        self.assertEqual(self.engine.said, ["A", "B"])

    def test_repeat_and_stop(self):
        tts = self.make()
        tts.speak("Hello")
        tts.wait_idle()
        self.assertTrue(tts.repeat())
        tts.wait_idle()
        self.assertEqual(self.engine.said, ["Hello", "Hello"])
        tts.stop()
        self.assertGreaterEqual(self.engine.stopped, 1)

    def test_lists_voices_and_applies_settings(self):
        tts = self.make(rate=150, volume=0.5, voice_id="v2")
        self.assertEqual(tts.list_voices(), [("v1", "Voice One"), ("v2", "Voice Two")])
        self.assertEqual(self.engine.props["rate"], 150)
        self.assertEqual(self.engine.props["voice"], "v2")

    def test_unavailable_engine_never_crashes(self):
        def broken():
            raise RuntimeError("no speech driver")
        tts = TextToSpeech(engine_factory=broken)
        self.assertFalse(tts.wait_ready(3))
        self.assertFalse(tts.available)
        self.assertIn("no speech driver", tts.error)
        self.assertFalse(tts.speak("hello"))
        self.assertEqual(tts.list_voices(), [])
        tts.stop()
        tts.shutdown()


class VoiceInputTests(unittest.TestCase):
    def test_missing_model_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            ok, message = check_voice_support(tmp)
            self.assertFalse(ok)                      # no model (or no vosk) in an empty folder
            self.assertTrue(message)
            self.assertIsNone(find_vosk_model(tmp))

    def test_listener_start_fails_gracefully(self):
        with tempfile.TemporaryDirectory() as tmp:
            statuses = []
            listener = VoiceListener(on_status=statuses.append, models_dir=tmp)
            ok, message = listener.start()
            self.assertFalse(ok)
            self.assertFalse(listener.listening)
            self.assertTrue(message)
            listener.stop()                           # must be safe

    def test_finds_model_folder(self):
        import os
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "vosk-model-small-en-us-0.15", "am"))
            self.assertIsNotNone(find_vosk_model(tmp))


if __name__ == "__main__":
    unittest.main()
