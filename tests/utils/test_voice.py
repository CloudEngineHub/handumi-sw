import json
import time
import unittest

from handumi.utils.voice import (
    COMMAND_PHRASES,
    VoiceCommandListener,
    speech_duration_s,
)


def _result(text: str, conf: float = 1.0) -> str:
    return json.dumps(
        {"text": text, "result": [{"word": w, "conf": conf} for w in text.split()]}
    )


class VoiceCommandListenerTest(unittest.TestCase):
    """Exercise the decode path directly; the mic and Vosk stay out of it."""

    def setUp(self) -> None:
        self.listener = VoiceCommandListener(confidence=0.7, debounce_s=1.5)

    def test_each_control_phrase_maps_to_its_action(self):
        for phrase, action in COMMAND_PHRASES.items():
            listener = VoiceCommandListener()
            listener._handle_result(_result(phrase))
            self.assertEqual(listener.poll(), action, phrase)

    def test_unknown_padding_around_a_phrase_is_stripped(self):
        """Breath and room noise decode as "[unk]" tokens beside the phrase."""
        self.listener._handle_result(_result("[unk] restart"))
        self.assertEqual(self.listener.poll(), "restart")

    def test_a_command_word_alone_is_not_a_command(self):
        """"we should start the machine" decodes to "[unk] start [unk]"."""
        self.listener._handle_result(_result("[unk] start [unk]"))
        self.assertIsNone(self.listener.poll())

    def test_unknown_tokens_do_not_drag_confidence_down(self):
        raw = json.dumps(
            {
                "text": "[unk] stop recording",
                "result": [
                    {"word": "[unk]", "conf": 0.1},
                    {"word": "stop", "conf": 0.95},
                    {"word": "recording", "conf": 0.92},
                ],
            }
        )
        self.listener._handle_result(raw)
        self.assertEqual(self.listener.poll(), "stop")

    def test_speech_outside_the_grammar_is_ignored(self):
        self.listener._handle_result(_result("we should start the next one"))
        self.assertIsNone(self.listener.poll())

    def test_low_confidence_is_ignored(self):
        self.listener._handle_result(_result("stop recording", conf=0.3))
        self.assertIsNone(self.listener.poll())

    def test_repeated_phrase_is_debounced(self):
        self.listener._handle_result(_result("stop recording"))
        self.listener._handle_result(_result("stop recording"))
        self.assertEqual(self.listener.poll(), "stop")
        self.assertIsNone(self.listener.poll())

    def test_a_different_command_is_not_debounced(self):
        self.listener._handle_result(_result("start recording"))
        self.listener._handle_result(_result("restart"))
        self.assertEqual(self.listener.poll(), "start")
        self.assertEqual(self.listener.poll(), "restart")

    def test_poll_drains_in_order_and_empties(self):
        self.listener._handle_result(_result("start recording"))
        self.assertEqual(self.listener.poll(), "start")
        self.assertIsNone(self.listener.poll())

    def test_drain_discards_pending_commands(self):
        self.listener._handle_result(_result("start recording"))
        self.listener.drain()
        self.assertIsNone(self.listener.poll())

    def test_mute_clears_pending_and_holds_the_gate(self):
        self.listener._handle_result(_result("stop recording"))
        self.listener.mute(5.0)
        self.assertIsNone(self.listener.poll())
        self.assertGreater(self.listener._muted_until, time.monotonic())

    def test_mute_never_shortens_an_active_mute(self):
        self.listener.mute(5.0)
        long_gate = self.listener._muted_until
        self.listener.mute(0.1)
        self.assertEqual(self.listener._muted_until, long_gate)

    def test_grammar_is_closed_over_the_control_phrases(self):
        grammar = json.loads(self.listener._grammar())
        self.assertEqual(set(grammar), {*COMMAND_PHRASES, "[unk]"})

    def test_malformed_recognizer_output_is_ignored(self):
        self.listener._handle_result("not json")
        self.assertIsNone(self.listener.poll())


class SpeechDurationTest(unittest.TestCase):
    def test_longer_announcements_mute_for_longer(self):
        self.assertGreater(
            speech_duration_s("Episode 3 saved, 812 frames"),
            speech_duration_s("Restart recording"),
        )

    def test_empty_text_still_yields_a_positive_gate(self):
        self.assertGreater(speech_duration_s(""), 0.0)


if __name__ == "__main__":
    unittest.main()
