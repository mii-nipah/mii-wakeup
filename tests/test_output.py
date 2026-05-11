from __future__ import annotations

import json
import unittest

from mii_wakeup.detection import WakeEvent
from mii_wakeup.output import render_event


class OutputTests(unittest.TestCase):
    def test_text_output_is_confidence_only(self) -> None:
        event = WakeEvent(label="hey_meee", score=0.7254321, timestamp=10.0)

        self.assertEqual(render_event(event, "text"), "0.725432")

    def test_json_output_keeps_event_metadata(self) -> None:
        event = WakeEvent(label="hey_meee", score=0.72, timestamp=10.0)

        rendered = json.loads(render_event(event, "json"))

        self.assertEqual(rendered["label"], "hey_meee")
        self.assertEqual(rendered["score"], 0.72)
        self.assertEqual(rendered["timestamp"], 10.0)


if __name__ == "__main__":
    unittest.main()
