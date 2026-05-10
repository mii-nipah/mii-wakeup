from __future__ import annotations

import unittest

import numpy as np

from mii_wakeup.detection import DetectionConfig, iter_wake_events


class FakeModel:
    def __init__(self, scores: list[dict[str, float]]) -> None:
        self._scores = iter(scores)

    def predict(self, audio: np.ndarray) -> dict[str, float]:
        return next(self._scores)


class DetectionTests(unittest.TestCase):
    def test_yields_first_score_above_threshold(self) -> None:
        model = FakeModel(
            [
                {"hey_meee": 0.1},
                {"hey_meee": 0.72},
                {"hey_meee": 0.9},
            ]
        )
        frames = [np.zeros(1280, dtype=np.int16) for _ in range(3)]

        events = list(
            iter_wake_events(
                model,
                frames,
                DetectionConfig(threshold=0.5, cooldown_seconds=0, max_events=1),
                monotonic=iter([1.0, 2.0, 3.0]).__next__,
                wall_clock=iter([10.0, 20.0, 30.0]).__next__,
            )
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].label, "hey_meee")
        self.assertAlmostEqual(events[0].score, 0.72)
        self.assertAlmostEqual(events[0].timestamp, 10.0)

    def test_continuous_mode_uses_per_label_cooldown(self) -> None:
        model = FakeModel(
            [
                {"hey_meee": 0.8},
                {"hey_meee": 0.9},
                {"hey_meee": 0.95},
                {"other": 0.7},
            ]
        )
        frames = [np.zeros(1280, dtype=np.int16) for _ in range(4)]

        events = list(
            iter_wake_events(
                model,
                frames,
                DetectionConfig(threshold=0.5, cooldown_seconds=1.0, max_events=None),
                monotonic=iter([1.0, 1.5, 2.2, 2.3]).__next__,
                wall_clock=iter([10.0, 15.0, 22.0, 23.0]).__next__,
            )
        )

        self.assertEqual(
            [event.label for event in events],
            ["hey_meee", "hey_meee", "other"],
        )


if __name__ == "__main__":
    unittest.main()
