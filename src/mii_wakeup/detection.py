from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class WakeWordModel(Protocol):
    def predict(self, audio: np.ndarray) -> Mapping[str, float]:
        """Return wake-word scores for one audio frame."""


@dataclass(frozen=True)
class DetectionConfig:
    threshold: float = 0.5
    cooldown_seconds: float = 1.0
    max_events: int | None = 1


@dataclass(frozen=True)
class WakeEvent:
    label: str
    score: float
    timestamp: float


ScoresCallback = Callable[[Mapping[str, float]], None]
Clock = Callable[[], float]


def iter_wake_events(
    model: WakeWordModel,
    frames: Iterable[np.ndarray],
    config: DetectionConfig,
    *,
    on_scores: ScoresCallback | None = None,
    monotonic: Clock = time.monotonic,
    wall_clock: Clock = time.time,
) -> Iterator[WakeEvent]:
    events_seen = 0
    last_event_by_label: dict[str, float] = {}

    for frame in frames:
        scores = model.predict(frame)
        if on_scores is not None:
            on_scores(scores)
        if not scores:
            continue

        label, score = _best_score(scores)
        if score < config.threshold:
            continue

        now = monotonic()
        last_event_at = last_event_by_label.get(label)
        if (
            last_event_at is not None
            and now - last_event_at < config.cooldown_seconds
        ):
            continue

        last_event_by_label[label] = now
        events_seen += 1
        yield WakeEvent(label=label, score=score, timestamp=wall_clock())

        if config.max_events is not None and events_seen >= config.max_events:
            return


def _best_score(scores: Mapping[str, float]) -> tuple[str, float]:
    label, score = max(scores.items(), key=lambda item: float(item[1]))
    return label, float(score)
