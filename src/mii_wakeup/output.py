from __future__ import annotations

import json
from typing import Literal

from .detection import WakeEvent


OutputFormat = Literal["text", "json"]


def render_event(event: WakeEvent, output_format: OutputFormat) -> str:
    if output_format == "json":
        return json.dumps(
            {
                "event": "wake",
                "label": event.label,
                "score": event.score,
                "timestamp": event.timestamp,
            },
            separators=(",", ":"),
        )

    return f"{event.score:.6f}"
