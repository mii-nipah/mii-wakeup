"""Compatibility wrapper for running the project as ``uv run main.py``."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent / "src"))

from mii_wakeup.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
