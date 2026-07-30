#!/usr/bin/env python
"""Click-to-run entry point.

    python run.py check
    python run.py backtest
    python run.py run --cycles 5

Identical to `python -m moobot.cli`, just shorter to type.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from moobot.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
