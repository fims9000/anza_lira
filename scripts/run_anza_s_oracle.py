#!/usr/bin/env python3
"""Run the frozen ANZA-S zero-training oracle."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anza_s.run import run


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
