#!/usr/bin/env python3
"""Run bounded ANZA-2 Phase 3D-A/B without training."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anza2.phase3d.run import run


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2, sort_keys=True))
