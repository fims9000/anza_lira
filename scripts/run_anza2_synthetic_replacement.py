#!/usr/bin/env python3
"""Open the frozen independent ANZA-2 Phase-2B replacement confirm once."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anza2_experiment.synthetic_replacement import run_phase2b


if __name__ == "__main__":
    print(json.dumps(run_phase2b(), indent=2, sort_keys=True))
