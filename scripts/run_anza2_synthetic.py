#!/usr/bin/env python3
"""Run the frozen ANZA-2 Phase-2 zero-training mechanism experiment."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anza2_experiment.synthetic_mechanism import run_phase2


if __name__ == "__main__":
    print(json.dumps(run_phase2(), indent=2, sort_keys=True))
