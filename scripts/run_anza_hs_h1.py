#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anza_hs.run_h1 import run


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
