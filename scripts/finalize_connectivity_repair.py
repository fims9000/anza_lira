#!/usr/bin/env python3
"""Build the claim-safe negative closeout package after the oracle stop."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connectivity_repair.closeout import build_closeout_package


if __name__ == "__main__":
    print(json.dumps(build_closeout_package(ROOT), indent=2, sort_keys=True))

