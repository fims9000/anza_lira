#!/usr/bin/env python3
"""Validate the bounded Connectivity/Diffusion repair closeout."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connectivity_repair.closeout import validate_pretraining_gates


if __name__ == "__main__":
    result = validate_pretraining_gates(ROOT)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"\nCONNECTIVITY REPAIR STATUS: {result['status']}")

