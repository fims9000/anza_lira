#!/usr/bin/env python3
"""Generate and validate the claim-safe max-min path closeout package."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_completion.closeout import build_closeout


def main() -> int:
    receipt = build_closeout(PROJECT_ROOT, device="cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

