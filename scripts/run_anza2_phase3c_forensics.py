#!/usr/bin/env python3
"""Run ANZA-2 Phase 3C-A F0--F9 forensics without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anza2.forensics.run import run_forensics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    result = run_forensics(device=args.device)
    print(json.dumps({
        "status": result["status"], "root_cause": result["root_cause"],
        "training_performed": result["training_performed"],
        "confirm_opened": result["confirm_opened"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
