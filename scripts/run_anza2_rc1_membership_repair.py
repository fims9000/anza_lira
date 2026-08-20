#!/usr/bin/env python3
"""Run the bounded RC1 membership-only development repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anza2_experiment.rc1_membership_repair import run_rc1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = run_rc1(device=args.device)
    print(json.dumps({
        "status": result["status"],
        "selected_config": result.get("selected_config"),
        "confirm_opened": result["confirm_opened"],
    }, indent=2, sort_keys=True))
