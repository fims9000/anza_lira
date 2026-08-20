#!/usr/bin/env python3
"""Run the deterministic crowd-only spatial-disagreement audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from method_repair.crowd_audit import run_crowd_target_audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "method_repair" / "audit" / "crowd_target.json",
    )
    parser.add_argument("--sample-count", type=int, default=40)
    args = parser.parse_args()
    result = run_crowd_target_audit(args.output, sample_count=args.sample_count)
    print(json.dumps({
        "status": result["status"],
        "output": str(args.output),
        "sample_count": result["selection"]["sample_count"],
        "expert_data_accessed": result["expert_data_accessed"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
