#!/usr/bin/env python3
"""Create the read-only ANZA method-repair forensic baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from method_repair.audit import PROJECT_ROOT, run_forensic_audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "method_repair" / "audit" / "baseline.json",
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    payload = run_forensic_audit(
        args.output,
        include_profile=args.profile,
        profile_device=args.device,
    )
    print(json.dumps({
        "status": payload["status"],
        "output": str(args.output),
        "expert_data_accessed": payload["expert_data_accessed"],
        "training_started": payload["training_started"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
