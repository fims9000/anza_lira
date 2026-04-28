#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Check metric threshold in a run metrics.json.")
    parser.add_argument("--metrics", required=True, type=str, help="Path to metrics.json")
    parser.add_argument("--metric", default="test_precision", type=str, help="Metric key in metrics.json")
    parser.add_argument("--min-value", required=True, type=float, help="Minimum required value")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        print(f"[FAIL] metrics file not found: {metrics_path}")
        return 2

    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    if args.metric not in data:
        print(f"[FAIL] metric '{args.metric}' not present in {metrics_path}")
        return 2

    value = float(data[args.metric])
    ok = value >= float(args.min_value)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {args.metric}={value:.6f} (target >= {float(args.min_value):.6f})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

