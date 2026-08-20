#!/usr/bin/env python3
"""Run only the bounded ANZA-LIRA LEADS A0/A1 phases."""

from __future__ import annotations

import argparse
import json

from anza_leads.run import run_a0, run_a1, run_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("a0", "a1", "all"), default="all")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = run_a0() if args.phase == "a0" else run_a1(device=args.device) if args.phase == "a1" else run_all(device=args.device)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
