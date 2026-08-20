#!/usr/bin/env python3
"""Run only authorized ANZA-LIRA final phases F0 through F3."""

from __future__ import annotations

import argparse
import json

from lira_final.runner import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stop-after", choices=("F1", "F2", "F3"), default="F3")
    args = parser.parse_args()
    print(json.dumps(run(device=args.device, stop_after=args.stop_after), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

