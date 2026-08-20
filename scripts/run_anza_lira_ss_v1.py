#!/usr/bin/env python3
"""Run only Structural Stability V1 SS0-SS1."""

from __future__ import annotations

import argparse
import json

from structural_stability_v1.runner import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(device=args.device), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

