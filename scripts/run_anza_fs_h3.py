#!/usr/bin/env python3
"""Run the bounded ANZA-FS H3 seed-41 matrix."""

from __future__ import annotations

import argparse

from anza_fs.run_h3 import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = run(device=args.device)
    print(result["status"])


if __name__ == "__main__":
    main()
