#!/usr/bin/env python3
"""Run the bounded ANZA-KIR IR0--IR2 protocol."""

from __future__ import annotations

import argparse
import json

from anza_kir.runner import run


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--device", default="cuda"); args = parser.parse_args()
    metrics = run(device=args.device)
    print(json.dumps({"status": metrics["status"], "gates": metrics["gates"]}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
