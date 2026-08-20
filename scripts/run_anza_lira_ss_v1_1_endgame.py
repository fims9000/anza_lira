#!/usr/bin/env python3
"""Run the frozen ANZA-LIRA CRACKS Structural Stability V1.1 endgame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from structural_stability_v1_1.endgame import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(device=args.device, smoke=args.smoke), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
