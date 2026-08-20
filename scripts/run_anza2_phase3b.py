#!/usr/bin/env python3
"""Run the one bounded, causal ANZA-2 Phase-3B development repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anza2_experiment.learned_affinity_repair import run_phase3b


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    print(json.dumps(run_phase3b(device=args.device), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
