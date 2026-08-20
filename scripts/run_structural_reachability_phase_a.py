#!/usr/bin/env python3
"""Run the frozen zero-training ANZA Structural Reachability Phase A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from structural_reachability.phase_a import run_phase_a


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args()
    result = run_phase_a(device=args.device)
    print(json.dumps({
        "status": result["status"],
        "protocol_sha256": result["protocol_sha256"],
        "training_performed": result["training_performed"],
        "expert_data_accessed": result["expert_data_accessed"],
        "delta": result["primary_comparison"]["point_delta"],
        "ci95": result["primary_comparison"]["ci95"],
        "delta_A": result["delta_A"],
        "phase_b_authorized": result["phase_b_authorized"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
