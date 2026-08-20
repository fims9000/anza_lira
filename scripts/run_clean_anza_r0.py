#!/usr/bin/env python3
"""Run or resume only the three missing CleanANZA R0 seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cracks_experiment.clean_anza_r0 import audit_r0_reuse_contract, freeze_clean_thresholds, run_r0_training


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("audit", "smoke", "train", "validate", "evaluate"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.stage == "audit":
        result = audit_r0_reuse_contract()
    elif args.stage == "smoke":
        result = run_r0_training(device=args.device, epochs=1, max_train_sections=4)
    elif args.stage == "train":
        result = run_r0_training(device=args.device)
    elif args.stage == "validate":
        result = freeze_clean_thresholds(device=args.device)
    else:
        from cracks_experiment.clean_anza_evaluation import build_r0_statistics

        result = build_r0_statistics()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
