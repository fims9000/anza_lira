#!/usr/bin/env python3
import argparse

from anza_ks_k2.runner import freeze_pretraining, run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--freeze-only", action="store_true"); parser.add_argument("--device", default="cuda"); args = parser.parse_args()
    receipt = freeze_pretraining()
    print(f"ANZA-KS K2 PRETRAINING: {receipt['status']}")
    if not args.freeze_only:
        metrics = run(device=args.device); print(f"ANZA-KS K2 STATUS: {metrics['status']}")
