#!/usr/bin/env python3
import argparse
import json

from anza_tracegraph.candidate_audit_v2 import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--device", default="cuda"); args = parser.parse_args()
    print(json.dumps(run(device=args.device), indent=2, sort_keys=True))

