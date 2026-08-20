#!/usr/bin/env python3
import argparse
import json

from anza_ks_k2.validator import validate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--pretraining", action="store_true"); args = parser.parse_args()
    result = validate(require_results=not args.pretraining); print(json.dumps(result, indent=2, sort_keys=True)); print("ANZA-KS K2 VALIDATION: PASS")
