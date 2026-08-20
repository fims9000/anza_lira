#!/usr/bin/env python3
import argparse
import json
from anza_tracegraph.runner import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--device", default="cuda"); args = parser.parse_args(); value = run(device=args.device); print(json.dumps({"status": value["status"], "gates": value.get("gates")}, indent=2, sort_keys=True))
