#!/usr/bin/env python3
"""Run a bounded ANZA-LIRA intervention phase."""

from __future__ import annotations

import argparse
import json

from lira_intervention.runner import build_i1, diagnose_i2, freeze_i0, run_i2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("i0", "i1", "i2", "i2-diagnostic"), required=True)
    args = parser.parse_args()
    result = {"i0": freeze_i0, "i1": build_i1, "i2": run_i2, "i2-diagnostic": diagnose_i2}[args.phase]()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
