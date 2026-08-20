#!/usr/bin/env python3
"""Validate frozen LEADS artifacts without training."""

from __future__ import annotations

import argparse
import json

from anza_leads.validator import validate_a0, validate_a1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("a0", "a1", "all"), default="all")
    args = parser.parse_args()
    result = validate_a0() if args.phase == "a0" else validate_a1() if args.phase == "a1" else {"a0": validate_a0(), "a1": validate_a1()}
    print(json.dumps(result, indent=2, sort_keys=True))
    statuses = [result["status"]] if "status" in result else [value["status"] for value in result.values()]
    if any(value != "PASS" for value in statuses):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
