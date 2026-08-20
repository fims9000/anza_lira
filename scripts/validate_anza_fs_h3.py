#!/usr/bin/env python3
"""Validate ANZA-FS H3 before training or after metrics exist."""

from __future__ import annotations

import argparse

from anza_fs.validator import validate_final, validate_pregradient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    result = validate_final() if args.final else validate_pregradient()
    print(result.get("research_status", result.get("validator_status")))


if __name__ == "__main__":
    main()
