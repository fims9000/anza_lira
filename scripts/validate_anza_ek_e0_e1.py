#!/usr/bin/env python3
"""Validate ANZA-EK before or after the zero-training causal run."""

from __future__ import annotations

import argparse

from anza_ek.validator import validate_final, validate_pre_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    result = validate_final() if args.final else validate_pre_run()
    print(result["research_status"])


if __name__ == "__main__":
    main()
