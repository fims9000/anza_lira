#!/usr/bin/env python3
"""Validate ANZA-KS before or after the K1 run."""

import argparse

from anza_ks.validator import validate_final, validate_pre_run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    arguments = parser.parse_args()
    result = validate_final() if arguments.final else validate_pre_run()
    print(result["research_status"])
