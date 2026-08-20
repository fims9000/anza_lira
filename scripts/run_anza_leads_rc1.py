#!/usr/bin/env python3
from __future__ import annotations

import argparse

from anza_leads.rc1_run import run_rc1


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); result = run_rc1(device=args.device)
    print(result["metrics"]["status"])


if __name__ == "__main__":
    main()
