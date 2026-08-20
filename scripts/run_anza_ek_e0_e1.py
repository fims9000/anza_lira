#!/usr/bin/env python3
"""Run frozen ANZA-EK E0/E1 without training."""

from anza_ek.run_e0_e1 import run


if __name__ == "__main__":
    print(run()["status"])
