#!/usr/bin/env python3
"""Run the frozen ANZA-KS K0/K1 study and stop."""

from anza_ks.runner import run


if __name__ == "__main__":
    print(run()["status"])
