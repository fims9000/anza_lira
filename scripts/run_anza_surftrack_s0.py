#!/usr/bin/env python3
from anza_surftrack.run_s0 import run_s0

if __name__ == "__main__":
    result = run_s0(); print(result["metrics"]["status"])
