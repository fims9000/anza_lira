#!/usr/bin/env python3
from pathlib import Path

from anza_ks_k2.benchmark import freeze_benchmark


if __name__ == "__main__":
    manifest = freeze_benchmark(Path("results/anza_ks/k2/freeze"))
    print(f"ANZA-KS K2 BENCHMARK FREEZE: {manifest['manifest_sha256']}")
