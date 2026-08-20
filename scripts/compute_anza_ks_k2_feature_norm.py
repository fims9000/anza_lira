#!/usr/bin/env python3
from pathlib import Path

from anza_ks_k2.normalization import save_feature_norm


if __name__ == "__main__":
    result = save_feature_norm(Path("results/anza_ks/k2/freeze/feature_norm.json"))
    print(f"ANZA-KS K2 FEATURE NORM: {result['sample_count']} train scenes")
