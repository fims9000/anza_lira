#!/usr/bin/env python3
from anza_ks_k2.freeze import freeze_k1_5


if __name__ == "__main__":
    receipt = freeze_k1_5()
    print(f"ANZA-KS K1.5 FREEZE: {receipt['status']}")
