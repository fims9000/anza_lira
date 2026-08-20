#!/usr/bin/env python3
from anza_ks_k2.k1_5 import run_k1_5, save_k1_5


if __name__ == "__main__":
    result = run_k1_5()
    save_k1_5(result)
    print(f"ANZA-KS K1.5 STATUS: {result['status']}")
