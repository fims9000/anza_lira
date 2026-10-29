"""Check a source-level RNG confound without claiming any network accuracy."""
from __future__ import annotations
import json
import numpy as np

def first_frequencies(seed: int, instances: int):
    rng = np.random.default_rng(seed)
    for _ in range(instances):
        rng.uniform(1.2, 2.8)
    return rng.uniform(5.0, 15.0, size=5)

def audit(n: int = 128):
    if n < 1: raise ValueError("n must be positive")
    matches = sum(np.array_equal(first_frequencies(420000000+i, 1),
                                 first_frequencies(420000000+i, 2))
                  for i in range(n))
    return {
       "status": "RNG_DRAW_ORDER_CONFOUND_DETECTED" if matches < n else "NO_DRAW_ORDER_DIFFERENCE",
       "pairs": n, "matched_background_frequencies": matches,
       "interpretation": "Only a deterministic nuisance-draw control; label leakage and classifier effect remain untested."
    }
if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
