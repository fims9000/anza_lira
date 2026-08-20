#!/usr/bin/env python3
"""Open CrossingTraceBench-v5 test once after calibration freeze."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_completion.test_evaluation import write_v5_test


def main() -> int:
    result = write_v5_test(
        PROJECT_ROOT / "results/final_practical_cycle/path_v5_test",
        project_root=PROJECT_ROOT,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(json.dumps({key: result[key] for key in ("status", "pair_metrics", "summary", "checks")}, indent=2, sort_keys=True))
    return 0 if result["status"] == "PATH_CLASSIFIER_TEST_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

