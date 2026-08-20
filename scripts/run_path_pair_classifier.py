#!/usr/bin/env python3
"""Run the single frozen endpoint-pair classifier experiment."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_completion.pair_classifier import write_pair_classifier


def main() -> int:
    result = write_pair_classifier(
        PROJECT_ROOT / "results" / "path_completion" / "pair_classifier",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(json.dumps({"status": result["status"], "validation_metrics": result["validation_metrics"]}, indent=2))
    return 0 if result["status"] == "ENDPOINT_PAIR_CLASSIFIER_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

