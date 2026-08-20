#!/usr/bin/env python3
"""Freeze validation-only calibration for the unchanged path classifier."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_completion.calibration import write_validation_calibration


def main() -> int:
    result = write_validation_calibration(
        PROJECT_ROOT / "results/final_practical_cycle/path_calibration",
        project_root=PROJECT_ROOT,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(json.dumps({
        "status": result["status"],
        "temperature": result["temperature"],
        "selected_operating_point": result["selected_operating_point"],
        "freeze_sha256": result["freeze_sha256"],
        "v5_test_samples_opened": result["v5_test_samples_opened"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

