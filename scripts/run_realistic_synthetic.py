#!/usr/bin/env python3
"""Run predicted-endpoint v6 development without opening v6 test."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_completion.realistic_synthetic import write_v6_development


def main() -> int:
    result = write_v6_development(
        PROJECT_ROOT / "results/final_practical_cycle/realistic_synthetic",
        project_root=PROJECT_ROOT,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(json.dumps({"status": result["status"], "eligible_models": result["eligible_models"], "models": {
        name: {key: value for key, value in data.items() if key not in {"cells"}}
        for name, data in result["models"].items()
    }}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

