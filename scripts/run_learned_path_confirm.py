#!/usr/bin/env python3
"""Open and freeze the independent learned path confirmation once."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_completion.learned_confirm import write_learned_confirm


def main() -> int:
    result = write_learned_confirm(
        PROJECT_ROOT / "results" / "path_completion" / "learned_confirm",
        project_root=PROJECT_ROOT,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(json.dumps({key: result[key] for key in ("status", "pair_metrics", "summary", "checks")}, indent=2))
    return 0 if result["status"] == "LEARNED_PATH_SYNTHETIC_CONFIRM_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

