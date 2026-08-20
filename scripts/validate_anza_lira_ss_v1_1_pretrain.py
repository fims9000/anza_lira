#!/usr/bin/env python3
"""Validate the final SS1.5 freeze and all downstream access locks."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    result = json.loads((ROOT / "results/anza_lira_ss_v1_1/ANZA_LIRA_SS_V1_1_MASTER_RESULT.json").read_text())
    assert result["status"] == "SS1_5_PRETRAINING_FREEZE_PASS"
    assert all(result["checks"].values())
    assert result["training_started"] is False and result["development_opened"] is False
    assert result["confirm_opened"] is False and result["expert_label_pixels_loaded"] is False
    root = ROOT / "results/anza_lira_ss_v1_1"
    assert not any((root / name).exists() for name in ("training", "calibration", "development", "lira", "confirm", "expert_descriptive"))
    print("ANZA-LIRA SS V1.1 VALIDATION: PASS (SS1_5_PRETRAINING_FREEZE_PASS; 12 TRAININGS UNOPENED)")


if __name__ == "__main__":
    main()
