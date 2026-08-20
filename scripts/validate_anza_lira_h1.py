#!/usr/bin/env python3
"""Validate H1 scope, gates, and unopened downstream stages."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    master_path = ROOT / "results/lira_h1/final/ANZA_LIRA_H1_MASTER_RESULT.json"
    if not master_path.is_file():
        raise SystemExit("H1 VALIDATION: FAIL missing master result")
    master = json.loads(master_path.read_text())
    assert master["unit"]["status"] == "H1_RIBBON_IMPLEMENTATION_PASS"
    assert master["parent_stops_changed"] is False
    assert master["p0_opened"] is False and master["path_opened"] is False
    assert master["confirm_375_400_opened"] is False and master["expert_accessed"] is False
    auth = json.loads((ROOT / "results/lira_h1/freeze/H1_FRESH_SPLIT_AUTHORIZATION.json").read_text())
    assert auth["old_v2_confirm_contents_read"] is False
    assert auth["splits"]["h1_development"] == list(range(347, 373))
    assert auth["splits"]["h1_confirm"] == list(range(375, 401))
    assert not any((ROOT / "results/lira_h1/development_dense_cache" / f"section_{section:03d}.npy").exists() for section in range(375, 401))
    locked = (ROOT / "results/lira_h1/development_candidate/H1_FRESH_CANDIDATE_REPORT.md").read_text()
    assert "LOCKED_NOT_RUN_AFTER_STOP_H1_RIBBON_BENCHMARK_FAIL" in locked
    print(f"H1 VALIDATION: PASS ({master['status']})")


if __name__ == "__main__":
    main()
