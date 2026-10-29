"""No-network arithmetic and provenance checks for the Tver journal evidence ledger.

Run: python tools/tver2026/check_evidence.py
This script does not replay trained checkpoints or open any confirm data.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATHS = {
    "historical": "docs/RESEARCH_STATUS_FINAL.md",
    "sbpp": "docs/research_history/__home__lebedeffson__Code__anza_lira__results__anza_tracegraph__sbpp_v3_b__TRACEGRAPH_SBPP_V3_B_REPORT.md",
    "tg1": "docs/research_history/__home__lebedeffson__Code__anza_lira__results__anza_tracegraph__candidate_audit_v2__CANDIDATE_AUDIT_V2_REPORT.md",
    "p0": "docs/research_history/__home__lebedeffson__Code__anza_lira__results__anza_tracegraph__endgame_v1__e3_relation__TRACEGRAPH_P0_RELATION_REPORT.md"
}
ANCHORS = {
    "historical": ("0.9923", "67.2%", "0.78%"),
    "sbpp": ("2611/2688", "117/192", "B5"),
    "tg1": ("206", "148", "35"),
    "p0": ("STOP_P0_RELATION_SELECTOR", "0.447173", "0.016493")
}
def check_evidence(root: Path = ROOT) -> dict:
    for name, rel in PATHS.items():
        s = (root / rel).read_text(encoding="utf-8")
        for anchor in ANCHORS[name]:
            if anchor not in s: raise AssertionError(f"Missing {anchor} in {rel}")
    assert 2688 - 2611 == 2 + 21 + 54 == 77
    assert 192 - 117 == 75
    assert 1024 - 818 == 148 + 1 + 21 + 1 + 35 == 206
    assert abs(2611 / 2688 - 0.971354) < 1e-6
    assert abs(1202 / 2688 - 0.447173) < 1e-6
    assert abs(1202 / 2605 - 0.461420) < 1e-6
    assert abs(19 / 1152 - 0.016493) < 1e-6
    assert abs(7 / 2605 - 0.002687) < 1e-6
    return {
      "status": "FROZEN_SOURCE_ANCHORS_AND_ARITHMETIC_PASS",
      "type": "historical report validation, NOT new classifier inference",
      "candidate_recall": 2611 / 2688,
      "weak_branch_recall": 117 / 192,
      "tg1_endpoint_recall": 818 / 1024,
      "p0_relation_recovery": 1202 / 2688,
      "p0_false_bridge": 19 / 1152,
      "p0_fixed_decision_candidate_oracle_ceiling": 1285 / 2688,
      "source_reports": list(PATHS.values())
    }
if __name__ == "__main__":
    print(json.dumps(check_evidence(), indent=2, ensure_ascii=False))
