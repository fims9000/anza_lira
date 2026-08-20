#!/usr/bin/env python3
"""Validate SS0-SS1 and enforce downstream locks."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    master = json.loads((ROOT / "results/anza_lira_ss_v1/ANZA_LIRA_SS_V1_MASTER_RESULT.json").read_text())
    assert master["ss0"]["status"] == "SS_S0_PASS"
    assert master["ss1"]["status"] == "SS_S1_PASS"
    split = json.loads((ROOT / "results/anza_lira_ss_v1/s0_audit/split_manifest.json").read_text())
    sets = [set(values) for values in split["splits"].values()]
    assert all(not sets[i] & sets[j] for i in range(len(sets)) for j in range(i + 1, len(sets)))
    assert len(split["splits"]["SS_CALIBRATION"]) == 40
    expert = json.loads((ROOT / "results/anza_lira_ss_v1/s0_audit/EXPERT_PROVENANCE.json").read_text())
    assert expert["expert_previously_accessed"] is True
    assert expert["untouched_expert_claim_allowed"] is False
    assert expert["expert_label_pixels_loaded_by_ss0_ss1"] is False
    validation = json.loads((ROOT / "results/anza_lira_ss_v1/s1_perturbation/validator.json").read_text())
    assert validation["validator_status"] == "PASS" and all(validation["checks"].values())
    assert not any((ROOT / "results/anza_lira_ss_v1" / name).exists() for name in ("s2_b0_b1_s41", "s3_b2_b3_s41", "s4_multiseed", "s5_lira", "s7_confirm"))
    print("ANZA-LIRA SS V1 VALIDATION: PASS (SS_S1_PASS; B0-B3 LOCKED)")


if __name__ == "__main__":
    main()

