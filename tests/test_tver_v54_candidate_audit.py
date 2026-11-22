"""V54 scientific audit invariants on fabricated case populations only."""
from __future__ import annotations

import pytest
from scripts import tver_v54_candidate_audit as audit


def _cases(cohort):
    sections = 100 if cohort == "TRAIN" else 60
    examples = []
    for s in range(sections):
        for g in range(2):
            for world in ("LINK", "NO_LINK"):
                examples.append({"section": s, "id": f"fake:{cohort}:{s}:{g}", "kind": world})
    return examples


def _fake_policy_scene(scene, name, policy):
    count = (int(scene["section"]) * 2 +
             int(str(scene["id"]).rsplit(":", 1)[1]))
    baseline_n = 125 if ":TRAIN:" in scene["id"] else 75
    if name == "V53_frozen_4x12":
        available = int(count < baseline_n)
    else:
        available = int(count < baseline_n + 15)
    if scene["kind"] == "NO_LINK":
        available = 0
    return {
        "section": scene["section"], "gap_id": scene["id"], "world": scene["kind"],
        "policy": name, "status": "OK", "adaptive_tau": .65,
        "n_candidates": 48 if name == "V53_frozen_4x12" else 80,
        "n_proposals_before_cap": 48 if name == "V53_frozen_4x12" else 120,
        "candidate_available": available, "geometric_top1_correct": available,
        "any_directional_source_within_8px_of_true_end": available,
        "used_source_within_8px_of_true_end": available,
        "ports_total": 10, "ports_after_radius_filter": 8,
        "ports_after_direction_filter": 8, "ports_used": 4,
        "proposals_before_per_source_k": 120,
        "proposals_after_per_source_k": 80,
        "final_cap": 48 if name == "V53_frozen_4x12" else 192,
    }


def test_frozen_population_and_exact_recall_reconciliation(monkeypatch):
    monkeypatch.setattr(audit, "_policy_scene", _fake_policy_scene)
    cases = _cases("REL_CAL_ALREADY_VIEWED")
    frozen = [
        {"section": s["section"], "gap_id": s["id"],
         "method": "raw_pair_rank_score", "world": "LINK",
         "n_candidates": "48", "true_candidate_available": str(int(i < 75))}
        for i, s in enumerate(x for x in cases if x["kind"] == "LINK")
    ]
    rows, summary = audit.evaluate_scenes(cases, "REL_CAL_ALREADY_VIEWED", frozen)
    assert len(rows) == 240 * len(audit.POLICIES)
    assert summary["V53_frozen_4x12"]["correct_candidate_available"] == 75
    assert summary["V54_sources_and_k_8x24"]["correct_candidate_available"] == 90
    assert summary["V54_sources_and_k_8x24"]["rescued_legacy_misses"] == 15
    assert summary["V54_sources_and_k_8x24"]["lost_legacy_hits"] == 0


def test_changed_archived_candidate_receipt_blocks_audit(monkeypatch):
    monkeypatch.setattr(audit, "_policy_scene", _fake_policy_scene)
    cases = _cases("REL_CAL_ALREADY_VIEWED")
    frozen = [
        {"section": s["section"], "gap_id": s["id"],
         "method": "raw_pair_rank_score", "world": "LINK",
         "n_candidates": "47", "true_candidate_available": str(int(i < 75))}
        for i, s in enumerate(x for x in cases if x["kind"] == "LINK")
    ]
    with pytest.raises(RuntimeError, match="candidate receipt drift"):
        audit.evaluate_scenes(cases, "REL_CAL_ALREADY_VIEWED", frozen)


def test_incomplete_source_or_world_pair_fails_closed(monkeypatch):
    monkeypatch.setattr(audit, "_policy_scene", _fake_policy_scene)
    with pytest.raises(RuntimeError, match="cohort or controlled world drift"):
        audit.evaluate_scenes(_cases("TRAIN")[:-1], "TRAIN")
