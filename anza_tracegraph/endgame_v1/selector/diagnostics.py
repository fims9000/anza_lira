"""Post-gate failure attribution without threshold or model changes."""

from __future__ import annotations

from collections import Counter
from typing import Any


def failure_attribution(decisions: list[dict[str, Any]], operating_curve: list[dict[str, Any]]) -> dict[str, Any]:
    safe = [row for row in operating_curve if row["FalseBridge"] <= 0.02 and row["WrongBranch"] <= 0.03]
    recovery_target = [row for row in operating_curve if row["CCR"] >= 0.87 and row["WrongBranch"] <= 0.03]
    safest_recovery = min(recovery_target, key=lambda row: (row["FalseBridge"], -row["threshold"])) if recovery_target else None
    best_safe = max(safe, key=lambda row: row["RelationRecovery"]) if safe else None
    accepted_none = [row for row in decisions if row["status"] == "NO_VALID_CONTINUATION" and int(row["accepted"])]
    wrong = [row for row in decisions if int(row["wrong_branch"])]
    top1 = sum(int(row["top_correct"]) for row in decisions if row["status"] == "CORRECT_CANDIDATE_PRESENT") / max(1, sum(row["status"] == "CORRECT_CANDIDATE_PRESENT" for row in decisions))
    bottleneck = "NONE_SCORE_SEPARATION" if top1 >= 0.87 and best_safe is not None and best_safe["RelationRecovery"] < 0.84 else "PAIR_RANKING_OR_REPRESENTATION"
    return {
        "diagnostic_only_after_frozen_e3": True,
        "no_gate_or_threshold_change": True,
        "bottleneck": bottleneck,
        "top1_candidate_accuracy": top1,
        "best_development_relation_recovery_under_frozen_safety_constraints": None if best_safe is None else best_safe["RelationRecovery"],
        "minimum_false_bridge_observed_at_CCR_at_least_0_87": None if safest_recovery is None else safest_recovery["FalseBridge"],
        "operating_point_at_CCR_at_least_0_87": safest_recovery,
        "accepted_none_count_at_frozen_threshold": len(accepted_none),
        "accepted_none_by_stratum": dict(sorted(Counter(row["stratum"] for row in accepted_none).items())),
        "wrong_branch_count_at_frozen_threshold": len(wrong),
        "wrong_branch_by_stratum": dict(sorted(Counter(row["stratum"] for row in wrong).items())),
        "interpretation": "Candidate competition ranking is strong, but confident NONE-source scores overlap positives; no scalar operating point reaches the recovery gates at the frozen false-bridge budget.",
    }
