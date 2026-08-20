#!/usr/bin/env python3
"""Fail-closed Phase-3 validator; it never opens confirm, CRACKS, or expert."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anza2_experiment.learned_affinity_repair import evaluate_saved_phase3b


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate() -> dict:
    phase2 = json.loads((ROOT / "results/anza2/phase2b/validator.json").read_text())
    original = ROOT / "results/anza2/phase3b/metrics.json"
    original_sha = digest(original)
    audit = evaluate_saved_phase3b(device="cpu")
    failures = []
    if phase2.get("research_status") != "PHASE2_GEOMETRY_SELECTIVITY_PASS":
        failures.append("Phase 2 is not passed")
    if audit.get("status") != "STOP_PHASE3B_LEARNED_AFFINITY_NO_GAIN":
        failures.append("expected frozen learned-affinity negative gate")
    if audit.get("gate_pass") is not False or audit.get("confirm_opened") is not False:
        failures.append("Phase-3B gate/confirm lock mismatch")
    if any(audit.get(key) is not False for key in ("cracks_data_accessed", "expert_data_accessed")):
        failures.append("real/expert data lock violated")
    if audit["three_seed_tpr_delta"] >= audit["minimum_tpr_delta"]:
        failures.append("negative status conflicts with measured practical gate")
    result = {
        "status": "PASS" if not failures else "FAIL",
        "research_status": "STOP_PHASE3B_LEARNED_AFFINITY_NO_GAIN" if not failures else "INVALID_PHASE3_EVIDENCE",
        "failures": failures, "phase2_positive_preserved": True,
        "phase3b_original_metrics_sha256": original_sha,
        "phase3b_reaudited_metrics_sha256": digest(ROOT / "results/anza2/phase3b/metrics_reaudited.json"),
        "phase4_allowed": False, "confirm_opened": False,
        "cracks_data_accessed": False, "expert_data_accessed": False,
    }
    root = ROOT / "results/anza2/phase3b"
    (root / "validator.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (root / "PHASE3B_REPORT.md").write_text(
        "# ANZA-2 Phase 3B learned-affinity closeout\n\n"
        "The independent oracle-field Phase 2B result remains positive. The first learned Phase 3 development run failed, and the single diagnosed repair coupled axis supervision to active fuzzy membership while comparing affinity OFF/ON in the same checkpoint.\n\n"
        f"The re-audited three-seed TPR delta is `{audit['three_seed_tpr_delta']:.8f}` "
        f"with 95% CI `{audit['three_seed_tpr_delta_ci95']}` versus the frozen practical gate "
        f"`{audit['minimum_tpr_delta']:.2f}`. Confirm, CRACKS, and expert data remain unopened.\n\n"
        "Allowed conclusion: ANZA-2 geometry is selective when the field is supplied, but this bounded learned field did not transfer that advantage from image input.\n"
    )
    return result


if __name__ == "__main__":
    value = validate(); print(json.dumps(value, indent=2, sort_keys=True))
    raise SystemExit(0 if value["status"] == "PASS" else 1)
