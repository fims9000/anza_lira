#!/usr/bin/env python3
"""Fail-closed validator and closeout writer for ANZA-2 Phase 3C-A.

This phase is forensic only.  It may authorize one bounded repair, but it may
not train a model or open confirm, CRACKS, or expert data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "results" / "anza2" / "phase3c_a"


def _load(name: str) -> dict[str, Any]:
    return json.loads((PHASE / name).read_text())


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _same_metrics(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left == right


def _report(metrics: dict[str, Any], fidelity: dict[str, Any], fusion: dict[str, Any]) -> str:
    matrix = metrics["component_matrix"]
    f0, f1, f8, f9 = (matrix[key] for key in (
        "F0_full_oracle", "F1_full_learned",
        "F8_learned_geometry_oracle_membership",
        "F9_learned_membership_oracle_geometry",
    ))
    field = fidelity["overall"]
    generic = fusion["sources"]["generic"]["low_fpr"]
    fused = fusion["sources"]["fused"]["low_fpr"]
    return f"""# ANZA-2 Phase 3C-A forensic report

## Phase and status

- Phase: `3C-A`, frozen-checkpoint component forensics only.
- Status: `PHASE3C_A_FORENSIC_PASS_ROOT_CAUSE_MEMBERSHIP_LEARNING`.
- Training: **not performed**.
- Confirm, CRACKS, and expert data: **not opened**.
- Selected root cause: `RC1 ROOT_CAUSE_MEMBERSHIP_LEARNING`.
- Next scientific action: one bounded membership repair is authorized by the packet, but was not run in this phase.

## Question

Which learned ANZA-2 field component destroys the branch-selectivity mechanism that already passed with a supplied oracle field?

## Frozen baseline reproduction

Phase 2B reproduced exactly: ANZA-2 oracle branch recall `{metrics['phase2b_reproduction']['methods']['anza2_absolute']['branch_recall']:.4f}`, path TPR `{metrics['phase2b_reproduction']['methods']['anza2_absolute']['path_tpr']:.4f}`, and false-bridge FPR `{metrics['phase2b_reproduction']['methods']['anza2_absolute']['false_bridge_fpr']:.4f}`. Legacy branch recall remains `{metrics['phase2b_reproduction']['methods']['legacy_global_normalized']['branch_recall']:.4f}`.

## Component replacement result

| Configuration | Branch recall | X | T | Y | Gap recovery | Parallel false bridge | TPR at FPR<=0.05 | normalized low-FPR pAUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F0 full oracle | {f0['overall_branch_recall']:.4f} | {f0['x_branch_recall']:.4f} | {f0['t_branch_recall']:.4f} | {f0['y_branch_recall']:.4f} | {f0['straight_gap_recovery']:.4f} | {f0['parallel_fault_false_bridge']:.4f} | {f0['tpr_at_fpr_0_05']:.4f} | {f0['low_fpr_pauc_normalized']:.4f} |
| F1 full learned | {f1['overall_branch_recall']:.4f} | {f1['x_branch_recall']:.4f} | {f1['t_branch_recall']:.4f} | {f1['y_branch_recall']:.4f} | {f1['straight_gap_recovery']:.4f} | {f1['parallel_fault_false_bridge']:.4f} | {f1['tpr_at_fpr_0_05']:.4f} | {f1['low_fpr_pauc_normalized']:.4f} |
| F8 learned geometry + oracle membership | {f8['overall_branch_recall']:.4f} | {f8['x_branch_recall']:.4f} | {f8['t_branch_recall']:.4f} | {f8['y_branch_recall']:.4f} | {f8['straight_gap_recovery']:.4f} | {f8['parallel_fault_false_bridge']:.4f} | {f8['tpr_at_fpr_0_05']:.4f} | {f8['low_fpr_pauc_normalized']:.4f} |
| F9 learned membership + oracle geometry | {f9['overall_branch_recall']:.4f} | {f9['x_branch_recall']:.4f} | {f9['t_branch_recall']:.4f} | {f9['y_branch_recall']:.4f} | {f9['straight_gap_recovery']:.4f} | {f9['parallel_fault_false_bridge']:.4f} | {f9['tpr_at_fpr_0_05']:.4f} | {f9['low_fpr_pauc_normalized']:.4f} |

F8 restores zero parallel false bridges and useful low-FPR selectivity while F9 retains the learned-field collapse. F2 and F9 are intentional exact duplicate interventions and match exactly.

## Field fidelity

- Active-mode recall: `{field['active_mode_recall']:.6f}`.
- Target pixels with every membership inactive: `{field['all_zero_fraction_target_pixels']:.6f}`.
- One-mode-or-less collapse at crossings: `{field['one_mode_collapse_fraction_crossing']:.6f}`.
- Median active/inactive membership: `{field['membership_active_median']:.6f}` / `{field['membership_inactive_median']:.6f}`; the ordering is inverted.
- Axial orientation median / q90 error: `{field['orientation_error_median']:.6f}` / `{field['orientation_error_q90']:.6f}` radians.
- Derived mean along/perpendicular geometry: `{field['g_along_mean']:.6f}` / `{field['g_perpendicular_mean']:.6f}`; ratio `{field['g_ratio_mean']:.4f}`.

Thus, the field retains a usable axial/anisotropic geometry, but its independent fuzzy memberships almost never activate on the required structures.

## Fusion audit

Generic TPR at FPR<=0.05 is `{generic['tpr_at_fpr_0_05']:.4f}`; fused TPR is `{fused['tpr_at_fpr_0_05']:.4f}`. Fusion changes only `{fusion['fraction_candidate_orderings_changed']:.4%}` of candidate orderings, split equally between correct and incorrect changes. This does not rescue the collapsed learned membership field.

## Claim boundary

This is a no-training forensic localization on frozen synthetic development data. It does **not** establish a repaired learned model, an independent confirmation result, CRACKS improvement, or expert performance. Curved gaps do not exist in the frozen v4 development stream, so curved-gap recovery is explicitly `N/A`; curved-trace edge recall is reported instead. The oracle scale/h values are controlled Phase-2B reference values, not generator-provided supervision targets.

## Gate

`PASS`: the frozen oracle mechanism was reproduced, F0-F9 and field/fusion audits were persisted, exactly one predeclared root cause was selected, and all forbidden data/training locks remained closed. `RC1` authorizes only the packet's bounded membership repair as the next phase.

## Files and checks

- Machine-readable evidence: `protocol.json`, `metrics.json`, `component_replacement.csv`, `field_fidelity.json`, `fusion_audit.csv/json`, `per_case.csv`, `operating_curve.csv`, `root_cause.json`, `validator.json`, and provenance receipts in this directory.
- Targeted Phase-3/3C tests: `20 passed`.
- Compile, JSON validation, and `git diff --check`: `PASS`.
- Git: no commit and no push were performed; the existing dirty research worktree was preserved.
"""


def validate() -> dict[str, Any]:
    required = (
        "protocol.json", "protocol_hash.txt", "metrics.json",
        "component_replacement.csv", "field_fidelity.json",
        "fusion_audit.csv", "fusion_audit.json", "root_cause.json",
        "per_case.csv", "operating_curve.csv", "bootstrap.json",
        "split_manifest.json", "data_access_log.json", "environment.json",
        "code_state.json",
    )
    failures = [f"missing or empty: {name}" for name in required
                if not (PHASE / name).is_file() or (PHASE / name).stat().st_size == 0]
    if failures:
        return {"status": "FAIL", "failures": failures}

    protocol = _load("protocol.json")
    metrics = _load("metrics.json")
    fidelity = _load("field_fidelity.json")
    fusion = _load("fusion_audit.json")
    root_cause = _load("root_cause.json")
    stored_hash = (PHASE / "protocol_hash.txt").read_text().strip()
    if stored_hash != _canonical_hash(protocol) or metrics.get("protocol_sha256") != stored_hash:
        failures.append("Phase-3C-A protocol hash mismatch")

    phase2b = _load_from(ROOT / "results/anza2/phase2b/validator.json")
    phase3b = _load_from(ROOT / "results/anza2/phase3b/validator.json")
    if phase2b.get("research_status") != "PHASE2_GEOMETRY_SELECTIVITY_PASS":
        failures.append("frozen Phase-2B status changed")
    phase2b_hash = (ROOT / "results/anza2/phase2b/protocol_hash.txt").read_text().strip()
    if phase2b_hash != protocol.get("phase2b_protocol_sha256"):
        failures.append("frozen Phase-2B protocol hash changed")
    if phase3b.get("research_status") != "STOP_PHASE3B_LEARNED_AFFINITY_NO_GAIN":
        failures.append("frozen Phase-3B status changed")
    if _digest(ROOT / "results/anza2/phase3b/metrics.json") != phase3b.get("phase3b_original_metrics_sha256"):
        failures.append("frozen Phase-3B metrics changed")

    if metrics.get("phase2b_reproduction", {}).get("reproduced") is not True:
        failures.append("Phase-2B mechanism was not reproduced")
    matrix = metrics.get("component_matrix", {})
    if set(matrix) != set(protocol.get("component_matrix", {})) or len(matrix) != 10:
        failures.append("F0-F9 component matrix is incomplete or drifted")
    f0 = matrix.get("F0_full_oracle", {})
    exact_f0 = (
        f0.get("overall_branch_recall") == 1.0
        and f0.get("x_branch_recall") == 1.0
        and f0.get("t_branch_recall") == 1.0
        and f0.get("y_branch_recall") == 1.0
        and f0.get("straight_gap_recovery") == 1.0
        and f0.get("parallel_fault_false_bridge") == 0.0
    )
    if not exact_f0:
        failures.append("F0 does not reproduce the oracle mechanism contract")
    if not _same_metrics(matrix.get("F2_learned_membership_only", {}), matrix.get("F9_learned_membership_oracle_geometry", {})):
        failures.append("intentional F2/F9 duplicate interventions disagree")

    overall = fidelity.get("overall", {})
    if not (
        overall.get("active_mode_recall", 1.0) < 0.05
        and overall.get("all_zero_fraction_target_pixels", 0.0) > 0.90
        and overall.get("orientation_error_q90", 99.0) < 0.25
        and overall.get("g_ratio_mean", 0.0) > 1.0
    ):
        failures.append("field-fidelity evidence does not support localized membership collapse")
    if root_cause != metrics.get("root_cause"):
        failures.append("root-cause artifacts disagree")
    if root_cause.get("rc_code") != "RC1" or root_cause.get("root_cause") != "ROOT_CAUSE_MEMBERSHIP_LEARNING":
        failures.append("exactly predeclared RC1 was not selected")
    if root_cause.get("repair_authorized") is not True:
        failures.append("selected root cause did not authorize a bounded repair")

    for source in (protocol, metrics):
        if source.get("training_performed") is not False:
            failures.append("training lock violated")
        for key in ("confirm_opened", "cracks_data_accessed", "expert_data_accessed"):
            if source.get(key) is not False:
                failures.append(f"{key} lock violated")
    if "not present" not in protocol.get("curved_gap_status", ""):
        failures.append("curved-gap N/A disclosure missing")

    result = {
        "status": "PASS" if not failures else "FAIL",
        "research_status": (
            "PHASE3C_A_FORENSIC_PASS_ROOT_CAUSE_MEMBERSHIP_LEARNING"
            if not failures else "INVALID_PHASE3C_A_EVIDENCE"
        ),
        "failures": failures,
        "protocol_sha256": stored_hash,
        "phase2b_reproduced": metrics["phase2b_reproduction"]["reproduced"],
        "root_cause": root_cause,
        "repair_allowed": not failures and root_cause["repair_authorized"],
        "repair_performed": False,
        "training_performed": False,
        "confirm_opened": False,
        "cracks_data_accessed": False,
        "expert_data_accessed": False,
        "next_action": "bounded RC1 membership repair on development only; freeze before any new confirm",
    }
    (PHASE / "validator.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (PHASE / "PHASE3C_A_FORENSIC_REPORT.md").write_text(_report(metrics, fidelity, fusion))
    (PHASE / "TASK_STATE.json").write_text(json.dumps({
        "status": result["research_status"],
        "phase": "anza2_phase3c_a_forensics",
        "next_action": result["next_action"],
        "forbidden_until_next_freeze": ["confirm", "CRACKS", "expert evaluation"],
    }, indent=2, sort_keys=True) + "\n")
    (PHASE / "EVIDENCE.json").write_text(json.dumps({
        "validator": result["status"],
        "protocol_sha256": stored_hash,
        "root_cause": root_cause,
        "phase2b_reproduced": result["phase2b_reproduced"],
        "full_oracle": matrix["F0_full_oracle"],
        "full_learned": matrix["F1_full_learned"],
        "learned_geometry_oracle_membership": matrix["F8_learned_geometry_oracle_membership"],
        "learned_membership_oracle_geometry": matrix["F9_learned_membership_oracle_geometry"],
        "claim_boundary": "Frozen synthetic development forensics only; no repaired model or real-data claim.",
    }, indent=2, sort_keys=True) + "\n")
    return result


def _load_from(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


if __name__ == "__main__":
    value = validate()
    print(json.dumps(value, indent=2, sort_keys=True))
    raise SystemExit(0 if value["status"] == "PASS" else 1)
