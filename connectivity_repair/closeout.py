"""Fail-closed closeout for the bounded Connectivity/Diffusion repair cycle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any
import zipfile


TERMINAL_STATUS = "CONNECTIVITY_REPAIR_NEGATIVE_WITH_ROOT_CAUSE"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pretraining_gates(root: Path) -> dict[str, Any]:
    root = Path(root)
    pretraining = root / "results" / "connectivity_repair" / "pretraining"
    audit = json.loads((pretraining / "v1_formula_code_audit.json").read_text())
    rf = json.loads((pretraining / "receptive_field_probe.json").read_text())
    oracle = json.loads((pretraining / "gt_connectivity_diffusion_oracle.json").read_text())
    benchmark = json.loads((pretraining / "benchmark_v5_config.json").read_text())
    checks = {
        "formula_audit_pass": audit.get("status") == "V1_FORMULA_CODE_AUDIT_PASS_CLEAN_ANZA_REQUIRED",
        "legacy_v1_unchanged": _sha256(root / "models" / "azconv.py") == audit["legacy"]["sha256"],
        "rf_context_observable": rf.get("status") == "CONNECTIVITY_CONTEXT_OBSERVABLE",
        "rf_minimum_is_9": rf.get("minimum_passing_receptive_field") == 9,
        "rf_pair_disjoint": rf.get("pair_disjoint") is True,
        "oracle_failed_closed": oracle.get("status") == "DIFFUSION_OPERATOR_INSUFFICIENT",
        "oracle_grid_complete": len(oracle.get("rows", [])) == 15,
        "oracle_has_no_passing_cell": not any(row.get("all_gates_pass") for row in oracle.get("rows", [])),
        "oracle_selected_nothing": oracle.get("selected") is None,
        "frozen_checkpoint_unchanged": _sha256(Path(oracle["frozen_clean_anza"]["checkpoint"])) == oracle["frozen_clean_anza"]["checkpoint_sha256"],
        "v5_test_locked": benchmark.get("test_status") == "LOCKED_UNOPENED" and oracle.get("test_v5_samples_opened") == 0,
        "expert_not_accessed": oracle.get("expert_data_accessed") is False,
        "cracks_not_accessed": oracle.get("cracks_samples_opened") == 0,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"connectivity repair closeout failed: {failed}")
    max_gap = max(oracle["rows"], key=lambda row: row["gap_recovery_rate"])
    safest = max(
        (row for row in oracle["rows"] if row["check_visible_dice_safety"]),
        key=lambda row: row["gap_recovery_rate"],
    )
    return {
        "status": TERMINAL_STATUS,
        "checks": checks,
        "minimum_observable_receptive_field": rf["minimum_passing_receptive_field"],
        "rf9_validation_auroc": next(row["validation_auroc"] for row in rf["rows"] if row["receptive_field"] == 9),
        "oracle_baseline": oracle["baseline"],
        "oracle_max_gap_recovery": {
            "steps": max_gap["steps"],
            "alpha": max_gap["alpha"],
            "gap_recovery_rate": max_gap["gap_recovery_rate"],
            "false_bridge_rate": max_gap["false_bridge_rate"],
            "visible_dice": max_gap["visible_dice"],
            "visible_dice_loss": max_gap["visible_dice_loss"],
        },
        "oracle_best_visible_safe": {
            "steps": safest["steps"],
            "alpha": safest["alpha"],
            "gap_recovery_rate": safest["gap_recovery_rate"],
            "false_bridge_rate": safest["false_bridge_rate"],
            "visible_dice": safest["visible_dice"],
            "visible_dice_loss": safest["visible_dice_loss"],
        },
        "d0_d3_training": "NOT_AUTHORIZED_NOT_RUN",
        "cracks": "NOT_AUTHORIZED_NOT_RUN",
        "expert": "NOT_ACCESSED",
        "v5_test": "LOCKED_UNOPENED",
    }


def write_closeout(root: Path) -> dict[str, Any]:
    root = Path(root)
    result = validate_pretraining_gates(root)
    final = root / "results" / "connectivity_repair" / "final"
    final.mkdir(parents=True, exist_ok=True)
    numbers = {
        "status": result["status"],
        "minimum_observable_receptive_field": result["minimum_observable_receptive_field"],
        "rf9_validation_auroc": result["rf9_validation_auroc"],
        "oracle_baseline": result["oracle_baseline"],
        "oracle_max_gap_recovery": result["oracle_max_gap_recovery"],
        "oracle_best_visible_safe": result["oracle_best_visible_safe"],
        "model_training_result": None,
        "cracks_result": None,
    }
    (final / "THESIS_NUMBERS.json").write_text(json.dumps(numbers, indent=2, sort_keys=True) + "\n")
    report = f"""# ANZA Connectivity/Diffusion Repair — final report

Status: `{TERMINAL_STATUS}`

This bounded cycle stopped at the mandatory feasibility gate. No D0-D3 model
was trained, no CRACKS sample or expert label was opened, and CrossingTraceBench-v5
test remains locked.

## What the controlled gates established

1. **Cause of the earlier C2/C3 visible gain:** unresolved. The capacity-matched
   D1/D2 experiment was correctly not started after the diffusion oracle failed,
   so the gain must not be attributed to auxiliary supervision.
2. **Independent fuzzy memberships:** CleanANZA now matches the intended
   independent fuzzy semantics and passes its invariants. Prior C1 is supportive
   evidence, not a new independent confirmation.
3. **Required observable context:** the minimum passing receptive field was
   `9x9` with balanced, pair-disjoint validation AUROC `{result['rf9_validation_auroc']:.6f}`.
4. **Perfect-connectivity feasibility:** failed. The maximum gap recovery was
   `{result['oracle_max_gap_recovery']['gap_recovery_rate']:.6f}` at
   `T={result['oracle_max_gap_recovery']['steps']}, alpha={result['oracle_max_gap_recovery']['alpha']}`,
   with false bridge `{result['oracle_max_gap_recovery']['false_bridge_rate']:.6f}`
   and visible Dice loss `{result['oracle_max_gap_recovery']['visible_dice_loss']:.6f}`.
   Required gates were gap recovery >=0.70, false bridge <=0.20, and visible Dice
   loss <=0.005.
5. **Learned hard-negative discrimination:** not tested; training was not authorized.
6. **D2 segmentation gain:** not tested; no D1/D2 training was run.
7. **D3 causal diffusion gain:** not tested; the GT upper-bound gate failed first.
8. **Remaining hard case:** restarted row-stochastic local averaging cannot make
   the hidden corridor strongly foreground while preserving the visible ridge.
   Restart also re-injects pre-existing false bridges from h0 on every step.
9. **CRACKS:** not run, because the synthetic feasibility prerequisite failed.
10. **Claim boundary:** we may claim that matched gaps become observable by RF=9
    in this controlled stream and that this concrete GT-connectivity restarted
    diffusion instantiation is insufficient. We may not claim a trained-model or
    CRACKS improvement.

## Root cause

The failure is in the proposed propagation mathematics, not in learned
connectivity: the oracle supplied perfect latent-lineage connectivity. With a
row-stochastic transition, every update is a convex local average plus a restart
toward h0. Increasing alpha and T raises gap evidence only modestly, while high
alpha attenuates the visible prediction. Low alpha preserves visible Dice but
saturates near gap recovery `{result['oracle_best_visible_safe']['gap_recovery_rate']:.6f}`
and false bridge `{result['oracle_best_visible_safe']['false_bridge_rate']:.6f}`.

Per the frozen stop condition, there is no D4, no epsilon/threshold tuning after
seeing the grid, and no CRACKS run.
"""
    (final / "FINAL_REPORT.md").write_text(report)
    evidence = f"""# Thesis evidence

- Formula/code audit: `../pretraining/v1_formula_code_audit.json`.
- Receptive-field raw curve: `../pretraining/receptive_field_probe_curve.csv`.
- Diffusion oracle raw grid: `../pretraining/gt_connectivity_diffusion_oracle.csv`.
- Machine-readable thesis values: `THESIS_NUMBERS.json`.
- All numerical values in `FINAL_REPORT.md` are derived from those JSON/CSV artifacts.
- D0-D3 training: `NOT_AUTHORIZED_NOT_RUN`.
- CRACKS/expert/v5 test access: `0 / false / LOCKED_UNOPENED`.
"""
    (final / "THESIS_EVIDENCE.md").write_text(evidence)
    (final / "VALIDATION_RECEIPT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def build_closeout_package(root: Path) -> dict[str, Any]:
    root = Path(root)
    result = write_closeout(root)
    final = root / "results" / "connectivity_repair" / "final"
    include = [
        "docs/research/ANZA_V1_FORMULA_CODE_AUDIT.md",
        "results/connectivity_repair/pretraining",
        "results/connectivity_repair/final",
        "connectivity_repair",
        "models/azconv_clean.py",
        "synthetic/crossing_trace_bench_v5.py",
        "tests/test_clean_anza.py",
        "tests/test_connectivity_balanced_metrics.py",
        "tests/test_connectivity_receptive_field.py",
        "tests/test_connectivity_diffusion_oracle.py",
        ".agent-state/TASK_STATE.json",
        ".agent-state/EVIDENCE.json",
    ]
    package = root / "results" / "connectivity_repair" / "ANZA_CONNECTIVITY_DIFFUSION_REPAIR_NEGATIVE_20260818.zip"
    with tempfile.TemporaryDirectory() as temporary:
        staging = Path(temporary) / "ANZA_CONNECTIVITY_DIFFUSION_REPAIR_NEGATIVE_20260818"
        for relative_name in include:
            source = root / relative_name
            destination = staging / relative_name
            if source.is_dir():
                shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.zip"))
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        checksums: list[str] = []
        for path in sorted(item for item in staging.rglob("*") if item.is_file()):
            checksums.append(f"{_sha256(path)}  {path.relative_to(staging)}")
        (staging / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n")
        with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(item for item in staging.rglob("*") if item.is_file()):
                archive.write(path, Path(staging.name) / path.relative_to(staging))
    with zipfile.ZipFile(package) as archive:
        bad = archive.testzip()
        file_count = len(archive.namelist())
    if bad is not None:
        raise ValueError(f"package CRC failure: {bad}")
    return {
        **result,
        "package": str(package),
        "package_sha256": _sha256(package),
        "package_file_count": file_count,
        "package_crc": "PASS",
    }

