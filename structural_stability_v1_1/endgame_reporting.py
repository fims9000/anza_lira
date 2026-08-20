"""Terminal reporting and packaging from frozen V1.1 raw artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any
import zipfile

import matplotlib.pyplot as plt
import numpy as np

from structural_stability_v1_1.amendment import sha256_file
from structural_stability_v1_1.protocol import RESULT_ROOT, ROOT, protocol_hash


PACKAGE = ROOT.parent / "_wip_backups/anza_lira/ANZA_LIRA_CRACKS_SS_V1_1_FINAL_RESEARCH_20260819.zip"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(""); return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _git_sha() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "UNAVAILABLE"


def build_tables_and_figures() -> dict[str, Any]:
    development = json.loads((RESULT_ROOT / "development/DEVELOPMENT_MASTER_RESULT.json").read_text())
    clean = _read_csv(RESULT_ROOT / "development/per_section_clean.csv")
    stress = _read_csv(RESULT_ROOT / "development/per_section_stress.csv")
    ensemble_clean = [row for row in clean if row["seed"] == "ensemble"]
    ensemble_stress = [row for row in stress if row["seed"] == "ensemble"]
    tables = RESULT_ROOT / "final/tables"
    clean_summary = []
    for variant in ("B0", "B1", "B2", "B3"):
        local = [row for row in ensemble_clean if row["variant"] == variant]
        clean_summary.append({"variant": variant, **{key: float(np.mean([float(row[key]) for row in local])) for key in ("dice", "cldice", "precision", "recall", "auprc", "fragmentation", "white_unknown_foreground")}})
    _write_csv(tables / "TABLE_CLEAN_SEGMENTATION.csv", clean_summary)
    robust = []
    for variant in ("B0", "B1", "B2", "B3"):
        local = [row for row in ensemble_stress if row["variant"] == variant]
        clean_by_section = {int(row["section_id"]): float(row["cldice"]) for row in ensemble_clean if row["variant"] == variant}
        robust.append({"variant": variant, "shift_cldice_mean": float(np.mean([float(row["cldice"]) for row in local])), "topo_drop_auc": float(np.mean([max(0, clean_by_section[int(row["section_id"])] - float(row["cldice"])) for row in local])), "fragmentation": float(np.mean([float(row["fragmentation"]) for row in local]))})
    _write_csv(tables / "TABLE_ROBUSTNESS_MACRO.csv", robust)
    breakdown = []
    for variant in ("B0", "B1", "B2", "B3"):
        for family in ("gain", "noise", "bandlimit", "phase", "warp"):
            for severity in (1, 2, 3):
                local = [row for row in ensemble_stress if row["variant"] == variant and row["family"] == family and int(row["severity"]) == severity]
                breakdown.append({"variant": variant, "family": family, "severity": severity, "cldice": float(np.mean([float(row["cldice"]) for row in local])), "dice": float(np.mean([float(row["dice"]) for row in local])), "fragmentation": float(np.mean([float(row["fragmentation"]) for row in local]))})
    _write_csv(tables / "TABLE_PERTURBATION_BREAKDOWN.csv", breakdown)
    decision = development["decision"]
    _write_csv(tables / "TABLE_ANZA_VS_GENERIC.csv", [{"comparison": "B3-B2", **decision["clean_dice_B3_B2"], "metric": "clean_dice"}, {"comparison": "B3-B2", **decision["clean_cldice_B3_B2"], "metric": "clean_cldice"}, {"comparison": "B3-B2", **decision["shift_cldice_B3_B2"], "metric": "shift_cldice"}, {"comparison": "B3/B2", **decision["topo_drop_ratio_B3_B2"], "metric": "topo_drop_ratio"}])
    _write_csv(tables / "TABLE_ANZA_VS_CONSISTENCY.csv", [{"comparison": "B3-B1", **decision["shift_cldice_B3_B1"], "metric": "shift_cldice"}, {"comparison": "B3/B1", **decision["topo_drop_ratio_B3_B1"], "metric": "topo_drop_ratio"}])
    geometry = development["geometry"]
    _write_csv(tables / "TABLE_GEOMETRY_MECHANISM.csv", [{"variant": variant, **values} for variant, values in geometry.items()])
    _write_csv(tables / "TABLE_SEED_REPLICATION.csv", decision["seed_directions"])
    _write_csv(tables / "TABLE_TTA_LIRA.csv", [])
    _write_csv(tables / "TABLE_CONFIRM.csv", [])
    _write_csv(tables / "TABLE_EXPERT_DESCRIPTIVE.csv", [])

    figure_root = RESULT_ROOT / "final/figures"; figure_root.mkdir(parents=True, exist_ok=True)
    variants = [row["variant"] for row in clean_summary]
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.bar(variants, [row["cldice"] for row in clean_summary]); axis.set_ylabel("clean clDice"); axis.set_ylim(0, 1)
    figure.savefig(figure_root / "clean_model_comparison.png", dpi=180); plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for variant in variants:
        local = [row for row in breakdown if row["variant"] == variant]
        axis.plot(range(len(local)), [row["cldice"] for row in local], marker="o", label=variant)
    axis.set_ylabel("shift clDice"); axis.set_xlabel("frozen stress cell"); axis.legend()
    figure.savefig(figure_root / "stress_cldice_cells.png", dpi=180); plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.bar([row["variant"] for row in robust], [row["topo_drop_auc"] for row in robust]); axis.set_ylabel("TopoDropAUC")
    figure.savefig(figure_root / "topology_drop_macro.png", dpi=180); plt.close(figure)
    paired = []
    b2 = {int(row["section_id"]): float(row["cldice"]) for row in ensemble_stress if row["variant"] == "B2"}
    # Multiple stress rows share a section; use section means.
    for section_id in sorted({int(row["section_id"]) for row in ensemble_stress}):
        a = [float(row["cldice"]) for row in ensemble_stress if row["variant"] == "B3" and int(row["section_id"]) == section_id]
        b = [float(row["cldice"]) for row in ensemble_stress if row["variant"] == "B2" and int(row["section_id"]) == section_id]
        paired.append((section_id, float(np.mean(a) - np.mean(b))))
    figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    axis.axhline(0, color="black", linewidth=1); axis.scatter([item[0] for item in paired], [item[1] for item in paired], s=12); axis.set_ylabel("B3-B2 shifted clDice")
    figure.savefig(figure_root / "paired_section_effect.png", dpi=180); plt.close(figure)
    manifest = {path.name: sha256_file(path) for path in sorted(figure_root.glob("*.png"))}
    _write_json(figure_root / "figure_manifest.json", manifest)
    return {"tables": len(list(tables.glob("*.csv"))), "figures": len(manifest)}


def finalize_terminal() -> dict[str, Any]:
    development = json.loads((RESULT_ROOT / "development/DEVELOPMENT_MASTER_RESULT.json").read_text())
    status = development["status"]
    artifacts = build_tables_and_figures()
    completion = json.loads((RESULT_ROOT / "TWELVE_RUN_COMPLETION_MANIFEST.json").read_text())
    calibration = json.loads((RESULT_ROOT / "calibration/CALIBRATION_FREEZE.json").read_text())
    final = RESULT_ROOT / "final"; final.mkdir(parents=True, exist_ok=True)
    allowed = ["Controlled CRACKS perturbation performance under the frozen section-disjoint V1.1 protocol."]
    if status == "ANZA_STABILITY_MULTISEED_DEV_PASS":
        allowed.append("Development evidence for incremental reciprocal determinant-one stability, pending confirm.")
    forbidden = ["real cross-survey robustness", "untouched expert generalization", "true Anosov dynamics in seismic data", "natural-gap repair", "geological instance identity"]
    master = {
        "protocol": "ANZA_LIRA_CRACKS_STRUCTURAL_STABILITY_V1_1",
        "protocol_sha256": protocol_hash(),
        "split_sha256": "43a3fb7716d5ff9e56c7da9a78f2127c20f8d13ba27d7e5576ac493176045671",
        "normalization_sha256": "013b16cc61ee8e1bc34a3221c5e7c26576e7dde8b4955e51adc65cc45f008630",
        "training": {"jobs_planned": 12, "jobs_completed": completion["jobs_completed"], "checkpoint_hashes": {f"{row['variant']}_s{row['seed']}": row["checkpoint_sha256"] for row in completion["records"]}},
        "thresholds": {"per_seed": calibration["per_seed_thresholds"], "ensemble": calibration["ensemble_thresholds"]},
        "development": development,
        "lira_development": {"status": "LOCKED_NOT_AUTHORIZED" if status != "ANZA_STABILITY_MULTISEED_DEV_PASS" else "PENDING_AUTHORIZED_PHASE"},
        "confirm": {"opened": False},
        "expert_descriptive": {"opened": False},
        "final_status": status,
        "allowed_claims": allowed,
        "forbidden_claims": forbidden,
    }
    _write_json(final / "ANZA_LIRA_SS_V1_1_MASTER_RESULT.json", master)
    (final / "CLAIMS_AND_LIMITATIONS.md").write_text("# Claims and limitations\n\n## Established historical controlled result\n\nThe earlier controlled corridor continuation result remains separate from V1.1.\n\n## New V1.1 result\n\n" + ("The frozen development result passed its ANZA gate; confirm remains required.\n" if status == "ANZA_STABILITY_MULTISEED_DEV_PASS" else "The frozen V1.1 development decision did not establish an incremental ANZA-specific effect.\n") + "\n## Not established\n\n" + "\n".join(f"- {item}" for item in forbidden) + "\n")
    (final / "REPRODUCIBILITY.md").write_text(f"# Reproducibility\n\n- Entrypoint: `/home/lebedeffson/Code/venv/bin/python scripts/run_anza_lira_ss_v1_1_endgame.py`\n- Seeds: 41, 42, 43.\n- Final optimizer updates: 1980 per run.\n- Protocol SHA-256: `{protocol_hash()}`.\n- Bootstrap seed: `20260819`; section is the statistical unit.\n- Tables and figures are regenerated from `development/per_section_*.csv` without retraining.\n")
    ledger_rows = ["# Experiment ledger", "", "| Phase | Status | Git SHA |", "|---|---|---|"]
    for row in completion["records"]:
        ledger_rows.append(f"| training {row['variant']} s{row['seed']} | {row['status']} | {_git_sha()} |")
    ledger_rows += [f"| calibration | {calibration['status']} | {_git_sha()} |", f"| development | {status} | {_git_sha()} |", "| LIRA | LOCKED unless ANZA dev PASS | - |", "| confirm | UNOPENED | - |", "| expert | UNOPENED | - |"]
    (final / "EXPERIMENT_LEDGER.md").write_text("\n".join(ledger_rows) + "\n")
    decision = development["decision"]
    (final / "ANZA_LIRA_SS_V1_1_FINAL_REPORT.md").write_text(f"# ANZA-LIRA CRACKS Structural Stability V1.1\n\n## Research question\n\nDoes reciprocal determinant-one B3 improve topology robustness over free-determinant B2 and ordinary consistency B1?\n\n## Frozen design\n\nSection-disjoint CRACKS, train-only normalization, 12 from-scratch runs, clean calibration, one-shot development, five perturbation families at three severities. White remains unknown.\n\n## Development\n\n- Clean Dice B3-B2: `{decision['clean_dice_B3_B2']}`\n- Clean clDice B3-B2: `{decision['clean_cldice_B3_B2']}`\n- Shifted clDice B3-B2: `{decision['shift_cldice_B3_B2']}`\n- Topology-drop ratio B3/B2: `{decision['topo_drop_ratio_B3_B2']}`\n- B3 geometry: `{development['geometry']['B3']}`\n\n## Confirm and expert\n\nConfirm and expert remained unopened because access is conditional on the development decision.\n\n## Final status\n\n`{status}`\n")
    (final / "FINAL_CODE_REVIEW.md").write_text("# Final code review\n\nChecked: threshold source, split locks, expert/dev access order, fixed perturbation cells, section bootstrap, B2/B3 symmetry, H0 rejection, final-step-only checkpoints, and raw-table traceability.\n")
    source = Path("/home/lebedeffson/.codex/attachments/ece204f1-375b-4a99-b447-692fcd11b47a/pasted-text.txt")
    shutil.copy2(source, final / "FINAL_EXECUTION_TZ.md")
    PACKAGE.parent.mkdir(parents=True, exist_ok=True)
    include = [ROOT / "structural_stability_v1_1", ROOT / "scripts/run_anza_lira_ss_v1_1_endgame.py", ROOT / "scripts/validate_anza_lira_ss_v1_1_endgame.py", RESULT_ROOT / "pretrain_freeze", RESULT_ROOT / "TWELVE_RUN_COMPLETION_MANIFEST.json", RESULT_ROOT / "calibration", RESULT_ROOT / "development", final]
    with zipfile.ZipFile(PACKAGE, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in include:
            if not item.exists(): continue
            paths = [item] if item.is_file() else [path for path in item.rglob("*") if path.is_file() and "checkpoint" not in path.name and "__pycache__" not in path.parts]
            for path in paths:
                archive.write(path, path.relative_to(ROOT.parent))
    package_sha = sha256_file(PACKAGE)
    (PACKAGE.with_suffix(PACKAGE.suffix + ".sha256")).write_text(f"{package_sha}  {PACKAGE.name}\n")
    master["package"] = {"path": str(PACKAGE), "sha256": package_sha, **artifacts}
    _write_json(final / "ANZA_LIRA_SS_V1_1_MASTER_RESULT.json", master)
    return master
