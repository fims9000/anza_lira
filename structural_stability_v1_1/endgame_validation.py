"""Artifact validator for the V1.1 endgame."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
import zipfile

from structural_stability_v1_1.amendment import sha256_file
from structural_stability_v1_1.endgame_reporting import PACKAGE
from structural_stability_v1_1.protocol import RESULT_ROOT, SEEDS, VARIANTS, protocol_hash


def validate() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    pretrain = json.loads((RESULT_ROOT / "pretrain_freeze/validator.json").read_text())
    checks["ss1_5_unchanged"] = pretrain.get("status") == "SS1_5_PRETRAINING_FREEZE_PASS" and pretrain.get("protocol_sha256") == protocol_hash()
    completion_path = RESULT_ROOT / "TWELVE_RUN_COMPLETION_MANIFEST.json"
    checks["twelve_run_manifest"] = completion_path.is_file()
    completion = json.loads(completion_path.read_text()) if completion_path.is_file() else {"records": []}
    records = completion.get("records", [])
    checks["exact_jobs"] = {(row.get("variant"), row.get("seed")) for row in records} == {(variant, seed) for variant in VARIANTS for seed in SEEDS}
    checks["final_steps"] = len(records) == 12 and all(row.get("optimizer_step") == 1980 and row.get("epoch") == 36 for row in records)
    checks["unique_hashes"] = len({row.get("checkpoint_sha256") for row in records}) == 12
    checks["locks_during_training"] = all(not row.get("development_opened") and not row.get("confirm_opened") and not row.get("expert_opened") and not row.get("historical_H0_loaded") for row in records)
    calibration_path = RESULT_ROOT / "calibration/CALIBRATION_FREEZE.json"
    checks["calibration_frozen"] = calibration_path.is_file() and json.loads(calibration_path.read_text()).get("status") == "SS_CALIBRATION_FROZEN"
    dev_path = RESULT_ROOT / "development/DEVELOPMENT_MASTER_RESULT.json"
    checks["development_written"] = dev_path.is_file()
    if dev_path.is_file():
        development = json.loads(dev_path.read_text())
        stress_path = RESULT_ROOT / "development/per_section_stress.csv"
        with stress_path.open(newline="") as handle: stress = list(csv.DictReader(handle))
        ensembles = [row for row in stress if row["seed"] == "ensemble"]
        cells = {(row["variant"], row["family"], int(row["severity"])) for row in ensembles}
        checks["all_stress_cells"] = cells == {(variant, family, severity) for variant in VARIANTS for family in ("gain", "noise", "bandlimit", "phase", "warp") for severity in (1, 2, 3)}
        checks["section_bootstrap"] = development["decision"]["shift_cldice_B3_B2"].get("resamples") == 10000
        checks["status_tree"] = development["status"] in {"STOP_ANZA_GEOMETRY_NOT_LEARNED", "STOP_ANZA_STABILITY_CLEAN_COST", "STOP_ANZA_STABILITY_WHITE_SAFETY", "STOP_ANZA_STABILITY_NO_INCREMENTAL_VALUE", "STRUCTURAL_STABILITY_PASS_ANOSOV_NOT_SPECIFIC", "STOP_ANZA_STABILITY_SEED_UNSTABLE", "ANZA_STABILITY_MULTISEED_DEV_PASS"}
        if development["status"] != "ANZA_STABILITY_MULTISEED_DEV_PASS":
            checks["confirm_expert_locked_on_dev_fail"] = not (RESULT_ROOT / "SS_V1_1_CONFIRM_AUTHORIZATION.json").exists() and not (RESULT_ROOT / "expert_descriptive/EXPERT_ACCESS.json").exists()
    final_master = RESULT_ROOT / "final/ANZA_LIRA_SS_V1_1_MASTER_RESULT.json"
    if final_master.is_file():
        checks["master_json"] = True
        package = json.loads(final_master.read_text()).get("package", {})
        checks["package"] = PACKAGE.is_file() and package.get("sha256") == sha256_file(PACKAGE) and zipfile.is_zipfile(PACKAGE)
    status = "PASS" if checks and all(checks.values()) else "FAIL"
    result = {"validator_status": status, "checks": checks, "research_status": json.loads(final_master.read_text()).get("final_status") if final_master.is_file() else "IN_PROGRESS"}
    (RESULT_ROOT / "final_validator.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
