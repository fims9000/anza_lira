"""Restart-safe state machine for the frozen V1.1 endgame."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from structural_stability_v1_1.endgame_evaluation import run_calibration, run_development
from structural_stability_v1_1.endgame_reporting import finalize_terminal
from structural_stability_v1_1.endgame_training import complete_training_manifest, run_training_job
from structural_stability_v1_1.protocol import RESULT_ROOT, SEEDS, VARIANTS


def run(*, device: str = "cuda", smoke: bool = False) -> dict[str, Any]:
    if smoke:
        return run_training_job("B3", 41, device=device, max_steps=1, smoke=True)
    for seed in SEEDS:
        for variant in VARIANTS:
            final = RESULT_ROOT / f"training/{variant}/s{seed}/RUN_FINAL_VALIDATION.json"
            if final.is_file() and json.loads(final.read_text()).get("status") == "RUN_FINAL_VALIDATION_PASS":
                print(f"phase=SS2_TRAIN variant={variant} seed={seed} action=SKIP_FINAL", flush=True)
                continue
            run_training_job(variant, seed, device=device)
    completion = complete_training_manifest()
    if completion["status"] != "SS2_SS3_TRAINING_COMPLETE":
        return completion
    calibration_path = RESULT_ROOT / "calibration/CALIBRATION_FREEZE.json"
    calibration = json.loads(calibration_path.read_text()) if calibration_path.is_file() else run_calibration(device=device)
    if calibration["status"] != "SS_CALIBRATION_FROZEN":
        return calibration
    development_path = RESULT_ROOT / "development/DEVELOPMENT_MASTER_RESULT.json"
    development = json.loads(development_path.read_text()) if development_path.is_file() else run_development(device=device)
    if development["status"] == "ANZA_STABILITY_MULTISEED_DEV_PASS":
        # The frozen protocol authorizes a separate deterministic LIRA phase here.
        # Never silently substitute an unvalidated implementation.
        return {"status": "ANZA_STABILITY_MULTISEED_DEV_PASS", "next_phase": "LIRA_DEVELOPMENT_AUTHORIZED", "development": development}
    return finalize_terminal()
