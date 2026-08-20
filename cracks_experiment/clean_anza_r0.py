"""Independent three-seed CRACKS R0 for the corrected CleanANZA baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from cracks_experiment.evaluation import verify_threshold_freeze
from cracks_experiment.matrix import PROJECT_ROOT, SETTING_A_PROTOCOL, setting_a_protocol_hash
from cracks_experiment.training import run_setting_a_training
from cracks_experiment.validation import _sha256, run_setting_a_validation


LEGACY_TRAINING_ROOT = PROJECT_ROOT / "results" / "anza_v2_study" / "cracks" / "setting_a"
R0_ROOT = PROJECT_ROOT / "results" / "maxmin_path_study" / "cracks" / "r0_clean_anza"
CLEAN_SOURCE = PROJECT_ROOT / "models" / "azconv_clean.py"
R0_PROTOCOL = {
    "version": "clean_anza_cracks_r0_v1",
    "setting_a_protocol_hash": setting_a_protocol_hash(),
    "policy": SETTING_A_PROTOCOL["policy"],
    "epochs": SETTING_A_PROTOCOL["epochs"],
    "optimizer": SETTING_A_PROTOCOL["optimizer"],
    "learning_rate": SETTING_A_PROTOCOL["learning_rate"],
    "effective_batch_size": SETTING_A_PROTOCOL["effective_batch_size"],
    "crop_size": SETTING_A_PROTOCOL["crop_size"],
    "foreground_crop_probability": SETTING_A_PROTOCOL["foreground_crop_probability"],
    "real_loss": SETTING_A_PROTOCOL["real_loss"],
    "threshold_candidates": SETTING_A_PROTOCOL["threshold_candidates"],
    "seeds": [41, 42, 43],
    "reference_models": ["unet", "anza_v1"],
    "candidate": "clean_anza",
    "expert": "LOCKED_NOT_USED_FOR_SELECTION",
    "clean_source_sha256": hashlib.sha256(CLEAN_SOURCE.read_bytes()).hexdigest(),
}


@dataclass(frozen=True)
class CleanR0Spec:
    run_id: str
    model: str
    seed: int
    structural_replay: bool = False
    use_fuzzy: bool = True
    directional_half_modes: bool = True
    comparison_family: str = "clean_anza_r0"

    @property
    def run_hash(self) -> str:
        payload = {"spec": asdict(self), "protocol": R0_PROTOCOL}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def clean_r0_specs() -> tuple[CleanR0Spec, ...]:
    return tuple(CleanR0Spec(f"clean_anza_s{seed}", "clean_anza", seed) for seed in R0_PROTOCOL["seeds"])


def audit_r0_reuse_contract() -> dict[str, Any]:
    legacy_freeze = verify_threshold_freeze(LEGACY_TRAINING_ROOT)
    by_run = {row["run_id"]: row for row in legacy_freeze["runs"]}
    reused = []
    for model, prefix in (("unet", "unet"), ("anza_v1", "v1")):
        for seed in R0_PROTOCOL["seeds"]:
            run_id = f"{prefix}_s{seed}"
            row = by_run[run_id]
            reused.append({"model": model, "seed": seed, **row})
    result = {
        "status": "PASS",
        "r0_protocol": R0_PROTOCOL,
        "legacy_threshold_freeze_sha256": legacy_freeze["freeze_sha256"],
        "reused_runs": reused,
        "clean_runs": [asdict(spec) | {"run_hash": spec.run_hash} for spec in clean_r0_specs()],
        "expert_scores_used_for_selection": False,
        "expert_data_accessed": False,
    }
    R0_ROOT.mkdir(parents=True, exist_ok=True)
    path = R0_ROOT / "reuse_contract.json"
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != encoded:
        raise ValueError("R0 reuse contract drift")
    path.write_text(encoded)
    return result


def run_r0_training(*, device: str = "cuda", epochs: int | None = None, max_train_sections: int | None = None) -> list[dict[str, Any]]:
    audit_r0_reuse_contract()
    return [
        run_setting_a_training(
            spec,
            R0_ROOT,
            epochs=epochs,
            max_train_sections=max_train_sections,
            device=device,
        )
        for spec in clean_r0_specs()
    ]


def freeze_clean_thresholds(*, device: str = "cuda", max_sections: int | None = None) -> dict[str, Any]:
    audit = audit_r0_reuse_contract()
    rows = []
    for spec in clean_r0_specs():
        validation = run_setting_a_validation(
            spec,
            R0_ROOT,
            device=device,
            max_sections=max_sections,
        )
        if validation.get("expert_scores_used") is not False:
            raise ValueError("expert lock failed in CleanANZA R0 threshold selection")
        run_dir = R0_ROOT / f"{spec.run_id}-{spec.run_hash}"
        rows.append({
            "run_id": spec.run_id,
            "run_hash": spec.run_hash,
            "seed": spec.seed,
            "checkpoint_sha256": _sha256(run_dir / "checkpoint-last.pt"),
            "validation_sha256": _sha256(run_dir / "crowd_validation.json"),
            "selected_threshold": validation["selected_threshold"],
            "section_count": validation["section_count"],
        })
    core = {
        "status": "FROZEN",
        "r0_protocol": R0_PROTOCOL,
        "legacy_threshold_freeze_sha256": audit["legacy_threshold_freeze_sha256"],
        "selection_source": "held-out non-expert crowd annotations only",
        "section_limit": max_sections,
        "expert_scores_used": False,
        "runs": rows,
    }
    digest = hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    result = {**core, "freeze_sha256": digest}
    path = R0_ROOT / "clean_threshold_freeze.json"
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != encoded:
        raise ValueError("CleanANZA threshold freeze drift")
    path.write_text(encoded)
    return result

