"""Frozen protocol and cross-fit split for LEADS RC1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from cracks_experiment.matrix import PROJECT_ROOT

from .protocol import canonical_hash, file_sha256, write_json


ROOT = PROJECT_ROOT / "results" / "anza_leads" / "rc1"
PARENT = PROJECT_ROOT / "results" / "anza_leads" / "a1_10pct_seed41"
PARENT_A0 = PROJECT_ROOT / "results" / "anza_leads" / "a0"
CHECKPOINT_ROOT = PROJECT_ROOT.parent / "_wip_backups" / "anza_lira" / "anza_leads_rc1_checkpoints"
VARIANTS = ("L0_backbone", "L2_generic_aniso", "L3_anza_hs")


BASE_PROTOCOL: dict[str, Any] = {
    "name": "ANZA_LEADS_RC1_RISK_FRONTIER_V1",
    "parent_status_immutable": "STOP_ANZA_LABEL_EFFICIENCY_NO_SIGNAL",
    "scientific_change": "fresh cross-fit sections plus uncapped score calibration and unsupported-white safety only",
    "variants": list(VARIANTS),
    "seed": 41,
    "optimization_fraction": 0.10,
    "training": {
        "epochs": 20, "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4,
        "crop_size": 256, "foreground_crop_probability": 0.70,
        "annotators_per_section": 4, "effective_batch_size": 4,
        "loss": "parent mean-annotator masked BCE+Dice+0.2 clDice plus equal 0.10 orientation auxiliary",
    },
    "calibration": {
        "precision_target": 0.90,
        "explicit_thresholds": [0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999],
        "quantile_candidates": 4097,
        "selection": "max macro-section clDice subject macro precision >=0.90 and recall>0; ties Dice, recall, threshold",
    },
    "development_precision_min": 0.88,
    "gate": {
        "cldice_delta_min": 0.015, "cldice_ci_low_min_exclusive": 0.0,
        "dice_delta_min": -0.005, "backbone_cldice_delta_min": -0.005,
        "backbone_dice_delta_min": -0.005, "auprc_delta_min": -0.005,
        "unsupported_white_ratio_max": 1.10,
    },
    "unsupported_white": {
        "distance_px": 5, "connectivity": 8,
        "definition": "white predicted pixel farther than 5px from explicit positive and in component not touching 5px positive dilation",
    },
    "frontier_precision_points": [0.80, 0.82, 0.84, 0.86, 0.88, 0.90],
    "bootstrap_resamples": 10_000,
    "expert": "LOCKED_NOT_ACCESSED",
    "locks": {"seeds_42_43": False, "anza_ms": False, "ssl": False, "domain_shift": False, "lira": False, "expert": False},
}


def tree_hash(path: Path, *, exclude_npz: bool = False) -> str:
    digest = hashlib.sha256()
    for item in sorted(Path(path).rglob("*")):
        if not item.is_file() or (exclude_npz and item.suffix == ".npz"):
            continue
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(file_sha256(item).encode())
    return digest.hexdigest()


def build_split() -> dict[str, Any]:
    old_split = json.loads((PARENT_A0 / "split_manifest.json").read_text())
    old_subsets = json.loads((PARENT_A0 / "label_subset_manifest.json").read_text())
    old_active = set(old_subsets["subsets"]["41"]["10pct"])
    old_selection = set(old_split["calibration"]) | set(old_split["development"])
    fresh = [value for value in old_split["training_pool"] if value not in old_active]
    development = fresh[:50]
    development_buffer = fresh[50:54]
    calibration = fresh[54:86]
    calibration_buffer = fresh[86:90]
    training_pool = fresh[90:]
    groups = [set(training_pool), set(calibration_buffer), set(calibration), set(development_buffer), set(development)]
    if any(groups[i] & groups[j] for i in range(len(groups)) for j in range(i + 1, len(groups))):
        raise AssertionError("RC1 partitions overlap")
    if (set(calibration) | set(development)) & (old_active | old_selection):
        raise AssertionError("RC1 evaluation is not fresh from parent decisions")
    stats = {int(row["section_id"]): row for row in old_subsets["section_stats"]}
    bins: dict[int, list[int]] = {index: [] for index in range(5)}
    for section in training_pool:
        bins[int(stats[section]["positive_quintile"])].append(section)
    for index in bins:
        bins[index].sort(key=lambda section: hashlib.sha256(f"LEADS-RC1:41:{section}".encode()).hexdigest())
    ordering = []
    while any(bins.values()):
        for index in range(5):
            if bins[index]:
                ordering.append(bins[index].pop(0))
    active_count = max(1, int(round(0.10 * len(ordering))))
    active = ordering[:active_count]
    manifest = {
        "version": "anza_leads_rc1_crossfit_v1",
        "source_parent_split_sha256": old_split["sha256"],
        "training_pool": training_pool, "calibration_buffer": calibration_buffer,
        "calibration": calibration, "development_buffer": development_buffer,
        "development": development, "optimization_10pct_seed41": active,
        "old_a1_active_sections_excluded_from_rc1_evaluation": sorted(old_active),
        "old_a1_selection_sections_excluded_from_rc1": sorted(old_selection),
        "historical_train_overlap_note": "RC1 evaluation sections came from the old label-audited train pool but were not old optimization, calibration, or development sections.",
        "expert_data_accessed": False,
    }
    manifest["sha256"] = canonical_hash(manifest)
    return manifest


def freeze_protocol() -> dict[str, Any]:
    ROOT.mkdir(parents=True, exist_ok=True)
    parent_hash = tree_hash(PARENT, exclude_npz=True)
    protocol = {**BASE_PROTOCOL, "parent_a1_tree_sha256_excluding_caches": parent_hash}
    split = build_split()
    protocol_sha = canonical_hash(protocol)
    targets = {"protocol.json": protocol, "split_manifest.json": split}
    if (ROOT / "protocol.json").exists():
        existing = json.loads((ROOT / "protocol.json").read_text())
        if canonical_hash(existing) != protocol_sha:
            raise ValueError("RC1 protocol drift")
        existing_split = json.loads((ROOT / "split_manifest.json").read_text())
        if existing_split != split:
            raise ValueError("RC1 split drift")
        return {"protocol": existing, "split": existing_split, "action": "SKIP"}
    for name, payload in targets.items():
        write_json(ROOT / name, payload)
    (ROOT / "protocol_hash.txt").write_text(protocol_sha + "\n")
    stats = {int(row["section_id"]): row for row in json.loads((PARENT_A0 / "label_subset_manifest.json").read_text())["section_stats"]}
    active_rows = [stats[value] for value in split["optimization_10pct_seed41"]]
    write_json(ROOT / "label_budget.json", {
        "optimization_fraction": 0.10, "optimization_sections": len(active_rows),
        "training_pool_sections": len(split["training_pool"]), "calibration_sections": len(split["calibration"]),
        "development_sections": len(split["development"]),
        "blue_pixels": sum(int(row["blue_pixels"]) for row in active_rows),
        "green_pixels": sum(int(row["green_pixels"]) for row in active_rows),
        "orange_pixels": sum(int(row["orange_pixels"]) for row in active_rows),
        "same_for_all_variants": True, "expert_data_accessed": False,
    })
    write_json(ROOT / "parent_freeze.json", {
        "path": str(PARENT), "tree_sha256_excluding_caches": parent_hash,
        "canonical_status": "STOP_ANZA_LABEL_EFFICIENCY_NO_SIGNAL",
    })
    return {"protocol": protocol, "split": split, "action": "RUN"}


def load_frozen() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = json.loads((ROOT / "protocol.json").read_text())
    split = json.loads((ROOT / "split_manifest.json").read_text())
    if canonical_hash(protocol) != (ROOT / "protocol_hash.txt").read_text().strip():
        raise ValueError("RC1 protocol hash drift")
    if split["sha256"] != canonical_hash({k: v for k, v in split.items() if k != "sha256"}):
        raise ValueError("RC1 split hash drift")
    return protocol, split


def verify_parent_immutable() -> bool:
    frozen = json.loads((ROOT / "parent_freeze.json").read_text())
    return tree_hash(PARENT, exclude_npz=True) == frozen["tree_sha256_excluding_caches"]
