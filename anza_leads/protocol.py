"""Frozen A0 protocol and label-only CRACKS split construction."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from cracks_experiment.matrix import PROJECT_ROOT
from cracks_experiment.partial_label_training import CRACKS_PROTOCOL
from datasets.cracks import BLUE, GREEN, ORANGE


A0_ROOT = PROJECT_ROOT / "results" / "anza_leads" / "a0"
A1_ROOT = PROJECT_ROOT / "results" / "anza_leads" / "a1_10pct_seed41"
CHECKPOINT_ROOT = PROJECT_ROOT.parent / "_wip_backups" / "anza_lira" / "anza_leads_a1_checkpoints"
VARIANTS = ("L0_backbone", "L1_isotropic", "L2_generic_aniso", "L3_anza_hs")
SEEDS = (41, 42, 43)
LABEL_FRACTIONS = (0.05, 0.10, 0.25, 1.00)


PROTOCOL: dict[str, Any] = {
    "name": "ANZA_LIRA_LEADS_V1",
    "active_phase": "A0_to_A1_only",
    "hypothesis": "fixed reciprocal ANZA geometry reduces sample complexity relative to initialized-equivalent free anisotropy",
    "matrix": list(VARIANTS),
    "operator": {
        "source": "anza_hs.operators imported unchanged",
        "orientations": 8,
        "kernel_size": 9,
        "base_scale": 1.5,
        "hyperbolicity": 0.35,
        "placements": ["decoder_quarter", "decoder_half"],
        "independent_sigmoid": True,
        "residual_gamma_initial": 0.0,
    },
    "data": {
        "dataset": "CRACKS",
        "split": "contiguous spatial section blocks with four-section exclusion buffers",
        "development_fraction": 0.20,
        "calibration_sections": 32,
        "buffer_sections_each_boundary": 4,
        "training_annotators": CRACKS_PROTOCOL["setting_a"]["training_annotators"],
        "heldout_annotators": CRACKS_PROTOCOL["setting_a"]["held_out_annotators"]["all"],
        "partial_semantics": {
            "blue": [1.0, 1.0], "green": [1.0, 0.5],
            "orange": [0.0, 1.0], "white": [0.0, 0.0],
        },
        "expert": "LOCKED_NOT_ACCESSED",
    },
    "subsets": {
        "fractions": list(LABEL_FRACTIONS),
        "seeds": list(SEEDS),
        "stratification": "label-only explicit-positive fraction quintiles; deterministic within-bin SHA256 ordering; nested prefixes",
    },
    "orientation_auxiliary": {
        "enabled_equally_all_variants": True,
        "weight": 0.10,
        "radius": 5,
        "minimum_positive_neighbors": 5,
        "sigma_theta": 0.20,
        "valid": "blue weight 1; green weight 0.5; orange and white invalid",
    },
    "training": {
        "active_seed": 41,
        "active_fraction": 0.10,
        "epochs": 20,
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "crop_size": 256,
        "foreground_crop_probability": 0.70,
        "annotators_per_section": 4,
        "effective_batch_size": 4,
        "loss": "mean annotator masked BCE+Dice+0.2 soft-clDice plus 0.10 equal orientation auxiliary",
        "topology_iterations": 5,
        "variant_specific_augmentation": False,
    },
    "calibration": {
        "threshold_grid": [round(0.05 * value, 2) for value in range(1, 20)],
        "precision_target": 0.90,
        "objective": "maximum macro-section clDice subject to macro explicit precision >= 0.90; infeasible uses highest precision",
        "development_access": False,
    },
    "gate": {
        "primary": "L3_anza_hs versus L2_generic_aniso",
        "dice_delta_min": -0.005,
        "cldice_delta_min": 0.015,
        "fragmentation_ratio_max": 0.80,
        "unknown_white_foreground_ratio_max": 1.10,
        "near_equal_repair_diagnostic": {"cldice_delta_min": -0.003, "fragmentation_ratio_max": 1.05},
        "scale_repair": {
            "median_lambda_min": 0.20,
            "median_lambda_max": 0.50,
            "systematic_scale_difference_min": 0.15,
            "evidence_activation_min": 0.02,
            "evidence_activation_max": 0.98,
        },
    },
    "locks": {
        "seeds_42_43": False,
        "anza_ms": False,
        "ssl": False,
        "domain_shift": False,
        "oof": False,
        "expert": False,
        "lira": False,
    },
}


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_hash() -> str:
    return canonical_hash(PROTOCOL)


def _available(section_ids: Iterable[int], annotators: Iterable[str]) -> list[int]:
    root = PROJECT_ROOT / "data" / "cracks" / "annotations"
    names = tuple(str(value) for value in annotators)
    return [
        int(section_id) for section_id in section_ids
        if (PROJECT_ROOT / "data" / "cracks" / "images" / f"section_{int(section_id):03d}.png").is_file()
        and any((root / name / f"section_{int(section_id):03d}.png").is_file() for name in names)
    ]


def build_split_manifest() -> dict[str, Any]:
    training_ids = set(_available(
        CRACKS_PROTOCOL["setting_a"]["training_section_ids"],
        PROTOCOL["data"]["training_annotators"],
    ))
    evaluation_ids = set(_available(
        CRACKS_PROTOCOL["setting_a"]["held_out_validation_section_ids"],
        PROTOCOL["data"]["heldout_annotators"],
    ))
    eligible = sorted(training_ids & evaluation_ids)
    if len(eligible) < 160:
        raise ValueError("CRACKS does not support the frozen section-disjoint LEADS split")
    development_count = int(round(len(eligible) * float(PROTOCOL["data"]["development_fraction"])))
    buffer_count = int(PROTOCOL["data"]["buffer_sections_each_boundary"])
    calibration_count = int(PROTOCOL["data"]["calibration_sections"])
    development = eligible[-development_count:]
    dev_buffer = eligible[-development_count - buffer_count:-development_count]
    cal_end = -development_count - buffer_count
    calibration = eligible[cal_end - calibration_count:cal_end]
    train_buffer = eligible[cal_end - calibration_count - buffer_count:cal_end - calibration_count]
    training_pool = eligible[:cal_end - calibration_count - buffer_count]
    groups = [set(training_pool), set(train_buffer), set(calibration), set(dev_buffer), set(development)]
    if any(groups[i] & groups[j] for i in range(len(groups)) for j in range(i + 1, len(groups))):
        raise AssertionError("LEADS section partitions overlap")
    manifest = {
        "version": "anza_leads_cracks_split_v1",
        "eligible_section_count": len(eligible),
        "training_pool": training_pool,
        "train_calibration_buffer": train_buffer,
        "calibration": calibration,
        "calibration_development_buffer": dev_buffer,
        "development": development,
        "spatial_order": "ascending official section ID",
        "expert_data_accessed": False,
        "training_evaluation_annotators_disjoint": not bool(
            set(PROTOCOL["data"]["training_annotators"]) & set(PROTOCOL["data"]["heldout_annotators"])
        ),
    }
    manifest["sha256"] = canonical_hash(manifest)
    return manifest


def _section_label_stats(section_id: int) -> dict[str, Any]:
    root = PROJECT_ROOT / "data" / "cracks" / "annotations"
    counts = Counter()
    available = 0
    name = f"section_{section_id:03d}.png"
    colors = {"blue": np.asarray(BLUE), "green": np.asarray(GREEN), "orange": np.asarray(ORANGE)}
    for annotator in PROTOCOL["data"]["training_annotators"]:
        path = root / annotator / name
        if not path.is_file():
            continue
        with Image.open(path) as handle:
            rgb = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        for key, color in colors.items():
            counts[key] += int(np.count_nonzero(np.all(rgb == color, axis=-1)))
        available += 1
    explicit = counts["blue"] + counts["green"] + counts["orange"]
    positive = counts["blue"] + counts["green"]
    return {
        "section_id": int(section_id),
        "available_training_annotators": int(available),
        "blue_pixels": int(counts["blue"]),
        "green_pixels": int(counts["green"]),
        "orange_pixels": int(counts["orange"]),
        "explicit_positive_fraction": float(positive / explicit) if explicit else 0.0,
        "green_positive_fraction": float(counts["green"] / positive) if positive else 0.0,
    }


def build_label_subset_manifest(split_manifest: dict[str, Any]) -> dict[str, Any]:
    stats = [_section_label_stats(section_id) for section_id in split_manifest["training_pool"]]
    values = np.asarray([row["explicit_positive_fraction"] for row in stats], dtype=np.float64)
    boundaries = np.quantile(values, [0.2, 0.4, 0.6, 0.8]).tolist()
    for row in stats:
        row["positive_quintile"] = int(np.searchsorted(boundaries, row["explicit_positive_fraction"], side="right"))
    subsets: dict[str, Any] = {}
    for seed in SEEDS:
        bins: dict[int, list[int]] = {index: [] for index in range(5)}
        for row in stats:
            bins[int(row["positive_quintile"])].append(int(row["section_id"]))
        for index in bins:
            bins[index].sort(key=lambda section: hashlib.sha256(f"LEADS:{seed}:{section}".encode()).hexdigest())
        ordering = []
        while any(bins.values()):
            for index in range(5):
                if bins[index]:
                    ordering.append(bins[index].pop(0))
        by_fraction = {}
        previous: set[int] = set()
        for fraction in LABEL_FRACTIONS:
            count = len(ordering) if fraction == 1.0 else max(1, int(round(len(ordering) * fraction)))
            selected = ordering[:count]
            if not previous.issubset(selected):
                raise AssertionError("LEADS label subsets are not nested")
            previous = set(selected)
            by_fraction[f"{int(fraction * 100)}pct"] = selected
        subsets[str(seed)] = by_fraction
    manifest = {
        "version": "anza_leads_nested_label_subsets_v1",
        "training_pool_sha256": canonical_hash(split_manifest["training_pool"]),
        "quintile_boundaries": boundaries,
        "section_stats": stats,
        "subsets": subsets,
        "selection_uses_model_outputs": False,
        "expert_data_accessed": False,
    }
    manifest["sha256"] = canonical_hash(manifest)
    return manifest


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def freeze_a0() -> dict[str, Any]:
    A0_ROOT.mkdir(parents=True, exist_ok=True)
    protocol_path = A0_ROOT / "protocol.json"
    split_path = A0_ROOT / "split_manifest.json"
    subset_path = A0_ROOT / "label_subset_manifest.json"
    if protocol_path.exists() or split_path.exists() or subset_path.exists():
        current = json.loads(protocol_path.read_text())
        if canonical_hash(current) != protocol_hash():
            raise ValueError("existing A0 protocol drift")
        split = json.loads(split_path.read_text())
        subsets = json.loads(subset_path.read_text())
        return {"protocol": current, "split": split, "subsets": subsets, "action": "SKIP"}
    split = build_split_manifest()
    subsets = build_label_subset_manifest(split)
    write_json(protocol_path, PROTOCOL)
    (A0_ROOT / "protocol_hash.txt").write_text(protocol_hash() + "\n")
    write_json(split_path, split)
    write_json(subset_path, subsets)
    write_json(A0_ROOT / "data_access_log.json", {
        "images": True, "training_nonexpert_annotations": True,
        "heldout_nonexpert_annotations": True, "expert_directory_traversed": False,
        "expert_masks_read": False, "development_model_outputs_read": False,
    })
    return {"protocol": PROTOCOL, "split": split, "subsets": subsets, "action": "RUN"}


def active_manifests() -> tuple[dict[str, Any], dict[str, Any]]:
    split = json.loads((A0_ROOT / "split_manifest.json").read_text())
    subsets = json.loads((A0_ROOT / "label_subset_manifest.json").read_text())
    if split.get("sha256") != canonical_hash({key: value for key, value in split.items() if key != "sha256"}):
        raise ValueError("split manifest hash drift")
    if subsets.get("sha256") != canonical_hash({key: value for key, value in subsets.items() if key != "sha256"}):
        raise ValueError("label subset manifest hash drift")
    return split, subsets


def expected_fixed_scales() -> tuple[float, float]:
    ell = float(PROTOCOL["operator"]["base_scale"])
    lam = float(PROTOCOL["operator"]["hyperbolicity"])
    return ell * math.exp(lam), ell * math.exp(-lam)
