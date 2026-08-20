"""Deterministic paired crop/perturbation manifests shared across B0-B3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch

from cracks_experiment.partial_labels import map_partial_annotation
from lira_final.protocol import TRAIN_ANNOTATORS
from structural_stability_v1.perturb.seeds import perturbation_seed
from structural_stability_v1.protocol import FAMILIES
from structural_stability_v1_1.amendment import sha256_file
from structural_stability_v1_1.protocol import PROTOCOL, PROTOCOL_ID, ROOT, SEEDS, VARIANTS, canonical_hash, protocol_hash


def _available(section_id: int) -> tuple[str, ...]:
    return tuple(
        annotator for annotator in TRAIN_ANNOTATORS
        if (ROOT / "data/cracks/annotations" / annotator / f"section_{section_id:03d}.png").is_file()
    )


def _aggregate_coordinates(section_id: int, annotators: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    positive = np.zeros((255, 701), dtype=bool)
    explicit = np.zeros((255, 701), dtype=bool)
    for annotator in annotators:
        path = ROOT / "data/cracks/annotations" / annotator / f"section_{section_id:03d}.png"
        with Image.open(path) as handle:
            target, weight = map_partial_annotation(np.asarray(handle.convert("RGB"), dtype=np.uint8))
        positive |= (target > 0.5) & (weight > 0)
        explicit |= weight > 0
    return np.argwhere(positive), np.argwhere(explicit)


def _pair_choice(seed: int, epoch: int, position: int) -> tuple[str, int]:
    text = f"{PROTOCOL_ID}|{seed}|{epoch}|{position}|pair"
    local = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    rng = np.random.default_rng(local)
    return FAMILIES[int(rng.integers(len(FAMILIES)))], int(rng.integers(1, 3))


def build_pair_manifests(section_ids: Sequence[int], output: Path) -> dict[str, Any]:
    ordered = [int(value) for value in section_ids]
    if len(ordered) != 220:
        raise ValueError("paired manifests require exactly 220 SS_TRAIN sections")
    epochs = int(PROTOCOL["training"]["epochs"])
    output.mkdir(parents=True, exist_ok=True)
    orders: dict[tuple[int, int], dict[int, int]] = {}
    rows: dict[int, list[dict[str, Any]]] = {seed: [] for seed in SEEDS}
    for seed in SEEDS:
        for epoch in range(epochs):
            order = torch.randperm(len(ordered), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
            orders[(seed, epoch)] = {dataset_index: position for position, dataset_index in enumerate(order)}
    for dataset_index, section_id in enumerate(ordered):
        available = _available(section_id)
        if len(available) < 4:
            raise ValueError(f"section {section_id} has fewer than four train annotators")
        positive, explicit = _aggregate_coordinates(section_id, available)
        if not explicit.size:
            raise ValueError(f"section {section_id} has no explicit nonexpert labels")
        for seed in SEEDS:
            for epoch in range(epochs):
                position = orders[(seed, epoch)][dataset_index]
                local_seed = seed + epoch * len(ordered) + dataset_index
                annotator_rng = np.random.default_rng(local_seed)
                chosen = sorted(annotator_rng.choice(len(available), size=4, replace=False).tolist())
                annotators = [available[index] for index in chosen]
                crop_rng = np.random.default_rng(local_seed)
                anchor = None
                if len(positive) and crop_rng.random() < 0.7:
                    anchor = positive[int(crop_rng.integers(len(positive)))]
                else:
                    top = int(crop_rng.integers(0, 1))
                    left = int(crop_rng.integers(0, 449))
                    inside = (
                        (explicit[:, 0] >= top) & (explicit[:, 0] < top + 256)
                        & (explicit[:, 1] >= left) & (explicit[:, 1] < left + 256)
                    )
                    if not np.any(inside):
                        anchor = explicit[int(crop_rng.integers(len(explicit)))]
                if anchor is not None:
                    top = min(max(int(anchor[0]) - 128, 0), 0)
                    left = min(max(int(anchor[1]) - 128, 0), 448)
                family, severity = _pair_choice(seed, epoch, position)
                crop_id = f"e{epoch:02d}_p{position:03d}_s{section_id:03d}_y{top}_x{left}"
                rows[seed].append({
                    "protocol": PROTOCOL_ID,
                    "seed": seed,
                    "epoch": epoch + 1,
                    "order_position": position,
                    "dataset_index": dataset_index,
                    "section_id": section_id,
                    "crop_top": top,
                    "crop_left": left,
                    "crop_size": 256,
                    "annotators": annotators,
                    "family": family,
                    "severity": severity,
                    "perturbation_seed": perturbation_seed(section_id, crop_id, family, severity),
                    "consumers": list(VARIANTS),
                })
        if (dataset_index + 1) % 25 == 0 or dataset_index + 1 == len(ordered):
            print(f"phase=SS1.5_MANIFEST section={dataset_index + 1}/{len(ordered)} expert=LOCKED", flush=True)
    records = []
    for seed in SEEDS:
        local_rows = sorted(rows[seed], key=lambda row: (row["epoch"], row["order_position"]))
        path = output / f"train_pair_manifest_seed{seed}.jsonl"
        with path.open("w") as handle:
            for row in local_rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        severity_values = {int(row["severity"]) for row in local_rows}
        records.append({
            "seed": seed,
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(path),
            "rows": len(local_rows),
            "epochs": epochs,
            "sections_per_epoch": len(ordered),
            "optimizer_updates": epochs * ((len(ordered) + 3) // 4),
            "severities": sorted(severity_values),
            "consumers": list(VARIANTS),
        })
    payload = {
        "status": "SHARED_PAIR_MANIFESTS_FROZEN",
        "protocol_sha256": protocol_hash(),
        "section_list_sha256": canonical_hash(ordered),
        "records": records,
        "expert_data_accessed": False,
        "development_confirm_accessed": False,
    }
    (output / "TRAINING_MANIFEST_PROTOCOL.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def assert_manifest_shared(record: dict[str, Any]) -> None:
    if tuple(record.get("consumers", ())) != VARIANTS:
        raise ValueError("pair manifest is not shared by B0-B3")
    if 3 in set(record.get("severities", ())):
        raise ValueError("severity 3 leaked into training manifest")


def _explicit_column_prefix(section_id: int, annotator: str) -> np.ndarray:
    path = ROOT / "data/cracks/annotations" / annotator / f"section_{section_id:03d}.png"
    with Image.open(path) as handle:
        _target, weight = map_partial_annotation(np.asarray(handle.convert("RGB"), dtype=np.uint8))
    column_counts = np.pad((weight > 0).sum(axis=0), (0, 3))
    return np.concatenate(([0], np.cumsum(column_counts)))


def selected_crop_has_explicit(prefixes: dict[str, np.ndarray], annotators: Sequence[str], left: int, size: int = 256) -> bool:
    right = int(left) + int(size)
    return sum(int(prefixes[name][right] - prefixes[name][int(left)]) for name in annotators) > 0


def validate_pair_manifest_crops(manifest_root: Path, output_path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        path = manifest_root / f"train_pair_manifest_seed{seed}.jsonl"
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    by_section: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_section.setdefault(int(row["section_id"]), []).append(row)
    failures = []
    for section_id, local_rows in sorted(by_section.items()):
        names = sorted({name for row in local_rows for name in row["annotators"]})
        prefixes = {name: _explicit_column_prefix(section_id, name) for name in names}
        for row in local_rows:
            if not selected_crop_has_explicit(prefixes, row["annotators"], int(row["crop_left"]), int(row["crop_size"])):
                failures.append({key: row[key] for key in ("seed", "epoch", "section_id", "crop_left", "annotators")})
    result = {
        "status": "PAIR_MANIFEST_CROP_VALIDATION_PASS" if not failures else "STOP_PAIR_MANIFEST_EMPTY_SELECTED_LABELS",
        "rows": len(rows), "sections": len(by_section),
        "selected_annotator_empty_crop_count": len(failures),
        "failures": failures[:100], "expert_data_accessed": False,
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
