#!/usr/bin/env python3
"""Build and audit the source-grouped ``geocrack_small_v1`` split."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.geocrack import compute_train_normalization, discover_pairs, sha256_file
from scripts.audit_geocrack_sites import audit_site_mapping
from scripts.check_geocrack_split import assert_no_source_leakage, freeze_or_verify_test_split


DEFAULT_TARGETS = {"train": 1245, "val": 300, "test": 450}
DEFAULT_MIN_SOURCES = {"train": 6, "val": 3, "test": 3}
FIELDS = ["patch_id", "source_image_id", "image_path", "mask_path"]


def grouped_small_split(
    rows: Sequence[Mapping[str, str]],
    *,
    targets: Mapping[str, int] = DEFAULT_TARGETS,
    seed: int = 2026,
    min_sources: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[str(row["source_image_id"])].append(dict(row))
    if len(groups) < len(targets):
        raise ValueError(f"Need at least {len(targets)} source groups, found {len(groups)}")
    rng = random.Random(seed)
    remaining = list(groups)
    rng.shuffle(remaining)
    tie_rank = {source: rank for rank, source in enumerate(remaining)}
    output = {name: [] for name in targets}
    min_sources = dict(min_sources or {name: 1 for name in targets})
    if set(min_sources) != set(targets) or any(int(value) < 1 for value in min_sources.values()):
        raise ValueError("min_sources must contain one positive value for every split")
    if sum(int(value) for value in min_sources.values()) > len(groups):
        raise ValueError(f"Requested {sum(min_sources.values())} source groups but only {len(groups)} exist")

    # Seed every partition with several source images. This makes source-level
    # uncertainty estimable instead of accidentally choosing one large source
    # that happens to match the requested patch count.
    for round_index in range(max(min_sources.values())):
        for split_name in sorted(targets, key=lambda name: targets[name]):
            required = int(min_sources[split_name])
            if round_index >= required:
                continue
            desired_now = targets[split_name] * float(round_index + 1) / required
            source = min(
                remaining,
                key=lambda item: (abs((len(output[split_name]) + len(groups[item])) - desired_now), tie_rank[item]),
            )
            remaining.remove(source)
            output[split_name].extend(groups[source])

    target_total = sum(targets.values())
    while remaining:
        assigned_total = sum(len(part) for part in output.values())
        best: tuple[float, str, int, str] | None = None
        for source in remaining:
            group_size = len(groups[source])
            total_cost = abs((assigned_total + group_size) - target_total)
            for split_name, target in targets.items():
                split_cost = abs((len(output[split_name]) + group_size) - target) / max(target, 1)
                candidate = (total_cost + split_cost, split_name, tie_rank[source], source)
                if best is None or candidate < best:
                    best = candidate
        assert best is not None
        _, split_name, _, source = best
        before = abs(assigned_total - target_total)
        after = abs((assigned_total + len(groups[source])) - target_total)
        if assigned_total >= target_total and after >= before:
            break
        output[split_name].extend(groups[source])
        remaining.remove(source)

    for part in output.values():
        part.sort(key=lambda row: row["patch_id"])
    source_sets = {name: {row["source_image_id"] for row in part} for name, part in output.items()}
    assert_no_source_leakage(source_sets["train"], source_sets["val"], source_sets["test"])
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _split_stats(root: Path, rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    foreground = 0
    pixels = 0
    for row in rows:
        mask = np.asarray(Image.open(root / row["mask_path"]).convert("L"), dtype=np.uint8) > 0
        foreground += int(mask.sum())
        pixels += int(mask.size)
    return {
        "source_image_count": len({row["source_image_id"] for row in rows}),
        "patch_count": len(rows),
        "foreground_pixels": foreground,
        "foreground_fraction": foreground / max(pixels, 1),
        "source_image_ids": sorted({row["source_image_id"] for row in rows}),
    }


def prepare(
    root: Path,
    output_dir: Path,
    *,
    seed: int = 2026,
    official_site_mapping: Path | None = None,
) -> dict[str, Any]:
    rows = discover_pairs(root)
    splits = grouped_small_split(rows, targets=DEFAULT_TARGETS, seed=seed, min_sources=DEFAULT_MIN_SOURCES)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, part in splits.items():
        path = output_dir / f"geocrack_small_v1_{name}.csv"
        _write_csv(path, part)
        paths[name] = path
    source_sets = {name: {row["source_image_id"] for row in part} for name, part in splits.items()}
    assert_no_source_leakage(source_sets["train"], source_sets["val"], source_sets["test"])
    frozen_test_hash = freeze_or_verify_test_split(paths["test"], output_dir / "test_split.sha256")
    site_audit = audit_site_mapping(
        {row["source_image_id"] for row in rows},
        official_site_mapping,
        output_path=output_dir / "site_mapping_audit.json",
    )
    manifest = {
        "name": "geocrack_small_v1",
        "split_seed": seed,
        "dataset_pair_count": len(rows),
        "targets": DEFAULT_TARGETS,
        "minimum_source_images": DEFAULT_MIN_SOURCES,
        "splits": {
            name: {**_split_stats(root, splits[name]), "csv": path.as_posix(), "csv_sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "unused_patch_count": len(rows) - sum(len(part) for part in splits.values()),
        "source_leakage": 0,
        "frozen_test_csv_sha256": frozen_test_hash,
        "test_split_contract": (output_dir / "test_split.sha256").as_posix(),
        "site_mapping_status": site_audit["site_mapping_status"],
    }
    manifest_path = output_dir / "geocrack_small_v1_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    compute_train_normalization(root, paths["train"], output_dir / "train_normalization.json")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/geocrack"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/geocrack/splits"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--official-site-mapping",
        type=Path,
        help="Official CSV with exact source_image_id,geological_site columns; filenames are never guessed",
    )
    args = parser.parse_args()
    manifest = prepare(args.root, args.output_dir, seed=args.seed, official_site_mapping=args.official_site_mapping)
    for name, stats in manifest["splits"].items():
        print(f"{name.upper()}: {stats['patch_count']} patches, {stats['source_image_count']} sources")
    print("SOURCE LEAKAGE: 0")
    print("STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
