#!/usr/bin/env python3
"""Audit the coordinate-blocked CRACKS split and freeze test IDs only when feasible."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "cracks" / "splits" / "split_manifest.json"
DEFAULT_TEST_HASH = PROJECT_ROOT / "data" / "cracks" / "splits" / "test_split.sha256"
INTERVALS = {
    "train": (1, 260),
    "guard_1": (261, 280),
    "validation": (281, 320),
    "guard_2": (321, 340),
    "test": (341, 400),
}
MINIMUM_COUNTS = {"train": 200, "validation": 25, "test": 40}


def canonical_test_bytes(test_ids: Sequence[int]) -> bytes:
    return (json.dumps(sorted(int(value) for value in test_ids), separators=(",", ":")) + "\n").encode("utf-8")


def validate_split_manifest(manifest: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    assignments = {name: [int(value) for value in manifest.get("assignments", {}).get(name, [])] for name in INTERVALS}
    sets = {name: set(values) for name, values in assignments.items()}
    names = tuple(sets)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            overlap = sets[first] & sets[second]
            if overlap:
                failures.append(f"{first}/{second} overlap: {sorted(overlap)}")
    for name, (lower, upper) in INTERVALS.items():
        out_of_interval = sorted(value for value in sets[name] if not lower <= value <= upper)
        if out_of_interval:
            failures.append(f"{name} IDs outside [{lower}, {upper}]: {out_of_interval}")
    for name, minimum in MINIMUM_COUNTS.items():
        if len(sets[name]) < minimum:
            failures.append(f"{name} has {len(sets[name])} sections; minimum is {minimum}")
    if sets["train"] and sets["validation"] and max(sets["train"]) >= min(sets["validation"]):
        failures.append("max(train ID) is not below min(validation ID)")
    if sets["validation"] and sets["test"] and max(sets["validation"]) >= min(sets["test"]):
        failures.append("max(validation ID) is not below min(test ID)")
    declared_counts = manifest.get("counts", {})
    for name, values in assignments.items():
        if name in declared_counts and int(declared_counts[name]) != len(values):
            failures.append(f"{name} declared count disagrees with actual IDs")
    return failures


def freeze_or_verify_test_ids(test_ids: Sequence[int], checksum_path: Path) -> str:
    digest = hashlib.sha256(canonical_test_bytes(test_ids)).hexdigest()
    if checksum_path.exists():
        frozen = checksum_path.read_text(encoding="utf-8").strip()
        if frozen != digest:
            raise ValueError(f"Frozen CRACKS test IDs changed: expected {frozen}, got {digest}")
    else:
        checksum_path.parent.mkdir(parents=True, exist_ok=True)
        checksum_path.write_text(digest + "\n", encoding="utf-8")
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--test-hash", type=Path, default=DEFAULT_TEST_HASH)
    parser.add_argument("--freeze-test", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    failures = validate_split_manifest(manifest)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        print("CRACKS SPLIT: INFEASIBLE")
        return 1
    if args.freeze_test:
        digest = freeze_or_verify_test_ids(manifest["assignments"]["test"], args.test_hash)
        print(f"CRACKS TEST SHA256: {digest}")
    print("CRACKS SPLIT: VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
