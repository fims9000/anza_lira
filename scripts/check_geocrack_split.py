#!/usr/bin/env python3
"""Reject any source-image overlap between frozen GeoCrack CSV splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Iterable


def load_sources(path: str | Path) -> set[str]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "source_image_id" not in rows[0]:
        raise ValueError(f"Missing source_image_id rows in {path}")
    return {row["source_image_id"] for row in rows}


def source_overlaps(train: Iterable[str], val: Iterable[str], test: Iterable[str]) -> dict[str, set[str]]:
    train_set, val_set, test_set = set(train), set(val), set(test)
    return {
        "TRAIN/VAL": train_set & val_set,
        "TRAIN/TEST": train_set & test_set,
        "VAL/TEST": val_set & test_set,
    }


def assert_no_source_leakage(train: Iterable[str], val: Iterable[str], test: Iterable[str]) -> None:
    overlaps = source_overlaps(train, val, test)
    leaking = {name: values for name, values in overlaps.items() if values}
    if leaking:
        detail = "; ".join(f"{name}: {sorted(values)}" for name, values in leaking.items())
        raise ValueError(f"GeoCrack source leakage detected: {detail}")


def freeze_or_verify_test_split(test_csv: str | Path, checksum_path: str | Path) -> str:
    """Create the immutable test contract once, then fail on every mutation."""
    test_csv = Path(test_csv)
    checksum_path = Path(checksum_path)
    digest = hashlib.sha256(test_csv.read_bytes()).hexdigest()
    if checksum_path.exists():
        frozen = checksum_path.read_text(encoding="utf-8").strip()
        if frozen != digest:
            raise ValueError(f"Frozen GeoCrack test CSV hash changed: expected {frozen}, got {digest}")
    else:
        checksum_path.parent.mkdir(parents=True, exist_ok=True)
        checksum_path.write_text(digest + "\n", encoding="utf-8")
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--test-checksum", type=Path)
    args = parser.parse_args()
    overlaps = source_overlaps(load_sources(args.train), load_sources(args.val), load_sources(args.test))
    for name, values in overlaps.items():
        print(f"{name} source overlap: {len(values)}")
    assert_no_source_leakage(load_sources(args.train), load_sources(args.val), load_sources(args.test))
    checksum_path = args.test_checksum or args.test.with_name("test_split.sha256")
    digest = freeze_or_verify_test_split(args.test, checksum_path)
    print(f"FROZEN TEST SHA256: {digest}")
    print("STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
