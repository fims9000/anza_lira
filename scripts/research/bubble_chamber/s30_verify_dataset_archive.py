#!/usr/bin/env python3
"""Verify byte-for-byte integrity of the separately archived private source datasets.

Public Git contains this verifier and hashes, NEVER the source BMP or labels.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

SCHEMA = "anza_bubble_chamber_dataset_integrity_v1"
SOURCE_NAMES = frozenset({"H50Z_030_01_006.7z", "1mH2Data.7z", "070_008.7z"})


def verify_dataset_sources(manifest_path: Path, source_dir: Path) -> dict:
    """Return a serializable manifest check, never infer human labels from proposals."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Wrong versioned source archive manifest")
    rows = manifest.get("source_archives")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("Exactly three original source archives expected")
    names = [r["file"] for r in rows]
    if len(set(names)) != len(names) or frozenset(names) != SOURCE_NAMES:
        raise ValueError("Unexpected, repeated or missing original archive names")
    summary = []
    for record in rows:
        name = record["file"]
        expected_size = record["size_bytes"]
        expected_digest = record["sha256"]
        if not isinstance(expected_size, int) or expected_size <= 0 or not isinstance(expected_digest, str) or len(expected_digest) != 64:
            raise ValueError("Corrupted manifest size or sha256")
        path = Path(source_dir) / name
        if not path.is_file():
            summary.append({"file": name, "status": "MISSING"})
            continue
        h = sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(2 * 1024 * 1024), b""):
                h.update(chunk)
        ok = path.stat().st_size == expected_size and h.hexdigest() == expected_digest
        summary.append({"file": name, "status": "VERIFIED" if ok else "HASH_MISMATCH",
                        "actual_sha256": h.hexdigest(), "actual_size_bytes": path.stat().st_size})
    return {"schema": "anza_source_archive_byte_verification_v1", "verified": all(x["status"] == "VERIFIED" for x in summary),
            "sources": summary, "expert_pixel_track_labels_added": 0,
            "automatic_particle_joins_authorized": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    args = parser.parse_args()
    result = verify_dataset_sources(args.manifest, args.source_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())