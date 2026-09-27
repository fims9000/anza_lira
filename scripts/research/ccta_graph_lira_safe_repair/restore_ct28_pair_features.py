#!/usr/bin/env python3
"""Restore the committed CT28 PAIR feature table from the compact Git payload.

This avoids requiring raw multi-GB CCTA data just to reproduce the already-frozen
local PAIR baselines.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import lzma
from pathlib import Path

EXPECTED_SHA256 = "819181674acb9a358dd822e55acf3f5721cf8d46b82c0e2655e7af842b0df864"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--payload",
        type=Path,
        default=Path("artifacts/varvara/ct28_pair/payload/expanded_relation_features.csv.xz.b64"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/varvara/ct28_pair/expanded_relation_features.csv"),
    )
    args = ap.parse_args()

    packed = base64.b64decode(args.payload.read_bytes())
    raw = lzma.decompress(packed)
    sha = hashlib.sha256(raw).hexdigest()
    if sha != EXPECTED_SHA256:
        raise SystemExit(f"SHA256 mismatch: {sha} != {EXPECTED_SHA256}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(raw)
    print(f"restored {args.out} ({len(raw)} bytes)")
    print(f"sha256 {sha}")

if __name__ == "__main__":
    main()
