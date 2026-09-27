#!/usr/bin/env python3
"""Restore the regenerated CT28 PAIR joblib snapshots committed as text payloads.

These snapshots are convenience artifacts. The canonical scientific source of truth
remains the frozen feature table + training script, because joblib compatibility can
depend on the Python/scikit-learn environment.
"""
from __future__ import annotations

import argparse
import base64
import gzip
from pathlib import Path

MODELS = [
    "geometry",
    "radial_hu_summary_v1",
    "geometry_plus_radial_v1",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--payload-dir",
        type=Path,
        default=Path("artifacts/varvara/ct28_pair/models_payload"),
    )
    ap.add_argument("--out-dir", type=Path, default=Path("ct28_pair_models_snapshot"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in MODELS:
        payload = args.payload_dir / f"{name}.joblib.gz.b64"
        raw = gzip.decompress(base64.b64decode(payload.read_bytes()))
        out = args.out_dir / f"{name}.joblib"
        out.write_bytes(raw)
        print(f"restored {out} ({len(raw)} bytes)")

    print("Note: retraining from expanded_relation_features.csv is preferred for environment-portable reproduction.")


if __name__ == "__main__":
    main()
