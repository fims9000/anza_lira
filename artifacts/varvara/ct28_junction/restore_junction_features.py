#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAYLOAD = ROOT / "payload"
OUT = ROOT / "junction_relation_features_train_val.csv"
EXPECTED_SHA256 = "c58f0aef645ea0cf4552b846cf4140c052908f6457efeb4bc6cdc3efa7532483"

parts = sorted(PAYLOAD.glob("junction_relation_features_train_val.part*.csv"))
if not parts:
    raise SystemExit("No CT28 JUNCTION feature payload parts found.")

with OUT.open("wb") as dst:
    for path in parts:
        dst.write(path.read_bytes())

digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
if digest != EXPECTED_SHA256:
    OUT.unlink(missing_ok=True)
    raise SystemExit(f"SHA256 mismatch: {digest} != {EXPECTED_SHA256}")

print(f"RESTORED: {OUT}")
print(f"SHA256:   {digest}")
