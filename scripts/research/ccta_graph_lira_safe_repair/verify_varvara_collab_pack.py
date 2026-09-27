#!/usr/bin/env python3
"""Validate the collaborator-facing CT28 PAIR pack without raw CCTA.

This is a packaging/integrity check, not a scientific re-analysis. It verifies that
all current handoff files exist, compact payloads decode, frozen hashes match, the
1,360-row pair plan has the expected patient/class split, and the held-out headline
metrics match the frozen protocol.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import lzma
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PACK = ROOT / "artifacts/varvara/ct28_pair"

EXPECTED_FEATURE_SHA = "819181674acb9a358dd822e55acf3f5721cf8d46b82c0e2655e7af842b0df864"
EXPECTED_SPLITS = {"train": (17, 758, 379, 379), "val": (5, 268, 134, 134), "test": (6, 334, 167, 167)}
EXPECTED_COMBINED = {
    "auroc": 0.9846893040266772,
    "recall": 0.8263473053892215,
    "fpr": 0.0179640718562874,
    "precision": 0.9787234042553192,
    "tp": 138,
    "fp": 3,
    "fn": 29,
    "tn": 164,
}

REQUIRED = [
    "README.md",
    "GENERATED_ARTIFACT_STATUS.md",
    "ct_alignment.csv",
    "expected_ct_geometry.csv",
    "pair_plan_counts.csv",
    "headline_test.csv",
    "paired_bootstrap_ci.csv",
    "risk_coverage.csv",
    "relation_pair_plan.csv",
    "expanded_relation_predictions.csv",
    "expanded_relation_summary.csv",
    "protocol.json",
    "payload/expanded_relation_features.csv.xz.b64",
    "models_payload/geometry.joblib.gz.b64",
    "models_payload/radial_hu_summary_v1.joblib.gz.b64",
    "models_payload/geometry_plus_radial_v1.joblib.gz.b64",
]

DOCS = [
    "docs/varvara/FINAL_HANDOFF_2026-09-27.md",
    "docs/varvara/CURRENT_TASK.md",
    "docs/varvara/ANSWER_TO_QUESTIONS_2026-09-26.md",
    "docs/varvara/NEGATIVE_RESULTS_THAT_MATTER.md",
    "docs/varvara/REPRODUCE_CT28_PAIR_BASELINE.md",
]

SCRIPTS = [
    "scripts/research/ccta_graph_lira_safe_repair/restore_ct28_pair_features.py",
    "scripts/research/ccta_graph_lira_safe_repair/restore_ct28_pair_models.py",
    "scripts/research/ccta_graph_lira_safe_repair/train_ct28_pair_from_features.py",
    "scripts/research/ccta_graph_lira_safe_repair/train_geometry_hgb_from_pair_plan.py",
]


def fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def check_exists() -> None:
    missing = [str(PACK / p) for p in REQUIRED if not (PACK / p).is_file()]
    missing += [str(ROOT / p) for p in DOCS + SCRIPTS if not (ROOT / p).is_file()]
    if missing:
        fail("missing files:\n  " + "\n  ".join(missing))


def check_protocol_and_hashes() -> None:
    protocol = json.loads((PACK / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("expanded_relation_features_sha256") != EXPECTED_FEATURE_SHA:
        fail("protocol feature SHA does not match frozen SHA")

    packed = base64.b64decode((PACK / "payload/expanded_relation_features.csv.xz.b64").read_bytes())
    raw = lzma.decompress(packed)
    sha = hashlib.sha256(raw).hexdigest()
    if sha != EXPECTED_FEATURE_SHA:
        fail(f"feature payload SHA mismatch: {sha}")

    pred_sha = hashlib.sha256((PACK / "expanded_relation_predictions.csv").read_bytes()).hexdigest()
    if pred_sha != protocol.get("expanded_relation_predictions_sha256"):
        fail("row-level prediction SHA mismatch")

    plan_sha = hashlib.sha256((PACK / "relation_pair_plan.csv").read_bytes()).hexdigest()
    if plan_sha != protocol.get("relation_pair_plan_sha256"):
        fail("pair-plan SHA mismatch")

    for name in ["geometry", "radial_hu_summary_v1", "geometry_plus_radial_v1"]:
        payload = PACK / f"models_payload/{name}.joblib.gz.b64"
        raw_model = gzip.decompress(base64.b64decode(payload.read_bytes()))
        if len(raw_model) < 100:
            fail(f"model payload {name} decoded to an implausibly small file")


def check_pair_plan() -> None:
    df = pd.read_csv(PACK / "relation_pair_plan.csv")
    if len(df) != 1360:
        fail(f"pair plan has {len(df)} rows instead of 1360")
    for split, (patients, rows, pos, neg) in EXPECTED_SPLITS.items():
        d = df[df["split"] == split]
        got = (d["scan_id"].nunique(), len(d), int((d["y"] == 1).sum()), int((d["y"] == 0).sum()))
        exp = (patients, rows, pos, neg)
        if got != exp:
            fail(f"{split} split mismatch: got {got}, expected {exp}")


def check_headline() -> None:
    h = pd.read_csv(PACK / "headline_test.csv")
    r = h[h["model"] == "geometry_plus_radial_v1"]
    if len(r) != 1:
        fail("headline_test.csv must contain exactly one geometry_plus_radial_v1 row")
    row = r.iloc[0]
    for k, v in EXPECTED_COMBINED.items():
        got = row[k]
        if isinstance(v, int):
            if int(got) != v:
                fail(f"headline {k}: got {got}, expected {v}")
        elif abs(float(got) - v) > 1e-12:
            fail(f"headline {k}: got {got}, expected {v}")


def main() -> None:
    check_exists()
    check_protocol_and_hashes()
    check_pair_plan()
    check_headline()
    print("PASS: Varvara CT28 collaborator pack is internally consistent.")
    print("Frozen PAIR rows: 1360 (758 train / 268 val / 334 test).")
    print("Frozen geometry+radial test: AUROC 0.9846893040, recall 0.8263473054, FPR 0.0179640719, precision 0.9787234043.")
    print("Raw CCTA is intentionally outside Git; it is required only for new image-feature extraction such as JUNCTION+CT.")


if __name__ == "__main__":
    main()