#!/usr/bin/env python3
"""Train the three lightweight CT28 PAIR baselines from expanded_relation_features.csv.

This script is deliberately separate from the CT extractor so a collaborator can
retrain/re-save the models without touching raw CCTA once the feature table exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def choose_threshold(y, score, max_fpr=0.05):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    best = None
    for t in np.r_[np.inf, np.sort(np.unique(score))[::-1], -np.inf]:
        pred = score >= t
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        npos = max(int(np.sum(y == 1)), 1)
        nneg = max(int(np.sum(y == 0)), 1)
        recall = tp / npos
        fpr = fp / nneg
        precision = tp / max(int(np.sum(pred)), 1)
        if fpr <= max_fpr + 1e-12:
            key = (recall, precision, -fpr, -float(t))
            if best is None or key > best[0]:
                best = (key, float(t))
    if best is None:
        raise RuntimeError("no threshold satisfies FPR budget")
    return best[1]


def metrics(y, score, threshold):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = score >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum((~pred) & (y == 1)))
    tn = int(np.sum((~pred) & (y == 0)))
    return {
        "n": int(len(y)),
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": tp / max(tp + fn, 1),
        "fpr": fp / max(fp + tn, 1),
        "precision": tp / max(tp + fp, 1),
    }


def make_model():
    return Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
        )),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("ct28_pair_models"))
    ap.add_argument("--max-val-fpr", type=float, default=0.05)
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    required = {"scan_id", "split", "y"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"missing columns: {sorted(missing)}")

    gcols = sorted(
        [c for c in df.columns if c.startswith("g") and c[1:].isdigit()],
        key=lambda x: int(x[1:]),
    )
    ctcols = sorted(
        [c for c in df.columns if c.startswith("ct") and c[2:].isdigit()],
        key=lambda x: int(x[2:]),
    )
    if not gcols or not ctcols:
        raise SystemExit(f"feature detection failed: geometry={len(gcols)}, ct={len(ctcols)}")

    specs = {
        "geometry": gcols,
        "radial_hu_summary_v1": ctcols,
        "geometry_plus_radial_v1": gcols + ctcols,
    }

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()

    if train["scan_id"].nunique() != 17 or val["scan_id"].nunique() != 5 or test["scan_id"].nunique() != 6:
        raise SystemExit("unexpected patient split; refusing to train")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    predictions = df[["scan_id", "split", "y"]].copy()

    for name, cols in specs.items():
        model = make_model()
        model.fit(train[cols], train["y"])

        val_score = model.predict_proba(val[cols])[:, 1]
        threshold = choose_threshold(val["y"], val_score, args.max_val_fpr)

        joblib.dump(
            {
                "model": model,
                "feature_columns": cols,
                "threshold": threshold,
                "threshold_source": "validation_only",
                "max_val_fpr": args.max_val_fpr,
            },
            args.out_dir / f"{name}.joblib",
        )

        full_score = model.predict_proba(df[cols])[:, 1]
        predictions[f"{name}_score"] = full_score
        predictions[f"{name}_pred"] = full_score >= threshold

        for split_name, part in [("train", train), ("val", val), ("test", test)]:
            score = model.predict_proba(part[cols])[:, 1]
            row = metrics(part["y"], score, threshold)
            row.update({"model": name, "split": split_name})
            rows.append(row)

    pd.DataFrame(rows).to_csv(args.out_dir / "summary.csv", index=False)
    predictions.to_csv(args.out_dir / "predictions.csv", index=False)
    (args.out_dir / "protocol.json").write_text(json.dumps({
        "models": list(specs),
        "classifier": "StandardScaler + LogisticRegression(C=1, class_weight=balanced)",
        "threshold_rule": "validation only; maximize recall subject to FPR <= configured budget",
        "max_val_fpr": args.max_val_fpr,
        "patient_split_expected": {"train": 17, "val": 5, "test": 6},
    }, indent=2), encoding="utf-8")

    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
