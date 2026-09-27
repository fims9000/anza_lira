#!/usr/bin/env python3
"""Train/save the strong CT28 geometry_hgb PAIR baseline from the frozen pair plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


def choose_threshold(y, score, max_fpr=0.05):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    best = None
    for t in np.r_[np.inf, np.sort(np.unique(score))[::-1], -np.inf]:
        pred = score >= t
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        recall = tp / max(int(np.sum(y == 1)), 1)
        fpr = fp / max(int(np.sum(y == 0)), 1)
        precision = tp / max(int(np.sum(pred)), 1)
        if fpr <= max_fpr + 1e-12:
            key = (recall, precision, -fpr, -float(t))
            if best is None or key > best[0]:
                best = (key, float(t))
    if best is None:
        raise RuntimeError("no threshold satisfies validation FPR budget")
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
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "recall": tp / max(tp + fn, 1),
        "fpr": fp / max(fp + tn, 1),
        "precision": tp / max(tp + fp, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "pair_plan",
        nargs="?",
        type=Path,
        default=Path("artifacts/varvara/ct28_pair/relation_pair_plan.csv"),
    )
    ap.add_argument("--out-dir", type=Path, default=Path("ct28_geometry_hgb"))
    ap.add_argument("--max-val-fpr", type=float, default=0.05)
    args = ap.parse_args()

    df = pd.read_csv(args.pair_plan)
    cols = [f"g{i}" for i in range(7)]
    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()

    if train["scan_id"].nunique() != 17 or val["scan_id"].nunique() != 5 or test["scan_id"].nunique() != 6:
        raise SystemExit("unexpected patient split; refusing to train")

    model = HistGradientBoostingClassifier(
        max_iter=150,
        max_leaf_nodes=7,
        learning_rate=0.05,
        l2_regularization=1.0,
        random_state=0,
    )
    model.fit(train[cols], train["y"])

    val_score = model.predict_proba(val[cols])[:, 1]
    threshold = choose_threshold(val["y"], val_score, args.max_val_fpr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": cols,
            "threshold": threshold,
            "threshold_source": "validation_only",
            "max_val_fpr": args.max_val_fpr,
        },
        args.out_dir / "geometry_hgb.joblib",
    )

    rows = []
    for split_name, part in [("train", train), ("val", val), ("test", test)]:
        score = model.predict_proba(part[cols])[:, 1]
        row = metrics(part["y"], score, threshold)
        row.update({"model": "geometry_hgb", "split": split_name})
        rows.append(row)

    pd.DataFrame(rows).to_csv(args.out_dir / "summary.csv", index=False)
    (args.out_dir / "protocol.json").write_text(
        json.dumps(
            {
                "model": "HistGradientBoostingClassifier",
                "params": {
                    "max_iter": 150,
                    "max_leaf_nodes": 7,
                    "learning_rate": 0.05,
                    "l2_regularization": 1.0,
                    "random_state": 0,
                },
                "features": cols,
                "threshold_rule": "validation only; maximize recall subject to FPR <= configured budget",
                "max_val_fpr": args.max_val_fpr,
                "patient_split_expected": {"train": 17, "val": 5, "test": 6},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
