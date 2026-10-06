#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ID_COLS = (
    "candidate_id", "source_junction_id", "scan_id", "split",
    "side", "degree", "y", "benchmark_group",
)


def model() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=5000,
            solver="lbfgs", random_state=0,
        )),
    ])


def choose_threshold(y: np.ndarray, score: np.ndarray, max_fpr: float) -> float:
    best = None
    for t in np.r_[np.inf, np.sort(np.unique(score))[::-1], -np.inf]:
        pred = score >= t
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum((~pred) & (y == 1)))
        tn = int(np.sum((~pred) & (y == 0)))
        recall = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        precision = tp / max(tp + fp, 1)
        if fpr <= max_fpr + 1e-12:
            key = (recall, precision, -fpr, -float(t))
            if best is None or key > best[0]:
                best = (key, float(t))
    if best is None:
        raise RuntimeError("no threshold satisfies FPR budget")
    return best[1]


def candidate_metrics(frame: pd.DataFrame, score_col: str, threshold: float) -> dict:
    y = frame["y"].to_numpy(int)
    score = frame[score_col].to_numpy(float)
    pred = score >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum((~pred) & (y == 1)))
    tn = int(np.sum((~pred) & (y == 0)))
    return {
        "candidate_count": int(len(frame)),
        "source_junction_count": int(frame["source_junction_id"].nunique()),
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": tp / max(tp + fn, 1),
        "fpr": fp / max(fp + tn, 1),
        "precision": tp / max(tp + fp, 1),
    }


def source_rankings(frame: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows = []
    for sid, group in frame.groupby("source_junction_id", sort=False):
        pos = group[group["y"] == 1]
        if len(pos) != 1:
            raise RuntimeError(f"{sid}: expected exactly one positive")
        true_score = float(pos.iloc[0][score_col])
        false = group.loc[group["y"] == 0, score_col].to_numpy(float)
        best_false = float(false.max()) if len(false) else np.nan
        margin = true_score - best_false if len(false) else np.nan
        rank = 1 + int(np.sum(false >= true_score))
        rows.append({
            "source_junction_id": sid,
            "scan_id": int(group.iloc[0]["scan_id"]),
            "split": str(group.iloc[0]["split"]),
            "benchmark_group": str(group.iloc[0]["benchmark_group"]),
            "true_score": true_score,
            "best_false_score": best_false,
            "margin": margin,
            "rank": rank,
            "top1": int(rank == 1),
        })
    return pd.DataFrame(rows)


def ranking_metrics(rankings: pd.DataFrame) -> dict:
    competitive = rankings[rankings["margin"].notna()]
    return {
        "top1": float(rankings["top1"].mean()),
        "top1_count": int(rankings["top1"].sum()),
        "source_count": int(len(rankings)),
        "competitive_top1": (
            float(competitive["top1"].mean()) if len(competitive) else None
        ),
        "competitive_top1_count": int(competitive["top1"].sum()),
        "competitive_source_count": int(len(competitive)),
        "isolated_source_count": int(rankings["margin"].isna().sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("features", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("junction_lira_v0"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-fpr", type=float, default=0.05)
    ap.add_argument("--ambiguous-quantile", type=float, default=0.25)
    args = ap.parse_args()

    frame = pd.read_csv(args.features)
    missing = set(ID_COLS) - set(frame.columns)
    if missing:
        raise RuntimeError(f"missing columns: {sorted(missing)}")
    if set(frame["split"]) != {"train", "val"}:
        raise RuntimeError("train/val only: held-out test must stay closed")
    if set(frame["degree"]) != {3}:
        raise RuntimeError("current JUNCTION-LIRA v0 is degree-3 only")

    geom = sorted(c for c in frame.columns if c.startswith("jg_"))
    ct = sorted(c for c in frame.columns if c.startswith("ct_"))
    if len(geom) != 9 or len(ct) != 152:
        raise RuntimeError(f"unexpected feature counts: {len(geom)} geometry, {len(ct)} CT")

    train = frame[frame["split"] == "train"].reset_index(drop=True)
    val = frame[frame["split"] == "val"].reset_index(drop=True)

    folds = min(args.folds, train["scan_id"].nunique())
    splitter = GroupKFold(n_splits=folds)
    oof_g = np.full(len(train), np.nan)
    oof_ct = np.full(len(train), np.nan)

    for fit_idx, hold_idx in splitter.split(train, train["y"], groups=train["scan_id"]):
        mg = model().fit(train.loc[fit_idx, geom], train.loc[fit_idx, "y"])
        mc = model().fit(train.loc[fit_idx, ct], train.loc[fit_idx, "y"])
        oof_g[hold_idx] = mg.predict_proba(train.loc[hold_idx, geom])[:, 1]
        oof_ct[hold_idx] = mc.predict_proba(train.loc[hold_idx, ct])[:, 1]

    if not (np.isfinite(oof_g).all() and np.isfinite(oof_ct).all()):
        raise RuntimeError("incomplete OOF base scores")

    mg = model().fit(train[geom], train["y"])
    mc = model().fit(train[ct], train["y"])
    val_g = mg.predict_proba(val[geom])[:, 1]
    val_ct = mc.predict_proba(val[ct])[:, 1]

    fusion = model().fit(np.c_[oof_g, oof_ct], train["y"])
    train_f = fusion.predict_proba(np.c_[oof_g, oof_ct])[:, 1]
    val_f = fusion.predict_proba(np.c_[val_g, val_ct])[:, 1]

    train_pred = train[list(ID_COLS)].copy()
    train_pred["geometry_score_oof"] = oof_g
    train_pred["ct_score_oof"] = oof_ct
    train_pred["junction_lira_score"] = train_f

    val_pred = val[list(ID_COLS)].copy()
    val_pred["geometry_score"] = val_g
    val_pred["ct_score"] = val_ct
    val_pred["junction_lira_score"] = val_f

    train_geom_rank = source_rankings(train_pred, "geometry_score_oof")
    train_margins = train_geom_rank["margin"].dropna()
    margin_cutoff = float(train_margins.quantile(args.ambiguous_quantile))

    val_geom_rank = source_rankings(val_pred, "geometry_score")
    ambiguous_ids = set(
        val_geom_rank.loc[
            val_geom_rank["margin"].notna()
            & (val_geom_rank["margin"] <= margin_cutoff),
            "source_junction_id",
        ].astype(str)
    )
    val_pred["geometry_model_ambiguous"] = (
        val_pred["source_junction_id"].astype(str).isin(ambiguous_ids)
    )

    threshold = choose_threshold(
        val_pred["y"].to_numpy(int),
        val_pred["junction_lira_score"].to_numpy(float),
        args.max_fpr,
    )
    val_pred["junction_lira_pred"] = val_pred["junction_lira_score"] >= threshold

    rankings = source_rankings(val_pred, "junction_lira_score")
    amb_rankings = rankings[
        rankings["source_junction_id"].astype(str).isin(ambiguous_ids)
    ]

    metrics = {
        "status": "development_train_val_only",
        "test_opened": False,
        "method": "junction_lira_v0_score_fusion",
        "base_models": "StandardScaler + balanced LogisticRegression",
        "fusion_inputs": ["geometry_score", "ct_score"],
        "fusion_training": f"patient-level {folds}-fold OOF train scores",
        "validation": {
            "candidate": candidate_metrics(val_pred, "junction_lira_score", threshold),
            "ranking": ranking_metrics(rankings),
            "geometry_model_ambiguous_ranking": (
                ranking_metrics(amb_rankings) if len(amb_rankings) else None
            ),
        },
        "geometry_model_ambiguous": {
            "definition": "geometry true-score - best false-score <= train OOF margin quantile",
            "quantile": args.ambiguous_quantile,
            "cutoff": margin_cutoff,
            "validation_sources": len(ambiguous_ids),
        },
        "threshold": {
            "source": "validation_only_development",
            "max_fpr": args.max_fpr,
            "value": threshold,
            "warning": "freeze before held-out test; never retune on test",
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_pred.to_csv(args.out_dir / "train_oof_predictions.csv", index=False)
    val_pred.to_csv(args.out_dir / "validation_predictions.csv", index=False)
    rankings.to_csv(args.out_dir / "validation_source_rankings.csv", index=False)
    (args.out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    joblib.dump({
        "geometry_model": mg,
        "ct_model": mc,
        "fusion_model": fusion,
        "geometry_features": geom,
        "ct_features": ct,
        "validation_threshold": threshold,
        "geometry_model_ambiguous_cutoff": margin_cutoff,
    }, args.out_dir / "junction_lira_v0.joblib")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
