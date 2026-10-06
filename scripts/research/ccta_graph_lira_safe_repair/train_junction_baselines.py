#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs/research/ccta_graph_lira_safe_repair/junction_baselines.json"
)
IDENTITY_COLUMNS = (
    "candidate_id",
    "source_junction_id",
    "scan_id",
    "split",
    "side",
    "degree",
    "y",
    "benchmark_group",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_sets(frame: pd.DataFrame, config: dict) -> dict[str, list[str]]:
    geometry = sorted(column for column in frame if column.startswith("jg_"))
    ct = sorted(column for column in frame if column.startswith("ct_"))
    if len(geometry) != int(config["expected_geometry_features"]):
        raise RuntimeError(f"unexpected geometry feature count: {len(geometry)}")
    if len(ct) != int(config["expected_ct_features"]):
        raise RuntimeError(f"unexpected CT feature count: {len(ct)}")
    return {
        "geometry": geometry,
        "ct": ct,
        "geometry_plus_ct": geometry + ct,
    }


def validate_input(frame: pd.DataFrame, config: dict) -> None:
    missing = set(IDENTITY_COLUMNS) - set(frame.columns)
    if missing:
        raise RuntimeError(f"missing input columns: {sorted(missing)}")
    if set(frame["split"]) != {"train", "val"}:
        raise RuntimeError("input must contain train and validation only")
    if len(frame) != int(config["expected_candidate_rows"]):
        raise RuntimeError(f"unexpected candidate row count: {len(frame)}")
    if not frame["candidate_id"].is_unique:
        raise RuntimeError("candidate_id is not unique")
    if set(frame["y"]) != {0, 1}:
        raise RuntimeError("y must be binary")
    if set(frame["degree"]) != {3}:
        raise RuntimeError("baseline currently supports degree-3 only")
    expected_counts = config["expected_patient_counts"]
    for split, expected in expected_counts.items():
        split_frame = frame[frame["split"] == split]
        actual_ids = set(split_frame["scan_id"].astype(int).unique())
        if len(actual_ids) != int(expected):
            raise RuntimeError(f"unexpected {split} patient count: {len(actual_ids)}")
        if actual_ids != set(map(int, config["expected_patient_ids"][split])):
            raise RuntimeError(f"unexpected frozen patient IDs for {split}")
        expected_rows = int(config["expected_split_rows"][split])
        if len(split_frame) != expected_rows:
            raise RuntimeError(f"unexpected {split} candidate count: {len(split_frame)}")
        expected_sources = int(config["expected_source_junction_counts"][split])
        actual_sources = int(split_frame["source_junction_id"].nunique())
        if actual_sources != expected_sources:
            raise RuntimeError(
                f"unexpected {split} source junction count: {actual_sources}"
            )
    source_summary = frame.groupby(["split", "source_junction_id"])["y"].agg(
        ["sum", "count"]
    )
    if not bool((source_summary["sum"] == 1).all()):
        raise RuntimeError("each source junction must have exactly one positive")


def make_model(classifier: dict) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=float(classifier["C"]),
                    class_weight=classifier["class_weight"],
                    max_iter=int(classifier["max_iter"]),
                    solver=str(classifier["solver"]),
                    random_state=int(classifier["random_state"]),
                ),
            ),
        ]
    )


def choose_threshold(
    y: Sequence[int], score: Sequence[float], max_fpr: float
) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if not 0 <= max_fpr <= 1:
        raise ValueError("max_fpr must lie within [0, 1]")
    best = None
    for threshold in np.r_[np.inf, np.sort(np.unique(score))[::-1], -np.inf]:
        prediction = score >= threshold
        tp = int(np.sum(prediction & (y == 1)))
        fp = int(np.sum(prediction & (y == 0)))
        recall = tp / max(int(np.sum(y == 1)), 1)
        fpr = fp / max(int(np.sum(y == 0)), 1)
        precision = tp / max(int(np.sum(prediction)), 1)
        if fpr <= max_fpr + 1e-12:
            key = (recall, precision, -fpr, -float(threshold))
            if best is None or key > best[0]:
                best = (key, float(threshold))
    if best is None:
        raise RuntimeError("no threshold satisfies the FPR budget")
    return best[1]


def candidate_metrics(
    frame: pd.DataFrame, score_column: str, threshold: float
) -> dict[str, float | int]:
    y = frame["y"].to_numpy(dtype=int)
    score = frame[score_column].to_numpy(dtype=float)
    prediction = score >= threshold
    tp = int(np.sum(prediction & (y == 1)))
    fp = int(np.sum(prediction & (y == 0)))
    fn = int(np.sum((~prediction) & (y == 1)))
    tn = int(np.sum((~prediction) & (y == 0)))
    return {
        "candidate_count": int(len(frame)),
        "source_junction_count": int(frame["source_junction_id"].nunique()),
        "positives": int(np.sum(y == 1)),
        "negatives": int(np.sum(y == 0)),
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


def source_rankings(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    rows = []
    for (split, source_id), group in frame.groupby(
        ["split", "source_junction_id"], sort=False
    ):
        positive = group.loc[group["y"] == 1]
        if len(positive) != 1:
            raise RuntimeError(f"source {source_id} does not have one positive")
        true_score = float(positive.iloc[0][score_column])
        false_scores = group.loc[group["y"] == 0, score_column].to_numpy(float)
        rank = 1 + int(np.sum(false_scores >= true_score))
        has_false = len(false_scores) > 0
        best_false = float(false_scores.max()) if has_false else np.nan
        margin = true_score - best_false if has_false else np.nan
        rows.append(
            {
                "split": split,
                "source_junction_id": source_id,
                "scan_id": int(group.iloc[0]["scan_id"]),
                "benchmark_group": str(group.iloc[0]["benchmark_group"]),
                "candidate_count": int(len(group)),
                "true_score": true_score,
                "best_false_score": best_false,
                "margin": margin,
                "rank": rank,
                "top1": int(rank == 1),
                "reciprocal_rank": 1.0 / rank,
            }
        )
    return pd.DataFrame(rows)


def ranking_metrics(rankings: pd.DataFrame) -> dict[str, float | int]:
    finite_margins = rankings["margin"].dropna()
    return {
        "source_junction_count": int(len(rankings)),
        "competitive_source_count": int(rankings["margin"].notna().sum()),
        "isolated_source_count": int(rankings["margin"].isna().sum()),
        "top1": float(rankings["top1"].mean()),
        "mrr": float(rankings["reciprocal_rank"].mean()),
        "margin_median": (
            float(finite_margins.median()) if len(finite_margins) else np.nan
        ),
        "margin_min": (
            float(finite_margins.min()) if len(finite_margins) else np.nan
        ),
    }


def geometry_ambiguous_sources(
    candidate_plan: pd.DataFrame,
    *,
    quantile: float,
) -> tuple[float, dict[str, set[str]], pd.DataFrame]:
    if not 0 < quantile < 1:
        raise ValueError("ambiguous quantile must lie within (0, 1)")
    required = {
        "candidate_id",
        "source_junction_id",
        "split",
        "y",
        "geometry_match_distance",
    }
    missing = required - set(candidate_plan.columns)
    if missing:
        raise RuntimeError(f"candidate plan is missing columns: {sorted(missing)}")
    negatives = candidate_plan[candidate_plan["y"] == 0]
    closest = (
        negatives.groupby(["split", "source_junction_id"], as_index=False)
        ["geometry_match_distance"]
        .min()
        .rename(columns={"geometry_match_distance": "closest_geometry_distance"})
    )
    train_distances = closest.loc[
        closest["split"] == "train", "closest_geometry_distance"
    ]
    if train_distances.empty:
        raise RuntimeError("no competitive train junctions for ambiguity cutoff")
    cutoff = float(train_distances.quantile(quantile))
    groups = {}
    for split in ("train", "val"):
        selected = closest.loc[
            (closest["split"] == split)
            & (closest["closest_geometry_distance"] <= cutoff),
            "source_junction_id",
        ]
        groups[split] = set(selected.astype(str))
    return cutoff, groups, closest


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    features_path = resolve_repo_path(config["features"])
    candidate_plan_path = resolve_repo_path(config["candidate_plan"])
    out_dir = resolve_repo_path(config["out_dir"])
    frame = pd.read_csv(features_path)
    candidate_plan = pd.read_csv(candidate_plan_path)
    validate_input(frame, config)
    if set(frame["candidate_id"].astype(str)) != set(
        candidate_plan["candidate_id"].astype(str)
    ):
        raise RuntimeError("feature table and candidate plan IDs differ")
    specs = feature_sets(frame, config)
    numeric_columns = sorted(
        {column for columns in specs.values() for column in columns}
    )
    numeric = frame[numeric_columns]
    if not bool(np.isfinite(numeric.to_numpy(dtype=float)).all()):
        raise RuntimeError("non-finite model feature detected")

    train = frame["split"] == "train"
    validation = frame["split"] == "val"
    predictions = frame[list(IDENTITY_COLUMNS)].copy()
    thresholds = {}
    models = {}
    for model_name, columns in specs.items():
        model = make_model(config["classifier"])
        model.fit(frame.loc[train, columns], frame.loc[train, "y"])
        scores = model.predict_proba(frame[columns])[:, 1]
        predictions[f"{model_name}_score"] = scores
        threshold = choose_threshold(
            frame.loc[validation, "y"],
            scores[validation],
            float(config["max_validation_fpr"]),
        )
        predictions[f"{model_name}_pred"] = scores >= threshold
        thresholds[model_name] = threshold
        models[model_name] = {
            "model": model,
            "feature_columns": columns,
            "threshold": threshold,
            "threshold_source": "validation_only",
        }

    ambiguity_cutoff, ambiguity_sources, closest_geometry = (
        geometry_ambiguous_sources(
            candidate_plan,
            quantile=float(config["geometry_ambiguous_train_distance_quantile"]),
        )
    )
    closest_distance = {
        (str(row.split), str(row.source_junction_id)): float(
            row.closest_geometry_distance
        )
        for row in closest_geometry.itertuples(index=False)
    }
    source_to_ambiguous = {
        (split, source): True
        for split, sources in ambiguity_sources.items()
        for source in sources
    }
    predictions["geometry_ambiguous"] = [
        source_to_ambiguous.get((split, str(source)), False)
        for split, source in zip(
            predictions["split"], predictions["source_junction_id"]
        )
    ]

    candidate_rows = []
    ranking_rows = []
    source_rows = []
    for model_name in specs:
        score_column = f"{model_name}_score"
        rankings = source_rankings(predictions, score_column)
        rankings["model"] = model_name
        rankings["closest_geometry_distance"] = [
            closest_distance.get((str(split), str(source)), np.nan)
            for split, source in zip(
                rankings["split"], rankings["source_junction_id"]
            )
        ]
        rankings["geometry_ambiguous"] = [
            source_to_ambiguous.get((split, str(source)), False)
            for split, source in zip(
                rankings["split"], rankings["source_junction_id"]
            )
        ]
        source_rows.extend(rankings.to_dict("records"))
        for split in ("train", "val"):
            split_candidates = predictions[predictions["split"] == split]
            split_rankings = rankings[rankings["split"] == split]
            groups = {
                "all_junctions": (split_candidates, split_rankings),
                "geometry_ambiguous": (
                    split_candidates[split_candidates["geometry_ambiguous"]],
                    split_rankings[split_rankings["geometry_ambiguous"]],
                ),
            }
            for group_name, (candidate_group, ranking_group) in groups.items():
                if candidate_group["y"].nunique() != 2:
                    raise RuntimeError(
                        f"{split}/{group_name} does not contain both classes"
                    )
                candidate_row = candidate_metrics(
                    candidate_group, score_column, thresholds[model_name]
                )
                candidate_row.update(
                    {"model": model_name, "split": split, "group": group_name}
                )
                candidate_rows.append(candidate_row)
                ranking_row = ranking_metrics(ranking_group)
                ranking_row.update(
                    {"model": model_name, "split": split, "group": group_name}
                )
                ranking_rows.append(ranking_row)

    candidate_table = pd.DataFrame(candidate_rows)
    ranking_table = pd.DataFrame(ranking_rows)
    source_table = pd.DataFrame(source_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_dir / "junction_baseline_predictions.csv", index=False)
    candidate_table.to_csv(
        out_dir / "junction_baseline_candidate_metrics.csv", index=False
    )
    ranking_table.to_csv(
        out_dir / "junction_baseline_ranking_metrics.csv", index=False
    )
    source_table.to_csv(out_dir / "junction_baseline_source_scores.csv", index=False)
    input_hashes = {
        "features_sha256": file_sha256(features_path),
        "candidate_plan_sha256": file_sha256(candidate_plan_path),
    }
    joblib.dump(
        {
            "models": models,
            "ambiguity_cutoff": ambiguity_cutoff,
            "ambiguity_source": "train_closest_geometry_distance_quantile",
            "input_hashes": input_hashes,
            "config": config,
        },
        out_dir / "junction_baseline_models.joblib",
    )
    metrics_bundle = {
        "models": list(specs),
        "classifier": config["classifier"],
        "input_hashes": input_hashes,
        "split_counts": {
            split: {
                "patients": int(part["scan_id"].nunique()),
                "source_junctions": int(part["source_junction_id"].nunique()),
                "candidates": int(len(part)),
                "positives": int(part["y"].sum()),
            }
            for split, part in frame.groupby("split")
        },
        "feature_counts": {name: len(columns) for name, columns in specs.items()},
        "thresholds": thresholds,
        "threshold_rule": (
            "validation only; maximize recall subject to configured FPR budget"
        ),
        "max_validation_fpr": float(config["max_validation_fpr"]),
        "geometry_ambiguous": {
            "definition": (
                "minimum geometry_match_distance from the source positive to "
                "a retained false candidate <= cutoff"
            ),
            "cutoff_source": "competitive train junction distances only",
            "train_distance_quantile": float(
                config["geometry_ambiguous_train_distance_quantile"]
            ),
            "cutoff": ambiguity_cutoff,
            "source_counts": {
                split: len(sources) for split, sources in ambiguity_sources.items()
            },
            "role": "evaluation subgroup only",
        },
        "candidate_metrics": candidate_rows,
        "ranking_metrics": ranking_rows,
        "held_out_test_accessed": False,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(json_ready(metrics_bundle), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("Validation candidate metrics")
    print(
        candidate_table[candidate_table["split"] == "val"]
        [["model", "group", "auroc", "auprc", "recall", "fpr", "precision"]]
        .to_string(index=False)
    )
    print("\nValidation junction ranking")
    print(
        ranking_table[ranking_table["split"] == "val"]
        [["model", "group", "source_junction_count", "top1", "mrr"]]
        .to_string(index=False)
    )
    print(f"\ngeometry ambiguous distance cutoff: {ambiguity_cutoff:.10f}")
    print(f"output: {out_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()