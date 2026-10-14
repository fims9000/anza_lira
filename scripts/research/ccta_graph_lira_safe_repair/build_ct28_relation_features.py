#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCENE_PLAN = REPO_ROOT / "artifacts/varvara/ct28_graph_lira/scene_plan_train_val.csv"
DEFAULT_SCENE_CANDIDATES = REPO_ROOT / "artifacts/varvara/ct28_graph_lira/scene_candidates_train_val.csv"
CANDIDATE_TYPES = ("PAIR", "JUNCTION")
HISTORICAL_FEATURES_PER_TYPE = (
    "top1", "top2", "top3", "top4", "top5", "top6", "top7",
    "gap12", "gap23", "gap13", "mean", "std", "q50", "q75", "q90",
    "minimum", "candidate_count", "top_over_sum",
)
CROSS_FEATURES = (
    "top_difference", "top_sum", "pair_gap12_repeat",
    "junction_gap12_repeat", "pair_count_repeat", "junction_count_repeat",
)


class RelationFeatureError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the canonical historical 42-feature CT28 relation input."
    )
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--score-column", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol-out", type=Path)
    parser.add_argument("--scene-plan", type=Path, default=DEFAULT_SCENE_PLAN)
    parser.add_argument("--scene-candidates", type=Path, default=DEFAULT_SCENE_CANDIDATES)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise RelationFeatureError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_columns(
    rows: Sequence[dict[str, str]], required: set[str], source: Path
) -> None:
    if not rows:
        raise RelationFeatureError(f"Empty CSV: {source}")
    missing = required - set(rows[0])
    if missing:
        raise RelationFeatureError(f"Missing columns in {source}: {sorted(missing)}")


def historical_score_features(values: Sequence[float]) -> list[float]:
    scores = np.asarray(sorted(values, reverse=True), dtype=float)
    if len(scores) == 0:
        return [0.0] * len(HISTORICAL_FEATURES_PER_TYPE)
    top = np.zeros(7, dtype=float)
    top[: min(7, len(scores))] = scores[:7]
    first, second, third = top[:3]
    return [
        *top.tolist(), first - second, second - third, first - third,
        float(scores.mean()), float(scores.std()),
        float(np.quantile(scores, 0.50)), float(np.quantile(scores, 0.75)),
        float(np.quantile(scores, 0.90)), float(scores.min()),
        float(len(scores)), float(first / (scores.sum() + 1e-6)),
    ]


def prepare_inputs(plans, candidates, scores, score_columns):
    plan_by_scene = {row["scene_id"]: row for row in plans}
    if len(plan_by_scene) != len(plans):
        raise RelationFeatureError("Duplicate scene_id in scene plan")
    if {row["split"] for row in plans} != {"train", "val"}:
        raise RelationFeatureError("Relation features must contain train/validation only")
    if not {row["relation_class"] for row in plans} <= {
        "NONE", "PAIR", "JUNCTION", "BOTH"
    }:
        raise RelationFeatureError("Unknown relation class")

    candidate_by_key = {
        (row["scene_id"], row["candidate_id"]): row for row in candidates
    }
    if len(candidate_by_key) != len(candidates):
        raise RelationFeatureError("Duplicate scene/candidate key")
    if any(key[0] not in plan_by_scene for key in candidate_by_key):
        raise RelationFeatureError("Candidate references an unknown scene")
    if not {row["candidate_type"] for row in candidates} <= set(CANDIDATE_TYPES):
        raise RelationFeatureError("Unknown candidate type")
    if not {row["y"] for row in candidates} <= {"0", "1"}:
        raise RelationFeatureError("Candidate labels must be binary")

    score_by_key = {}
    for row in scores:
        key = (row["scene_id"], row["candidate_id"])
        if key in score_by_key:
            raise RelationFeatureError(f"Duplicate score row: {key}")
        if key not in candidate_by_key:
            raise RelationFeatureError(f"Score references an unknown candidate: {key}")
        if row["candidate_type"] != candidate_by_key[key]["candidate_type"]:
            raise RelationFeatureError(f"Candidate type mismatch: {key}")
        parsed = {}
        for column in score_columns:
            value = float(row[column])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise RelationFeatureError(f"{column} must be in [0, 1]: {key}")
            parsed[column] = value
        score_by_key[key] = parsed
    if set(score_by_key) != set(candidate_by_key):
        missing = set(candidate_by_key) - set(score_by_key)
        raise RelationFeatureError(f"Local scores miss {len(missing)} candidates")

    keys_by_scene_type = defaultdict(list)
    for key, candidate in candidate_by_key.items():
        keys_by_scene_type[(key[0], candidate["candidate_type"])].append(key)
    return plan_by_scene, score_by_key, keys_by_scene_type


def build_historical_features(
    plan_by_scene, score_by_key, keys_by_scene_type, score_columns
):
    feature_rows = []
    model_columns = []
    for score_column in score_columns:
        for candidate_type in CANDIDATE_TYPES:
            prefix = f"{score_column}_{candidate_type.lower()}"
            model_columns.extend(
                f"{prefix}_{name}" for name in HISTORICAL_FEATURES_PER_TYPE
            )
        model_columns.extend(f"{score_column}_{name}" for name in CROSS_FEATURES)

    for scene_id, plan in plan_by_scene.items():
        row = {
            "scene_id": scene_id, "scan_id": plan["scan_id"],
            "split": plan["split"], "scene_family": plan["scene_family"],
            "relation_class": plan["relation_class"],
        }
        for score_column in score_columns:
            vectors = {}
            for candidate_type in CANDIDATE_TYPES:
                keys = keys_by_scene_type.get((scene_id, candidate_type), [])
                vector = historical_score_features(
                    [score_by_key[key][score_column] for key in keys]
                )
                vectors[candidate_type] = vector
                prefix = f"{score_column}_{candidate_type.lower()}"
                row.update({
                    f"{prefix}_{name}": value
                    for name, value in zip(HISTORICAL_FEATURES_PER_TYPE, vector)
                })
            pair, junction = vectors["PAIR"], vectors["JUNCTION"]
            cross_values = (
                pair[0] - junction[0], pair[0] + junction[0],
                pair[7], junction[7], pair[16], junction[16],
            )
            row.update({
                f"{score_column}_{name}": value
                for name, value in zip(CROSS_FEATURES, cross_values)
            })
        feature_rows.append(row)
    return feature_rows, model_columns


def main() -> None:
    args = parse_args()
    score_columns = list(dict.fromkeys(args.score_column))
    plans = read_csv(args.scene_plan)
    candidates = read_csv(args.scene_candidates)
    scores = read_csv(args.scores)
    validate_columns(
        plans,
        {"scene_id", "scan_id", "split", "scene_family", "relation_class"},
        args.scene_plan,
    )
    validate_columns(
        candidates, {"scene_id", "candidate_id", "candidate_type", "y"},
        args.scene_candidates,
    )
    validate_columns(
        scores, {"scene_id", "candidate_id", "candidate_type", *score_columns},
        args.scores,
    )
    plan_by_scene, score_by_key, keys_by_scene_type = prepare_inputs(
        plans, candidates, scores, score_columns
    )
    feature_rows, model_columns = build_historical_features(
        plan_by_scene, score_by_key, keys_by_scene_type, score_columns
    )
    write_csv(args.out, feature_rows)

    protocol = {
        "representation": "historical_42_score_landscape",
        "scenes": len(feature_rows),
        "score_columns": score_columns,
        "required_score_provenance": (
            "patient-level OOF local scores for train scenes; "
            "frozen train-fitted local models for validation"
        ),
        "metadata_columns_not_for_model": [
            "scene_id", "scan_id", "split", "scene_family", "relation_class"
        ],
        "features_per_score_column": 42,
        "model_feature_columns": model_columns,
        "candidate_count_features_included": True,
        "full_candidate_pool_retained_for_graph_lira": True,
        "feature_definition": {
            "per_candidate_type": list(HISTORICAL_FEATURES_PER_TYPE),
            "cross_type": list(CROSS_FEATURES),
        },
    }
    if args.protocol_out:
        args.protocol_out.parent.mkdir(parents=True, exist_ok=True)
        args.protocol_out.write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
