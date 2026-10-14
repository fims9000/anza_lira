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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import build_ct28_relation_features as relation_features


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCENE_DIR = REPO_ROOT / "artifacts/varvara/ct28_graph_lira"
RELATION_CLASSES = ("NONE", "PAIR", "JUNCTION", "BOTH")
EVIDENCE_STREAMS = ("geometry", "ct", "geometry_plus_ct")
PAIR_GEOMETRY_FEATURES = tuple(f"pair_g{index}" for index in range(7))
JUNCTION_GEOMETRY_FEATURES = (
    "junction_span_mm",
    "junction_residual_mean_mm",
    "junction_residual_max_mm",
    "junction_forward_min_mm",
    "junction_forward_mean_mm",
    "junction_alignment_min",
    "junction_alignment_mean",
    "junction_center_distance_max_mm",
    "junction_center_distance_std_mm",
)
EXPECTED_PATIENT_IDS = {
    "train": {953, 955, 956, 959, 960, 963, 964, 967, 969, 970, 971, 975, 976, 977, 979, 982, 983},
    "val": {957, 961, 965, 966, 974},
}
LOCAL_MODEL_PARAMS = {
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 5000,
    "solver": "lbfgs",
    "random_state": 0,
    "missing_ct_policy": "train-fitted median imputation without missing indicators",
}
LOCAL_HGB_PARAMS = {
    "max_iter": 140,
    "max_leaf_nodes": 15,
    "learning_rate": 0.06,
    "l2_regularization": 2.0,
    "class_weight": "balanced",
    "early_stopping": False,
    "random_state": 0,
    "missing_ct_policy": "train-fitted median imputation without missing indicators",
}
RELATION_MODEL_PARAMS = {
    "max_iter": 140,
    "max_leaf_nodes": 15,
    "learning_rate": 0.06,
    "l2_regularization": 2.0,
    "class_weight": "balanced",
    "early_stopping": False,
    "random_state": 0,
}


class RelationTrainingError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CT28 patient-OOF local evidence and paired relation heads."
    )
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_SCENE_DIR)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--tau", type=float, default=0.85)
    parser.add_argument(
        "--local-model-family",
        choices=("logistic", "hgb"),
        default="logistic",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def local_model(family: str = "logistic") -> Pipeline:
    if family == "logistic":
        logistic_params = {
            key: value
            for key, value in LOCAL_MODEL_PARAMS.items()
            if key != "missing_ct_policy"
        }
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=False)),
                ("scale", StandardScaler()),
                ("lr", LogisticRegression(**logistic_params)),
            ]
        )
    if family == "hgb":
        hgb_params = {
            key: value
            for key, value in LOCAL_HGB_PARAMS.items()
            if key != "missing_ct_policy"
        }
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=False)),
                ("hgb", HistGradientBoostingClassifier(**hgb_params)),
            ]
        )
    raise RelationTrainingError(f"unknown local model family: {family}")


def relation_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**RELATION_MODEL_PARAMS)


def validate_scene_inputs(plans: pd.DataFrame, candidates: pd.DataFrame) -> None:
    if set(plans["split"]) != {"train", "val"}:
        raise RelationTrainingError("scene plan must contain train/validation only")
    for split, expected in EXPECTED_PATIENT_IDS.items():
        actual = set(plans.loc[plans["split"] == split, "scan_id"].astype(int))
        if actual != expected:
            raise RelationTrainingError(f"unexpected {split} patients")
    if set(plans["relation_class"]) != set(RELATION_CLASSES):
        raise RelationTrainingError("unexpected relation classes")
    if len(plans) != 1404 or len(candidates) != 15092:
        raise RelationTrainingError("unexpected frozen scene/candidate row count")
    if candidates[["scene_id", "candidate_id"]].duplicated().any():
        raise RelationTrainingError("duplicate scene/candidate key")
    if set(candidates["candidate_type"]) != {"PAIR", "JUNCTION"}:
        raise RelationTrainingError("unexpected candidate types")
    if set(candidates["y"].astype(int)) != {0, 1}:
        raise RelationTrainingError("candidate labels must be binary")


def patient_folds(train: pd.DataFrame, folds: int) -> dict[int, int]:
    if not 2 <= folds <= len(EXPECTED_PATIENT_IDS["train"]):
        raise RelationTrainingError("invalid patient fold count")
    splitter = GroupKFold(n_splits=folds)
    mapping: dict[int, int] = {}
    for fold, (_, hold_indices) in enumerate(
        splitter.split(train, train["y"], groups=train["scan_id"])
    ):
        held_patients = set(train.iloc[hold_indices]["scan_id"].astype(int))
        if held_patients & set(mapping):
            raise RelationTrainingError("patient assigned to more than one OOF fold")
        mapping.update({patient: fold for patient in held_patients})
    if set(mapping) != EXPECTED_PATIENT_IDS["train"]:
        raise RelationTrainingError("OOF folds do not cover frozen train patients")
    return mapping


def feature_sets(frame: pd.DataFrame) -> dict[str, dict[str, tuple[str, ...]]]:
    pair_ct = tuple(column for column in frame if column.startswith("pair_ct"))
    junction_ct = tuple(
        column
        for column in frame
        if column.startswith("junction_") and column not in JUNCTION_GEOMETRY_FEATURES
    )
    if len(pair_ct) != 38 or len(junction_ct) != 152:
        raise RelationTrainingError(
            f"unexpected CT feature counts: PAIR={len(pair_ct)}, JUNCTION={len(junction_ct)}"
        )
    return {
        "PAIR": {
            "geometry": PAIR_GEOMETRY_FEATURES,
            "ct": pair_ct,
            "geometry_plus_ct": PAIR_GEOMETRY_FEATURES + pair_ct,
        },
        "JUNCTION": {
            "geometry": JUNCTION_GEOMETRY_FEATURES,
            "ct": junction_ct,
            "geometry_plus_ct": JUNCTION_GEOMETRY_FEATURES + junction_ct,
        },
    }


def generate_local_scores(
    plans: pd.DataFrame,
    candidates: pd.DataFrame,
    folds: int,
    model_family: str = "logistic",
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    validate_scene_inputs(plans, candidates)
    if {"scan_id", "split"} - set(candidates):
        frame = candidates.merge(
            plans[["scene_id", "scan_id", "split"]],
            on="scene_id",
            how="left",
            validate="many_to_one",
        )
    else:
        frame = candidates.copy()
        expected_meta = plans.set_index("scene_id")[["scan_id", "split"]]
        for row in frame[["scene_id", "scan_id", "split"]].itertuples(index=False):
            expected = expected_meta.loc[row.scene_id]
            if int(row.scan_id) != int(expected["scan_id"]) or row.split != expected["split"]:
                raise RelationTrainingError("candidate/scene metadata mismatch")
    if frame[["scan_id", "split"]].isna().any().any():
        raise RelationTrainingError("candidate without scene metadata")
    frame["scan_id"] = frame["scan_id"].astype(int)
    frame["y"] = frame["y"].astype(int)
    train = frame[frame["split"] == "train"]
    validation = frame[frame["split"] == "val"]
    fold_by_patient = patient_folds(train, folds)
    frame["local_oof_fold"] = (
        frame["scan_id"].map(fold_by_patient).fillna(-1).astype(int)
    )
    for stream in EVIDENCE_STREAMS:
        frame[f"{stream}_score"] = np.nan

    feature_columns = feature_sets(frame)
    diagnostics: dict[str, object] = {}
    saved_models: dict[str, object] = {}
    for candidate_type in ("PAIR", "JUNCTION"):
        type_train = train[train["candidate_type"] == candidate_type]
        type_validation = validation[validation["candidate_type"] == candidate_type]
        diagnostics[candidate_type] = {}
        saved_models[candidate_type] = {}
        for stream in EVIDENCE_STREAMS:
            columns = list(feature_columns[candidate_type][stream])
            train_values = type_train[columns].to_numpy(dtype=float)
            validation_values = type_validation[columns].to_numpy(dtype=float)
            if np.isinf(train_values).any() or np.isinf(validation_values).any():
                raise RelationTrainingError(
                    f"infinite {candidate_type} {stream} model feature"
                )
            if stream == "geometry" and (
                np.isnan(train_values).any() or np.isnan(validation_values).any()
            ):
                raise RelationTrainingError(
                    f"missing {candidate_type} geometry model feature"
                )
            score_column = f"{stream}_score"
            for fold in range(folds):
                held_patients = {
                    patient
                    for patient, assigned in fold_by_patient.items()
                    if assigned == fold
                }
                fit = type_train[~type_train["scan_id"].isin(held_patients)]
                held = type_train[type_train["scan_id"].isin(held_patients)]
                if fit["y"].nunique() != 2 or held["y"].nunique() != 2:
                    raise RelationTrainingError(
                        f"fold {fold} lacks a class for {candidate_type} {stream}"
                    )
                fold_model = local_model(model_family).fit(fit[columns], fit["y"])
                frame.loc[held.index, score_column] = fold_model.predict_proba(
                    held[columns]
                )[:, 1]
            final_model = local_model(model_family).fit(
                type_train[columns], type_train["y"]
            )
            frame.loc[type_validation.index, score_column] = final_model.predict_proba(
                type_validation[columns]
            )[:, 1]
            saved_models[candidate_type][stream] = final_model
            stream_diagnostics = {
                "feature_count": len(columns),
                "feature_columns": columns,
                "train_candidates": int(len(type_train)),
                "validation_candidates": int(len(type_validation)),
                "train_positives": int(type_train["y"].sum()),
                "validation_positives": int(type_validation["y"].sum()),
                "train_candidates_with_missing_features": int(
                    np.isnan(train_values).any(axis=1).sum()
                ),
                "validation_candidates_with_missing_features": int(
                    np.isnan(validation_values).any(axis=1).sum()
                ),
            }
            for split in ("train", "val"):
                part = frame[
                    (frame["candidate_type"] == candidate_type)
                    & (frame["split"] == split)
                ]
                stream_diagnostics[f"{split}_auroc"] = float(
                    roc_auc_score(part["y"], part[score_column])
                )
                stream_diagnostics[f"{split}_auprc"] = float(
                    average_precision_score(part["y"], part[score_column])
                )
            diagnostics[candidate_type][stream] = stream_diagnostics

    score_columns = [f"{stream}_score" for stream in EVIDENCE_STREAMS]
    if not np.isfinite(frame[score_columns].to_numpy(dtype=float)).all():
        raise RelationTrainingError("incomplete local OOF scores")
    score_rows = frame[
        [
            "scene_id",
            "candidate_id",
            "candidate_type",
            "scan_id",
            "split",
            "local_oof_fold",
            "y",
            *score_columns,
        ]
    ].copy()
    model_params = (
        LOCAL_MODEL_PARAMS if model_family == "logistic" else LOCAL_HGB_PARAMS
    )
    model_description = (
        "MedianImputer + StandardScaler + LogisticRegression"
        if model_family == "logistic"
        else "MedianImputer + HistGradientBoostingClassifier"
    )
    protocol = {
        "training": "patient-group OOF for train; all-train fitted model for validation",
        "model_family": model_family,
        "model": model_description,
        "params": model_params,
        "folds": folds,
        "fold_by_patient": {
            str(key): value for key, value in sorted(fold_by_patient.items())
        },
        "models": diagnostics,
    }
    return score_rows, protocol, saved_models


def build_relation_feature_rows(
    plan_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    score_rows: list[dict[str, str]],
    score_columns: Sequence[str],
) -> tuple[list[dict[str, object]], list[str]]:
    prepared = relation_features.prepare_inputs(
        plan_rows, candidate_rows, score_rows, list(score_columns)
    )
    return relation_features.build_historical_features(*prepared, list(score_columns))


def evaluate_predictions(
    truth: np.ndarray, probabilities: np.ndarray, tau: float
) -> dict[str, object]:
    raw = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    selective = np.where(confidence >= tau, raw, 0)
    relation_present = truth != 0
    none_truth = truth == 0

    def summary(prediction: np.ndarray) -> dict[str, object]:
        return {
            "accuracy": float(accuracy_score(truth, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
            "macro_f1": float(f1_score(truth, prediction, average="macro")),
            "per_class_recall": {
                label: float(value)
                for label, value in zip(
                    RELATION_CLASSES,
                    recall_score(
                        truth,
                        prediction,
                        labels=np.arange(4),
                        average=None,
                        zero_division=0,
                    ),
                )
            },
            "confusion_matrix": confusion_matrix(
                truth, prediction, labels=np.arange(4)
            ).tolist(),
            "relation_class_accuracy_when_present": float(
                np.mean(prediction[relation_present] == truth[relation_present])
            ),
            "non_none_prediction_on_true_none": float(
                np.mean(prediction[none_truth] != 0)
            ),
        }

    return {
        "raw": summary(raw),
        "tau_0_85": {
            **summary(selective),
            "confidence_coverage": float(np.mean(confidence >= tau)),
            "confident_scenes": int(np.sum(confidence >= tau)),
        },
        "multiclass_log_loss": float(
            log_loss(truth, probabilities, labels=np.arange(4))
        ),
    }


def fit_relation_head(
    rows: list[dict[str, object]], columns: list[str], tau: float, model_name: str
) -> tuple[dict[str, object], list[dict[str, object]], HistGradientBoostingClassifier]:
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "val"]
    x_train = np.asarray(
        [[float(row[column]) for column in columns] for row in train_rows]
    )
    x_validation = np.asarray(
        [[float(row[column]) for column in columns] for row in validation_rows]
    )
    y_train = np.asarray(
        [RELATION_CLASSES.index(str(row["relation_class"])) for row in train_rows]
    )
    y_validation = np.asarray(
        [RELATION_CLASSES.index(str(row["relation_class"])) for row in validation_rows]
    )
    if not (np.isfinite(x_train).all() and np.isfinite(x_validation).all()):
        raise RelationTrainingError("non-finite relation features")
    model = relation_model().fit(x_train, y_train)
    probabilities = model.predict_proba(x_validation)
    if not np.array_equal(model.classes_, np.arange(4)):
        raise RelationTrainingError("relation model class order changed")
    metrics = evaluate_predictions(y_validation, probabilities, tau)
    raw = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    selective = np.where(confidence >= tau, raw, 0)
    predictions = []
    for row, probability, raw_class, selective_class, conf in zip(
        validation_rows, probabilities, raw, selective, confidence
    ):
        predictions.append(
            {
                "model": model_name,
                "scene_id": row["scene_id"],
                "scan_id": row["scan_id"],
                "scene_family": row["scene_family"],
                "relation_class": row["relation_class"],
                "raw_prediction": RELATION_CLASSES[int(raw_class)],
                "confidence": float(conf),
                "tau_prediction": RELATION_CLASSES[int(selective_class)],
                **{
                    f"probability_{label.lower()}": float(probability[index])
                    for index, label in enumerate(RELATION_CLASSES)
                },
            }
        )
    return metrics, predictions, model


def main() -> None:
    args = parse_args()
    if args.tau != 0.85:
        raise RelationTrainingError("frozen relation tau must remain 0.85")
    if (
        args.local_model_family == "hgb"
        and args.out_dir.resolve() == args.scene_dir.resolve()
    ):
        raise RelationTrainingError(
            "local HGB is an ablation; provide a separate --out-dir"
        )
    scene_plan_path = args.scene_dir / "scene_plan_train_val.csv"
    scene_candidate_path = args.scene_dir / "scene_candidates_train_val.csv"
    candidate_feature_path = args.scene_dir / "scene_candidate_features_train_val.csv"
    scene_feature_protocol_path = args.scene_dir / "scene_candidate_feature_protocol.json"
    plans = pd.read_csv(scene_plan_path)
    scene_candidates = pd.read_csv(scene_candidate_path)
    candidate_features = pd.read_csv(
        candidate_feature_path,
        dtype={"ct_sampling_error": "string"},
        low_memory=False,
    )
    scene_feature_protocol = (
        json.loads(scene_feature_protocol_path.read_text())
        if scene_feature_protocol_path.exists()
        else {}
    )
    validate_scene_inputs(plans, scene_candidates)
    validate_scene_inputs(plans, candidate_features)
    source_keys = set(map(tuple, scene_candidates[["scene_id", "candidate_id"]].to_numpy()))
    feature_keys = set(map(tuple, candidate_features[["scene_id", "candidate_id"]].to_numpy()))
    if source_keys != feature_keys:
        raise RelationTrainingError("candidate feature table does not match scene candidates")

    local_scores, local_protocol, local_models = generate_local_scores(
        plans, candidate_features, args.folds, args.local_model_family
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    local_score_path = args.out_dir / "relation_local_scores_train_val.csv"
    local_scores.to_csv(local_score_path, index=False)

    plan_rows = relation_features.read_csv(scene_plan_path)
    candidate_rows = relation_features.read_csv(scene_candidate_path)
    score_rows = relation_features.read_csv(local_score_path)
    geometry_rows, geometry_columns = build_relation_feature_rows(
        plan_rows, candidate_rows, score_rows, ["geometry_score"]
    )
    fused_rows, fused_columns = build_relation_feature_rows(
        plan_rows, candidate_rows, score_rows, ["geometry_plus_ct_score"]
    )
    conditioned_rows, conditioned_columns = build_relation_feature_rows(
        plan_rows, candidate_rows, score_rows, ["geometry_score", "ct_score"]
    )
    if (
        len(geometry_columns) != 42
        or len(fused_columns) != 42
        or len(conditioned_columns) != 84
    ):
        raise RelationTrainingError(
            "unexpected relation feature counts: "
            f"{len(geometry_columns)}, {len(fused_columns)}, {len(conditioned_columns)}"
        )
    geometry_metrics, geometry_predictions, geometry_head = fit_relation_head(
        geometry_rows, geometry_columns, args.tau, "geometry_only"
    )
    fused_metrics, fused_predictions, fused_head = fit_relation_head(
        fused_rows, fused_columns, args.tau, "geometry_plus_ct"
    )
    conditioned_metrics, conditioned_predictions, conditioned_head = fit_relation_head(
        conditioned_rows, conditioned_columns, args.tau, "ct_conditioned"
    )
    predictions_path = args.out_dir / "relation_predictions_val.csv"
    pd.DataFrame(
        geometry_predictions + fused_predictions + conditioned_predictions
    ).to_csv(predictions_path, index=False)
    majority = float(
        plans.loc[plans["split"] == "val", "relation_class"]
        .value_counts(normalize=True)
        .max()
    )
    metrics_payload = {
        "status": "development_train_val_only",
        "held_out_test_accessed": False,
        "validation_scenes": int((plans["split"] == "val").sum()),
        "validation_majority_accuracy": majority,
        "tau": args.tau,
        "models": {
            "geometry_only": {
                "representation": "historical_42_score_landscape",
                "feature_count": len(geometry_columns),
                **geometry_metrics,
            },
            "geometry_plus_ct": {
                "representation": "historical_42_early_fusion_score_landscape",
                "feature_count": len(fused_columns),
                **fused_metrics,
            },
            "ct_conditioned": {
                "representation": "parallel_geometry_and_ct_historical42_landscapes",
                "feature_count": len(conditioned_columns),
                **conditioned_metrics,
            },
        },
        "metric_scope_warning": (
            "Scene relation-class metrics before Graph-LIRA identity selection, "
            "perturbation consistency, and structural repair evaluation."
        ),
    }
    metrics_path = args.out_dir / "relation_metrics_val.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2) + "\n")

    models_path = args.out_dir / "relation_models.joblib"
    joblib.dump(
        {
            "local_models": local_models,
            "relation_heads": {
                "geometry_only": geometry_head,
                "geometry_plus_ct": fused_head,
                "ct_conditioned": conditioned_head,
            },
            "local_model_family": args.local_model_family,
            "local_model_params": (
                LOCAL_MODEL_PARAMS
                if args.local_model_family == "logistic"
                else LOCAL_HGB_PARAMS
            ),
            "relation_model_params": RELATION_MODEL_PARAMS,
            "relation_classes": RELATION_CLASSES,
            "geometry_relation_features": geometry_columns,
            "geometry_plus_ct_relation_features": fused_columns,
            "ct_conditioned_relation_features": conditioned_columns,
        },
        models_path,
    )

    output_paths = [
        local_score_path,
        predictions_path,
        metrics_path,
        models_path,
    ]
    protocol = {
        "protocol_version": (
            "ct28_parallel_geometry_ct_relation_v1"
            if args.local_model_family == "logistic"
            else "ct28_parallel_geometry_ct_relation_local_hgb_ablation_v1"
        ),
        "held_out_test_accessed": False,
        "source_files": {
            "scene_plan": provenance_path(scene_plan_path),
            "scene_candidates": provenance_path(scene_candidate_path),
            "scene_candidate_features": provenance_path(candidate_feature_path),
            "scene_candidate_feature_protocol": (
                provenance_path(scene_feature_protocol_path)
                if scene_feature_protocol_path.exists()
                else None
            ),
        },
        "source_sha256": {
            "scene_plan": sha256(scene_plan_path),
            "scene_candidates": sha256(scene_candidate_path),
            "scene_candidate_features": sha256(candidate_feature_path),
            "scene_candidate_feature_protocol": (
                sha256(scene_feature_protocol_path)
                if scene_feature_protocol_path.exists()
                else None
            ),
        },
        "input_base_stress": scene_feature_protocol.get(
            "base_stress", {"enabled": False, "role": "not recorded"}
        ),
        "local_evidence": local_protocol,
        "relation_model": {
            "model": "HistGradientBoostingClassifier",
            "params": RELATION_MODEL_PARAMS,
            "class_order": list(RELATION_CLASSES),
            "tau": args.tau,
        },
        "relation_features": {
            "geometry_only": {
                "representation": "historical_42_score_landscape",
                "feature_count": len(geometry_columns),
                "model_feature_columns": geometry_columns,
            },
            "geometry_plus_ct": {
                "representation": "historical_42_early_fusion_score_landscape",
                "feature_count": len(fused_columns),
                "model_feature_columns": fused_columns,
            },
            "ct_conditioned": {
                "representation": "parallel_geometry_and_ct_historical42_landscapes",
                "feature_count": len(conditioned_columns),
                "model_feature_columns": conditioned_columns,
            },
            "candidate_count_features_included": True,
            "geometry_plus_ct_local_score_role": (
                "input to the separate 42-feature early-fusion relation baseline; "
                "excluded from the 84-feature separated-evidence head"
            ),
        },
        "outputs_sha256": {
            path.name: sha256(path) for path in output_paths
        },
    }
    protocol_path = args.out_dir / "relation_model_protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main()
