#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd

import extract_ct28_scene_candidate_features as ct_features
import run_ct28_graph_lira as graph_lira
import train_ct28_relation_head as relation_training
from ct28_perturbations import perturb_scene_endpoints


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCENE_DIR = REPO_ROOT / "artifacts/varvara/ct28_graph_lira"
DEFAULT_CLEAN_EVIDENCE_DIR = (
    DEFAULT_SCENE_DIR / "ablations/local_hgb"
)
DEFAULT_ARCHIVE_DIR = (
    REPO_ROOT / "varvara/anza_lira_local_data/imagecas_801_1000"
)
DEFAULT_OUT_DIR = (
    DEFAULT_SCENE_DIR / "ablations/relation_perturbation_augmented_hgb"
)
DEFAULT_STRESS_SCENE_DIR = (
    DEFAULT_SCENE_DIR / "ablations/stress30_jitter1"
)
HELD_OUT_PATIENTS = {954, 958, 972, 973, 980, 984}
SCORE_COLUMNS = ("geometry_score", "ct_score", "geometry_plus_ct_score")


class AugmentationError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train only the CT-conditioned HGB relation head on clean plus "
            "perturbed train scenes. Clean patient-OOF local HGB models stay frozen."
        )
    )
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    parser.add_argument(
        "--clean-evidence-dir", type=Path, default=DEFAULT_CLEAN_EVIDENCE_DIR
    )
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--stress-scene-dir", type=Path, default=DEFAULT_STRESS_SCENE_DIR
    )
    parser.add_argument("--perturbations", type=int, default=15)
    parser.add_argument("--position-noise-mm", type=float, default=0.8)
    parser.add_argument("--tangent-noise-deg", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--tau", type=float, default=0.85)
    return parser.parse_args()


def require_frozen_settings(args: argparse.Namespace) -> None:
    expected = (15, 0.8, 20.0, 0.85)
    actual = (
        args.perturbations,
        args.position_noise_mm,
        args.tangent_noise_deg,
        args.tau,
    )
    if actual != expected:
        raise AugmentationError(
            "frozen experiment requires 15 perturbations, 0.8 mm, 20 degrees "
            "and tau=0.85"
        )
    if args.out_dir.resolve() in {
        args.scene_dir.resolve(),
        args.clean_evidence_dir.resolve(),
    }:
        raise AugmentationError("augmentation outputs require an isolated directory")


def load_inputs(args: argparse.Namespace) -> dict[str, object]:
    paths = {
        "plans": args.scene_dir / "scene_plan_train_val.csv",
        "endpoints": args.scene_dir / "scene_endpoints_train_val.csv",
        "candidates": args.scene_dir / "scene_candidates_train_val.csv",
        "features": args.scene_dir / "scene_candidate_features_train_val.csv",
        "scores": args.clean_evidence_dir / "relation_local_scores_train_val.csv",
        "predictions": args.clean_evidence_dir / "relation_predictions_val.csv",
        "metrics": args.clean_evidence_dir / "relation_metrics_val.json",
        "models": args.clean_evidence_dir / "relation_models.joblib",
        "protocol": args.clean_evidence_dir / "relation_model_protocol.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise AugmentationError(f"missing input files: {missing}")
    plans = pd.read_csv(paths["plans"])
    endpoints = pd.read_csv(paths["endpoints"])
    candidates = pd.read_csv(paths["candidates"])
    features = pd.read_csv(
        paths["features"], dtype={"ct_sampling_error": "string"}, low_memory=False
    )
    scores = pd.read_csv(paths["scores"])
    predictions = pd.read_csv(paths["predictions"])
    models = joblib.load(paths["models"])
    protocol = json.loads(paths["protocol"].read_text())
    metrics = json.loads(paths["metrics"].read_text())
    relation_training.validate_scene_inputs(plans, candidates)
    relation_training.validate_scene_inputs(plans, features)
    patient_ids = set(plans["scan_id"].astype(int))
    if patient_ids & HELD_OUT_PATIENTS:
        raise AugmentationError("held-out test patient entered the experiment")
    if models.get("local_model_family") != "hgb":
        raise AugmentationError("clean evidence pack must contain local HGB models")
    if protocol.get("local_evidence", {}).get("model_family", "hgb") != "hgb":
        raise AugmentationError("clean protocol is not the local-HGB ablation")
    if set(map(tuple, candidates[["scene_id", "candidate_id"]].to_numpy())) != set(
        map(tuple, features[["scene_id", "candidate_id"]].to_numpy())
    ):
        raise AugmentationError("candidate feature table does not match candidates")
    return {
        "paths": paths,
        "plans": plans,
        "endpoints": endpoints,
        "candidates": candidates,
        "features": features,
        "scores": scores,
        "predictions": predictions,
        "models": models,
        "protocol": protocol,
        "metrics": metrics,
    }


def fit_oof_local_models(
    features: pd.DataFrame,
    clean_scores: pd.DataFrame,
    fold_by_patient: dict[int, int],
) -> tuple[dict[int, dict[str, dict[str, object]]], float]:
    train = features[features["split"] == "train"].copy()
    train["scan_id"] = train["scan_id"].astype(int)
    train["y"] = train["y"].astype(int)
    feature_columns = relation_training.feature_sets(features)
    fold_models: dict[int, dict[str, dict[str, object]]] = {}
    reproduced_rows: list[pd.DataFrame] = []
    for fold in sorted(set(fold_by_patient.values())):
        held_patients = {
            patient for patient, assigned_fold in fold_by_patient.items()
            if assigned_fold == fold
        }
        fold_models[fold] = {}
        for candidate_type in ("PAIR", "JUNCTION"):
            type_train = train[train["candidate_type"] == candidate_type]
            fit = type_train[~type_train["scan_id"].isin(held_patients)]
            held = type_train[type_train["scan_id"].isin(held_patients)]
            fold_models[fold][candidate_type] = {}
            predicted = held[["scene_id", "candidate_id", "candidate_type"]].copy()
            for stream in relation_training.EVIDENCE_STREAMS:
                columns = list(feature_columns[candidate_type][stream])
                model = relation_training.local_model("hgb").fit(
                    fit[columns], fit["y"]
                )
                fold_models[fold][candidate_type][stream] = model
                predicted[f"{stream}_score"] = model.predict_proba(
                    held[columns]
                )[:, 1]
            reproduced_rows.append(predicted)
    reproduced = pd.concat(reproduced_rows, ignore_index=True)
    expected = clean_scores[clean_scores["split"] == "train"][
        ["scene_id", "candidate_id", "candidate_type", *SCORE_COLUMNS]
    ]
    comparison = expected.merge(
        reproduced,
        on=["scene_id", "candidate_id", "candidate_type"],
        validate="one_to_one",
        suffixes=("_expected", "_reproduced"),
    )
    if len(comparison) != len(expected):
        raise AugmentationError("retrained OOF models do not cover clean train scores")
    differences = []
    for column in SCORE_COLUMNS:
        differences.append(
            np.abs(
                comparison[f"{column}_expected"].to_numpy(dtype=float)
                - comparison[f"{column}_reproduced"].to_numpy(dtype=float)
            )
        )
    maximum_difference = float(np.max(np.concatenate(differences)))
    if maximum_difference > 1e-12:
        raise AugmentationError(
            f"clean OOF local-score reproduction failed: {maximum_difference}"
        )
    return fold_models, maximum_difference


def clean_relation_rows(
    plans: pd.DataFrame,
    candidates: pd.DataFrame,
    scores: pd.DataFrame,
) -> tuple[list[dict[str, object]], list[str]]:
    rows, columns = graph_lira.relation_feature_rows(
        plans,
        candidates,
        scores,
        ("geometry_score", "ct_score"),
    )
    if len(columns) != 84:
        raise AugmentationError(f"expected 84 relation features, got {len(columns)}")
    return rows, columns


def build_augmented_train_rows(
    args: argparse.Namespace,
    plans: pd.DataFrame,
    endpoints: pd.DataFrame,
    candidates: pd.DataFrame,
    fold_models: dict[int, dict[str, dict[str, object]]],
    fold_by_patient: dict[int, int],
    *,
    score_transform: Callable | None = None,
) -> tuple[list[dict[str, object]], list[str], list[dict[str, object]]]:
    train_plans = plans[plans["split"] == "train"].copy()
    train_scene_ids = set(train_plans["scene_id"])
    train_endpoints = endpoints[endpoints["scene_id"].isin(train_scene_ids)]
    train_candidates = candidates[candidates["scene_id"].isin(train_scene_ids)]
    expected_geometry = pd.read_csv(ct_features.EXPECTED_GEOMETRY).set_index("scan_id")
    expected_alignment = pd.read_csv(ct_features.EXPECTED_ALIGNMENT).set_index("scan_id")
    junction_arm_names = ct_features.arm_feature_names(ct_features.RADII_MM)
    augmented_rows: list[dict[str, object]] = []
    audit_by_scene = {
        scene_id: {
            "scene_id": scene_id,
            "scan_id": int(scan_id),
            "perturbed_variants": 0,
            "variants_with_missing_ct_candidates": 0,
            "missing_ct_candidate_rows": 0,
        }
        for scene_id, scan_id in train_plans[["scene_id", "scan_id"]].itertuples(
            index=False, name=None
        )
    }
    relation_columns: list[str] | None = None
    for scan_id in sorted(relation_training.EXPECTED_PATIENT_IDS["train"]):
        patient_plans = train_plans[train_plans["scan_id"].astype(int) == scan_id]
        patient_scene_ids = set(patient_plans["scene_id"])
        patient_endpoints = train_endpoints[
            train_endpoints["scene_id"].isin(patient_scene_ids)
        ]
        patient_candidates = train_candidates[
            train_candidates["scene_id"].isin(patient_scene_ids)
        ]
        ct, inverse_affine = graph_lira.load_verified_ct(
            scan_id, args.archive_dir, expected_geometry, expected_alignment
        )
        local_models = fold_models[fold_by_patient[scan_id]]
        for repetition in range(args.perturbations):
            feature_rows: list[dict[str, object]] = []
            perturbed_frames: list[pd.DataFrame] = []
            missing_by_scene = {scene_id: 0 for scene_id in patient_scene_ids}
            for scene_id, scene_endpoint_rows in patient_endpoints.groupby(
                "scene_id", sort=False
            ):
                perturbed = perturb_scene_endpoints(
                    scene_endpoint_rows,
                    seed=args.seed,
                    repetition=repetition,
                    position_noise_mm=args.position_noise_mm,
                    tangent_noise_deg=args.tangent_noise_deg,
                )
                perturbed_frames.append(perturbed)
                endpoint_lookup = perturbed.set_index("endpoint_id")
                scene_candidates = patient_candidates[
                    patient_candidates["scene_id"] == scene_id
                ]
                for _, candidate in scene_candidates.iterrows():
                    row = graph_lira.perturbed_candidate_features(
                        candidate,
                        endpoint_lookup,
                        ct,
                        inverse_affine,
                        junction_arm_names,
                    )
                    model_columns = [
                        column
                        for column in row
                        if column.startswith("pair_ct")
                        or column.startswith("junction_")
                    ]
                    if any(pd.isna(row[column]) for column in model_columns):
                        missing_by_scene[str(scene_id)] += 1
                    feature_rows.append(row)
            feature_frame = pd.DataFrame(feature_rows)
            scores = graph_lira.score_feature_frame(feature_frame, local_models)
            if score_transform is not None:
                original = scores.copy(deep=True)
                scores = score_transform(
                    int(scan_id), int(repetition),
                    pd.concat(perturbed_frames, ignore_index=True),
                    patient_candidates,
                    ct, inverse_affine, original.copy(deep=True),
                )
                keys = ("scene_id", "candidate_id", "candidate_type")
                if not isinstance(scores, pd.DataFrame) or len(scores) != len(original):
                    raise AugmentationError("Augmented CT-score callback changed candidate count")
                if not scores[list(keys)].reset_index(drop=True).equals(
                    original[list(keys)].reset_index(drop=True)
                ):
                    raise AugmentationError("Augmented CT-score callback changed candidate identity/order")
                for column in ("geometry_score", "geometry_plus_ct_score"):
                    if not np.array_equal(
                        scores[column].to_numpy(), original[column].to_numpy()
                    ):
                        raise AugmentationError(
                            f"Augmented CT-score callback unexpectedly changed {column}"
                        )
                if not np.isfinite(scores["ct_score"].to_numpy(dtype=float)).all():
                    raise AugmentationError("Augmented learned CT-score contains nonfinite values")
            relation_rows, columns = graph_lira.relation_feature_rows(
                patient_plans,
                patient_candidates,
                scores,
                ("geometry_score", "ct_score"),
            )
            if relation_columns is None:
                relation_columns = columns
            elif relation_columns != columns:
                raise AugmentationError("relation feature order changed across variants")
            for row in relation_rows:
                row["augmentation_index"] = repetition + 1
                row["sample_weight"] = 1.0 / (args.perturbations + 1)
                augmented_rows.append(row)
                audit = audit_by_scene[str(row["scene_id"])]
                audit["perturbed_variants"] += 1
                missing = missing_by_scene[str(row["scene_id"])]
                audit["missing_ct_candidate_rows"] += missing
                audit["variants_with_missing_ct_candidates"] += int(missing > 0)
        del ct
        print(
            f"augmentation: patient {scan_id} complete "
            f"({len(patient_plans)} scenes x {args.perturbations})",
            flush=True,
        )
    if relation_columns is None:
        raise AugmentationError("no perturbed relation rows were generated")
    return augmented_rows, relation_columns, list(audit_by_scene.values())


def fit_augmented_head(
    train_rows: list[dict[str, object]],
    validation_rows: list[dict[str, object]],
    columns: list[str],
    tau: float,
) -> tuple[object, dict[str, object], list[dict[str, object]]]:
    x_train = np.asarray(
        [[float(row[column]) for column in columns] for row in train_rows]
    )
    y_train = np.asarray(
        [relation_training.RELATION_CLASSES.index(str(row["relation_class"])) for row in train_rows]
    )
    weights = np.asarray([float(row["sample_weight"]) for row in train_rows])
    x_validation = np.asarray(
        [[float(row[column]) for column in columns] for row in validation_rows]
    )
    y_validation = np.asarray(
        [relation_training.RELATION_CLASSES.index(str(row["relation_class"])) for row in validation_rows]
    )
    if not np.isfinite(x_train).all() or not np.isfinite(x_validation).all():
        raise AugmentationError("non-finite relation features")
    model = relation_training.relation_model().fit(
        x_train, y_train, sample_weight=weights
    )
    if not np.array_equal(model.classes_, np.arange(4)):
        raise AugmentationError("relation model class order changed")
    probabilities = model.predict_proba(x_validation)
    metrics = relation_training.evaluate_predictions(
        y_validation, probabilities, tau
    )
    raw = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    selective = np.where(confidence >= tau, raw, 0)
    predictions = []
    for row, probability, raw_class, selective_class, conf in zip(
        validation_rows, probabilities, raw, selective, confidence
    ):
        predictions.append(
            {
                "model": "ct_conditioned",
                "scene_id": row["scene_id"],
                "scan_id": row["scan_id"],
                "scene_family": row["scene_family"],
                "relation_class": row["relation_class"],
                "raw_prediction": relation_training.RELATION_CLASSES[int(raw_class)],
                "confidence": float(conf),
                "tau_prediction": relation_training.RELATION_CLASSES[
                    int(selective_class)
                ],
                **{
                    f"probability_{label.lower()}": float(probability[index])
                    for index, label in enumerate(relation_training.RELATION_CLASSES)
                },
            }
        )
    return model, metrics, predictions


def validate_augmentation_weights(
    rows: list[dict[str, object]], perturbations: int
) -> None:
    frame = pd.DataFrame(rows)
    counts = frame.groupby("scene_id").size()
    weight_sums = frame.groupby("scene_id")["sample_weight"].sum()
    expected_variants = perturbations + 1
    if not counts.eq(expected_variants).all():
        raise AugmentationError("each train scene must have clean plus 15 variants")
    if not np.allclose(weight_sums.to_numpy(dtype=float), 1.0, atol=1e-12):
        raise AugmentationError("per-source-scene sample weights do not sum to one")
    if set(frame["augmentation_index"].astype(int)) != set(range(expected_variants)):
        raise AugmentationError("augmentation indices are incomplete")


def write_stress_validation_evidence(
    scene_dir: Path,
    out_dir: Path,
    model_pack: dict[str, object],
    tau: float,
) -> None:
    plan_path = scene_dir / "scene_plan_train_val.csv"
    candidate_path = scene_dir / "scene_candidates_train_val.csv"
    feature_path = scene_dir / "scene_candidate_features_train_val.csv"
    plans = pd.read_csv(plan_path)
    candidates = pd.read_csv(candidate_path)
    features = pd.read_csv(
        feature_path, dtype={"ct_sampling_error": "string"}, low_memory=False
    )
    relation_training.validate_scene_inputs(plans, candidates)
    relation_training.validate_scene_inputs(plans, features)
    val_plans = plans[plans["split"] == "val"].copy()
    val_scene_ids = set(val_plans["scene_id"])
    val_candidates = candidates[candidates["scene_id"].isin(val_scene_ids)]
    val_features = features[features["scene_id"].isin(val_scene_ids)]
    if set(val_plans["scan_id"].astype(int)) != relation_training.EXPECTED_PATIENT_IDS["val"]:
        raise AugmentationError("Stress30 validation patient set changed")
    if set(plans["scan_id"].astype(int)) & HELD_OUT_PATIENTS:
        raise AugmentationError("held-out test patient entered Stress30 evaluation")
    scores = graph_lira.score_feature_frame(
        val_features, model_pack["local_models"]
    )
    predicted = graph_lira.predict_relations(
        val_plans, val_candidates, scores, model_pack, tau
    )
    prediction_rows: list[dict[str, object]] = []
    metric_payload = {
        "status": "development_stress30_validation_only",
        "held_out_test_accessed": False,
        "training_on_stress30": False,
        "tau": tau,
        "models": {},
    }
    for model_name in graph_lira.MODEL_CONFIGS:
        probabilities = []
        truth = []
        for plan in val_plans.itertuples(index=False):
            item = predicted[model_name][str(plan.scene_id)]
            probability = np.asarray(item["probabilities"], dtype=float)
            probabilities.append(probability)
            truth.append(
                relation_training.RELATION_CLASSES.index(str(plan.relation_class))
            )
            raw_index = int(np.argmax(probability))
            selected_index = (
                raw_index if float(item["confidence"]) >= tau else 0
            )
            prediction_rows.append(
                {
                    "model": model_name,
                    "scene_id": plan.scene_id,
                    "scan_id": int(plan.scan_id),
                    "scene_family": plan.scene_family,
                    "relation_class": plan.relation_class,
                    "raw_prediction": relation_training.RELATION_CLASSES[raw_index],
                    "confidence": float(item["confidence"]),
                    "tau_prediction": relation_training.RELATION_CLASSES[
                        selected_index
                    ],
                    **{
                        f"probability_{label.lower()}": float(probability[index])
                        for index, label in enumerate(
                            relation_training.RELATION_CLASSES
                        )
                    },
                }
            )
        metric_payload["models"][model_name] = relation_training.evaluate_predictions(
            np.asarray(truth), np.asarray(probabilities), tau
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    score_path = out_dir / "relation_local_scores_train_val.csv"
    prediction_path = out_dir / "relation_predictions_val.csv"
    metrics_path = out_dir / "relation_metrics_val.json"
    models_path = out_dir / "relation_models.joblib"
    protocol_path = out_dir / "relation_model_protocol.json"
    scores.to_csv(score_path, index=False)
    pd.DataFrame(prediction_rows).to_csv(prediction_path, index=False)
    metrics_path.write_text(json.dumps(metric_payload, indent=2) + "\n")
    joblib.dump(model_pack, models_path)
    protocol = {
        "protocol_version": "ct28_augmented_relation_head_stress30_evaluation_v1",
        "held_out_test_accessed": False,
        "training_on_stress30": False,
        "local_models": "clean all-train HGB models",
        "relation_head": "clean-plus-15 augmented CT-conditioned HGB",
        "scene_base_stress": {"tangent_noise_deg": 30.0, "position_noise_mm": 1.0},
        "validation_rows_only_in_local_score_file": True,
        "source_sha256": {
            "scene_plan": relation_training.sha256(plan_path),
            "scene_candidates": relation_training.sha256(candidate_path),
            "scene_candidate_features": relation_training.sha256(feature_path),
        },
        "outputs_sha256": {
            path.name: relation_training.sha256(path)
            for path in (score_path, prediction_path, metrics_path, models_path)
        },
    }
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    require_frozen_settings(args)
    loaded = load_inputs(args)
    paths = loaded["paths"]
    plans = loaded["plans"]
    endpoints = loaded["endpoints"]
    candidates = loaded["candidates"]
    features = loaded["features"]
    clean_scores = loaded["scores"]
    clean_predictions = loaded["predictions"]
    clean_models = loaded["models"]
    clean_protocol = loaded["protocol"]
    clean_metrics = loaded["metrics"]

    fold_by_patient = {
        int(patient): int(fold)
        for patient, fold in clean_protocol["local_evidence"][
            "fold_by_patient"
        ].items()
    }
    fold_models, reproduction_error = fit_oof_local_models(
        features, clean_scores, fold_by_patient
    )
    print(
        f"clean OOF local HGB reproduction: max abs diff={reproduction_error:.3g}",
        flush=True,
    )

    all_clean_rows, relation_columns = clean_relation_rows(
        plans, candidates, clean_scores
    )
    clean_train_rows = [row for row in all_clean_rows if row["split"] == "train"]
    validation_rows = [row for row in all_clean_rows if row["split"] == "val"]
    per_variant_weight = 1.0 / (args.perturbations + 1)
    for row in clean_train_rows:
        row["augmentation_index"] = 0
        row["sample_weight"] = per_variant_weight

    perturbed_rows, perturbed_columns, audit_rows = build_augmented_train_rows(
        args,
        plans,
        endpoints,
        candidates,
        fold_models,
        fold_by_patient,
    )
    if perturbed_columns != relation_columns:
        raise AugmentationError("clean and perturbed relation feature orders differ")
    augmented_train_rows = clean_train_rows + perturbed_rows
    validate_augmentation_weights(augmented_train_rows, args.perturbations)

    augmented_head, augmented_metrics, augmented_predictions = fit_augmented_head(
        augmented_train_rows, validation_rows, relation_columns, args.tau
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.out_dir / "relation_local_scores_train_val.csv"
    shutil.copy2(paths["scores"], score_path)

    predictions = clean_predictions[
        clean_predictions["model"] != "ct_conditioned"
    ].copy()
    predictions = pd.concat(
        [predictions, pd.DataFrame(augmented_predictions)], ignore_index=True
    )
    prediction_path = args.out_dir / "relation_predictions_val.csv"
    predictions.to_csv(prediction_path, index=False)

    metric_payload = copy.deepcopy(clean_metrics)
    metric_payload["models"]["ct_conditioned"] = {
        "representation": "parallel_geometry_and_ct_historical42_landscapes",
        "feature_count": len(relation_columns),
        "training": "clean plus 15 train perturbations; equal per-scene total weight",
        **augmented_metrics,
    }
    metric_payload["augmentation"] = {
        "relation_head_only": True,
        "local_models_retrained_with_perturbations": False,
        "perturbations": args.perturbations,
        "position_noise_mm": args.position_noise_mm,
        "tangent_noise_deg": args.tangent_noise_deg,
        "train_source_scenes": len(clean_train_rows),
        "weighted_train_rows": len(augmented_train_rows),
        "weight_per_variant": per_variant_weight,
        "weight_sum_per_source_scene": 1.0,
        "clean_oof_max_abs_reproduction_error": reproduction_error,
    }
    metrics_path = args.out_dir / "relation_metrics_val.json"
    metrics_path.write_text(json.dumps(metric_payload, indent=2) + "\n")

    model_pack = copy.deepcopy(clean_models)
    model_pack["relation_heads"]["ct_conditioned"] = augmented_head
    model_pack["ct_conditioned_relation_training"] = {
        "augmentation": "clean_plus_15_perturbations",
        "sample_weight_per_variant": per_variant_weight,
        "local_models": "unchanged clean patient-OOF HGB",
    }
    models_path = args.out_dir / "relation_models.joblib"
    joblib.dump(model_pack, models_path)

    audit = pd.DataFrame(audit_rows).sort_values(["scan_id", "scene_id"])
    audit["clean_variants"] = 1
    audit["total_variants"] = args.perturbations + 1
    audit["weight_per_variant"] = per_variant_weight
    audit["total_scene_weight"] = 1.0
    audit_path = args.out_dir / "augmentation_audit_train.csv"
    audit.to_csv(audit_path, index=False)

    protocol = {
        "protocol_version": "ct28_relation_perturbation_augmentation_hgb_v1",
        "status": "development_train_val_only",
        "held_out_test_accessed": False,
        "path_stage_implemented": False,
        "local_evidence": {
            "family": "HGB",
            "streams": [
                "PAIR geometry",
                "PAIR CT",
                "PAIR geometry+CT",
                "JUNCTION geometry",
                "JUNCTION CT",
                "JUNCTION geometry+CT",
            ],
            "training": "clean only; five-fold patient-level OOF",
            "perturbation_retraining": False,
            "clean_oof_max_abs_reproduction_error": reproduction_error,
        },
        "relation_head": {
            "family": "HistGradientBoostingClassifier",
            "input": "PAIR/JUNCTION geometry and CT score landscapes",
            "feature_count": len(relation_columns),
            "feature_columns": relation_columns,
            "clean_source_scenes": len(clean_train_rows),
            "perturbed_copies_per_scene": args.perturbations,
            "fit_rows": len(augmented_train_rows),
            "sample_weight_per_variant": per_variant_weight,
            "weight_sum_per_source_scene": 1.0,
        },
        "perturbation": {
            "seed": args.seed,
            "position_noise_mm": args.position_noise_mm,
            "tangent_noise_deg": args.tangent_noise_deg,
            "mechanism": "endpoint position Gaussian noise and tangent angular Gaussian noise before feature extraction",
        },
        "validation": {
            "clean_scenes": int((plans["split"] == "val").sum()),
            "patients": sorted(relation_training.EXPECTED_PATIENT_IDS["val"]),
            "tau": args.tau,
            "candidate_identity": "geometry_plus_ct_score",
            "graph_consistency_gate": 0.60,
        },
        "source_files": {
            key: relation_training.provenance_path(path)
            for key, path in paths.items()
            if key not in {"metrics", "predictions", "models", "protocol"}
        },
        "source_sha256": {
            key: relation_training.sha256(path) for key, path in paths.items()
        },
        "outputs_sha256": {
            path.name: relation_training.sha256(path)
            for path in (score_path, prediction_path, metrics_path, models_path, audit_path)
        },
    }
    protocol_path = args.out_dir / "relation_model_protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    if args.stress_scene_dir.exists():
        write_stress_validation_evidence(
            args.stress_scene_dir,
            args.out_dir / "stress30_validation_evidence",
            model_pack,
            args.tau,
        )
    print(json.dumps(metric_payload["models"]["ct_conditioned"], indent=2))


if __name__ == "__main__":
    main()
