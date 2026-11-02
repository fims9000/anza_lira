#!/usr/bin/env python3
"""Train relation-only perturbation ablations on the selected CT28 HGB pack."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import train_ct28_perturbation_augmented_relation_head as previous
import train_ct28_position_robust_junction as junction_training
import train_ct28_position_robust_pair_geometry as pair_training
import train_ct28_relation_head as training


ROOT = training.DEFAULT_SCENE_DIR
DEFAULT_EVIDENCE = ROOT / "ablations/pair_junction_geometry_robust_unrounded"
DEFAULT_OUTPUT = training.REPO_ROOT / "varvara/anza_lira_local_results/ct28_selected_relation_augmentation"


def fold_models_for_patients(loaded: dict[str, object]) -> tuple[dict[int, dict], float]:
    features = loaded["features"]
    scores = loaded["scores"]
    plans = loaded["plans"]
    candidates = loaded["candidates"]
    endpoints = loaded["endpoints"]
    protocol = loaded["protocol"]
    train = features[features.split.eq("train")].copy()
    train["scan_id"] = train.scan_id.astype(int)
    score_train = scores[scores.split.eq("train")]
    original_folds = score_train.groupby("scan_id").local_oof_fold.first().astype(int).to_dict()
    geometry_folds = {
        "PAIR": {int(k): int(v) for k, v in protocol["patient_oof_folds_pair_geometry"].items()},
        "JUNCTION": {int(k): int(v) for k, v in protocol["patient_oof_folds_junction_geometry"].items()},
    }
    if set(original_folds) != training.EXPECTED_PATIENT_IDS["train"]:
        raise ValueError("unexpected patient set in saved OOF scores")
    by_type = {}
    for candidate_type, builder in (("PAIR", pair_training), ("JUNCTION", junction_training)):
        geometry = list(training.PAIR_GEOMETRY_FEATURES if candidate_type == "PAIR" else training.JUNCTION_GEOMETRY_FEATURES)
        kind_candidates = candidates[candidates.candidate_type.eq(candidate_type)].merge(
            plans[["scene_id", "scan_id", "split"]], on="scene_id", validate="many_to_one"
        )
        train_candidates = kind_candidates[kind_candidates.split.eq("train")]
        train_endpoints = endpoints.merge(
            plans[["scene_id", "scan_id", "split"]], on="scene_id", validate="many_to_one"
        ).query('split == "train"')
        clean = train[train.candidate_type.eq(candidate_type)].copy()
        clean["weight"] = 0.5
        parts = [clean]
        for repetition in range(5):
            function = builder.pair_geometry_rows if candidate_type == "PAIR" else builder.geometry_rows
            noisy = function(train_candidates, train_endpoints, repetition=repetition,
                             seed=20260920, position_noise_mm=0.8)
            noisy = noisy.merge(clean[["scene_id", "candidate_id", "scan_id", "y"]],
                                on=["scene_id", "candidate_id"], validate="one_to_one")
            if len(noisy) != len(clean):
                raise ValueError(f"incomplete {candidate_type} geometry augmentation")
            noisy["weight"] = 0.1
            parts.append(noisy)
        by_type[candidate_type] = (clean, pd.concat(parts, ignore_index=True), geometry)

    models_by_patient = {patient: {kind: {} for kind in ("PAIR", "JUNCTION")}
                         for patient in original_folds}
    max_error = 0.0
    for candidate_type, (clean, augmented, geometry) in by_type.items():
        for stream in training.EVIDENCE_STREAMS:
            columns = list(loaded["models"]["local_models"][candidate_type][stream].feature_names_in_)
            folds = geometry_folds[candidate_type] if stream == "geometry" else original_folds
            fit_data = augmented if stream == "geometry" else clean
            for fold in range(5):
                held_patients = {patient for patient, assigned in folds.items() if assigned == fold}
                fit = fit_data[~fit_data.scan_id.isin(held_patients)]
                held = clean[clean.scan_id.isin(held_patients)]
                model = training.local_model("hgb").fit(
                    fit[columns], fit.y,
                    **({"hgb__sample_weight": fit.weight} if stream == "geometry" else {}),
                )
                prediction = model.predict_proba(held[columns])[:, 1]
                expected = held[["scene_id", "candidate_id"]].merge(
                    score_train[["scene_id", "candidate_id", f"{stream}_score"]],
                    on=["scene_id", "candidate_id"], validate="one_to_one"
                )[f"{stream}_score"].to_numpy()
                max_error = max(max_error, float(np.max(np.abs(prediction - expected))))
                for patient in held_patients:
                    models_by_patient[patient][candidate_type][stream] = model
    if max_error > 1e-10:
        raise ValueError(f"selected-baseline OOF reproduction failed: {max_error}")
    return models_by_patient, max_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=ROOT)
    parser.add_argument("--clean-evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--archive-dir", type=Path, default=previous.DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--position-noise-mm", type=float, default=0.8)
    parser.add_argument("--tangent-noise-deg", type=float, default=20.0)
    args = parser.parse_args()
    if (args.position_noise_mm, args.tangent_noise_deg) not in {(0.8, 20.0), (0.8, 0.0), (0.0, 20.0)}:
        raise ValueError("only predeclared joint, position-only and tangent-only modes are allowed")
    args.perturbations = 15
    args.seed = 20260920
    args.tau = 0.85
    loaded = previous.load_inputs(args)
    models_by_patient, max_error = fold_models_for_patients(loaded)
    print(f"selected-baseline OOF reproduction max error: {max_error:.3g}", flush=True)
    clean_rows, columns = previous.clean_relation_rows(
        loaded["plans"], loaded["candidates"], loaded["scores"]
    )
    train_clean = [row for row in clean_rows if row["split"] == "train"]
    val_clean = [row for row in clean_rows if row["split"] == "val"]
    for row in train_clean:
        row["augmentation_index"] = 0
    noisy_rows, noisy_columns, audit = previous.build_augmented_train_rows(
        args, loaded["plans"], loaded["endpoints"], loaded["candidates"],
        models_by_patient, {patient: patient for patient in models_by_patient},
    )
    if columns != noisy_columns:
        raise ValueError("relation feature columns changed")
    mode = "joint" if args.position_noise_mm and args.tangent_noise_deg else (
        "position_only" if args.position_noise_mm else "tangent_only"
    )
    for name, clean_weight in (("half", 0.5), ("equal", 1.0 / 16.0)):
        if mode != "joint" and name == "equal":
            continue
        out = args.out_dir / mode / name
        out.mkdir(parents=True, exist_ok=True)
        train_rows = copy.deepcopy(train_clean + noisy_rows)
        for row in train_rows:
            row["sample_weight"] = clean_weight if row["augmentation_index"] == 0 else (1-clean_weight)/15
        previous.validate_augmentation_weights(train_rows, 15)
        head, metrics, new_predictions = previous.fit_augmented_head(train_rows, val_clean, columns, 0.85)
        shutil.copy2(loaded["paths"]["scores"], out / "relation_local_scores_train_val.csv")
        saved = loaded["predictions"].query('model != "ct_conditioned"')
        pd.concat([saved, pd.DataFrame(new_predictions)], ignore_index=True).to_csv(
            out / "relation_predictions_val.csv", index=False
        )
        model_pack = copy.deepcopy(loaded["models"])
        model_pack["relation_heads"]["ct_conditioned"] = head
        joblib.dump(model_pack, out / "relation_models.joblib")
        (out / "relation_metrics_val.json").write_text(json.dumps(metrics, indent=2) + "\n")
        source = loaded["paths"]
        (out / "relation_model_protocol.json").write_text(json.dumps({
            "status": "validation_only", "held_out_test_accessed": False,
            "base_model_sha256": training.sha256(source["models"]),
            "noise_mode": mode, "position_noise_mm": args.position_noise_mm,
            "tangent_noise_deg": args.tangent_noise_deg,
            "clean_weight": clean_weight, "perturbation_weight_each": (1-clean_weight)/15,
            "oof_reproduction_max_error": max_error,
            "source_sha256": {k: training.sha256(args.scene_dir / f) for k, f in (
                ("scene_plan", "scene_plan_train_val.csv"),
                ("scene_candidates", "scene_candidates_train_val.csv"),
                ("scene_candidate_features", "scene_candidate_features_train_val.csv"))},
        }, indent=2) + "\n")
        print(mode, name, "clean relation metrics", metrics, flush=True)
    pd.DataFrame(audit).to_csv(args.out_dir / mode / "augmentation_audit_train.csv", index=False)


if __name__ == "__main__":
    main()
