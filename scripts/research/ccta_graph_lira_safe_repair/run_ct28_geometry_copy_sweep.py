#!/usr/bin/env python3
"""Sweep position-only local geometry copies with fresh patient-OOF relation inputs."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import run_ct28_graph_lira as graph
import train_ct28_geometry_augmentation_matrix as geometry
import train_ct28_perturbation_augmented_relation_head as augmentation
import train_ct28_relation_head as training
import train_ct28_selected_relation_augmentation as selected
from ct28_perturbations import perturb_scene_endpoints


REPO = training.REPO_ROOT
SCENES = graph.DEFAULT_SCENE_DIR
ROBUST = SCENES / "ablations/pair_junction_geometry_robust_unrounded"
BASE = REPO / "varvara/anza_lira_local_results/ct28_fusion_identity_augmentation/joint/pair_fusion/clean_evidence"
OUTPUT = REPO / "varvara/anza_lira_local_results/ct28_copy_count_ablations/geometry"
COUNTS = (0, 3, 5, 10, 15)
TEST_IDS = {954, 958, 972, 973, 980, 984}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=SCENES)
    parser.add_argument("--robust-evidence", type=Path, default=ROBUST)
    parser.add_argument("--base-evidence", type=Path, default=BASE)
    parser.add_argument("--archive-dir", type=Path, default=graph.DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    parser.add_argument("--reuse-saved-copies", action="store_true",
                        help="Reuse previously extracted train-only copies after validating their row counts.")
    return parser.parse_args()


def local_geometry_sweep(config, loaded, base_scores):
    features, candidates, endpoints = (loaded[key] for key in ("features", "candidates", "endpoints"))
    folds = loaded["protocol"]
    models = {k: {kind: {} for kind in ("PAIR", "JUNCTION")} for k in COUNTS}
    scores = {k: base_scores.copy().set_index(["scene_id", "candidate_id"]) for k in COUNTS}
    for kind in ("PAIR", "JUNCTION"):
        columns = geometry.geometry_columns(kind)
        frame = features[features.candidate_type.eq(kind)]
        clean = frame[frame.split.eq("train")][["scene_id", "candidate_id", "scan_id", "y", *columns]].copy()
        validation = frame[frame.split.eq("val")]
        scene_ids = set(clean.scene_id)
        kind_candidates = candidates[candidates.candidate_type.eq(kind) & candidates.scene_id.isin(scene_ids)]
        kind_endpoints = endpoints[endpoints.scene_id.isin(scene_ids)]
        path = config.out_dir / f"{kind.lower()}_position_train_copies.csv"
        if config.reuse_saved_copies and path.exists():
            noisy = pd.read_csv(path, float_precision="round_trip")
        else:
            copies = []
            for repeat in range(15):
                part = geometry.geometry_rows(kind_candidates, kind_endpoints, kind, "position_only", repeat)
                part = part.merge(clean[["scene_id", "candidate_id", "scan_id", "y"]],
                                  on=["scene_id", "candidate_id"], validate="one_to_one")
                if len(part) != len(clean):
                    raise ValueError(f"incomplete {kind} geometry copy {repeat}")
                part["augmentation_index"] = repeat + 1
                copies.append(part)
            noisy = pd.concat(copies, ignore_index=True)
            noisy.to_csv(path, index=False)
        if len(noisy) != 15*len(clean) or not noisy.groupby(["scene_id", "candidate_id"]).size().eq(15).all():
            raise ValueError(f"{kind} saved geometry copies incomplete")
        fold_map = {int(p): int(f) for p, f in folds[f"patient_oof_folds_{kind.lower()}_geometry"].items()}
        if set(fold_map) != training.EXPECTED_PATIENT_IDS["train"] or set(fold_map.values()) != set(range(5)):
            raise ValueError(f"{kind} OOF fold mapping changed")
        for k in COUNTS:
            fit_clean = clean.copy()
            fit_clean["weight"] = 1.0 if k == 0 else 0.5
            if k:
                extra = noisy[noisy.augmentation_index.le(k)].copy()
                extra["weight"] = 0.5/k
                fit = pd.concat([fit_clean, extra], ignore_index=True)
                if not np.allclose(fit.groupby(["scene_id", "candidate_id"]).weight.sum(), 1.0):
                    raise ValueError(f"{kind} K={k} sample weights changed")
            else:
                fit = fit_clean
            train_predictions = pd.Series(index=pd.MultiIndex.from_frame(clean[["scene_id", "candidate_id"]]), dtype=float)
            for fold in range(5):
                held_patients = {p for p, f in fold_map.items() if f == fold}
                train_fold = fit[~fit.scan_id.isin(held_patients)]
                held = clean[clean.scan_id.isin(held_patients)]
                model = training.local_model("hgb").fit(
                    train_fold[columns], train_fold.y,
                    **({"hgb__sample_weight": train_fold.weight} if k else {}))
                keys = pd.MultiIndex.from_frame(held[["scene_id", "candidate_id"]])
                train_predictions.loc[keys] = model.predict_proba(held[columns])[:, 1]
                for patient in held_patients:
                    models[k][kind][patient] = model
            final = training.local_model("hgb").fit(
                fit[columns], fit.y, **({"hgb__sample_weight": fit.weight} if k else {}))
            models[k][kind]["final"] = final
            val_keys = pd.MultiIndex.from_frame(validation[["scene_id", "candidate_id"]])
            all_predictions = pd.concat([
                train_predictions,
                pd.Series(final.predict_proba(validation[columns])[:, 1], index=val_keys)])
            mask = scores[k].candidate_type.eq(kind)
            scores[k].loc[mask, "geometry_score"] = all_predictions.loc[scores[k].index[mask]].to_numpy()
            if k == 5:
                original = base_scores[base_scores.candidate_type.eq(kind)].geometry_score.to_numpy()
                observed = scores[k].loc[mask, "geometry_score"].to_numpy()
                error = float(np.max(np.abs(observed-original)))
                if error > 1e-10:
                    raise ValueError(f"{kind} K=5 OOF/final reproduction failed: {error}")
            print(f"{kind} geometry K={k} fitted", flush=True)
    return models, {k: frame.reset_index() for k, frame in scores.items()}


def relation_copies(config, loaded, models, clean_relation, ct_folds):
    plans, candidates, endpoints = (loaded[key] for key in ("plans", "candidates", "endpoints"))
    train_plans = plans[plans.split.eq("train")]
    scene_ids = set(train_plans.scene_id)
    train_candidates = candidates[candidates.scene_id.isin(scene_ids)]
    train_endpoints = endpoints[endpoints.scene_id.isin(scene_ids)]
    expected_geometry = pd.read_csv(graph.ct_features.EXPECTED_GEOMETRY).set_index("scan_id")
    expected_alignment = pd.read_csv(graph.ct_features.EXPECTED_ALIGNMENT).set_index("scan_id")
    arm_names = graph.ct_features.arm_feature_names(graph.ct_features.RADII_MM)
    paths = {k: config.out_dir / f"relation_train_copies_k{k:02d}.csv" for k in COUNTS}
    if config.reuse_saved_copies and all(path.exists() for path in paths.values()):
        expected = 15*len(train_plans)
        if all(len(pd.read_csv(path, usecols=["scene_id"])) == expected for path in paths.values()):
            return paths
        raise ValueError("saved relation copy row counts changed")
    for path in paths.values():
        path.unlink(missing_ok=True)
    first_patient = min(training.EXPECTED_PATIENT_IDS["train"])
    for patient in sorted(training.EXPECTED_PATIENT_IDS["train"]):
        patient_plans = train_plans[train_plans.scan_id.eq(patient)]
        patient_scene_ids = set(patient_plans.scene_id)
        patient_candidates = train_candidates[train_candidates.scene_id.isin(patient_scene_ids)]
        patient_endpoints = train_endpoints[train_endpoints.scene_id.isin(patient_scene_ids)]
        by_scene = dict(tuple(patient_candidates.groupby("scene_id", sort=False)))
        ct, inverse_affine = graph.load_verified_ct(
            patient, config.archive_dir, expected_geometry, expected_alignment)
        rows_by_k = {k: [] for k in COUNTS}
        for repeat in range(15):
            feature_rows = []
            for scene_id, scene_endpoints in patient_endpoints.groupby("scene_id", sort=False):
                perturbed = perturb_scene_endpoints(
                    scene_endpoints, seed=20260920, repetition=repeat,
                    position_noise_mm=0.8, tangent_noise_deg=20.0).set_index("endpoint_id")
                for candidate in by_scene[scene_id].itertuples(index=False):
                    feature_rows.append(graph.perturbed_candidate_features(
                        pd.Series(candidate._asdict()), perturbed, ct, inverse_affine, arm_names))
            frame = pd.DataFrame(feature_rows)
            ct_scores = {}
            for kind in ("PAIR", "JUNCTION"):
                mask = frame.candidate_type.eq(kind)
                model = ct_folds[patient][kind]["ct"]
                ct_scores[kind] = model.predict_proba(
                    frame.loc[mask, list(model.feature_names_in_)])[:, 1]
            for k in COUNTS:
                local = frame[["scene_id", "candidate_id", "candidate_type"]].copy()
                local["geometry_score"] = np.nan
                local["ct_score"] = np.nan
                for kind in ("PAIR", "JUNCTION"):
                    mask = frame.candidate_type.eq(kind)
                    model = models[k][kind][patient]
                    local.loc[mask, "geometry_score"] = model.predict_proba(
                        frame.loc[mask, list(model.feature_names_in_)])[:, 1]
                    local.loc[mask, "ct_score"] = ct_scores[kind]
                rows, columns = graph.relation_feature_rows(
                    patient_plans, patient_candidates, local,
                    ("geometry_score", "ct_score"))
                if columns != clean_relation[k][1]:
                    raise ValueError("relation feature order changed")
                for row in rows:
                    row["augmentation_index"] = repeat + 1
                rows_by_k[k].extend(rows)
        for k, rows in rows_by_k.items():
            pd.DataFrame(rows).to_csv(
                paths[k], mode="a", header=patient == first_patient, index=False)
        print(f"geometry relation copies: patient {patient} complete", flush=True)
        del ct
    return paths


def main() -> None:
    config = arguments()
    config.out_dir.mkdir(parents=True, exist_ok=True)
    loaded = augmentation.load_inputs(argparse.Namespace(
        scene_dir=config.scene_dir, clean_evidence_dir=config.robust_evidence))
    plans, candidates = loaded["plans"], loaded["candidates"]
    if set(plans.scan_id.astype(int)) & TEST_IDS:
        raise ValueError("held-out test entered scene pack")
    base_scores = pd.read_csv(config.base_evidence / "relation_local_scores_train_val.csv")
    base_pack = joblib.load(config.base_evidence / "relation_models.joblib")
    base_protocol = json.loads((config.base_evidence / "relation_model_protocol.json").read_text())
    if base_protocol["source_sha256"] != loaded["protocol"]["source_sha256"]:
        raise ValueError("frozen scene SHA changed")
    actual_hashes = {
        key: training.sha256(config.scene_dir / filename)
        for key, filename in (
            ("scene_plan", "scene_plan_train_val.csv"),
            ("scene_candidates", "scene_candidates_train_val.csv"),
            ("scene_candidate_features", "scene_candidate_features_train_val.csv"),
        )
    }
    if actual_hashes != base_protocol["source_sha256"]:
        raise ValueError("scene files no longer match frozen SHA")
    if not base_scores[["scene_id", "candidate_id"]].equals(
        loaded["scores"][["scene_id", "candidate_id"]]):
        raise ValueError("candidate order changed")
    models, scores = local_geometry_sweep(config, loaded, base_scores)
    if not np.array_equal(scores[5][["scene_id", "candidate_id"]].to_numpy(),
                          base_scores[["scene_id", "candidate_id"]].to_numpy()):
        raise ValueError("K=5 candidate order changed")
    scores[5]["geometry_score"] = base_scores["geometry_score"].to_numpy()
    ct_folds, oof_error = selected.fold_models_for_patients(loaded)
    clean_relation = {}
    for k in COUNTS:
        rows, columns = graph.relation_feature_rows(
            plans, candidates, scores[k], ("geometry_score", "ct_score"))
        if len(columns) != 84:
            raise ValueError("relation feature dimension changed")
        clean_relation[k] = (rows, columns)
    noisy_paths = relation_copies(config, loaded, models, clean_relation, ct_folds)
    reference = pd.read_csv(config.base_evidence / "relation_predictions_val.csv")
    reference = reference[reference.model.eq("ct_conditioned")].reset_index(drop=True)
    for k in COUNTS:
        rows, columns = clean_relation[k]
        train = [dict(row, augmentation_index=0, sample_weight=0.5)
                 for row in rows if row["split"] == "train"]
        validation = [row for row in rows if row["split"] == "val"]
        noisy = [dict(row, sample_weight=0.5/15)
                 for row in pd.read_csv(noisy_paths[k], float_precision="round_trip").to_dict("records")]
        augmented_train = train + noisy
        augmentation.validate_augmentation_weights(augmented_train, 15)
        head, metrics, predictions = augmentation.fit_augmented_head(
            augmented_train, validation, columns, 0.85)
        geometry_rows, geometry_columns = graph.relation_feature_rows(
            plans, candidates, scores[k], ("geometry_score",))
        _, geometry_predictions, geometry_head = training.fit_relation_head(
            geometry_rows, geometry_columns, 0.85, "geometry_only")
        if k == 5:
            observed = pd.DataFrame(predictions)
            for column in ("probability_none", "probability_pair", "probability_junction", "probability_both"):
                difference = float(np.max(np.abs(observed[column].to_numpy()-reference[column].to_numpy())))
                if difference > 1e-10:
                    raise ValueError(f"K=5 relation reproduction failed: {column} {difference}")
        out = config.out_dir / f"k{k:02d}"
        out.mkdir(exist_ok=True)
        pack = copy.deepcopy(base_pack)
        for kind in ("PAIR", "JUNCTION"):
            pack["local_models"][kind]["geometry"] = models[k][kind]["final"]
        pack["relation_heads"]["ct_conditioned"] = head
        pack["relation_heads"]["geometry_only"] = geometry_head
        pack["geometry_relation_features"] = geometry_columns
        joblib.dump(pack, out / "relation_models.joblib")
        scores[k].to_csv(out / "relation_local_scores_train_val.csv", index=False)
        pd.DataFrame(geometry_predictions+predictions).to_csv(
            out / "relation_predictions_val.csv", index=False)
        (out / "relation_metrics_val.json").write_text(json.dumps(metrics, indent=2)+"\n")
        (out / "relation_model_protocol.json").write_text(json.dumps({
            "status": "validation_ablation_not_frozen", "held_out_test_accessed": False,
            "source_sha256": base_protocol["source_sha256"],
            "reference_evidence": str(config.base_evidence),
            "geometry_train_copy_count": k, "geometry_noise": "position_only_0.8mm",
            "geometry_clean_weight": 1.0 if k == 0 else 0.5,
            "geometry_per_copy_weight": 0 if k == 0 else 0.5/k,
            "relation_train_copy_count": 15, "relation_noise": "joint_0.8mm_20deg",
            "relation_clean_weight": 0.5,
            "base_oof_reproduction_max_error": oof_error,
            "pair_identity": "geometry_plus_ct_score", "junction_identity": "geometry_score",
            "tau": 0.85, "consistency": 0.60, "evaluation_perturbations": 15,
        }, indent=2)+"\n")
        print(f"geometry K={k} evidence complete", flush=True)


if __name__ == "__main__":
    main()
