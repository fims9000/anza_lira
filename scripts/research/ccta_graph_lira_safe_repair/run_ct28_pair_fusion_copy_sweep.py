#!/usr/bin/env python3
"""Sweep PAIR geometry+CT identity train-copy count on the frozen CT28 48/0 pack."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import run_ct28_graph_lira as graph
import train_ct28_fusion_identity_augmentation as fusion
import train_ct28_relation_head as training


REPO = training.REPO_ROOT
SCENES = graph.DEFAULT_SCENE_DIR
BASE = REPO / "varvara/anza_lira_local_results/ct28_fusion_identity_augmentation/joint/pair_fusion/clean_evidence"
ORIGINAL = REPO / "varvara/anza_lira_local_results/ct28_selected_relation_augmentation/joint/half"
OUTPUT = REPO / "varvara/anza_lira_local_results/ct28_copy_count_ablations/fusion"
COUNTS = (0, 3, 5, 10, 15)


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=SCENES)
    parser.add_argument("--base-evidence", type=Path, default=BASE)
    parser.add_argument("--original-evidence", type=Path, default=ORIGINAL)
    parser.add_argument("--archive-dir", type=Path, default=graph.DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    config = args()
    config.out_dir.mkdir(parents=True, exist_ok=True)
    plans = pd.read_csv(config.scene_dir / "scene_plan_train_val.csv")
    candidates = pd.read_csv(config.scene_dir / "scene_candidates_train_val.csv")
    endpoints = pd.read_csv(config.scene_dir / "scene_endpoints_train_val.csv")
    features = pd.read_csv(config.scene_dir / "scene_candidate_features_train_val.csv", low_memory=False)
    training.validate_scene_inputs(plans, candidates)
    if set(plans.scan_id.astype(int)) & fusion.TEST_PATIENTS:
        raise ValueError("held-out test entered scene pack")
    base_scores = pd.read_csv(config.base_evidence / "relation_local_scores_train_val.csv")
    original_scores = pd.read_csv(config.original_evidence / "relation_local_scores_train_val.csv")
    base_pack = joblib.load(config.base_evidence / "relation_models.joblib")
    base_protocol = json.loads((config.base_evidence / "relation_model_protocol.json").read_text())
    hashes = fusion.source_hashes(config.scene_dir)
    if base_protocol["source_sha256"] != hashes:
        raise ValueError("scene source SHA changed")
    if not base_scores[["scene_id", "candidate_id"]].equals(features[["scene_id", "candidate_id"]]):
        raise ValueError("feature/score candidate order changed")
    if not np.array_equal(base_scores[["geometry_score", "ct_score"]].to_numpy(),
                          original_scores[["geometry_score", "ct_score"]].to_numpy()):
        raise ValueError("relation evidence changed")
    columns = list(base_pack["local_models"]["PAIR"]["geometry_plus_ct"].feature_names_in_)
    if len(columns) != 45:
        raise ValueError("PAIR fusion feature list changed")
    clean = features[(features.split.eq("train")) & features.candidate_type.eq("PAIR")]
    clean = clean[["scene_id", "candidate_id", "scan_id", "y", *columns]].copy()
    validation = features[(features.split.eq("val")) & features.candidate_type.eq("PAIR")]
    pair_candidates = candidates[candidates.candidate_type.eq("PAIR")].merge(
        plans[["scene_id", "scan_id", "split"]], on="scene_id", validate="many_to_one")
    train_pairs = pair_candidates[pair_candidates.split.eq("train")]
    train_endpoints = endpoints[endpoints.scene_id.isin(train_pairs.scene_id)]
    patient_folds = original_scores[original_scores.split.eq("train")].groupby("scan_id").local_oof_fold.first().astype(int).to_dict()
    if set(patient_folds) != training.EXPECTED_PATIENT_IDS["train"] or set(patient_folds.values()) != set(range(5)):
        raise ValueError("patient OOF fold map changed")
    expected_geometry = pd.read_csv(graph.ct_features.EXPECTED_GEOMETRY).set_index("scan_id")
    expected_alignment = pd.read_csv(graph.ct_features.EXPECTED_ALIGNMENT).set_index("scan_id")
    arm_names = graph.ct_features.arm_feature_names(graph.ct_features.RADII_MM)
    noisy_rows = []
    for patient in sorted(patient_folds):
        patient_pairs = train_pairs[train_pairs.scan_id.eq(patient)]
        patient_endpoints = train_endpoints[train_endpoints.scene_id.isin(patient_pairs.scene_id)]
        by_scene = dict(tuple(patient_pairs.groupby("scene_id", sort=False)))
        ct, inverse_affine = graph.load_verified_ct(patient, config.archive_dir, expected_geometry, expected_alignment)
        for repetition in range(15):
            for scene_id, scene_endpoints in patient_endpoints.groupby("scene_id", sort=False):
                perturbed = fusion.perturb(scene_endpoints, repetition, "joint").set_index("endpoint_id")
                for candidate in by_scene[scene_id].itertuples(index=False):
                    values = graph.perturbed_candidate_features(
                        pd.Series(candidate._asdict()), perturbed, ct, inverse_affine, arm_names)
                    noisy_rows.append({
                        "scene_id": scene_id, "candidate_id": candidate.candidate_id,
                        "scan_id": patient, "y": int(candidate.y),
                        "augmentation_index": repetition + 1,
                        **{column: values[column] for column in columns},
                    })
        print(f"PAIR fusion copies: patient {patient} complete", flush=True)
        del ct
    noisy = pd.DataFrame(noisy_rows)
    if len(noisy) != 15*len(clean) or not noisy.groupby(["scene_id", "candidate_id"]).size().eq(15).all():
        raise ValueError("PAIR fusion copies incomplete")
    noisy.to_csv(config.out_dir / "joint_pair_train_copies.csv", index=False)

    for count in COUNTS:
        fit_clean = clean.copy()
        fit_clean["weight"] = 1.0 if count == 0 else 0.5
        if count:
            fit_noisy = noisy[noisy.augmentation_index.le(count)].copy()
            fit_noisy["weight"] = 0.5/count
            fit = pd.concat([fit_clean, fit_noisy], ignore_index=True)
            if not np.allclose(fit.groupby(["scene_id", "candidate_id"]).weight.sum(), 1.0):
                raise ValueError("PAIR fusion sample weights changed")
        else:
            fit = fit_clean
        train_scores = pd.Series(index=pd.MultiIndex.from_frame(clean[["scene_id", "candidate_id"]]), dtype=float)
        for fold in range(5):
            held_ids = {patient for patient, assigned in patient_folds.items() if assigned == fold}
            train_fold = fit[~fit.scan_id.isin(held_ids)]
            held = clean[clean.scan_id.isin(held_ids)]
            model = training.local_model("hgb").fit(
                train_fold[columns], train_fold.y,
                **({"hgb__sample_weight": train_fold.weight} if count else {}))
            keys = pd.MultiIndex.from_frame(held[["scene_id", "candidate_id"]])
            train_scores.loc[keys] = model.predict_proba(held[columns])[:, 1]
        final = training.local_model("hgb").fit(
            fit[columns], fit.y,
            **({"hgb__sample_weight": fit.weight} if count else {}))
        val_keys = pd.MultiIndex.from_frame(validation[["scene_id", "candidate_id"]])
        val_scores = pd.Series(final.predict_proba(validation[columns])[:, 1], index=val_keys)
        combined = pd.concat([train_scores, val_scores])
        if combined.isna().any():
            raise ValueError("PAIR fusion OOF/final scores incomplete")
        output_scores = base_scores.copy().set_index(["scene_id", "candidate_id"])
        mask = output_scores.candidate_type.eq("PAIR")
        output_scores.loc[mask, "geometry_plus_ct_score"] = combined.loc[output_scores.index[mask]].to_numpy()
        output_scores = output_scores.reset_index()
        if count == 5:
            pair_mask = base_scores.candidate_type.eq("PAIR")
            error = float(np.max(np.abs(output_scores.loc[pair_mask, "geometry_plus_ct_score"].to_numpy()
                                        - base_scores.loc[pair_mask, "geometry_plus_ct_score"].to_numpy())))
            if error > 1e-10:
                raise ValueError(f"K=5 PAIR fusion failed to reproduce 48/0 source: {error}")
        out = config.out_dir / f"k{count:02d}"
        out.mkdir(exist_ok=True)
        pack = copy.deepcopy(base_pack)
        pack["local_models"]["PAIR"]["geometry_plus_ct"] = final
        joblib.dump(pack, out / "relation_models.joblib")
        output_scores.to_csv(out / "relation_local_scores_train_val.csv", index=False)
        (out / "relation_predictions_val.csv").write_bytes(
            (config.base_evidence / "relation_predictions_val.csv").read_bytes())
        (out / "relation_model_protocol.json").write_text(json.dumps({
            "status": "validation_ablation_not_frozen", "held_out_test_accessed": False,
            "source_sha256": hashes, "reference_evidence": str(config.base_evidence),
            "pair_fusion_train_copy_count": count, "noise": "joint_0.8mm_20deg",
            "clean_weight": 1.0 if count == 0 else 0.5,
            "per_copy_weight": 0.0 if count == 0 else 0.5/count,
            "relation_head_unchanged": True,
            "pair_identity": "geometry_plus_ct_score", "junction_identity": "geometry_score",
            "tau": 0.85, "consistency": 0.60, "evaluation_perturbations": 15,
        }, indent=2) + "\n")
        print(f"PAIR fusion K={count} fitted", flush=True)


if __name__ == "__main__":
    main()
