#!/usr/bin/env python3
"""Sweep train-scene copy count for the frozen CT28 48/0 relation baseline."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import run_ct28_graph_lira as graph
import train_ct28_perturbation_augmented_relation_head as augmentation
import train_ct28_relation_head as training
import train_ct28_selected_relation_augmentation as selected


REPO = training.REPO_ROOT
SCENES = graph.DEFAULT_SCENE_DIR
ROBUST = SCENES / "ablations/pair_junction_geometry_robust_unrounded"
BASE = REPO / "varvara/anza_lira_local_results/ct28_fusion_identity_augmentation/joint/pair_fusion/clean_evidence"
OUTPUT = REPO / "varvara/anza_lira_local_results/ct28_copy_count_ablations/relation"
COUNTS = (0, 3, 5, 10, 15)
TEST_IDS = {954, 958, 972, 973, 980, 984}


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=SCENES)
    parser.add_argument("--robust-evidence", type=Path, default=ROBUST)
    parser.add_argument("--base-evidence", type=Path, default=BASE)
    parser.add_argument("--archive-dir", type=Path, default=graph.DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    config = args()
    config.out_dir.mkdir(parents=True, exist_ok=True)
    loaded = augmentation.load_inputs(argparse.Namespace(
        scene_dir=config.scene_dir, clean_evidence_dir=config.robust_evidence))
    plans, candidates = loaded["plans"], loaded["candidates"]
    if set(plans.scan_id.astype(int)) & TEST_IDS:
        raise ValueError("held-out test entered scene pack")
    base_scores = pd.read_csv(config.base_evidence / "relation_local_scores_train_val.csv")
    base_predictions = pd.read_csv(config.base_evidence / "relation_predictions_val.csv")
    base_pack = joblib.load(config.base_evidence / "relation_models.joblib")
    base_protocol = json.loads((config.base_evidence / "relation_model_protocol.json").read_text())
    if base_protocol["source_sha256"] != loaded["protocol"]["source_sha256"]:
        raise ValueError("source scene hashes differ")
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
    if not base_scores[["scene_id", "candidate_id"]].equals(loaded["scores"][["scene_id", "candidate_id"]]):
        raise ValueError("candidate order changed")
    if not np.array_equal(base_scores[["geometry_score", "ct_score"]].to_numpy(),
                          loaded["scores"][["geometry_score", "ct_score"]].to_numpy()):
        raise ValueError("relation evidence changed")
    fold_models, oof_error = selected.fold_models_for_patients(loaded)
    clean_rows, columns = graph.relation_feature_rows(
        plans, candidates, base_scores, ("geometry_score", "ct_score"))
    if len(columns) != 84:
        raise ValueError("relation feature dimension changed")
    generation_args = argparse.Namespace(
        archive_dir=config.archive_dir, perturbations=15, seed=20260920,
        position_noise_mm=0.8, tangent_noise_deg=20.0)
    noisy, noisy_columns, audit = augmentation.build_augmented_train_rows(
        generation_args, plans, loaded["endpoints"], candidates,
        fold_models, {patient: patient for patient in fold_models})
    if noisy_columns != columns:
        raise ValueError("perturbed relation feature order changed")
    pd.DataFrame(audit).to_csv(config.out_dir / "train_extraction_audit.csv", index=False)
    pd.DataFrame(noisy).to_csv(config.out_dir / "joint_train_relation_copies.csv", index=False)
    validation_rows = [row for row in clean_rows if row["split"] == "val"]
    geometry_predictions = base_predictions[base_predictions.model.eq("geometry_only")]
    reference = base_predictions[base_predictions.model.eq("ct_conditioned")].reset_index(drop=True)
    for count in COUNTS:
        out = config.out_dir / f"k{count:02d}"
        out.mkdir(exist_ok=True)
        if count == 0:
            metrics, predictions, head = training.fit_relation_head(
                clean_rows, columns, 0.85, "ct_conditioned")
        else:
            train_rows = [dict(row, augmentation_index=0, sample_weight=0.5)
                          for row in clean_rows if row["split"] == "train"]
            train_rows.extend(dict(row, sample_weight=0.5/count)
                              for row in noisy if int(row["augmentation_index"]) <= count)
            augmentation.validate_augmentation_weights(train_rows, count)
            head, metrics, predictions = augmentation.fit_augmented_head(
                train_rows, validation_rows, columns, 0.85)
        if count == 15:
            observed = pd.DataFrame(predictions)
            for column in ("probability_none", "probability_pair", "probability_junction", "probability_both"):
                error = float(np.max(np.abs(observed[column].to_numpy() - reference[column].to_numpy())))
                if error > 1e-10:
                    raise ValueError(f"K=15 failed to reproduce reference relation probabilities: {column} {error}")
        pack = copy.deepcopy(base_pack)
        pack["relation_heads"]["ct_conditioned"] = head
        joblib.dump(pack, out / "relation_models.joblib")
        base_scores.to_csv(out / "relation_local_scores_train_val.csv", index=False)
        pd.concat([geometry_predictions, pd.DataFrame(predictions)], ignore_index=True).to_csv(
            out / "relation_predictions_val.csv", index=False)
        (out / "relation_metrics_val.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (out / "relation_model_protocol.json").write_text(json.dumps({
            "status": "validation_ablation_not_frozen", "held_out_test_accessed": False,
            "source_sha256": base_protocol["source_sha256"],
            "reference_evidence": str(config.base_evidence),
            "train_copy_count": count, "noise": "joint_0.8mm_20deg",
            "clean_weight": 1.0 if count == 0 else 0.5,
            "per_copy_weight": 0.0 if count == 0 else 0.5/count,
            "base_oof_reproduction_max_error": oof_error,
            "pair_identity": "geometry_plus_ct_score",
            "junction_identity": "geometry_score",
            "tau": 0.85, "consistency": 0.60,
            "evaluation_perturbations": 15,
        }, indent=2) + "\n")
        print(f"relation K={count} fitted", flush=True)


if __name__ == "__main__":
    main()
