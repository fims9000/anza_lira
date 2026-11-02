#!/usr/bin/env python3
"""Train PAIR/JUNCTION fusion identity ablations on clean CT28 train patients."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import run_ct28_graph_lira as graph
import train_ct28_relation_head as training
from ct28_perturbations import perturb_scene_endpoints


ROOT = graph.DEFAULT_SCENE_DIR
BASE = training.REPO_ROOT / "varvara/anza_lira_local_results/ct28_selected_relation_augmentation/joint/half"
STRESS_SCENES = ROOT / "ablations/stress30_jitter1"
OUTPUT = training.REPO_ROOT / "varvara/anza_lira_local_results/ct28_fusion_identity_augmentation"
MODES = {
    "position_only": (0.8, 0.0),
    "tangent_only": (0.0, 20.0),
    "joint": (0.8, 20.0),
}
SCOPES = {
    "pair_fusion": ("PAIR",),
    "junction_fusion": ("JUNCTION",),
    "both_fusion": ("PAIR", "JUNCTION"),
}
TEST_PATIENTS = {954, 958, 972, 973, 980, 984}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--scene-dir", type=Path, default=ROOT)
    parser.add_argument("--base-evidence", type=Path, default=BASE)
    parser.add_argument("--stress-scene-dir", type=Path, default=STRESS_SCENES)
    parser.add_argument("--stress-evidence", type=Path, default=BASE / "stress30_evidence")
    parser.add_argument("--archive-dir", type=Path, default=graph.DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    return parser.parse_args()


def source_hashes(scene_dir: Path) -> dict[str, str]:
    return {
        key: training.sha256(scene_dir / filename)
        for key, filename in (
            ("scene_plan", "scene_plan_train_val.csv"),
            ("scene_candidates", "scene_candidates_train_val.csv"),
            ("scene_candidate_features", "scene_candidate_features_train_val.csv"),
        )
    }


def perturb(endpoints: pd.DataFrame, repetition: int, mode: str) -> pd.DataFrame:
    position_sigma, tangent_sigma = MODES[mode]
    return perturb_scene_endpoints(
        endpoints,
        seed=20260920,
        repetition=repetition,
        position_noise_mm=position_sigma,
        tangent_noise_deg=tangent_sigma,
    )


def replace_fusion_scores(
    source: pd.DataFrame,
    features: pd.DataFrame,
    final_models: dict[str, object],
    selected_types: tuple[str, ...],
) -> pd.DataFrame:
    output = source.copy().set_index(["scene_id", "candidate_id"])
    indexed_features = features.set_index(["scene_id", "candidate_id"])
    for kind in selected_types:
        kind_rows = source[source["candidate_type"] == kind]
        keys = pd.MultiIndex.from_frame(kind_rows[["scene_id", "candidate_id"]])
        columns = list(final_models[kind].feature_names_in_)
        output.loc[keys, "geometry_plus_ct_score"] = final_models[kind].predict_proba(
            indexed_features.loc[keys, columns]
        )[:, 1]
    result = output.reset_index()
    if len(result) != len(source) or not result[["scene_id", "candidate_id"]].equals(
        source[["scene_id", "candidate_id"]]
    ):
        raise ValueError("candidate identity/order changed")
    if not np.array_equal(result["geometry_score"], source["geometry_score"]) or not np.array_equal(
        result["ct_score"], source["ct_score"]
    ):
        raise ValueError("relation evidence changed")
    if not np.isfinite(result[["geometry_score", "ct_score", "geometry_plus_ct_score"]]).all().all():
        raise ValueError("incomplete local scores")
    return result


def write_evidence(
    dest: Path,
    source_dir: Path,
    scores: pd.DataFrame,
    base_models: dict[str, object],
    final_models: dict[str, object],
    selected_types: tuple[str, ...],
    hashes: dict[str, str],
    mode: str,
    stress: bool,
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    scores.to_csv(dest / "relation_local_scores_train_val.csv", index=False)
    pack = copy.deepcopy(base_models)
    for kind in selected_types:
        pack["local_models"][kind]["geometry_plus_ct"] = final_models[kind]
    joblib.dump(pack, dest / "relation_models.joblib")
    shutil.copy2(source_dir / "relation_predictions_val.csv", dest / "relation_predictions_val.csv")
    protocol = {
        "status": "validation_ablation_not_frozen",
        "held_out_test_accessed": False,
        "train_on_stress30": False,
        "mode": mode,
        "selected_types": list(selected_types),
        "local_change": "Only selected geometry+CT fusion identity model(s); relation HGB and all geometry/CT local models unchanged",
        "train_augmentation": {
            "clean_weight": 0.5,
            "perturbed_copies": 5,
            "weight_per_copy": 0.1,
            "position_sigma_mm": MODES[mode][0],
            "tangent_sigma_deg": MODES[mode][1],
            "seed": 20260920,
        },
        "evaluation_base_stress30": stress,
        "source_sha256": hashes,
        "frozen_relation_head_source": str(source_dir),
        "output_sha256": {
            filename: training.sha256(dest / filename)
            for filename in (
                "relation_local_scores_train_val.csv",
                "relation_models.joblib",
                "relation_predictions_val.csv",
            )
        },
    }
    (dest / "relation_model_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")


def main() -> None:
    args = arguments()
    mode_dir = args.out_dir / args.mode
    clean_protocol = json.loads((args.base_evidence / "relation_model_protocol.json").read_text())
    clean_hashes = source_hashes(args.scene_dir)
    if clean_hashes != clean_protocol["source_sha256"]:
        raise ValueError("clean frozen scene source changed")
    stress_protocol = json.loads((args.stress_evidence / "relation_model_protocol.json").read_text())
    stress_hashes = source_hashes(args.stress_scene_dir)
    if stress_hashes != stress_protocol["source_sha256"]:
        raise ValueError("Stress30 frozen scene source changed")

    plans = pd.read_csv(args.scene_dir / "scene_plan_train_val.csv")
    candidates = pd.read_csv(args.scene_dir / "scene_candidates_train_val.csv")
    training.validate_scene_inputs(plans, candidates)
    if set(plans["scan_id"].astype(int)) & TEST_PATIENTS:
        raise ValueError("held-out test appeared in clean scene pack")
    stress_plans = pd.read_csv(args.stress_scene_dir / "scene_plan_train_val.csv")
    stress_candidates = pd.read_csv(args.stress_scene_dir / "scene_candidates_train_val.csv")
    training.validate_scene_inputs(stress_plans, stress_candidates)
    if set(stress_plans["scan_id"].astype(int)) & TEST_PATIENTS:
        raise ValueError("held-out test appeared in Stress30 scene pack")
    if not plans[["scene_id", "scan_id", "split", "relation_class"]].equals(
        stress_plans[["scene_id", "scan_id", "split", "relation_class"]]
    ):
        raise ValueError("clean and Stress30 scene identities differ")

    clean_features = pd.read_csv(args.scene_dir / "scene_candidate_features_train_val.csv", low_memory=False)
    stress_features = pd.read_csv(args.stress_scene_dir / "scene_candidate_features_train_val.csv", low_memory=False)
    clean_scores = pd.read_csv(args.base_evidence / "relation_local_scores_train_val.csv")
    stress_scores = pd.read_csv(args.stress_evidence / "relation_local_scores_train_val.csv")
    if not clean_scores[["scene_id", "candidate_id"]].equals(
        clean_features[["scene_id", "candidate_id"]]
    ):
        raise ValueError("clean features/scores are misaligned")
    stress_val_features = stress_features[stress_features["split"] == "val"].reset_index(drop=True)
    if not stress_scores[["scene_id", "candidate_id"]].equals(
        stress_val_features[["scene_id", "candidate_id"]]
    ):
        raise ValueError("Stress30 features/scores are misaligned")
    base_models = joblib.load(args.base_evidence / "relation_models.joblib")
    stress_models = joblib.load(args.stress_evidence / "relation_models.joblib")
    if base_models["local_model_family"] != "hgb" or stress_models["local_model_family"] != "hgb":
        raise ValueError("HGB local model packs required")
    columns = {
        kind: list(base_models["local_models"][kind]["geometry_plus_ct"].feature_names_in_)
        for kind in ("PAIR", "JUNCTION")
    }
    if (len(columns["PAIR"]), len(columns["JUNCTION"])) != (45, 161):
        raise ValueError("unexpected fusion feature counts")
    train_scores = clean_scores[clean_scores["split"] == "train"]
    patient_folds = train_scores.groupby("scan_id")["local_oof_fold"].first().astype(int).to_dict()
    if set(patient_folds) != training.EXPECTED_PATIENT_IDS["train"] or set(patient_folds.values()) != set(range(5)):
        raise ValueError("unexpected patient OOF folds")
    if not train_scores.groupby("scan_id")["local_oof_fold"].nunique().eq(1).all():
        raise ValueError("patient spans OOF folds")

    endpoints = pd.read_csv(args.scene_dir / "scene_endpoints_train_val.csv")
    train_candidates = candidates.merge(
        plans[["scene_id", "scan_id", "split"]], on="scene_id", validate="many_to_one"
    ).query('split == "train"')
    train_endpoints = endpoints[endpoints["scene_id"].isin(train_candidates["scene_id"])]
    augmented: dict[str, list[pd.DataFrame]] = {"PAIR": [], "JUNCTION": []}
    audit = []
    expected_geometry = pd.read_csv(graph.ct_features.EXPECTED_GEOMETRY).set_index("scan_id")
    expected_alignment = pd.read_csv(graph.ct_features.EXPECTED_ALIGNMENT).set_index("scan_id")
    arm_names = graph.ct_features.arm_feature_names(graph.ct_features.RADII_MM)
    for patient in sorted(patient_folds):
        patient_candidates = train_candidates[train_candidates["scan_id"] == patient]
        patient_endpoints = train_endpoints[train_endpoints["scene_id"].isin(patient_candidates["scene_id"])]
        by_scene = dict(tuple(patient_candidates.groupby("scene_id", sort=False)))
        ct, inverse_affine = graph.load_verified_ct(
            patient, args.archive_dir, expected_geometry, expected_alignment
        )
        rows = {"PAIR": [], "JUNCTION": []}
        for repetition in range(5):
            for scene_id, scene_endpoints in patient_endpoints.groupby("scene_id", sort=False):
                lookup = perturb(scene_endpoints, repetition, args.mode).set_index("endpoint_id")
                for candidate in by_scene[scene_id].itertuples(index=False):
                    values = graph.perturbed_candidate_features(
                        pd.Series(candidate._asdict()), lookup, ct, inverse_affine, arm_names
                    )
                    kind = candidate.candidate_type
                    rows[kind].append({
                        "scene_id": scene_id,
                        "candidate_id": candidate.candidate_id,
                        "scan_id": patient,
                        "y": int(candidate.y),
                        "sample_weight": 0.1,
                        **{column: values[column] for column in columns[kind]},
                    })
        for kind in ("PAIR", "JUNCTION"):
            part = pd.DataFrame(rows[kind])
            expected = 5 * int(patient_candidates["candidate_type"].eq(kind).sum())
            if len(part) != expected:
                raise ValueError(f"incomplete {kind} augmentation for patient {patient}")
            augmented[kind].append(part)
            audit.append({
                "scan_id": patient,
                "candidate_type": kind,
                "clean_candidates": expected // 5,
                "perturbed_candidates": len(part),
                "missing_ct_rows": int(part.isna().any(axis=1).sum()),
            })
        del ct
        print(f"{args.mode}: train CT patient {patient} complete", flush=True)

    final_models: dict[str, object] = {}
    revised_scores = clean_scores.copy().set_index(["scene_id", "candidate_id"])
    for kind in ("PAIR", "JUNCTION"):
        clean = clean_features[
            (clean_features["split"] == "train") & (clean_features["candidate_type"] == kind)
        ][["scene_id", "candidate_id", "scan_id", "y", *columns[kind]]].copy()
        clean["sample_weight"] = 0.5
        fit_data = pd.concat([clean, *augmented[kind]], ignore_index=True)
        if len(fit_data) != 6 * len(clean):
            raise ValueError(f"incorrect {kind} augmented row count")
        weight_sum = fit_data.groupby(["scene_id", "candidate_id"])["sample_weight"].sum()
        if not np.allclose(weight_sum, 1.0, atol=1e-12, rtol=0):
            raise ValueError(f"{kind} candidate weights do not sum to one")
        for fold in range(5):
            held_patients = {patient for patient, assigned in patient_folds.items() if assigned == fold}
            fit = fit_data[~fit_data["scan_id"].isin(held_patients)]
            held = clean[clean["scan_id"].isin(held_patients)]
            if set(fit["scan_id"].astype(int)) & held_patients or fit["y"].nunique() != 2:
                raise ValueError(f"{kind} OOF leakage or missing class")
            model = training.local_model("hgb")
            model.fit(fit[columns[kind]], fit["y"], hgb__sample_weight=fit["sample_weight"])
            keys = pd.MultiIndex.from_frame(held[["scene_id", "candidate_id"]])
            revised_scores.loc[keys, "geometry_plus_ct_score"] = model.predict_proba(
                held[columns[kind]]
            )[:, 1]
            print(f"{args.mode}: {kind} OOF fold {fold + 1}/5 complete", flush=True)
        final = training.local_model("hgb")
        final.fit(fit_data[columns[kind]], fit_data["y"], hgb__sample_weight=fit_data["sample_weight"])
        final_models[kind] = final
        print(f"{args.mode}: {kind} final model complete", flush=True)

    # Validation gets predictions from all-train models; neither split is fit here.
    for kind in ("PAIR", "JUNCTION"):
        val = clean_features[
            (clean_features["split"] == "val") & (clean_features["candidate_type"] == kind)
        ]
        keys = pd.MultiIndex.from_frame(val[["scene_id", "candidate_id"]])
        revised_scores.loc[keys, "geometry_plus_ct_score"] = final_models[kind].predict_proba(
            val[columns[kind]]
        )[:, 1]
    revised_scores = revised_scores.reset_index()
    if len(revised_scores) != len(clean_scores):
        raise ValueError("fusion scores are incomplete")
    if not revised_scores[["scene_id", "candidate_id"]].equals(
        clean_scores[["scene_id", "candidate_id"]]
    ) or not np.array_equal(revised_scores["geometry_score"], clean_scores["geometry_score"]):
        raise ValueError("fusion augmentation changed candidate order or geometry evidence")

    for scope, kinds in SCOPES.items():
        scope_dir = mode_dir / scope
        clean_source = clean_scores.copy()
        for kind in kinds:
            mask = clean_source["candidate_type"] == kind
            clean_source.loc[mask, "geometry_plus_ct_score"] = revised_scores.loc[
                mask, "geometry_plus_ct_score"
            ].to_numpy()
        stress_source = replace_fusion_scores(
            stress_scores, stress_val_features, final_models, kinds
        )
        write_evidence(
            scope_dir / "clean_evidence", args.base_evidence,
            clean_source, base_models, final_models, kinds, clean_hashes, args.mode, False,
        )
        write_evidence(
            scope_dir / "stress30_evidence", args.stress_evidence,
            stress_source, stress_models, final_models,
            kinds, stress_hashes, args.mode, True,
        )
        print(f"{args.mode}: {scope} evidence packs complete", flush=True)
    mode_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit).to_csv(mode_dir / "train_extraction_audit.csv", index=False)


if __name__ == "__main__":
    main()
