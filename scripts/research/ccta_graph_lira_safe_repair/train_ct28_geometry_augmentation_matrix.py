#!/usr/bin/env python3
"""Train CT28 geometry-local noise ablations for clean validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import build_ct28_scene_plan as scene_builder
import run_ct28_graph_lira as graph
import train_ct28_perturbation_augmented_relation_head as augmented_relation
import train_ct28_relation_head as training
from ct28_perturbations import perturb_scene_endpoints


REPO = training.REPO_ROOT
DEFAULT_BASE = REPO / "artifacts/varvara/ct28_graph_lira/ablations/local_hgb"
DEFAULT_OUTPUT = REPO / "varvara/anza_lira_local_results/ct28_geometry_augmentation_matrix"
MODES = {
    "position_only": (0.8, 0.0),
    "tangent_only": (0.0, 20.0),
    "joint": (0.8, 20.0),
}
SCOPES = {
    "pair": ("PAIR",),
    "junction": ("JUNCTION",),
    "both": ("PAIR", "JUNCTION"),
}
TRAIN_REPEATS = 5
RELATION_REPEATS = 15
SEED = 20260920
TAU = 0.85
TEST_PATIENTS = {954, 958, 972, 973, 980, 984}


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=graph.DEFAULT_SCENE_DIR)
    parser.add_argument("--base-evidence", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--archive-dir", type=Path, default=graph.DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def source_hashes(path: Path) -> dict[str, str]:
    return {
        key: training.sha256(path / name)
        for key, name in (
            ("scene_plan", "scene_plan_train_val.csv"),
            ("scene_candidates", "scene_candidates_train_val.csv"),
            ("scene_candidate_features", "scene_candidate_features_train_val.csv"),
        )
    }


def geometry_columns(kind: str) -> list[str]:
    return list(training.PAIR_GEOMETRY_FEATURES if kind == "PAIR" else training.JUNCTION_GEOMETRY_FEATURES)


def geometry_rows(candidates: pd.DataFrame, endpoints: pd.DataFrame, kind: str, mode: str, repetition: int) -> pd.DataFrame:
    rows = []
    by_scene = dict(tuple(candidates.groupby("scene_id", sort=False)))
    position_noise, tangent_noise = MODES[mode]
    columns = geometry_columns(kind)
    for scene_id, scene_endpoints in endpoints.groupby("scene_id", sort=False):
        if scene_id not in by_scene:
            continue
        perturbed = perturb_scene_endpoints(
            scene_endpoints, seed=SEED, repetition=repetition,
            position_noise_mm=position_noise, tangent_noise_deg=tangent_noise,
        ).set_index("endpoint_id")
        for candidate in by_scene[scene_id].itertuples(index=False):
            arms = []
            for endpoint_id in candidate.endpoint_ids.split("|"):
                point = perturbed.loc[endpoint_id]
                arms.append({
                    "position": point[["x_lps", "y_lps", "z_lps"]].to_numpy(dtype=float),
                    "tangent": point[["tangent_x", "tangent_y", "tangent_z"]].to_numpy(dtype=float),
                })
            values = (
                scene_builder.pair_geometry(*arms) if kind == "PAIR"
                else scene_builder.oriented_junction_geometry(arms)
            )
            rows.append({"scene_id": scene_id, "candidate_id": candidate.candidate_id,
                         **dict(zip(columns, values, strict=True))})
    return pd.DataFrame(rows)


def checked_fold_maps(base_scores: pd.DataFrame) -> dict[str, dict[int, int]]:
    mapping = {}
    for kind in ("PAIR", "JUNCTION"):
        rows = base_scores.query('split == "train" and candidate_type == @kind')
        if not rows.groupby("scan_id")["local_oof_fold"].nunique().eq(1).all():
            raise ValueError(f"{kind} patient spans local OOF folds")
        mapping[kind] = rows.groupby("scan_id")["local_oof_fold"].first().astype(int).to_dict()
        if set(mapping[kind]) != training.EXPECTED_PATIENT_IDS["train"] or set(mapping[kind].values()) != set(range(5)):
            raise ValueError(f"{kind} frozen patient folds changed")
    return mapping


def local_training(
    features: pd.DataFrame, candidates: pd.DataFrame, endpoints: pd.DataFrame,
    base_scores: pd.DataFrame, base_pack: dict, folds: dict[str, dict[int, int]],
) -> tuple[dict, dict, dict]:
    fitted = {}
    base_fold = {int(patient): {kind: {} for kind in ("PAIR", "JUNCTION")}
                 for patient in training.EXPECTED_PATIENT_IDS["train"]}
    scores = {}
    for kind in ("PAIR", "JUNCTION"):
        all_kind = features[features["candidate_type"].eq(kind)]
        clean = all_kind[all_kind["split"].eq("train")].copy()
        validation = all_kind[all_kind["split"].eq("val")].copy()
        kind_candidates = candidates[candidates["candidate_type"].eq(kind)]
        train_scenes = set(clean["scene_id"])
        kind_endpoints = endpoints[endpoints["scene_id"].isin(train_scenes)]
        noisy = {}
        for mode in MODES:
            copies = []
            for repetition in range(TRAIN_REPEATS):
                part = geometry_rows(kind_candidates[kind_candidates["scene_id"].isin(train_scenes)],
                                     kind_endpoints, kind, mode, repetition)
                part = part.merge(clean[["scene_id", "candidate_id", "scan_id", "y"]],
                                  on=["scene_id", "candidate_id"], validate="one_to_one")
                if len(part) != len(clean):
                    raise ValueError(f"incomplete {kind}/{mode} geometry copy")
                part["weight"] = 0.1
                copies.append(part)
            noisy[mode] = copies
            print(f"{kind} {mode}: {TRAIN_REPEATS} local copies", flush=True)

        for stream in ("geometry", "ct"):
            columns = list(base_pack["local_models"][kind][stream].feature_names_in_)
            for fold in range(5):
                held_patients = {patient for patient, assigned in folds[kind].items() if assigned == fold}
                fit = clean[~clean["scan_id"].isin(held_patients)]
                held = clean[clean["scan_id"].isin(held_patients)]
                model = training.local_model("hgb").fit(fit[columns], fit["y"])
                expected = held[["scene_id", "candidate_id"]].merge(
                    base_scores[["scene_id", "candidate_id", f"{stream}_score"]],
                    on=["scene_id", "candidate_id"], validate="one_to_one",
                )[f"{stream}_score"].to_numpy()
                error = np.max(np.abs(model.predict_proba(held[columns])[:, 1] - expected))
                if error > 1e-10:
                    raise ValueError(f"base {kind}/{stream} OOF reproduction error {error}")
                for patient in held_patients:
                    base_fold[patient][kind][stream] = model

        columns = geometry_columns(kind)
        weighted_clean = clean.copy()
        weighted_clean["weight"] = 0.5
        for mode, copies in noisy.items():
            fit_data = pd.concat([weighted_clean, *copies], ignore_index=True)
            weights = fit_data.groupby(["scene_id", "candidate_id"])["weight"].sum()
            if len(fit_data) != 6 * len(clean) or not np.allclose(weights, 1.0):
                raise ValueError(f"{kind}/{mode} training weights invalid")
            fold_models = {}
            train_scores = pd.Series(index=pd.MultiIndex.from_frame(clean[["scene_id", "candidate_id"]]), dtype=float)
            for fold in range(5):
                held_patients = {patient for patient, assigned in folds[kind].items() if assigned == fold}
                fit = fit_data[~fit_data["scan_id"].isin(held_patients)]
                held = clean[clean["scan_id"].isin(held_patients)]
                model = training.local_model("hgb").fit(
                    fit[columns], fit["y"], hgb__sample_weight=fit["weight"]
                )
                keys = pd.MultiIndex.from_frame(held[["scene_id", "candidate_id"]])
                train_scores.loc[keys] = model.predict_proba(held[columns])[:, 1]
                for patient in held_patients:
                    fold_models[patient] = model
            final = training.local_model("hgb").fit(
                fit_data[columns], fit_data["y"], hgb__sample_weight=fit_data["weight"]
            )
            val_keys = pd.MultiIndex.from_frame(validation[["scene_id", "candidate_id"]])
            val_scores = pd.Series(final.predict_proba(validation[columns])[:, 1], index=val_keys)
            fitted[kind, mode] = {"final": final, "fold": fold_models}
            scores[kind, mode] = pd.concat([train_scores, val_scores])
            print(f"{kind} {mode}: 5 OOF folds and final model complete", flush=True)
    return fitted, base_fold, scores


def score_geometry_ct(frame: pd.DataFrame, patient_models: dict) -> pd.DataFrame:
    scores = frame[["scene_id", "candidate_id", "candidate_type"]].copy()
    for stream in ("geometry", "ct"):
        scores[f"{stream}_score"] = np.nan
        for kind in ("PAIR", "JUNCTION"):
            mask = frame["candidate_type"].eq(kind)
            model = patient_models[kind][stream]
            columns = list(model.feature_names_in_)
            scores.loc[mask, f"{stream}_score"] = model.predict_proba(frame.loc[mask, columns])[:, 1]
    if not np.isfinite(scores[["geometry_score", "ct_score"]]).all().all():
        raise ValueError("incomplete perturbed geometry/CT scores")
    return scores


def main() -> None:
    config = args()
    config.out_dir.mkdir(parents=True, exist_ok=True)
    clean_hashes = source_hashes(config.scene_dir)
    base_protocol = json.loads((config.base_evidence / "relation_model_protocol.json").read_text())
    if base_protocol["source_sha256"] != clean_hashes:
        raise ValueError("clean frozen source differs from base evidence")
    plans = pd.read_csv(config.scene_dir / "scene_plan_train_val.csv")
    candidates = pd.read_csv(config.scene_dir / "scene_candidates_train_val.csv")
    endpoints = pd.read_csv(config.scene_dir / "scene_endpoints_train_val.csv")
    features = pd.read_csv(config.scene_dir / "scene_candidate_features_train_val.csv", low_memory=False)
    base_scores = pd.read_csv(config.base_evidence / "relation_local_scores_train_val.csv")
    base_pack = joblib.load(config.base_evidence / "relation_models.joblib")
    training.validate_scene_inputs(plans, candidates)
    if set(plans["scan_id"].astype(int)) & TEST_PATIENTS:
        raise ValueError("held-out test patient in clean scene pack")
    if not features[["scene_id", "candidate_id"]].equals(base_scores[["scene_id", "candidate_id"]]):
        raise ValueError("clean features and OOF scores misaligned")
    if base_pack["local_model_family"] != "hgb":
        raise ValueError("HGB base pack required")
    folds = checked_fold_maps(base_scores)
    fitted, base_fold, new_scores = local_training(features, candidates, endpoints, base_scores, base_pack, folds)

    variants = {"no_augmentation": ()}
    variants.update({f"{mode}/{scope}": tuple((kind, mode) for kind in kinds)
                     for mode in MODES for scope, kinds in SCOPES.items()})
    scores_by_variant = {}
    pack_by_variant = {}
    clean_relation = {}
    for name, selected in variants.items():
        table = base_scores.copy().set_index(["scene_id", "candidate_id"])
        pack = dict(base_pack)
        pack["local_models"] = {kind: dict(base_pack["local_models"][kind]) for kind in ("PAIR", "JUNCTION")}
        pack["relation_heads"] = dict(base_pack["relation_heads"])
        for kind, mode in selected:
            mask = table["candidate_type"].eq(kind)
            keys = table.index[mask]
            table.loc[keys, "geometry_score"] = new_scores[kind, mode].loc[keys].to_numpy()
            pack["local_models"][kind]["geometry"] = fitted[kind, mode]["final"]
        table = table.reset_index()
        if not table[["scene_id", "candidate_id"]].equals(base_scores[["scene_id", "candidate_id"]]):
            raise ValueError(f"{name} candidate order changed")
        if not table[["ct_score", "geometry_plus_ct_score"]].equals(base_scores[["ct_score", "geometry_plus_ct_score"]]):
            raise ValueError(f"{name} CT/fusion evidence changed")
        path = config.out_dir / name
        path.mkdir(parents=True, exist_ok=True)
        table.to_csv(path / "relation_local_scores_train_val.csv", index=False)
        table = pd.read_csv(path / "relation_local_scores_train_val.csv")
        scores_by_variant[name], pack_by_variant[name] = table, pack
        rows, columns = graph.relation_feature_rows(plans, candidates, table, ("geometry_score", "ct_score"))
        if len(columns) != 84:
            raise ValueError("CT-conditioned relation dimension changed")
        clean_relation[name] = (rows, columns)
        (path / "augmented_relation_rows_train.csv").unlink(missing_ok=True)

    train_plans = plans[plans["split"].eq("train")]
    train_scene_ids = set(train_plans["scene_id"])
    train_candidates = candidates[candidates["scene_id"].isin(train_scene_ids)]
    train_endpoints = endpoints[endpoints["scene_id"].isin(train_scene_ids)]
    expected_geometry = pd.read_csv(graph.ct_features.EXPECTED_GEOMETRY).set_index("scan_id")
    expected_alignment = pd.read_csv(graph.ct_features.EXPECTED_ALIGNMENT).set_index("scan_id")
    arm_names = graph.ct_features.arm_feature_names(graph.ct_features.RADII_MM)
    for patient in sorted(training.EXPECTED_PATIENT_IDS["train"]):
        patient_plans = train_plans[train_plans["scan_id"].eq(patient)]
        scene_ids = set(patient_plans["scene_id"])
        patient_candidates = train_candidates[train_candidates["scene_id"].isin(scene_ids)]
        patient_endpoints = train_endpoints[train_endpoints["scene_id"].isin(scene_ids)]
        by_scene = dict(tuple(patient_candidates.groupby("scene_id", sort=False)))
        ct, inverse_affine = graph.load_verified_ct(patient, config.archive_dir, expected_geometry, expected_alignment)
        patient_rows = {name: [] for name in variants}
        for repetition in range(RELATION_REPEATS):
            feature_rows = []
            for scene_id, scene_endpoints in patient_endpoints.groupby("scene_id", sort=False):
                perturbed = perturb_scene_endpoints(
                    scene_endpoints, seed=SEED, repetition=repetition,
                    position_noise_mm=0.8, tangent_noise_deg=20.0,
                ).set_index("endpoint_id")
                for candidate in by_scene[scene_id].itertuples(index=False):
                    feature_rows.append(graph.perturbed_candidate_features(
                        pd.Series(candidate._asdict()), perturbed, ct, inverse_affine, arm_names
                    ))
            frame = pd.DataFrame(feature_rows)
            base_scores_noise = score_geometry_ct(frame, base_fold[patient])
            for name, selected in variants.items():
                local_scores = base_scores_noise.copy()
                for kind, mode in selected:
                    mask = frame["candidate_type"].eq(kind)
                    model = fitted[kind, mode]["fold"][patient]
                    columns = list(model.feature_names_in_)
                    local_scores.loc[mask, "geometry_score"] = model.predict_proba(frame.loc[mask, columns])[:, 1]
                rows, columns = graph.relation_feature_rows(
                    patient_plans, patient_candidates, local_scores, ("geometry_score", "ct_score")
                )
                if columns != clean_relation[name][1]:
                    raise ValueError(f"{name} relation feature order changed")
                for row in rows:
                    row["augmentation_index"] = repetition + 1
                    row["sample_weight"] = 0.5 / RELATION_REPEATS
                patient_rows[name].extend(rows)
        for name, rows in patient_rows.items():
            pd.DataFrame(rows).to_csv(
                config.out_dir / name / "augmented_relation_rows_train.csv",
                mode="a", header=patient == min(training.EXPECTED_PATIENT_IDS["train"]), index=False,
            )
        print(f"relation features: train patient {patient} complete", flush=True)
        del ct

    for name, selected in variants.items():
        out = config.out_dir / name
        clean_rows, columns = clean_relation[name]
        train_clean = [dict(row, augmentation_index=0, sample_weight=0.5)
                       for row in clean_rows if row["split"] == "train"]
        val_clean = [row for row in clean_rows if row["split"] == "val"]
        noisy = pd.read_csv(out / "augmented_relation_rows_train.csv").to_dict("records")
        train_rows = train_clean + noisy
        augmented_relation.validate_augmentation_weights(train_rows, RELATION_REPEATS)
        head, metrics, ct_predictions = augmented_relation.fit_augmented_head(train_rows, val_clean, columns, TAU)
        geom_rows, geom_columns = graph.relation_feature_rows(
            plans, candidates, scores_by_variant[name], ("geometry_score",)
        )
        _, geometry_predictions, geometry_head = training.fit_relation_head(
            geom_rows, geom_columns, TAU, "geometry_only"
        )
        pack = pack_by_variant[name]
        pack["relation_heads"]["ct_conditioned"] = head
        pack["relation_heads"]["geometry_only"] = geometry_head
        pack["ct_conditioned_relation_features"] = columns
        pack["geometry_relation_features"] = geom_columns
        joblib.dump(pack, out / "relation_models.joblib")
        pd.DataFrame(geometry_predictions + ct_predictions).to_csv(out / "relation_predictions_val.csv", index=False)
        (out / "relation_metrics_val.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (out / "relation_model_protocol.json").write_text(json.dumps({
            "status": "validation_ablation_not_frozen", "held_out_test_accessed": False,
            "selected_geometry_local_models": list(selected),
            "local_train_copies": TRAIN_REPEATS, "local_clean_weight": 0.5,
            "relation_train_copies": RELATION_REPEATS, "relation_clean_weight": 0.5,
            "relation_noise": {"position_sigma_mm": 0.8, "tangent_sigma_deg": 20.0},
            "patient_folds": folds, "tau": TAU, "consistency": 0.60,
            "source_sha256": clean_hashes,
        }, indent=2) + "\n")

        print(f"{name}: clean evidence complete", flush=True)


if __name__ == "__main__":
    main()
