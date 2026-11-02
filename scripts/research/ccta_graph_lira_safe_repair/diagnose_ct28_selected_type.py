#!/usr/bin/env python3
"""Validation-only selected K10/K15 type and confidence diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import run_ct28_graph_lira as graph
from ct28_perturbations import perturb_scene_endpoints
from extract_junction_ct_features import estimate_junction_center


ROOT = graph.REPO_ROOT
SCENES = graph.DEFAULT_SCENE_DIR
EVIDENCE = ROOT / "varvara/anza_lira_local_results/ct28_copy_count_ablations/joint_check/fusion_10_relation_15"
REFERENCE = ROOT / "varvara/anza_lira_local_results/ct28_copy_count_ablations/selected_diagnostics"
OUTPUT = ROOT / "varvara/anza_lira_local_results/ct28_copy_count_ablations/selected_type_diagnostics"
TARGET = {"junction_only", "mixed"}
XYZ = ["x_lps", "y_lps", "z_lps"]
TANGENT = ["tangent_x", "tangent_y", "tangent_z"]
STREAMS = {
    "freeze_pair_geometry": ("PAIR", "geometry_score"),
    "freeze_junction_geometry": ("JUNCTION", "geometry_score"),
    "freeze_pair_ct": ("PAIR", "ct_score"),
    "freeze_junction_ct": ("JUNCTION", "ct_score"),
}
IDENTITY = {
    "geometry_only": "geometry_score",
    "ct_conditioned": {"PAIR": "geometry_plus_ct_score", "JUNCTION": "geometry_score"},
}


def perturb(endpoints: pd.DataFrame, rep: int, mode: str) -> pd.DataFrame:
    result = perturb_scene_endpoints(
        endpoints, seed=20260920, repetition=rep,
        position_noise_mm=0.8, tangent_noise_deg=20.0,
    )
    if mode == "position_only":
        result.loc[:, TANGENT] = endpoints[TANGENT].to_numpy()
    elif mode == "tangent_only":
        result.loc[:, XYZ] = endpoints[XYZ].to_numpy()
    elif mode != "joint":
        raise ValueError(mode)
    return result


def frozen_relation_scores(dynamic: pd.DataFrame, clean: pd.DataFrame, name: str) -> pd.DataFrame:
    kind, column = STREAMS[name]
    result = dynamic.copy()
    mask = result.candidate_type.eq(kind)
    keys = pd.MultiIndex.from_frame(result.loc[mask, ["scene_id", "candidate_id"]])
    values = clean.set_index(["scene_id", "candidate_id"])[column].reindex(keys).to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("missing clean scores")
    result.loc[mask, column] = values
    return result


def record(plan: pd.Series, mode: str, rep: int, intervention: str, decision: dict, base: dict) -> dict:
    if decision["raw_relation"] != base["raw_relation"]:
        layer = "raw_type_flip"
    elif decision["relation"] != base["relation"]:
        layer = ("confidence_fell_below_tau" if decision["relation"] == "NONE"
                 else "confidence_rose_above_tau")
    elif decision["relation_key"] != base["relation_key"]:
        layer = "candidate_identity_flip"
    else:
        layer = "stable"
    row = {
        "scene_id": plan.scene_id, "scan_id": int(plan.scan_id),
        "scene_family": plan.scene_family, "truth": plan.relation_class,
        "mode": mode, "repetition": rep, "intervention": intervention,
        "raw_relation": decision["raw_relation"], "tau_relation": decision["relation"],
        "selected_relation": decision["selected_relation"],
        "confidence": float(decision["confidence"]), "failure_layer": layer,
        "relation_key": decision["relation_key"], "base_relation_key": base["relation_key"],
        "base_raw_relation": base["raw_relation"], "base_tau_relation": base["relation"],
        "pair_candidate_id": decision["pair_candidate_id"] or "",
        "junction_candidate_id": decision["junction_candidate_id"] or "",
    }
    row.update({f"p_{name.lower()}": float(decision["probabilities"][i])
                for i, name in enumerate(graph.RELATION_CLASSES)})
    return row


def junction_trace(candidates: pd.DataFrame, endpoints: pd.DataFrame, scored: pd.DataFrame,
                   mode: str, rep: int) -> list[dict]:
    result = []
    endpoint_groups = dict(tuple(endpoints.groupby("scene_id")))
    lookup = scored.set_index(["scene_id", "candidate_id"])
    for candidate in candidates[candidates.candidate_type.eq("JUNCTION")].itertuples():
        arms = endpoint_groups[candidate.scene_id].set_index("endpoint_id").loc[
            str(candidate.endpoint_ids).split("|")
        ]
        center = estimate_junction_center(arms[XYZ].to_numpy(), arms[TANGENT].to_numpy())
        score = lookup.loc[(candidate.scene_id, candidate.candidate_id)]
        result.append({
            "scene_id": candidate.scene_id, "candidate_id": candidate.candidate_id,
            "y": int(candidate.y), "mode": mode, "repetition": rep,
            "center_x": center[0], "center_y": center[1], "center_z": center[2],
            "geometry_score": score.geometry_score, "ct_score": score.ct_score,
            "geometry_plus_ct_score": score.geometry_plus_ct_score,
        })
    return result


def save_gzip(rows: list[dict], path: Path) -> None:
    pd.DataFrame(rows).to_csv(
        path, index=False, compression={"method": "gzip", "mtime": 0}
    )


def summarize_saved() -> None:
    """Rebuild compact causal and failure summaries without resampling CT."""
    frame = pd.read_csv(OUTPUT / "decision_traces_val.csv.gz").fillna("")
    if len(frame) != 20640 or set(frame.scan_id) != graph.EXPECTED_VALIDATION_PATIENTS:
        raise ValueError("saved validation trace is incomplete or has unexpected patients")
    type_stable = frame.raw_relation.eq(frame.base_raw_relation)
    tau_stable = frame.tau_relation.eq(frame.base_tau_relation)
    key_stable = frame.relation_key.eq(frame.base_relation_key)
    frame["failure_layer"] = np.select(
        [~type_stable, ~tau_stable & frame.tau_relation.eq("NONE"),
         ~tau_stable, ~key_stable],
        ["raw_type_flip", "confidence_fell_below_tau",
         "confidence_rose_above_tau", "candidate_identity_flip"],
        default="stable",
    )
    save_gzip(frame.to_dict("records"), OUTPUT / "decision_traces_val.csv.gz")
    frame.groupby(["mode", "intervention", "scene_family", "failure_layer"]).agg(
        repeats=("scene_id", "size"), scenes=("scene_id", "nunique"),
        mean_confidence=("confidence", "mean"),
    ).reset_index().to_csv(OUTPUT / "failure_layer_summary_val.csv", index=False)
    correct = frame[frame.base_tau_relation.eq(frame.truth) & frame.truth.ne("NONE")]
    summary = correct.groupby(["mode", "intervention", "scene_family"]).agg(
        repeats=("scene_id", "size"), scenes=("scene_id", "nunique"),
        raw_correct=("raw_relation", lambda value: int(value.eq(correct.loc[value.index, "truth"]).sum())),
        tau_correct=("tau_relation", lambda value: int(value.eq(correct.loc[value.index, "truth"]).sum())),
        same_candidate_set=("relation_key", lambda value: int(value.eq(correct.loc[value.index, "base_relation_key"]).sum())),
        confidence_fell_below_tau=("failure_layer", lambda value: int(value.eq("confidence_fell_below_tau").sum())),
    ).reset_index()
    summary.to_csv(OUTPUT / "base_correct_scene_summary_val.csv", index=False)
    patient = correct[correct.intervention.eq("dynamic")].groupby(
        ["scan_id", "scene_family", "mode"]
    ).agg(repeats=("scene_id", "size"), scenes=("scene_id", "nunique"),
          tau_correct=("tau_relation", lambda value: int(value.eq(correct.loc[value.index, "truth"]).sum())),
          same_candidate_set=("relation_key", lambda value: int(value.eq(correct.loc[value.index, "base_relation_key"]).sum())))
    patient.reset_index().to_csv(OUTPUT / "per_patient_val.csv", index=False)
    centers = pd.read_csv(OUTPUT / "junction_centers_val.csv.gz")
    source_candidates = pd.read_csv(SCENES / "scene_candidates_train_val.csv")
    source_endpoints = pd.read_csv(SCENES / "scene_endpoints_train_val.csv")
    source_scores = pd.read_csv(EVIDENCE / "relation_local_scores_train_val.csv")
    target_ids = set(frame.scene_id[frame.scene_family.isin(TARGET)])
    source_candidates = source_candidates[
        source_candidates.scene_id.isin(target_ids) & source_candidates.candidate_type.eq("JUNCTION")
    ]
    source_endpoints = source_endpoints[source_endpoints.scene_id.isin(target_ids)]
    source_scores = source_scores[source_scores.scene_id.isin(target_ids)]
    endpoint_groups = {key: group.set_index("endpoint_id")
                       for key, group in source_endpoints.groupby("scene_id")}
    clean_center = {}
    for candidate in source_candidates[source_candidates.y.eq(1)].itertuples():
        arms = endpoint_groups[candidate.scene_id].loc[str(candidate.endpoint_ids).split("|")]
        clean_center[candidate.scene_id] = estimate_junction_center(
            arms[XYZ].to_numpy(), arms[TANGENT].to_numpy()
        )
    clean_scores = source_scores.set_index(["scene_id", "candidate_id"])
    rank_rows = []
    for (scene_id, mode, repetition), group in centers.groupby(
        ["scene_id", "mode", "repetition"], sort=False
    ):
        true = group[group.y.eq(1)]
        false = group[group.y.eq(0)]
        if len(true) != 1:
            raise ValueError(f"expected one true junction candidate: {scene_id}")
        if false.empty:
            continue
        true_row = true.iloc[0]
        false_row = false.sort_values("geometry_score", ascending=False).iloc[0]
        true_id, false_id = true_row.candidate_id, false_row.candidate_id
        rank_rows.append({
            "scene_id": scene_id, "mode": mode, "repetition": repetition,
            "true_candidate_id": true_id, "best_false_candidate_id": false_id,
            "true_geometry_score": true_row.geometry_score,
            "best_false_geometry_score": false_row.geometry_score,
            "geometry_margin": true_row.geometry_score - false_row.geometry_score,
            "true_ct_score": true_row.ct_score, "best_false_ct_score": false_row.ct_score,
            "true_geometry_score_change": true_row.geometry_score - clean_scores.loc[(scene_id, true_id)].geometry_score,
            "best_false_geometry_score_change": false_row.geometry_score - clean_scores.loc[(scene_id, false_id)].geometry_score,
            "true_center_displacement_mm": float(np.linalg.norm(
                true_row[["center_x", "center_y", "center_z"]].to_numpy(dtype=float)
                - clean_center[scene_id]
            )),
        })
    pd.DataFrame(rank_rows).to_csv(OUTPUT / "junction_true_best_false_val.csv", index=False)
    print(summary[summary.scene_family.isin(TARGET)].to_string(index=False))


def main() -> None:
    source = {
        "plans": SCENES / "scene_plan_train_val.csv",
        "candidates": SCENES / "scene_candidates_train_val.csv",
        "endpoints": SCENES / "scene_endpoints_train_val.csv",
        "features": SCENES / "scene_candidate_features_train_val.csv",
        "models": EVIDENCE / "relation_models.joblib",
        "scores": EVIDENCE / "relation_local_scores_train_val.csv",
        "reference": REFERENCE / "perturbation_decisions_val.csv",
    }
    saved_protocol = json.loads((EVIDENCE / "relation_model_protocol.json").read_text())
    for field, name in (("scene_plan", "plans"), ("scene_candidates", "candidates"),
                        ("scene_candidate_features", "features")):
        if graph.sha256(source[name]) != saved_protocol["source_sha256"][field]:
            raise ValueError(f"source mismatch: {field}")
    plans = pd.read_csv(source["plans"])
    val = plans[plans.split.eq("val")].copy()
    if len(val) != 368 or set(val.scan_id) != graph.EXPECTED_VALIDATION_PATIENTS:
        raise ValueError("validation cohort changed")
    if set(plans.scan_id) & {954, 958, 972, 973, 980, 984}:
        raise ValueError("held-out test appeared in scene pack")
    candidates = pd.read_csv(source["candidates"])
    candidates = candidates[candidates.scene_id.isin(val.scene_id)]
    endpoints = pd.read_csv(source["endpoints"])
    endpoints = endpoints[endpoints.scene_id.isin(val.scene_id)]
    clean = pd.read_csv(source["scores"])
    clean = clean[clean.scene_id.isin(val.scene_id)]
    models = joblib.load(source["models"])
    if models.get("local_model_family") != "hgb":
        raise ValueError("selected local models are not HGB")
    base = graph.select_scene_decisions(
        val, candidates, clean,
        graph.predict_relations(val, candidates, clean, models, 0.85),
        IDENTITY,
    )["ct_conditioned"]
    reference = pd.read_csv(source["reference"]).fillna("")
    reference = reference[reference.model.eq("ct_conditioned")].set_index(["scene_id", "repetition"])
    if len(reference) != 368 * 15:
        raise ValueError("reference decisions incomplete")
    expected_geometry = pd.read_csv(graph.ct_features.EXPECTED_GEOMETRY).set_index("scan_id")
    expected_alignment = pd.read_csv(graph.ct_features.EXPECTED_ALIGNMENT).set_index("scan_id")
    arm_names = graph.ct_features.arm_feature_names(graph.ct_features.RADII_MM)
    plan_index = val.set_index("scene_id", drop=False)
    decisions, features_out, local_out, centers_out = [], [], [], []
    max_error = 0.0
    for patient in sorted(graph.EXPECTED_VALIDATION_PATIENTS):
        patient_plans = val[val.scan_id.eq(patient)]
        ids = set(patient_plans.scene_id)
        patient_candidates = candidates[candidates.scene_id.isin(ids)]
        patient_endpoints = endpoints[endpoints.scene_id.isin(ids)]
        patient_clean = clean[clean.scene_id.isin(ids)]
        target_plans = patient_plans[patient_plans.scene_family.isin(TARGET)]
        target_ids = set(target_plans.scene_id)
        target_candidates = patient_candidates[patient_candidates.scene_id.isin(target_ids)]
        target_clean = patient_clean[patient_clean.scene_id.isin(target_ids)]
        ep_groups = dict(tuple(patient_endpoints.groupby("scene_id", sort=False)))
        candidate_groups = dict(tuple(patient_candidates.groupby("scene_id", sort=False)))
        ct, inverse_affine = graph.load_verified_ct(
            patient, graph.DEFAULT_ARCHIVE_DIR, expected_geometry, expected_alignment
        )
        for rep in range(15):
            for mode in ("joint", "position_only", "tangent_only"):
                included = ids if mode == "joint" else target_ids
                rows, changed_groups = [], {}
                for scene_id in sorted(included):
                    changed = perturb(ep_groups[scene_id], rep, mode)
                    changed_groups[scene_id] = changed
                    lookup = changed.set_index("endpoint_id")
                    for _, candidate in candidate_groups[scene_id].iterrows():
                        rows.append(graph.perturbed_candidate_features(
                            candidate, lookup, ct, inverse_affine, arm_names
                        ))
                feat = pd.DataFrame(rows)
                scores = graph.score_feature_frame(feat, models["local_models"])
                pp = patient_plans if mode == "joint" else target_plans
                cc = patient_candidates if mode == "joint" else target_candidates
                relation_rows, columns = graph.relation_feature_rows(
                    pp, cc, scores, ("geometry_score", "ct_score")
                )
                expected = models["ct_conditioned_relation_features"]
                if [column for column in columns if column in set(expected)] != expected or len(expected) != 84:
                    raise ValueError("84 relation columns changed")
                features_out.extend({"mode": mode, "repetition": rep, **row}
                                    for row in relation_rows)
                local_out.extend({"mode": mode, "repetition": rep, **row}
                                 for row in scores.to_dict(orient="records"))
                prediction = graph.predict_relations(pp, cc, scores, models, 0.85)
                selected = graph.select_scene_decisions(
                    pp, cc, scores, prediction, IDENTITY
                )["ct_conditioned"]
                for scene_id in sorted(included):
                    decision = selected[scene_id]
                    decisions.append(record(plan_index.loc[scene_id], mode, rep,
                                            "dynamic", decision, base[scene_id]))
                    if mode == "joint":
                        old = reference.loc[(scene_id, rep)]
                        if (decision["relation_key"] != old.relation_key
                                or decision["raw_relation"] != old.raw_relation
                                or decision["relation"] != old.tau_relation):
                            raise ValueError(f"repeat mismatch: {scene_id} {rep}")
                        max_error = max(max_error, abs(decision["confidence"] - float(old.confidence)))
                        if max_error > 1e-10:
                            raise ValueError("confidence mismatch")
                for scene_id in sorted(included & target_ids):
                    centers_out.extend(junction_trace(
                        candidate_groups[scene_id], changed_groups[scene_id],
                        scores[scores.scene_id.eq(scene_id)], mode, rep,
                    ))
                target_scores = scores[scores.scene_id.isin(target_ids)]
                for intervention in STREAMS:
                    hybrid = frozen_relation_scores(target_scores, target_clean, intervention)
                    probability = graph.predict_relations(
                        target_plans, target_candidates, hybrid, models, 0.85
                    )
                    # The candidate selector keeps perturbed identity scores.
                    changed = graph.select_scene_decisions(
                        target_plans, target_candidates, target_scores,
                        probability, IDENTITY,
                    )["ct_conditioned"]
                    for scene_id in sorted(target_ids):
                        decisions.append(record(plan_index.loc[scene_id], mode, rep,
                                                intervention, changed[scene_id], base[scene_id]))
            if rep in (4, 9, 14):
                print(f"patient {patient}: repeats {rep + 1}/15", flush=True)
        del ct
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save_gzip(decisions, OUTPUT / "decision_traces_val.csv.gz")
    save_gzip(features_out, OUTPUT / "relation_features_val.csv.gz")
    save_gzip(local_out, OUTPUT / "local_scores_val.csv.gz")
    save_gzip(centers_out, OUTPUT / "junction_centers_val.csv.gz")
    summary = pd.DataFrame(decisions).groupby(
        ["mode", "intervention", "scene_family", "failure_layer"]
    ).agg(repeats=("scene_id", "size"), scenes=("scene_id", "nunique"),
          mean_confidence=("confidence", "mean")).reset_index()
    summary.to_csv(OUTPUT / "failure_layer_summary_val.csv", index=False)
    (OUTPUT / "protocol.json").write_text(json.dumps({
        "status": "validation_diagnostic_only", "held_out_test_accessed": False,
        "training_performed": False, "tau": 0.85, "consistency": 0.60,
        "seed": 20260920, "evaluation_repetitions": 15,
        "position_noise_mm": 0.8, "tangent_noise_deg": 20.0,
        "identity": IDENTITY["ct_conditioned"],
        "freeze_scope": "relation evidence only; identity stays dynamic",
        "joint_reference_max_confidence_error": max_error,
        "source_sha256": {name: graph.sha256(path) for name, path in source.items()},
    }, indent=2) + "\n")
    print(f"saved {len(decisions)} decisions; joint reference error {max_error:.3g}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        summarize_saved()
    else:
        main()
        summarize_saved()
