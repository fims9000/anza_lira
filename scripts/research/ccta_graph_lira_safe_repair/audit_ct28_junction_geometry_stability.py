#!/usr/bin/env python3
"""Localize selected K10/15 JUNCTION geometry sensitivity without retraining."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import build_ct28_scene_plan as scenes
from ct28_perturbations import perturb_scene_endpoints
import train_ct28_relation_head as relation


ROOT = relation.REPO_ROOT
SCENES = ROOT / "artifacts/varvara/ct28_graph_lira"
SELECTED = ROOT / "varvara/anza_lira_local_results/ct28_copy_count_ablations/joint_check/fusion_10_relation_15"
TRACE = ROOT / "varvara/anza_lira_local_results/ct28_copy_count_ablations/selected_type_diagnostics/local_scores_val.csv.gz"
OUTPUT = ROOT / "varvara/anza_lira_local_results/ct28_pretest_audit/junction_geometry_stability.json"
XYZ = ["x_lps", "y_lps", "z_lps"]
TANGENT = ["tangent_x", "tangent_y", "tangent_z"]


def candidate_features(candidates: pd.DataFrame, endpoints: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    by_scene = {key: group.set_index("endpoint_id") for key, group in endpoints.groupby("scene_id", sort=False)}
    for candidate in candidates.itertuples(index=False):
        lookup = by_scene[candidate.scene_id]
        items = []
        for endpoint_id in candidate.endpoint_ids.split("|"):
            endpoint = lookup.loc[endpoint_id]
            items.append({
                "position": endpoint[XYZ].to_numpy(dtype=float),
                "tangent": endpoint[TANGENT].to_numpy(dtype=float),
            })
        values = scenes.oriented_junction_geometry(items)
        rows.append({"scene_id": candidate.scene_id, "candidate_id": candidate.candidate_id,
                     **dict(zip(relation.JUNCTION_GEOMETRY_FEATURES, values))})
    return pd.DataFrame(rows)


def ranking(frame: pd.DataFrame, score: str, competitive: set[str]) -> dict:
    hits = total = 0
    margins = []
    by_patient: dict[int, list[int]] = {}
    for (_, scene_id), group in frame.groupby(["repetition", "scene_id"], sort=False):
        if scene_id not in competitive:
            continue
        total += 1
        hit = int(group.loc[group[score].idxmax(), "y"] == 1)
        hits += hit
        patient = int(group.scan_id.iloc[0])
        by_patient.setdefault(patient, [0, 0])
        by_patient[patient][0] += hit
        by_patient[patient][1] += 1
        margins.append(float(group.loc[group.y.eq(1), score].max()
                             - group.loc[group.y.eq(0), score].max()))
    return {"correct_top1": hits, "scene_repeats": total,
            "margin_quantiles": [float(x) for x in np.quantile(margins, [0, .25, .5, .75, 1])],
            "by_patient": {str(k): v for k, v in sorted(by_patient.items())}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    model_pack = joblib.load(SELECTED / "relation_models.joblib")
    model = model_pack["local_models"]["JUNCTION"]["geometry"]
    columns = list(model.feature_names_in_)
    if columns != list(relation.JUNCTION_GEOMETRY_FEATURES):
        raise ValueError("selected JUNCTION geometry feature order changed")
    plans = pd.read_csv(SCENES / "scene_plan_train_val.csv")
    val_ids = set(plans.loc[plans.split.eq("val"), "scene_id"])
    candidates = pd.read_csv(SCENES / "scene_candidates_train_val.csv")
    candidates = candidates[candidates.scene_id.isin(val_ids) & candidates.candidate_type.eq("JUNCTION")].copy()
    endpoints = pd.read_csv(SCENES / "scene_endpoints_train_val.csv")
    endpoints = endpoints[endpoints.scene_id.isin(val_ids)].copy()
    candidate_meta = candidates[["scene_id", "candidate_id", "y"]].merge(
        plans[["scene_id", "scan_id"]], on="scene_id", validate="many_to_one")
    competitive = {scene_id for scene_id, group in candidates.groupby("scene_id")
                   if group.y.eq(1).any() and group.y.eq(0).any()}
    clean = candidate_features(candidates, endpoints)
    clean_score = pd.read_csv(SELECTED / "relation_local_scores_train_val.csv")
    clean_score = clean_score[clean_score.split.eq("val") & clean_score.candidate_type.eq("JUNCTION")]
    clean = clean.merge(clean_score[["scene_id", "candidate_id", "geometry_score"]],
                        on=["scene_id", "candidate_id"], validate="one_to_one")
    clean_error = float(np.max(np.abs(model.predict_proba(clean[columns])[:, 1]
                                      - clean.geometry_score.to_numpy())))
    if clean_error > 1e-10:
        raise ValueError(f"clean score reproduction failed: {clean_error}")
    reference = pd.read_csv(TRACE)
    reference = reference[reference.candidate_type.eq("JUNCTION")].set_index(
        ["mode", "repetition", "scene_id", "candidate_id"])
    outputs = {"status": "validation_diagnostic_only", "held_out_test_accessed": False,
               "selected_model": "K10/15 JUNCTION geometry HGB", "clean_score_max_error": clean_error,
               "feature_names": columns, "competitive_scenes": len(competitive), "modes": {}}
    grouped_endpoints = list(endpoints.groupby("scene_id", sort=False))
    for mode in ("position_only", "joint"):
        repetitions = []
        for repetition in range(15):
            noisy_parts = []
            for _, scene_endpoints in grouped_endpoints:
                noisy = perturb_scene_endpoints(
                    scene_endpoints, seed=20260920, repetition=repetition,
                    position_noise_mm=0.8, tangent_noise_deg=20.0 if mode == "joint" else 0.0)
                noisy_parts.append(noisy)
            noisy_features = candidate_features(candidates, pd.concat(noisy_parts, ignore_index=True))
            scored = noisy_features.merge(candidate_meta, on=["scene_id", "candidate_id"], validate="one_to_one")
            scored["repetition"] = repetition
            scored["score"] = model.predict_proba(scored[columns])[:, 1]
            for column in columns:
                anchored = scored[columns].copy()
                anchored[column] = clean.set_index(["scene_id", "candidate_id"])[column].reindex(
                    pd.MultiIndex.from_frame(scored[["scene_id", "candidate_id"]])).to_numpy()
                scored["anchor_" + column] = model.predict_proba(anchored)[:, 1]
            repetitions.append(scored)
        all_rows = pd.concat(repetitions, ignore_index=True)
        trace_index = pd.MultiIndex.from_arrays(
            [[mode] * len(all_rows), all_rows.repetition, all_rows.scene_id, all_rows.candidate_id],
            names=["mode", "repetition", "scene_id", "candidate_id"],
        )
        expected = reference.geometry_score.reindex(trace_index).to_numpy()
        covered = np.isfinite(expected)
        if int(covered.sum()) != len(reference.loc[mode]):
            raise ValueError(f"{mode} saved trace coverage is incomplete")
        replay_error = float(np.max(np.abs(expected[covered] - all_rows.score.to_numpy()[covered])))
        if replay_error > 1e-8:
            raise ValueError(f"{mode} saved trace reproduction failed: {replay_error}")
        clean_key = clean.set_index(["scene_id", "candidate_id"])
        feature_delta = {}
        for column in columns:
            clean_values = clean_key[column].reindex(pd.MultiIndex.from_frame(
                all_rows[["scene_id", "candidate_id"]])).to_numpy()
            feature_delta[column] = {
                "median_absolute_change": float(np.median(np.abs(all_rows[column] - clean_values))),
                "clean_anchor_ranking": ranking(all_rows, "anchor_" + column, competitive),
            }
        outputs["modes"][mode] = {
            "saved_trace_max_error": replay_error,
            "ranking": ranking(all_rows, "score", competitive),
            "single_feature_clean_anchor": feature_delta,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(outputs, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({mode: {"top1": item["ranking"]["correct_top1"],
                             "total": item["ranking"]["scene_repeats"],
                             "best_anchor": max(item["single_feature_clean_anchor"],
                                                key=lambda name: item["single_feature_clean_anchor"][name]["clean_anchor_ranking"]["correct_top1"])}
                      for mode, item in outputs["modes"].items()}, indent=2))


if __name__ == "__main__":
    main()
