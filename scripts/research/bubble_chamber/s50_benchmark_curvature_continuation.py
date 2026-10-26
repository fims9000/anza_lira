#!/usr/bin/env python3
"""S50 locked study: curvature vs length-matched hard false continuation.

Private 2D data are read locally. Only aggregate synthetic-only metrics may be
written to the public report. NEVER mix native H50Z machine proposals into the
synthetic GT evaluation or treat 2D smoothness as a particle identity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from bubble_chamber.large_archive import audited_large_manifest
from bubble_chamber.s35_fragment_link import (
    GEOMETRY_FIELDS, calibrated_negative_threshold, feature_matrix)
from bubble_chamber.s35_simulated_gaps import extract_track_templates, make_scene
from scripts.research.bubble_chamber.s37_benchmark_review_risk import split_source_frames
from bubble_chamber.s37_risk_calibration import scene_stats
from bubble_chamber.s50_curvature_continuation import (
    S50_FIELDS, ALL_FIELDS, matched_candidates, pair_geometry_and_curvature)

SCHEMA = "s50_frozen_length_matched_2d_continuation_public_v1"
PROTOCOL = {
    "train_n": 80, "calibration_n": 100, "test_n_per_seed": 100,
    "minimum_margin": .04, "calibration_pair_fpr_budget": .02,
    "train_seeds": {"005": 20265001, "006": 20265011},
    "cal_seeds": {"005": 20265002, "006": 20265012},
    "test_seeds": {"005": (20265003, 20265004), "006": (20265013, 20265014)},
    "scenarios": ("parallel", "crossing", "scratched", "empty_gap"),
}
ARMS = {
    "geometry": GEOMETRY_FIELDS,
    "geometry+curvature": ALL_FIELDS,
    "geometry+shuffled_curvature_NEGATIVE_CONTROL": ALL_FIELDS,
}


def generate_cases(tiles: Path, photo: str, templates: list[dict], *, n: int, seed: int) -> list[dict]:
    paths = sorted(tiles.glob(f"H50Z_030_01_{photo}_*.png"))
    if len(paths) != (21 if photo == "005" else 18):
        raise ValueError("S50 requires exact frozen H50Z authenticated crop bank")
    if n < 4 or len(templates) < 24:
        raise ValueError("Too few source templates or scenes")
    rng = np.random.default_rng(seed)
    cases = []
    for number in range(n):
        path = paths[int(rng.integers(len(paths)))]
        scenario = PROTOCOL["scenarios"][number % 4]
        scene = make_scene(path, templates, seed=int(rng.integers(1, 2**31-1)), scenario=scenario)
        left, right = matched_candidates(scene.left, scene.right)
        rows = []
        for i, a in enumerate(left):
            for j, b in enumerate(right):
                r = {"left_index": i, "right_index": j,
                     "label_SYNTHETIC_ONLY": int((i,j) in scene.correct_pairs)}
                r.update(pair_geometry_and_curvature(a,b))
                rows.append(r)
        if (len(rows) != 6 or sum(r["label_SYNTHETIC_ONLY"] for r in rows) != 2
                or not all(np.isfinite([r[key] for key in ALL_FIELDS]).all() for r in rows)):
            raise ValueError("Unexpected topology, source or numerical failure")
        # Never publish authentic source IDs; they are retained only in this process.
        cases.append({"pairs": rows, "scenario": scenario,
                      "synthetic_measurement_error_px": scene.simulated_measurement_error_px,
                      "_PRIVATE_source_frames": scene.source_frame_ids})
    return cases


def flatten(cases: list[dict]) -> tuple[list[dict], np.ndarray]:
    rows = [r for case in cases for r in case["pairs"]]
    labels = np.asarray([r["label_SYNTHETIC_ONLY"] for r in rows], dtype=int)
    if len(rows) != len(cases)*6 or np.count_nonzero(labels) != len(cases)*2:
        raise ValueError("Inconsistent synthetic scene layout")
    return rows, labels


def shuffle_extra_features(rows: list[dict], rng: np.random.Generator) -> list[dict]:
    """Shuffles only image-independent 2D extension; labels remain unchanged."""
    if len(rows) % 6:
        raise ValueError("Must shuffle complete scenes")
    x = np.asarray([[r[k] for k in S50_FIELDS] for r in rows], dtype=float)
    reordered = x[rng.permutation(len(x))]
    return [dict(r, **dict(zip(S50_FIELDS, map(float, vals))))
            for r, vals in zip(rows, reordered)]


def fit_model(rows: list[dict], labels: np.ndarray, fields: tuple[str,...]):
    x = feature_matrix(rows, fields)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(max_iter=600, random_state=1337,
                               class_weight="balanced", C=.3)
    model.fit(scaler.transform(x), labels)
    def score(pairs):
        return model.predict_proba(scaler.transform(feature_matrix(pairs, fields)))[:,1]
    return score


def _public_decision_stats(cases: list[dict], scores: np.ndarray, threshold: float) -> dict:
    r = scene_stats(cases, scores, threshold, margin=PROTOCOL["minimum_margin"])
    return {"scenes": r["n_scenes"],
            "true_links_recovered_SYNTHETIC_ONLY": r["correct_links"],
            "synthetic_wrong_joins": r["wrong_links"],
            "scenes_with_any_wrong_join": r["any_false_scene_count"],
            "suggestions": r["accepted_suggestions"],
            "recoverable_true_links": len(cases)*2,
            "synthetic_true_link_recall": r["correct_links"] / (2*len(cases)),
            "precision_if_suggested_SYNTHETIC_ONLY": (r["correct_links"] / r["accepted_suggestions"]
                    if r["accepted_suggestions"] else None),
            "native_physics_automatic_links": 0,
            "scene_risk_certified": False}


def scene_paired_bootstrap_auc_difference(cases: list[dict],
                                          baseline: np.ndarray, extended: np.ndarray,
                                          *, seed: int, n_bootstrap: int = 1000) -> dict:
    """Uncertainty only conditional on this synthetic generator and its sources."""
    y = np.asarray([int(r["label_SYNTHETIC_ONLY"]) for case in cases for r in case["pairs"]])
    b = np.asarray(baseline, dtype=float)
    e = np.asarray(extended, dtype=float)
    if len(b) != len(y) or len(e) != len(y) or len(y) != len(cases)*6:
        raise ValueError("Paired bootstrap requires identical complete scenes")
    rng = np.random.default_rng(seed)
    point = float(roc_auc_score(y,e) - roc_auc_score(y,b))
    deltas = np.empty(n_bootstrap, dtype=float)
    for z in range(n_bootstrap):
        ix = rng.integers(0, len(cases), len(cases))
        take = (ix[:,None]*6 + np.arange(6)[None,:]).reshape(-1)
        deltas[z] = roc_auc_score(y[take],e[take]) - roc_auc_score(y[take],b[take])
    return {"delta_AUROC_curvature_minus_geometry": point,
            "conditional_synthetic_scene_bootstrap_95pct_percentile":
                [float(x) for x in np.quantile(deltas,[.025,.975])],
            "bootstrap_count":n_bootstrap,
            "independent_real_physical_run_confidence_interval":False}


def _scene_strata(cases: list[dict], scores: np.ndarray, threshold:float):
    output = {}
    for scenario in PROTOCOL["scenarios"]:
        inds = [i for i,c in enumerate(cases) if c["scenario"] == scenario]
        if not inds: continue
        pick = (np.asarray(inds)[:,None]*6 + np.arange(6)[None,:]).reshape(-1)
        output[scenario] = _public_decision_stats([cases[i] for i in inds], scores[pick], threshold)
    return output


def run(source: Path, tiles: Path, public_out: Path, *, train_n: int = 80,
        cal_n: int = 100, test_n: int = 100, bootstrap: int = 1000) -> dict:
    if train_n < 24 or cal_n < 12 or test_n < 12:
        raise ValueError("Too few train/cal/test synthetic scenes")
    manifest = audited_large_manifest(source)
    groups = split_source_frames(manifest)
    train_frames,cal_frames,test_frames,split_digest = groups
    templates = {k: extract_track_templates(manifest, chosen_frames=g) for k,g in (
        ("train",train_frames),("cal",cal_frames),("test",test_frames))}
    public = {"schema":SCHEMA,
              "scientific_scope":"2D image-plane prior tested only with artificial gaps on genuine H50Z backgrounds",
              "native_verified_track_count":0,"native_verified_join_count":0,
              "independent_physical_runs":0,
              "source_operator_measurements_are_not_H50Z_photo_GT":True,
              "photo_background_runs":1,
              "source_visualmetro_complete_frames":len(manifest["frames"]),
              "source_sha256":manifest["provenance_sha256"],
              "S37_frame_split_sha256_UNCHANGED":split_digest,
              "template_source_groups_frame_disjoint": {k:len(v) for k,v in (("train",train_frames),("cal",cal_frames),("test",test_frames))},
              "template_counts":{k:len(v) for k,v in templates.items()},
              "all_compared_candidate_centerlines_fixed_to_31_points":True,
              "calibration_uses_only_cal_scenes":True,
              "synthetic_scene_bootstrap_NOT_physical_CI":True,
              "protocol": {**PROTOCOL, "train_n": train_n, "calibration_n": cal_n,
                           "test_n_per_seed":test_n},
              "directions":{}}
    for photo,target in (("005","006"),("006","005")):
        train_cases = generate_cases(tiles,photo,templates["train"],
                         n=train_n,seed=PROTOCOL["train_seeds"][photo])
        cal_cases = generate_cases(tiles,photo,templates["cal"],
                         n=cal_n,seed=PROTOCOL["cal_seeds"][photo])
        train_rows,yt = flatten(train_cases)
        cal_rows,yc = flatten(cal_cases)
        assert set(x for c in train_cases for x in c["_PRIVATE_source_frames"]).issubset(train_frames)
        assert set(x for c in cal_cases for x in c["_PRIVATE_source_frames"]).issubset(cal_frames)
        scrambled_rng = np.random.default_rng(20265150 if photo=="005" else 20265160)
        tr_sc = shuffle_extra_features(train_rows,scrambled_rng)
        ca_sc = shuffle_extra_features(cal_rows,scrambled_rng)
        fits = {}
        cutoffs = {}
        calibration = {}
        for arm,cols in ARMS.items():
            current_train = tr_sc if "shuffled" in arm else train_rows
            current_cal = ca_sc if "shuffled" in arm else cal_rows
            score = fit_model(current_train,yt,cols)
            fits[arm] = score
            s_cal=score(current_cal)
            t=calibrated_negative_threshold(s_cal,yc,max_fpr=PROTOCOL["calibration_pair_fpr_budget"])
            cutoffs[arm] = t
            calibration[arm] = {"calibration_cutoff":float(t),
                                "empirical_cal_scene_counts":_public_decision_stats(cal_cases,s_cal,t)}
        direction = {"photo_train_and_cal":photo,"photo_heldout":target,
                     "calibration":calibration,"tests":{}}
        for seed in PROTOCOL["test_seeds"][photo]:
            cases=generate_cases(tiles,target,templates["test"],n=test_n,seed=seed)
            assert set(x for c in cases for x in c["_PRIVATE_source_frames"]).issubset(test_frames)
            test_rows, y = flatten(cases)
            test_sc = shuffle_extra_features(test_rows,np.random.default_rng(seed+2500))
            scored = {}
            for arm, cols in ARMS.items():
                s = fits[arm](test_sc if "shuffled" in arm else test_rows)
                scored[arm]=s
            results = {}
            for arm,s in scored.items():
                results[arm] = {
                    "AUROC_SYNTHETIC":float(roc_auc_score(y,s)),
                    "AUPRC_SYNTHETIC":float(average_precision_score(y,s)),
                    "cal_only_empirical_2pct_pair_FPR_policy":_public_decision_stats(cases,s,cutoffs[arm]),
                    "by_synthetic_scenario":_scene_strata(cases,s,cutoffs[arm]),
                }
            paired = scene_paired_bootstrap_auc_difference(
                cases,scored["geometry"],scored["geometry+curvature"],
                seed=seed+991, n_bootstrap=bootstrap)
            direction["tests"][str(seed)] = {"arms":results,
                                                 "paired_curvature_vs_geometry":paired}
            print(f"S50 {photo}->{target} seed={seed}: baseline {results['geometry']['AUROC_SYNTHETIC']:.4f} "
                  f"curvature {results['geometry+curvature']['AUROC_SYNTHETIC']:.4f} "
                  f"delta {paired['delta_AUROC_curvature_minus_geometry']:+.4f}",flush=True)
        public["directions"][f"{photo}_to_{target}"]=direction
        public_out.parent.mkdir(parents=True, exist_ok=True)
        public_out.write_text(json.dumps(public,indent=2,allow_nan=False,ensure_ascii=False)+"\n")
    return public


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--visualmetro",type=Path,required=True)
    p.add_argument("--photo-tiles",type=Path,required=True)
    p.add_argument("--public-out",type=Path,required=True)
    p.add_argument("--train-scenes",type=int,default=80)
    p.add_argument("--cal-scenes",type=int,default=100)
    p.add_argument("--test-scenes",type=int,default=100)
    p.add_argument("--bootstrap",type=int,default=1000)
    a=p.parse_args()
    r=run(a.visualmetro,a.photo_tiles,a.public_out,train_n=a.train_scenes,
          cal_n=a.cal_scenes,test_n=a.test_scenes,bootstrap=a.bootstrap)
    print("S50 public synthetic-only aggregate written:",a.public_out,"directions",list(r["directions"]))

if __name__ == "__main__":
    main()
