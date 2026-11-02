#!/usr/bin/env python3
"""Join selected CT28 abstention reasons with ground-truth candidate scores."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import train_ct28_relation_head as training


ROOT = training.REPO_ROOT
SCENES = ROOT / "artifacts/varvara/ct28_graph_lira"
SWEEP = ROOT / "varvara/anza_lira_local_results/ct28_copy_count_ablations"
SELECTED = SWEEP / "joint_check/fusion_10_relation_15"
DIAGNOSTICS = SWEEP / "selected_diagnostics"
TEST_IDS = {954, 958, 972, 973, 980, 984}


def main() -> None:
    plans = pd.read_csv(SCENES / "scene_plan_train_val.csv")
    candidates = pd.read_csv(SCENES / "scene_candidates_train_val.csv")
    scores = pd.read_csv(SELECTED / "relation_local_scores_train_val.csv")
    predictions = pd.read_csv(DIAGNOSTICS / "graph_lira_predictions_val.csv")
    reasons = pd.read_csv(DIAGNOSTICS / "abstention_reasons_val.csv")
    validation = plans[plans.split.eq("val")]
    if len(validation) != 368 or set(validation.scan_id.astype(int)) != training.EXPECTED_PATIENT_IDS["val"]:
        raise ValueError("validation scene cohort changed")
    if set(plans.scan_id.astype(int)) & TEST_IDS:
        raise ValueError("held-out test scene entered diagnostic")
    predictions = predictions[predictions.model.eq("ct_conditioned")]
    if len(predictions) != 368 or len(reasons) != 368:
        raise ValueError("selected diagnostics must cover every validation scene")
    frame = predictions.merge(
        reasons.drop(columns=[column for column in ("scan_id", "scene_family", "relation_class", "geometry_ambiguous")
                              if column in reasons.columns]),
        on="scene_id", validate="one_to_one", suffixes=("", "_diagnostic"),
    ).merge(
        validation[["scene_id", "pair_candidate_count", "junction_candidate_count"]],
        on="scene_id", validate="one_to_one",
    )
    scored = candidates[["scene_id", "candidate_id", "candidate_type", "y"]].merge(
        scores[["scene_id", "candidate_id", "geometry_score", "ct_score", "geometry_plus_ct_score"]],
        on=["scene_id", "candidate_id"], validate="one_to_one",
    )
    scored = scored[scored.scene_id.isin(validation.scene_id)].copy()
    scored["identity_score"] = scored["geometry_score"].where(
        scored.candidate_type.eq("JUNCTION"), scored["geometry_plus_ct_score"]
    )
    rows = []
    for (scene_id, candidate_type), group in scored.groupby(["scene_id", "candidate_type"], sort=False):
        true = group[group.y.eq(1)]
        false = group[group.y.eq(0)]
        if len(true) > 1:
            raise ValueError(f"multiple true {candidate_type} candidates in {scene_id}")
        rows.append({
            "scene_id": scene_id,
            "candidate_type": candidate_type,
            "true_candidate_id": true.candidate_id.iloc[0] if len(true) else "",
            "true_identity_score": float(true.identity_score.iloc[0]) if len(true) else float("nan"),
            "best_false_candidate_id": false.loc[false.identity_score.idxmax(), "candidate_id"] if len(false) else "",
            "best_false_identity_score": float(false.identity_score.max()) if len(false) else float("nan"),
        })
    audit = pd.DataFrame(rows)
    for candidate_type, prefix in (("PAIR", "pair"), ("JUNCTION", "junction")):
        subset = audit[audit.candidate_type.eq(candidate_type)].drop(columns="candidate_type")
        subset = subset.rename(columns={column: f"{prefix}_{column}" for column in subset if column != "scene_id"})
        frame = frame.merge(subset, on="scene_id", how="left", validate="one_to_one")
    failures = frame[
        (frame.relation_class.ne("NONE") & frame.accepted.eq(0))
        | (frame.relation_class.eq("NONE") & frame.selected_relation_before_consistency.ne("NONE"))
    ].copy()
    fields = [
        "scene_id", "scan_id", "scene_family", "relation_class", "raw_relation", "tau_relation",
        "selected_relation_before_consistency", "confidence", "combined_consistency",
        "primary_reason", "pre_consistency_exact", "pre_consistency_false",
        "proposed_pair_candidate_id", "pair_true_candidate_id", "pair_true_identity_score",
        "pair_best_false_candidate_id", "pair_best_false_identity_score",
        "proposed_junction_candidate_id", "junction_true_candidate_id", "junction_true_identity_score",
        "junction_best_false_candidate_id", "junction_best_false_identity_score",
        "selected_type_gate_value", "pair_identity_gate_value", "junction_identity_gate_value",
        "pair_candidate_count", "junction_candidate_count", "geometry_ambiguous",
        "junction_benchmark_group", "base_compatibility_satisfied",
    ]
    failures[fields].to_csv(DIAGNOSTICS / "failure_cases_val.csv", index=False)
    print("failure cases", len(failures))
    print(reasons[reasons.relation_class.ne("NONE")].primary_reason.value_counts().to_string())


if __name__ == "__main__":
    main()
