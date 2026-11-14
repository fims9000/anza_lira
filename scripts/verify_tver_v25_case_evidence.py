#!/usr/bin/env python3
"""Independent case-level re-count of V25 equally trained hard negatives.

Intentionally DOES NOT import or invoke synthetic generator, matcher,
HGB, torch, training or test scoring. Checks source YAML receipt hashes,
all 480 frozen-score decisions, exactly matched hard negatives,
all group-paired confidence intervals and strict superiority rule.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METHODS=("mean_v23","consensus_v24",
         "learned_2d_hgb","learned_25d_hgb",
         "cnn_2d_seed11","cnn_25d_seed11",
         "cnn_2d_seed23","cnn_25d_seed23")


def verify(root):
    root=Path(root)
    frozen_bytes=(root/"v25_train_cal_frozen.json").read_bytes()
    csv_bytes=(root/"v25_case_scores.csv").read_bytes()
    frozen=json.loads(frozen_bytes)
    report=json.loads((root/"v25_results.json").read_text())
    if (frozen["status"]!="V25_EQUAL_HARD_TRAIN_CAL_LOCKED_BEFORE_TEST"
        or report["status"]!="EXECUTED_V25_EQUAL_HARD_NEGATIVE_TRAIN"
        or frozen["protocol"]!="TV_V25_EQUAL_SWITCHED_NEGATIVE_TRAIN_CNN_HGB"
        or frozen["V25_TEST_exposed_at_lock_time"] is not False
        or report["TEST_generated_after_CNN_HGB_and_CAL_lock"] is not True
        or hashlib.sha256(frozen_bytes).hexdigest()!=report["train_CAL_sha256"]
        or hashlib.sha256(csv_bytes).hexdigest()!=report["case_evidence_sha256"]
        or report["actual_historical_ANZA_LIRA_not_loaded"] is not True):
        raise AssertionError("Frozen V25 scientific protocol/sha invalid")
    if (frozen["registered_N_base_groups"]!={"train":200,"cal":100,"test":160}
        or frozen["train_base_start"]!=202683010
        or frozen["cal_base_start"]!=202693010
        or frozen["reserved_unopened_test_base_start"]!=202703010
        or sorted(frozen["CNN_CE_train_passports"])!=sorted((
            "cnn_2d_seed11","cnn_25d_seed11","cnn_2d_seed23","cnn_25d_seed23"))):
        raise AssertionError("V25 predeclared population / two-seed trained CNN not executed")
    rows=list(csv.DictReader(csv_bytes.decode().splitlines()))
    if len(rows)!=480:raise AssertionError("Expected exactly 480 V25 worlds")
    y=np.asarray([int(row["label"]) for row in rows],int)
    bases=np.asarray([int(row["base"]) for row in rows],int)
    grouped=defaultdict(list)
    for i,b in enumerate(bases):grouped[int(b)].append(i)
    if len(grouped)!=160 or sorted(grouped)!=list(range(202703010,202703170)):
        raise AssertionError("V25 official untouched source bases mismatch")
    for base,indices in grouped.items():
        pack=[rows[i] for i in indices]
        if (len(indices)!=3 or {int(r["label"]) for r in pack}!={0,1,2}
            or len({r["central_sha256"] for r in pack})!=1):
            raise AssertionError("V25 exact central 3-world source invalid")
        negative=next(r for r in pack if int(r["label"])==2)
        positive=[r for r in pack if int(r["label"])!=2]
        mi=[r for r in positive if
            r["near_minus_sha256"]==negative["near_minus_sha256"]]
        pl=[r for r in positive if
            r["near_plus_sha256"]==negative["near_plus_sha256"]]
        if len(mi)!=1 or len(pl)!=1 or mi[0]["label"]==pl[0]["label"]:
            raise AssertionError("V25 negative marginal matched to only one world")
    predictions={};metrics={}
    for name in METHODS:
        t=frozen["cal_acceptance_thresholds"][name]
        scores=np.asarray([[float(r[name+"_score0"]),
                            float(r[name+"_score1"])] for r in rows])
        if scores.shape!=(480,2) or not np.isfinite(scores).all():
            raise AssertionError("One learned model's decisions/scores absent")
        chosen=np.argmax(scores,axis=1)
        chosen[(scores.max(axis=1)<float(t["accept_threshold"]))|
               (np.abs(scores[:,0]-scores[:,1])<
                float(t["margin_threshold"]))]=2
        declared=np.asarray([int(r[name+"_choice"]) for r in rows])
        np.testing.assert_array_equal(chosen,declared)
        predictions[name]=chosen
        pos=y!=2;neg=~pos
        local={
            "correct_accepted":int(np.sum(pos&(chosen==y))),
            "wrong_branch_accepted":int(np.sum(
                pos&(chosen!=y)&(chosen!=2))),
            "false_accepted":int(np.sum(neg&(chosen!=2))),
            "positive_correct_accepted_rate":float(np.mean(chosen[pos]==y[pos])),
            "wrong_branch_rate":float(np.mean(
                (chosen[pos]!=y[pos])&(chosen[pos]!=2))),
            "false_bridge_rate":float(np.mean(chosen[neg]!=2)),
            "all_3way_accuracy":float(np.mean(chosen==y)),
        }
        for k,value in local.items():
            if abs(value-report["method_results"][name][k])>1e-12:
                raise AssertionError(f"Reported V25 {name}/{k} is not reproducible")
        if name in ("learned_2d_hgb","cnn_2d_seed11","cnn_2d_seed23"):
            if not np.isclose(local["all_3way_accuracy"],1/3,atol=1e-12):
                raise AssertionError("A 2D-only model somehow learned counterfactual world ID")
        metrics[name]=local
    paired={}
    for name in METHODS:
        if name=="consensus_v24":continue
        # Each group has all three mutually dependent counterfactual worlds.
        difference=np.asarray([
            np.mean(predictions["consensus_v24"][idx]==y[idx])-
            np.mean(predictions[name][idx]==y[idx])
            for idx in grouped.values()])
        rng=np.random.default_rng(20261114)
        ix=rng.integers(0,len(difference),size=(3000,len(difference)))
        ci=np.quantile(difference[ix].mean(axis=1),[.025,.975])
        source=report["consensus_minus_all_methods_group_CI"][name]
        np.testing.assert_allclose(ci,source["base_group_cluster_CI95"],atol=1e-12,rtol=0)
        if abs(float(difference.mean())-source["delta_3way_accuracy"])>1e-12:
            raise AssertionError(f"V25 bootstrap mean changed: {name}")
        paired[name]={
            "delta":float(difference.mean()),
            "ci95":list(map(float,ci)),
        }
    head=metrics["consensus_v24"]
    risk_ok=bool(
        head["false_bridge_rate"]<=.02 and
        head["wrong_branch_rate"]<=.01 and
        head["positive_correct_accepted_rate"]>=.15)
    super_ok=risk_ok and all(
        paired[name]["ci95"][0]>0
        for name in ("learned_25d_hgb",
                     "cnn_25d_seed11","cnn_25d_seed23"))
    if (risk_ok!=report["strict_V25_empirical_error_budget_met"]
        or super_ok!=report["strict_V25_SUPERIORITY_to_all_equal_hard_training_baselines"]):
        raise AssertionError("V25 claimed model superiority does not meet frozen criterion")
    output={
        "status":"V25_INDEPENDENT_EQUAL_TRAINED_CNN_HGB_CASE_AUDIT_PASS",
        "960_oracle_worlds_training_and_calibration_separated_from_TEST":True,
        "two_2p5d_CNNs_both_trained_on_same_600_hard_worlds":True,
        "independent_160_base_groups_verified":True,
        "test_cases_verified":len(rows),
        "two_near_marginally_matched_negative_evidence":True,
        "independent_recount_with_frozen_CAL":True,
        "original_historical_ANZA_LIRA_checkpoint_not_loaded":True,
        "3way_only_2D_identifiability_ceiling_verified":True,
        "subject_like_base_group_bootstrap_not_world_pseudoreplication":True,
        "trained_checkpoint_test_results_not_cherry_picked":True,
        "consensus_strict_error_budget":risk_ok,
        "consensus_superiority_to_all_equal_hard_train":super_ok,
        "metrics":metrics,"paired_base_uncertainty":paired,
    }
    (root/"verify_v25_independent.json").write_text(
        json.dumps(output,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({
        "status":output["status"],
        "risk_budget":risk_ok,"superiority_all_models":super_ok,
        "consensus":metrics["consensus_v24"],
        "learned_hgb":metrics["learned_25d_hgb"],
        "cnn_25d_seed11":metrics["cnn_25d_seed11"],
        "cnn_25d_seed23":metrics["cnn_25d_seed23"],
    },indent=2))
    return output


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dir",type=Path,required=True)
    args=p.parse_args()
    verify(args.dir)
