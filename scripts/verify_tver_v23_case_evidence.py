#!/usr/bin/env python3
"""Independent case-level audit of executed Tver V23 synthetic experiment.

Does NOT call the V23 model, renderer, evaluator, or training functions.
All results derived from frozen CAL thresholds and individual TEST CSV,
including 3-world central equality checks and paired nuisance-base bootstrap.
Only supports the synthetic LINK/NO_LINK claim, never real ANZA-LIRA E2E.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

POLICIES=("geometry","central","adjacent_straight","far_sham",
          "continuity_25d","learned_2d","learned_25d")


def audit(root:Path):
    root=Path(root)
    freeze_raw=(root/"v23_train_cal_frozen.json").read_bytes()
    cases_raw=(root/"v23_test_case_scores.csv").read_bytes()
    freeze=json.loads(freeze_raw)
    result=json.loads((root/"v23_results.json").read_text())
    if (freeze["status"]!="TRAIN_CAL_FROZEN_BEFORE_ANY_TEST_OPENED"
        or result["status"]!="V23_SYNTHETIC_EXECUTED_AUDITED"
        or hashlib.sha256(freeze_raw).hexdigest()!=result["train_cal_frozen_SHA256"]
        or hashlib.sha256(cases_raw).hexdigest()!=result["test_case_scores_SHA256"]
        or freeze["hyperparameters_TEST_tuned"]
        or not freeze["geometry_oracle_source_and_two_candidates_supplied"]
        or not result["not_trained_historical_ANZA_LIRA"]):
        raise AssertionError("Frozen calibration/SHA/source-claim contract mismatch")
    rows=list(csv.DictReader(cases_raw.decode().splitlines()))
    if len(rows)!=960:raise AssertionError("Need 160×3 cases per IID/OOD")
    receipt={"status":"V23_INDEPENDENT_CASEWISE_SYNTHETIC_AUDIT_PASS",
             "calibration_selected_before_test":True,
             "geometry_oracle_source_and_candidates_used":True,
             "historical_ANZA_LIRA_not_executed":True,
             "per_case_SHA256_verified":True,
             "groups_resampled_not_three_counterfactual_worlds":True,
             "scores_and_decisions_recomputed_from_frozen_thresholds":True,
             "results":{}}
    for split in ("test","ood"):
        part=[r for r in rows if r["split"]==split]
        if len(part)!=480:raise AssertionError("Wrong test case count")
        groups=defaultdict(list)
        for r in part:groups[int(r["base"])].append(r)
        if len(groups)!=160:raise AssertionError("Only 160 independent base groups")
        for key,pack in groups.items():
            if (len(pack)!=3 or
                {int(q["label"]) for q in pack}!={0,1,2} or
                len({q["central_sha256"] for q in pack})!=1):
                raise AssertionError(f"Counterfactual central/pairing failure {key}")
        y=np.asarray([int(r["label"]) for r in part])
        grouped=np.asarray([int(r["base"]) for r in part])
        metrics={};prediction={}
        for policy in POLICIES:
            p=freeze["cal_thresholds"][policy]
            scores=np.asarray([[float(q[f"{policy}_score0"]),
                                float(q[f"{policy}_score1"])]
                               for q in part],float)
            if not np.isfinite(scores).all():
                raise AssertionError("Nonfinite TEST observed features")
            best=np.max(scores,axis=1)
            margin=np.abs(scores[:,0]-scores[:,1])
            selected=np.argmax(scores,axis=1)
            selected[(best<float(p["accept_threshold"]))|
                     (margin<float(p["margin_threshold"]))]=2
            predicted=np.asarray([
                int(q[f"{policy}_prediction"]) for q in part])
            np.testing.assert_array_equal(selected,predicted)
            prediction[policy]=predicted
            good=y!=2;bad=~good
            independent={
                "correct_accepted":int(np.sum(good&(predicted==y))),
                "wrong_branch_accepted":int(np.sum(
                    good&(predicted!=y)&(predicted!=2))),
                "false_accepted":int(np.sum(bad&(predicted!=2))),
                "all_3way_accuracy":float(np.mean(predicted==y)),
                "positive_correct_accepted_rate":float(
                    np.mean(predicted[good]==y[good])),
                "wrong_branch_rate":float(np.mean(
                    (predicted[good]!=y[good])&(predicted[good]!=2))),
                "false_bridge_rate":float(np.mean(predicted[bad]!=2)),
            }
            for key,v in independent.items():
                original=result["report"][split]["policy_results"][policy][key]
                if abs(v-original)>1e-12:
                    raise AssertionError(f"Drift {split}/{policy}/{key}")
            if policy in ("geometry","central","learned_2d"):
                if not np.isclose(independent["all_3way_accuracy"],1/3):
                    raise AssertionError("Counterfactual 2D identifiability violated")
            metrics[policy]=independent
        # True group-paired descriptive CI: never bootstrap 480 worlds.
        group_ids=sorted(groups)
        for baseline in ("geometry","central","adjacent_straight","far_sham",
                         "learned_2d","learned_25d"):
            diff=[]
            for g in group_ids:
                idx=np.flatnonzero(grouped==g)
                if len(idx)!=3:raise AssertionError("Broken dependent triplet")
                a=prediction["continuity_25d"][idx]
                b=prediction[baseline][idx]
                diff.append(float(np.mean(a==y[idx])-np.mean(b==y[idx])))
            diff=np.asarray(diff)
            rng=np.random.default_rng(20261114)
            draw=rng.integers(0,len(diff),size=(3000,len(diff)))
            ci=np.quantile(diff[draw].mean(axis=1),[.025,.975])
            org=result["report"][split][
                "continuity_minus_controls_cluster_bootstrap"][baseline]
            np.testing.assert_allclose(ci,org["base_group_cluster_CI95"],atol=1e-12,rtol=0)
            if abs(float(np.mean(diff))-org["delta_3way_accuracy"])>1e-12:
                raise AssertionError("Cluster pair difference drift")
        receipt["results"][split]={
            "n_independent_base_groups":len(groups),
            "n_counterfactual_worlds":len(part),
            "label_counts":[int(np.sum(y==i)) for i in (0,1,2)],
            "source_sha_identical_by_group":True,
            "policy_rates":metrics,
            "paired_CI_recomputed":True}
    outcome=result["report"]["ood"]["OOD_admissibility"]
    s=receipt["results"]["ood"]["policy_rates"]["continuity_25d"]
    baseline_delta=result["report"]["ood"][
        "continuity_minus_controls_cluster_bootstrap"]["learned_25d"][
        "base_group_cluster_CI95"][0]
    independent_pass=bool(
        s["false_bridge_rate"]<=.02 and s["wrong_branch_rate"]<=.01
        and s["positive_correct_accepted_rate"]>=.15 and baseline_delta>0)
    if outcome!=independent_pass:
        raise AssertionError("Reported OOD protocol fail/pass not independently derived")
    receipt["strict_OOD_pass_recomputed"]=independent_pass
    (root/"verify_v23_independent.json").write_text(
        json.dumps(receipt,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({
        "status":receipt["status"],"strict_ood_pass":independent_pass,
        "iid_accepted":receipt["results"]["test"]["policy_rates"][
            "continuity_25d"]["correct_accepted"],
        "ood_accepted":s["correct_accepted"]},indent=2))
    return receipt


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dir",type=Path,required=True)
    args=p.parse_args()
    audit(args.dir)
