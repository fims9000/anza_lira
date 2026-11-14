#!/usr/bin/env python3
"""Independent verification of V24 hybrid-NOLINK synthetic case outputs.

Reads immutable CAL receipt, per-world case CSV and published summary.
DOES NOT import V24 or V23 rendering/training/decision code. It verifies
counterfactual near-slice exact marginal matching, independently recalculates
each selective prediction, all rates, linked no-link and whole-base CI.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METHODS=("continuity_25d","learned_25d","consensus_v24",
         "single_neighbor_minus","single_neighbor_plus",
         "central","learned_2d")


def audit(root):
    root=Path(root)
    raw=(root/"v24_before_test_freeze.json").read_bytes()
    cases=(root/"v24_hard_negative_case_scores.csv").read_bytes()
    result=json.loads((root/"v24_results.json").read_text())
    lock=json.loads(raw)
    if (result["status"]!="EXECUTED_NEW_SYNTHETIC_HARD_NEGATIVE"
        or lock["status"]!="V24_ALL_THRESHOLDS_FROZEN_BEFORE_HYBRID_TEST_OPENED"
        or hashlib.sha256(raw).hexdigest()!=result["V24_CAL_lock_SHA256"]
        or hashlib.sha256(cases).hexdigest()!=result["cases_sha256"]
        or lock["new_hybrid_base_seed"]!=202673010 or
        result["N_base_groups"]!=160 or result["N_worlds"]!=480
        or not result["V24_TEST_opened_only_after_CAL_freeze"]
        or not result["NO_LINK_each_neighbor_exact_positive_world_match"]):
        raise AssertionError("V24 pre-test freeze/immutable SHA/registered cohort failure")
    rows=list(csv.DictReader(cases.decode().splitlines()))
    if len(rows)!=480:raise AssertionError("V24 synthetic case CSV incomplete")
    labels=np.asarray([int(r["label"]) for r in rows],int)
    groups=np.asarray([int(r["base"]) for r in rows],int)
    sections=defaultdict(list)
    for i,r in enumerate(rows):sections[int(r["base"])].append(i)
    if (len(sections)!=160 or len(np.unique(groups))!=160 or
        any(len(v)!=3 for v in sections.values())):
        raise AssertionError("V24 actual counterfactual cluster grouping unavailable")
    for group,indices in sections.items():
        rs=[rows[i] for i in indices]
        if ({int(r["label"]) for r in rs}!={0,1,2} or
            len({r["central_sha256"] for r in rs})!=1):
            raise AssertionError("World-label balance or bitwise center not identical")
        true=[r for r in rs if int(r["label"])!=2]
        none=next(r for r in rs if int(r["label"])==2)
        minus=[r for r in true if r["near_minus_sha256"]==none["near_minus_sha256"]]
        plus=[r for r in true if r["near_plus_sha256"]==none["near_plus_sha256"]]
        if (len(minus)!=1 or len(plus)!=1 or
            int(minus[0]["label"])==int(plus[0]["label"])):
            raise AssertionError("V24 synthetic negative not exactly one-slice-matched")
    all_preds={};metrics={}
    for name in METHODS:
        source=np.asarray([[float(r[name+"_score0"]),
                            float(r[name+"_score1"])]
                           for r in rows])
        if not np.isfinite(source).all():
            raise AssertionError("Nonfinite score")
        cal=lock["thresholds"][name]
        expected=np.argmax(source,axis=1)
        expected[(source.max(axis=1)<cal["accept_threshold"])|
                 (np.abs(source[:,0]-source[:,1])<
                  cal["margin_threshold"])]=2
        claimed=np.asarray([int(r[name+"_pred"]) for r in rows])
        np.testing.assert_array_equal(expected,claimed)
        all_preds[name]=expected
        positive=labels!=2
        negative=~positive
        fresh={
            "correct_accepted":int(np.sum(
                positive&(expected==labels))),
            "wrong_branch_accepted":int(np.sum(
                positive&(expected!=labels)&(expected!=2))),
            "false_accepted":int(np.sum(negative&(expected!=2))),
            "all_3way_accuracy":float(np.mean(expected==labels)),
            "positive_correct_accepted_rate":float(
                np.mean(expected[positive]==labels[positive])),
            "wrong_branch_rate":float(np.mean(
                (expected[positive]!=labels[positive])&
                (expected[positive]!=2))),
            "false_bridge_rate":float(np.mean(
                expected[negative]!=2)),
            "rejection_rate":float(np.mean(expected==2)),
        }
        for k,x in fresh.items():
            observed=result["method_results"][name][k]
            if abs(x-observed)>1e-12:
                raise AssertionError(f"V24 published metric drift {name}/{k}")
        metrics[name]=fresh
    # Full independent within-nuisance-base paired uncertainty computation.
    contrast={}
    for name in ("continuity_25d","learned_25d","central",
                 "single_neighbor_minus","single_neighbor_plus"):
        delta=np.asarray([
            np.mean(all_preds["consensus_v24"][idx]==labels[idx])-
            np.mean(all_preds[name][idx]==labels[idx])
            for idx in sections.values()],float)
        rng=np.random.default_rng(20261114)
        draw=rng.integers(0,len(delta),size=(3000,len(delta)))
        ci=np.quantile(delta[draw].mean(axis=1),[.025,.975])
        r=result["consensus_minus_baselines_group_bootstrap"][name]
        np.testing.assert_allclose(
            ci,r["base_group_cluster_CI95"],atol=1e-12,rtol=0)
        if abs(float(delta.mean())-r["delta_3way_accuracy"])>1e-12:
            raise AssertionError(f"V24 paired 160-base contrast drift: {name}")
        contrast[name]={
            "delta":float(delta.mean()),"ci95":list(map(float,ci))}
    c=metrics["consensus_v24"]
    admissible=bool(
        c["false_bridge_rate"]<=.02
        and c["wrong_branch_rate"]<=.01
        and c["positive_correct_accepted_rate"]>=.15
        and contrast["learned_25d"]["ci95"][0]>0)
    if admissible!=result["hard_negative_OOD_criterion_met"]:
        raise AssertionError("V24 prior OOD admissibility criterion not reproduced")
    receipt={
        "status":"V24_INDEPENDENT_MATCHED_MARGINAL_CASE_AUDIT_PASS",
        "all_480_cases_independently_scored_from_frozen_CAL":True,
        "n_independent_synthetic_base_groups":160,
        "n_dependent_counterfactual_worlds":480,
        "identical_central_2D_pixels_SHA_for_all_worlds":True,
        "negative_each_single_neighbor_exactly_one_positive_world":True,
        "negative_two_near_slices_correspond_to_opposite_positive_worlds":True,
        "160_base_group_bootstrap_recomputed":True,
        "V23_frozen_train_CAL_not_changed_using_V24_TEST":True,
        "methods":metrics,"paired_ci":contrast,
        "protocol_strict_hard_NO_LINK_criteria_recomputed":admissible,
        "original_ANZA_SBPP_LIRA_trained_pipeline_not_executed":True,
        "synthetic_no_natural_geological_identity_labels":True,
    }
    (root/"verify_v24_independent.json").write_text(
        json.dumps(receipt,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({
        "status":receipt["status"],
        "correct":c["correct_accepted"],
        "false":c["false_accepted"],
        "wrong":c["wrong_branch_accepted"],
        "strict_criteria":admissible,
        "delta_vs_trained_25d":contrast["learned_25d"],
    },indent=2))
    return receipt


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dir",type=Path,required=True)
    a=p.parse_args()
    audit(a.dir)
