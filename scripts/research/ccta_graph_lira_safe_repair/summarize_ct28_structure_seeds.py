#!/usr/bin/env python3
"""Strict three-seed comparison of five frozen CT28 development-val masks.

Reads only previously created JSON receipts. Never opens CT, weights, masks,
training labels, validation files or the six held-out CT28 test patients.
Refuses incomplete/mismatched comparisons and frozen-baseline drift. Does
NOT report independent clinical evidence or promote a research checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

EXPECTED_VAL=frozenset({957,961,965,966,974})
METRICS=(
    "ANZA_minus_CNN_dice_delta",
    "ANZA_minus_CNN_cldice_delta",
)
MODEL_METRICS=(
    "dice_delta","cldice_delta",
    "gt_false_bridge_pairs","unsupported_added_voxels",
)


def aggregate(reports:dict[int,Path])->dict:
    if len(reports)<3 or len(set(reports))!=len(reports):
        raise ValueError("Three distinct predeclared training seeds are required")
    evidence={}
    signature=None
    baseline=None
    for seed,path in sorted(reports.items()):
        report=json.loads(Path(path).read_text(encoding="utf-8"))
        pair=report.get("variant_pair")
        if (report.get("status")!="complete_five_patient_matched_CCTA_dev_val"
            or report.get("patient_completed")!=5
            or report.get("patient_failed_or_missing")!=0
            or report.get("held_out_test_accessed") is not False
            or not isinstance(pair,list) or len(pair)!=2
            or pair[0] not in {"conv3d_structure","conv3d_sep_structure"}
            or pair[1] not in {
                "anza3d_structure","anza3d_sep_structure",
                "anza3d_noaxial_structure","anza3d_noaxial_sep_structure"
            }
            or report.get("pair_identity_source")!="neural_pair"
            or report.get("relation_train_protocol")!=
                "experimental_patient_OOF_15copy_augmented_relation_head"):
            raise ValueError(f"Seed {seed}: incomplete, non-structural or mismatched clinical evaluation protocol")
        current=(tuple(pair),report.get("protocol_sha256"),
                 report.get("relation_train_protocol"),
                 report.get("tip_encoding","summed"),
                 bool(report.get("noaxial_control",False)))
        if signature is None:
            signature=current
        elif signature!=current:
            raise ValueError("Cannot compare incompatible ANZA/CNN geometry, model, or mask freeze")
        rows=report.get("per_patient")
        if not isinstance(rows,list) or len(rows)!=5:
            raise ValueError("Missing patient-level outputs")
        bypatient={int(x["patient_id"]):x for x in rows}
        if set(bypatient)!=EXPECTED_VAL or not all(x.get("status")=="complete" for x in rows):
            raise ValueError("Incomplete or wrong five development-validation subjects")
        perpatient={}
        for pid,row in sorted(bypatient.items()):
            required=(
                "K10_dice_delta","K10_cldice_delta",*METRICS,
                *[name+"_"+metric for name in pair for metric in MODEL_METRICS]
            )
            for key in required:
                value=row.get(key)
                if not isinstance(value,(int,float)) or not math.isfinite(value):
                    raise ValueError(f"Patient {pid} missing finite {key}")
            k10=(row["K10_dice_delta"],row["K10_cldice_delta"])
            if baseline is None:
                baseline={}
            if pid in baseline and any(abs(a-b)>1e-8 for a,b in zip(baseline[pid],k10)):
                raise ValueError("Frozen K10/15 baseline metrics changed across seed evaluations")
            baseline[pid]=k10
            cnn,anza=pair
            perpatient[pid]={
                "cldice_delta_an_minus_cnn":float(row["ANZA_minus_CNN_cldice_delta"]),
                "dice_delta_an_minus_cnn":float(row["ANZA_minus_CNN_dice_delta"]),
                "cnn_false_bridge_pairs":int(row[cnn+"_gt_false_bridge_pairs"]),
                "anza_false_bridge_pairs":int(row[anza+"_gt_false_bridge_pairs"]),
                "cnn_unsupported_added_voxels":int(row[cnn+"_unsupported_added_voxels"]),
                "anza_unsupported_added_voxels":int(row[anza+"_unsupported_added_voxels"]),
            }
        evidence[seed]={
            "per_patient":perpatient,
            "mean_ANZA_minus_CNN_cldice_delta":float(
                sum(x["cldice_delta_an_minus_cnn"] for x in perpatient.values())/5
            ),
            "mean_ANZA_minus_CNN_dice_delta":float(
                sum(x["dice_delta_an_minus_cnn"] for x in perpatient.values())/5
            ),
            "false_bridge_pairs_ANZA_minus_CNN":int(
                sum(x["anza_false_bridge_pairs"]-x["cnn_false_bridge_pairs"]
                    for x in perpatient.values())
            ),
            "unsupported_voxels_ANZA_minus_CNN":int(
                sum(x["anza_unsupported_added_voxels"]-x["cnn_unsupported_added_voxels"]
                    for x in perpatient.values())
            ),
        }
    scores=[r["mean_ANZA_minus_CNN_cldice_delta"] for r in evidence.values()]
    safe=[r["false_bridge_pairs_ANZA_minus_CNN"]<=0 for r in evidence.values()]
    result={
        "status":"three_seed_five_patient_development_only",
        "seeds":sorted(evidence),
        "variant_pair":list(signature[0]),
        "frozen_protocol_sha256":signature[1],
        "relation_training":signature[2],
        "tip_encoding":signature[3],
        "noaxial_control":signature[4],
        "held_out_CT28_test_accessed":False,
        "per_seed":evidence,
        "ANZA_better_cldice_seed_count":sum(x>0 for x in scores),
        "ANZA_no_extra_false_bridges_seed_count":sum(safe),
        "ANZA_better_cldice_and_no_extra_false_bridge_seed_count":sum(
            s>0 and safe[i] for i,s in enumerate(scores)
        ),
        "mean_ANZA_minus_CNN_cldice_delta_across_seeds":float(sum(scores)/len(scores)),
        "interpretation":"Descriptive dev-val only. No threshold tuning or clinical superiority claim.",
    }
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report",action="append",required=True,
                   help="Repeat SEED:/path/to/matched_val_comparison.json for at least 3 seeds")
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():
        raise FileExistsError("Immutable summary already exists")
    reports={}
    for spec in a.report:
        seed,path=spec.split(":",1)
        number=int(seed)
        if number in reports:
            raise ValueError("Duplicate seed")
        reports[number]=Path(path)
    result=aggregate(reports)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({
        "status":result["status"],
        "seeds":result["seeds"],
        "mean_cldice_difference":result["mean_ANZA_minus_CNN_cldice_delta_across_seeds"],
        "seeds_with_safe_cldice_win":result["ANZA_better_cldice_and_no_extra_false_bridge_seed_count"],
    },ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
