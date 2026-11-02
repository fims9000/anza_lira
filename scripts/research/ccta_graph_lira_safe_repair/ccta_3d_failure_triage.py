#!/usr/bin/env python3
"""Select observed CCTA 3D repair failures for a paper's failure analysis.

Uses only saved train/validation JSON from posthoc evaluation and actual
inference decisions. Never invents three failures when the run contains fewer.
GT-derived bridge findings remain strictly posthoc; no model or path retuning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TEST_IDS={"954","958","972","973","980","984"}


def load_json(path:Path)->dict:
    with path.open(encoding="utf-8") as input:
        return json.load(input)


def observed_failure_cases(
    *,
    patient:str,
    decisions:dict,
    metrics:dict,
    min_examples:int=3,
)->dict:
    if patient in TEST_IDS:
        raise ValueError("Held-out CT28 test remains closed")
    if min_examples<1:
        raise ValueError("min_examples must be positive")
    if decisions.get("patient_label")!=patient or metrics.get("patient_id")!=patient:
        raise ValueError("Mismatched patient decisions/ground truth")
    if metrics.get("truth_used_for_inference") is not False:
        raise ValueError("Ground truth leakage reported during inference")
    cases=[]
    # False bridges across ground-truth disconnected artery components,
    # with exemplar voxel locations for manual 3D Slicer inspection.
    for outcome in metrics.get("per_merge_examples",[]):
        wrong=int(outcome.get("gt_inconsistent_pairs",0))
        unknown=int(outcome.get("unassigned_pairs",0))
        if wrong>0:
            cases.append({
                "category":"GT_inconsistent_inter_component_bridge",
                "patient_id":patient,
                "posthoc_only":True,
                "severity":wrong,
                "after_component":outcome["after_component"],
                "before_component_representative_ijk":
                    outcome.get("before_component_representative_ijk",{}),
                "explanation":(
                    "Newly merged predicted fragments map to distinct disconnected "
                    "reference GT components. Inspect manually; GT may be incomplete."
                ),
            })
        elif unknown>0:
            cases.append({
                "category":"unassigned_reference_component_bridge",
                "patient_id":patient,
                "posthoc_only":True,
                "severity":unknown,
                "after_component":outcome["after_component"],
                "before_component_representative_ijk":
                    outcome.get("before_component_representative_ijk",{}),
                "explanation":(
                    "At least one merged fragment does not overlap any reference "
                    "GT component. This is unknown, not automatically a false bridge."
                ),
            })
    for item in decisions.get("decisions",[]):
        reason=str(item.get("reason",""))
        if reason in ("ok","K10_abstain"):
            if reason=="K10_abstain":
                category="selective_model_abstention"
                detail="Relation/identity confidence or perturbation consistency did not pass safety gates"
            else:
                continue
        elif reason.startswith("mask_gate:") or reason.startswith("CT path solver:"):
            category="topology_or_path_veto"
            detail="CT path or topology constraints prevented an unsafe candidate repair"
        elif "global" in reason or "count_support" in reason:
            category="global_or_distributional_rejection"
            detail="Global endpoint conflicts or unseen cardinality pattern led to abstention"
        else:
            category="runtime_or_candidate_failure"
            detail="Experimental system rejected an invalid or unsupported decision"
        cases.append({
            "category":category,
            "patient_id":patient,
            "posthoc_only":False,
            "scene_id":item.get("scene_id"),
            "reason":reason,
            "confidence":item.get("confidence"),
            "consistency":item.get("consistency"),
            "pair_candidate":item.get("candidate_pair"),
            "junction_candidate":item.get("candidate_junction"),
            "explanation":detail,
        })
    priority={
        "GT_inconsistent_inter_component_bridge":0,
        "unassigned_reference_component_bridge":1,
        "topology_or_path_veto":2,
        "global_or_distributional_rejection":3,
        "runtime_or_candidate_failure":4,
        "selective_model_abstention":5,
    }
    cases.sort(key=lambda case:(
        priority.get(case["category"],99),
        -case.get("severity",0),
        str(case.get("scene_id","")),
    ))
    # Diverse primary causes before repeat examples from the same category.
    selected=[]
    seen=set()
    for case in cases:
        if case["category"] not in seen:
            selected.append(case)
            seen.add(case["category"])
    for case in cases:
        if len(selected)>=min_examples:
            break
        if case not in selected:
            selected.append(case)
    selected=selected[:min_examples]
    return {
        "status":"observed_three_or_more_failure_examples" if len(selected)>=min_examples else "insufficient_observed_failure_cases",
        "patient_id":patient,
        "requested_examples":min_examples,
        "available_failure_events":len(cases),
        "selected_examples":selected,
        "manual_validation_required":True,
        "scope":"train_or_validation_only_readonly_failure_analysis",
        "held_out_test_accessed":False,
        "limitation":(
            "Automated selection does not prove clinical relevance. Manual CCTA "
            "and original GT review are required; zero invented examples."
        ),
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-id",required=True)
    parser.add_argument("--decisions",type=Path,required=True)
    parser.add_argument("--metrics",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--examples",type=int,default=3)
    args=parser.parse_args()
    if args.patient_id in TEST_IDS:
        raise ValueError("Held-out test patients may not be used")
    if args.output.exists() or args.output.resolve() in {
        args.decisions.resolve(),args.metrics.resolve()
    }:
        raise FileExistsError("Never overwrite original research evidence")
    report=observed_failure_cases(
        patient=args.patient_id,
        decisions=load_json(args.decisions),
        metrics=load_json(args.metrics),
        min_examples=args.examples,
    )
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status":report["status"],"selected":len(report["selected_examples"]),
        "observed":report["available_failure_events"],
    },indent=2))


if __name__=="__main__":
    main()
