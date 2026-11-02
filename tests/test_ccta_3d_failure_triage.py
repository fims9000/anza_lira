"""Failure triage does not invent pathology or open held-out patients."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/
                      "scripts/research/ccta_graph_lira_safe_repair"))
from ccta_3d_failure_triage import observed_failure_cases


def test_three_distinct_observed_3d_failure_categories():
    decisions={
        "patient_label":"953",
        "decisions":[
            {"scene_id":"s1","reason":"K10_abstain","confidence":.81,"consistency":.2},
            {"scene_id":"s2","reason":"mask_gate:unselected_component_contact","confidence":.93,"consistency":1.},
            {"scene_id":"s3","reason":"globally_conflicting_endpoint_proposal"},
        ],
    }
    metrics={
        "patient_id":"953",
        "truth_used_for_inference":False,
        "per_merge_examples":[{
            "after_component":1,
            "gt_inconsistent_pairs":2,
            "unassigned_pairs":0,
            "before_component_representative_ijk":{"1":[2,3,4],"2":[8,3,4]},
        }],
    }
    report=observed_failure_cases(patient="953",decisions=decisions,metrics=metrics)
    assert report["status"]=="observed_three_or_more_failure_examples"
    assert len(report["selected_examples"])==3
    assert report["selected_examples"][0]["category"]=="GT_inconsistent_inter_component_bridge"
    assert report["selected_examples"][0]["before_component_representative_ijk"]["1"]==[2,3,4]
    assert {case["category"] for case in report["selected_examples"]}=={
        "GT_inconsistent_inter_component_bridge",
        "topology_or_path_veto",
        "global_or_distributional_rejection",
    }


def test_one_observed_failure_does_not_invent_three():
    report=observed_failure_cases(
        patient="953",
        decisions={"patient_label":"953","decisions":[{"scene_id":"s1","reason":"K10_abstain"}]},
        metrics={"patient_id":"953","truth_used_for_inference":False},
        min_examples=3,
    )
    assert report["status"]=="insufficient_observed_failure_cases"
    assert len(report["selected_examples"])==1
    assert report["available_failure_events"]==1


def test_heldout_and_mismatched_patient_forbidden():
    with pytest.raises(ValueError,match="Held-out"):
        observed_failure_cases(
            patient="954",decisions={"patient_label":"954"},
            metrics={"patient_id":"954"},
        )
    with pytest.raises(ValueError,match="Mismatched"):
        observed_failure_cases(
            patient="953",decisions={"patient_label":"955"},
            metrics={"patient_id":"953"},
        )
