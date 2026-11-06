"""S98 independent visual trace-piece boundaries, adversarial 2D image geometry.

These artificial lines mimic only local observed strokes and manually drawn
2D appearance. No physical particle, no H50Z truth and no performance claim.
"""
from __future__ import annotations

import numpy as np
import pytest

from bubble_chamber.s98_trace_continuity import (
    SCHEMA,discover_human_trace_piece_boundaries,measure_proposal_recovery
)


def line(x0,x1,y):
    x=np.arange(x0,x1+1,dtype=float)
    return np.column_stack((x,np.full_like(x,y)))


def human(sid,instance,x0=20,x1=180,y=40,kind="TRACK_LIKE",certainty="CLEAR"):
    x=np.arange(x0,x1+1,10)
    if x[-1]!=x1:x=np.r_[x,x1]
    return {
        "stroke_id":sid,"visual_instance_id":instance,
        "kind":kind,"certainty":certainty,
        "points_xy_source_px":[[float(a),float(y)] for a in x],
    }


def source(name,x0,x1,y):
    return {"component_id":name,"points_xy":line(x0,x1,y)}


def proposal(a,b,kind,selected=True):
    decision={
        "S92_GAP":"PRIORITY_GAP_REVIEW_NOT_PHYSICAL_ACCEPT",
        "S96_CROSSING":"PROVISIONAL_2D_CROSSING_CONTINUATION",
    }[kind] if selected else "AMBIGUOUS_PARTNER_ABSTAIN"
    return {
        "component_a":a,"component_b":b,"status":"UNKNOWN",
        "review_decision":decision,"physical_auto_join":False,
    }


def test_fragmented_human_track_is_recovered_independently_of_s92_s96_proposals():
    h=[human("operator_stroke_one","T01")]
    c=[
        source("A",20,70,40),
        source("B",85,130,40),
        source("C",146,180,40),
        source("artifact",230,250,40),
    ]
    x=discover_human_trace_piece_boundaries(human_strokes=h,observed_components=c)
    assert x["schema"]==SCHEMA
    assert x["human_clear_positive_strokes"]==1
    assert x["assessable_unique_detected_piece_adjacencies"]==2
    assert x["ambiguous_unique_piece_adjacencies_abstained"]==0
    assert x["per_human_stroke"][0]["unique_stable_detected_pieces"]==3
    got={
        (row["component_a"],row["component_b"])
        for row in x["conditionally_assessable_piece_adjacencies"]
    }
    assert got=={("A","B"),("B","C")}
    assert all(row["physical_same_particle_verification"] is False
               for row in x["conditionally_assessable_piece_adjacencies"])
    assert all(row["example_reference_boundary_anchor_xy_source"][1]==40.
               for row in x["conditionally_assessable_piece_adjacencies"])

    r=measure_proposal_recovery(x,proposal_sets={
        "S92_GAP":[proposal("A","B","S92_GAP"),proposal("A","C","S92_GAP")],
        "S96_CROSSING":[proposal("B","C","S96_CROSSING")],
    })
    assert r["reference_first_unique_source_piece_boundaries"]==2
    assert r["models_compared"][0]["conditional_reference_piece_boundary_selected_coverage"]==.5
    assert r["models_compared"][1]["conditional_reference_piece_boundary_selected_coverage"]==.5
    assert r["union_selected_conditional_coverage"]==1.
    assert r["reference_boundaries_missed_by_all_selected_models"]==[]
    assert r["does_not_estimate_false_positive_rate_without_exhaustive_background_GT"] is True
    assert r["partial_human_visible_not_particle_GT"] is True


def test_no_machine_proposals_cannot_erase_reference_first_missed_links():
    x=discover_human_trace_piece_boundaries(
        human_strokes=[human("one","trace")],
        observed_components=[
            source("a",20,70,40),source("b",85,130,40),source("c",145,180,40)
        ]
    )
    empty=measure_proposal_recovery(x,proposal_sets={
        "S92_GAP":[],"S96_CROSSING":[]
    })
    assert empty["reference_first_unique_source_piece_boundaries"]==2
    assert empty["union_selected_conditional_coverage"]==0.
    assert len(empty["reference_boundaries_missed_by_all_selected_models"])==2
    for model in empty["models_compared"]:
        assert model["conditional_reference_piece_boundary_selected_coverage"]==0.
        assert model["reference_boundaries_not_selected"]


def test_reversed_source_polyline_and_permuted_component_order_are_equivalent():
    h=[human("expert","same")]
    c=[source("a",20,70,40),source("b",85,130,40),source("c",145,180,40)]
    orig=discover_human_trace_piece_boundaries(human_strokes=h,observed_components=c)
    reversed_source=[
        {"component_id":x["component_id"],"points_xy":x["points_xy"][::-1]}
        for x in reversed(c)
    ]
    alt=discover_human_trace_piece_boundaries(
        human_strokes=h,observed_components=reversed_source
    )
    assert {x["pair_id"] for x in orig["conditionally_assessable_piece_adjacencies"]}=={
        x["pair_id"] for x in alt["conditionally_assessable_piece_adjacencies"]
    }


def test_short_positive_track_retained_and_no_artifact_gt_leakage():
    x=discover_human_trace_piece_boundaries(
        human_strokes=[
            human("very_short","tiny",x0=20,x1=40,y=55),
            human("possible_hairpin","noise",x0=20,x1=100,y=85,
                  kind="ARTIFACT"),
            human("uncertain_true","unclear",x0=20,x1=100,y=125,
                  certainty="UNCERTAIN"),
        ],
        observed_components=[
            source("S1",20,40,55),
            source("POSSIBLE_ARTIFACT",20,100,85),
        ]
    )
    assert x["human_clear_positive_strokes"]==1
    assert x["per_human_stroke"][0]["unique_stable_detected_pieces"]==1
    assert x["assessable_unique_detected_piece_adjacencies"]==0
    outcome=measure_proposal_recovery(x,proposal_sets={"S92_GAP":[]})
    assert outcome["models_compared"][0][
        "conditional_reference_piece_boundary_selected_coverage"
    ] is None
    assert outcome["union_selected_conditional_coverage"] is None


def test_mixed_observed_component_across_independent_human_instances_abstains():
    # An ambiguous detected line appears to support *two* independently
    # labeled overlapping visual instances. It cannot establish one particle.
    h=[
        human("one","T01"),
        human("two","T02"),
    ]
    c=[source("a",20,70,40),source("shared",85,130,40),
       source("c",146,180,40)]
    result=discover_human_trace_piece_boundaries(
        human_strokes=h,observed_components=c
    )
    assert result["assessable_unique_detected_piece_adjacencies"]==0
    assert result["ambiguous_unique_piece_adjacencies_abstained"]==2
    assert result["source_components_touching_multiple_human_visual_instances"]==[
        "a","c","shared"
    ]


def test_gap_threshold_and_no_observed_fragments_do_not_fake_recall():
    h=[human("long","T01",x0=20,x1=300)]
    c=[source("A",20,70,40),source("B",220,300,40)]
    strict=discover_human_trace_piece_boundaries(
        human_strokes=h,observed_components=c,
        max_unobserved_gap_along_trace_px=50
    )
    assert strict["assessable_unique_detected_piece_adjacencies"]==0
    loose=discover_human_trace_piece_boundaries(
        human_strokes=h,observed_components=c,
        max_unobserved_gap_along_trace_px=200
    )
    assert loose["assessable_unique_detected_piece_adjacencies"]==1
    assert loose["fixed_method_parameters"]["max_unobserved_gap_along_trace_px"]==200
    absent=discover_human_trace_piece_boundaries(
        human_strokes=h,observed_components=[]
    )
    assert absent["human_clear_positive_strokes"]==1
    assert absent["per_human_stroke"][0]["unique_stable_detected_pieces"]==0
    assert absent["assessable_unique_detected_piece_adjacencies"]==0
    assert measure_proposal_recovery(absent,proposal_sets={
        "S92_GAP":[]
    })["union_selected_conditional_coverage"] is None


def test_source_budget_duplicate_id_bad_proposal_and_bad_settings_fail_closed():
    h=[human("one","T01")]
    c=[source("a",20,70,40),source("b",85,130,40)]
    with pytest.raises(ValueError,match="duplicated"):
        discover_human_trace_piece_boundaries(
            human_strokes=h,observed_components=c+[c[0]]
        )
    with pytest.raises(ValueError,match="threshold"):
        discover_human_trace_piece_boundaries(
            human_strokes=h,observed_components=c,ambiguity_margin_px=99.
        )
    x=discover_human_trace_piece_boundaries(
        human_strokes=h,observed_components=c
    )
    with pytest.raises(ValueError,match="physical truth"):
        measure_proposal_recovery(x,proposal_sets={
            "S92_GAP":[{**proposal("a","b","S92_GAP"),"physical_auto_join":True}]
        })
    with pytest.raises(ValueError,match="Unsupported"):
        measure_proposal_recovery(x,proposal_sets={"PHYSICS_TRUE":[]})
    with pytest.raises(ValueError,match="Self"):
        measure_proposal_recovery(x,proposal_sets={
            "S92_GAP":[proposal("a","a","S92_GAP")]
        })



def test_full_curved_coil_reference_detects_disconnected_observed_arms_without_parabola():
    # Curved coil is a growing polar spiral: cannot be represented by a
    # single endpoint-constrained quadratic without losing its turning.
    theta=np.linspace(0.,2.5*np.pi,140)
    radius=9.+4.*theta
    xy=np.column_stack((
        160.+radius*np.cos(theta),
        160.+radius*np.sin(theta),
    ))
    expert=[{
        "stroke_id":"human_coiling","visual_instance_id":"VISUAL_COIL_1",
        "kind":"COILING_TRACK_LIKE","certainty":"CLEAR",
        "points_xy_source_px":xy.tolist()
    }]
    fragments=[
        {"component_id":"arm1","points_xy":xy[:36].copy()},
        {"component_id":"arm2","points_xy":xy[43:90].copy()},
        {"component_id":"arm3","points_xy":xy[97:].copy()},
    ]
    scored=discover_human_trace_piece_boundaries(
        human_strokes=expert,observed_components=fragments,
        radius_px=1.8,min_run_support_px=3.,
        max_unobserved_gap_along_trace_px=80
    )
    assert scored["human_clear_positive_strokes"]==1
    assert scored["per_human_stroke"][0]["human_visible_type"]=="COILING_TRACK_LIKE"
    assert scored["per_human_stroke"][0]["unique_stable_detected_pieces"]==3
    assert scored["assessable_unique_detected_piece_adjacencies"]==2
    assert {
        frozenset((x["component_a"],x["component_b"]))
        for x in scored["conditionally_assessable_piece_adjacencies"]
    }=={frozenset(("arm1","arm2")),frozenset(("arm2","arm3"))}
    # The algorithm is independent of any fitted quadratic, and does not
    # infer a particle type, neutron cause or an accepted true physics join.
    recovery=measure_proposal_recovery(
        scored,proposal_sets={"S92_GAP":[]}
    )
    assert recovery["union_selected_conditional_coverage"]==0.
    assert recovery["partial_human_visible_not_particle_GT"] is True


def test_proposed_edge_outside_source_S86_fails_when_exact_source_inventory_provided():
    scored=discover_human_trace_piece_boundaries(
        human_strokes=[human("manual","visual")],
        observed_components=[
            source("sourceA",20,70,40),
            source("sourceB",85,130,40),
        ]
    )
    with pytest.raises(ValueError,match="unknown original S86 source"):
        measure_proposal_recovery(
            scored,proposal_sets={
                "S92_GAP":[proposal("sourceA","invented","S92_GAP")]
            },
            source_observed_component_ids={"sourceA","sourceB"},
        )
