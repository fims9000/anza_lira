"""S92 geometry contracts: do not erase short true tracks or falsely certify gaps."""
import math

import numpy as np
import pytest

from bubble_chamber.s92_curve_geometry import (
    TraceReviewConfig, construct_chain_hypotheses, describe_trace,
    propose_gap_links, resample,
)


def _line(x0,x1,y):
    return np.column_stack((np.arange(x0,x1+1,dtype=float),
                            np.full(x1-x0+1,y,dtype=float)))


def _curve(t, radius):
    return np.column_stack((radius*np.cos(t),radius*np.sin(t)))


def test_short_tick_and_hairpin_are_review_advisories_not_deleted():
    short=_line(10,26,20)
    outcome=describe_trace(short)
    assert outcome["shape_review_category"]=="LOW_PRIORITY_POSSIBLE_ARTIFACT"
    assert "SHORT_STROKE_NOT_VERIFIED" in outcome["review_flags"]
    assert outcome["physical_identity"]=="UNKNOWN"
    t=np.linspace(0,math.pi,65)
    hair=_curve(t,12)
    flagged=describe_trace(hair)
    assert flagged["shape_review_category"]=="LOW_PRIORITY_POSSIBLE_ARTIFACT"
    assert "COMPACT_HAIRPIN_ARTIFACT_POSSIBLE" in flagged["review_flags"]
    assert flagged["not_a_particle_classification"] is True


def test_long_coiling_spiral_not_suppressed_as_short_hair():
    t=np.linspace(0,4*math.pi,340)
    r=4+0.65*t
    curve=np.column_stack((r*np.cos(t)+150,r*np.sin(t)+120))
    report=describe_trace(curve)
    assert report["observed_length_px"]>72
    assert abs(report["total_signed_turn_deg"])>255
    assert report["shape_review_category"]=="COILING_TRACE_REVIEW"
    assert report["turn_direction_coherence"]>.72
    opposite=describe_trace(curve[::-1])
    assert opposite["shape_review_category"]=="COILING_TRACE_REVIEW"
    mirrored=describe_trace(np.column_stack((-curve[:,0],curve[:,1])))
    assert mirrored["shape_review_category"]=="COILING_TRACE_REVIEW"


def test_long_smooth_parabolic_segment_kept_for_review():
    x=np.linspace(0,120,160)
    curve=np.column_stack((x,.0015*(x-60)**2+20))
    m=describe_trace(curve)
    assert m["shape_review_category"]=="EXTENDED_IMAGE_TRACE_CANDIDATE"
    assert m["physical_identity"]=="UNKNOWN"


def test_two_dashed_parts_are_proposed_as_single_reversible_chain():
    pieces=[
        {"component_id":"a","points_xy":_line(15,45,30)},
        {"component_id":"b","points_xy":_line(54,85,30)},
    ]
    first=[p.copy() for p in (pieces[0]["points_xy"],pieces[1]["points_xy"])]
    proposals,review,meta=propose_gap_links(pieces)
    assert len(review)==1
    assert meta["gap_review_links"]==1
    assert review[0]["gap_px"]==pytest.approx(9)
    assert review[0]["inferred_only"] is True
    assert review[0]["physical_auto_join"] is False
    chains=construct_chain_hypotheses(pieces,review)
    assert len(chains)==1
    assert len(chains[0]["observed_sections_xy"])==2
    assert len(chains[0]["inferred_gaps"])==1
    assert chains[0]["not_a_verified_particle_track"] is True
    for a,b in zip(pieces,first):
        assert np.array_equal(a["points_xy"],b)


def test_curved_gap_geometry_supports_arc_not_just_linear_heading():
    t1=np.linspace(0,.62,40)
    t2=np.linspace(.94,1.5,40)
    p=_curve(t1,36)+np.array([80,80])
    q=_curve(t2,36)+np.array([80,80])
    proposals,chosen,_=propose_gap_links([
        {"component_id":"before","points_xy":p},
        {"component_id":"after","points_xy":q},
    ],TraceReviewConfig(max_gap_px=30))
    assert len(chosen)==1
    assert chosen[0]["bridge_turn_deg"]>0
    assert chosen[0]["gap_px"]<30
    assert all(row["status"]=="UNKNOWN" for row in proposals)


def test_two_plausible_competitors_abstain_without_random_tiebreak():
    parts=[
        {"component_id":"a","points_xy":_line(10,40,40)},
        {"component_id":"b","points_xy":_line(52,86,34)},
        {"component_id":"c","points_xy":_line(52,86,46)},
    ]
    proposals,selected,_=propose_gap_links(parts)
    assert selected==[]
    assert any(x["review_decision"] in (
        "AMBIGUOUS_PARTNER_ABSTAIN","THIRD_FRAGMENT_NEAR_GAP_ABSTAIN"
    ) for x in proposals)


def test_bridge_through_observed_third_track_is_not_selected():
    vertical=np.column_stack((np.full(41,50.),np.arange(20,61,dtype=float)))
    parts=[
        {"component_id":"left","points_xy":_line(14,44,40)},
        {"component_id":"right","points_xy":_line(56,90,40)},
        {"component_id":"crossing","points_xy":vertical},
    ]
    proposals,selected,_=propose_gap_links(parts)
    crossing_proposals=[
        x for x in proposals if {x["component_a"],x["component_b"]}=={"left","right"}
    ]
    assert crossing_proposals
    assert crossing_proposals[0]["crosses_other_observed_fragment"] is True
    assert crossing_proposals[0]["review_decision"]=="THIRD_FRAGMENT_NEAR_GAP_ABSTAIN"
    assert all({x["component_a"],x["component_b"]}!={"left","right"} for x in selected)


def test_dense_rivals_fail_closed_and_invalid_input_rejected():
    arr=[{"component_id":f"x{i}","points_xy":_line(10,50,30+i*.05)}
         for i in range(12)]
    _,sel,report=propose_gap_links(arr,TraceReviewConfig(max_near_pairs=10))
    assert sel==[]
    assert report["abstain_reason"]=="DENSE_ENDPOINT_NEIGHBORHOOD"
    with pytest.raises(ValueError,match="Duplicate"):
        propose_gap_links([arr[0],arr[0]])
    with pytest.raises(ValueError,match="Repeated"):
        describe_trace(np.array([[1,1],[1,1],[2,1],[3,1]]))


def test_bridge_endpoints_and_metric_are_translation_rotation_invariant():
    a=_line(10,40,40)
    b=_line(50,80,40)
    base,links,_=propose_gap_links([
        {"component_id":"x","points_xy":a},{"component_id":"y","points_xy":b}
    ])
    theta=.7
    R=np.array([[math.cos(theta),-math.sin(theta)],
                [math.sin(theta),math.cos(theta)]])
    transform=lambda z: z@R.T+np.array([400,300])
    rotated,rel,_=propose_gap_links([
        {"component_id":"x","points_xy":transform(a)},
        {"component_id":"y","points_xy":transform(b)}
    ])
    assert len(links)==len(rel)==1
    assert rel[0]["gap_px"]==pytest.approx(links[0]["gap_px"],abs=1e-3)
    assert rel[0]["geometry_score"]==pytest.approx(links[0]["geometry_score"],abs=.002)


def test_three_short_dashes_stitch_into_one_unknown_long_structure():
    from bubble_chamber.s92_curve_review import assign_display_priority
    pieces=[
        {"component_id":"dash_1","points_xy":_line(10,27,30)},
        {"component_id":"dash_2","points_xy":_line(39,56,30)},
        {"component_id":"dash_3","points_xy":_line(68,85,30)},
    ]
    proposals,review,stats=propose_gap_links(pieces)
    assert stats["gap_review_links"]==2
    chains=construct_chain_hypotheses(pieces,review)
    assert len(chains)==1
    assert len(chains[0]["ordered_component_ids"])==3
    assert len(chains[0]["inferred_gaps"])==2
    assert all(len(x)>5 for x in chains[0]["observed_sections_xy"])
    assert chains[0]["shape_if_all_gaps_real"]["observed_length_px"]>70
    rows=[{"component_id":c["component_id"],
           "shape_review_category":describe_trace(c["points_xy"])["shape_review_category"]}
          for c in pieces]
    stats=assign_display_priority(rows,chains)
    assert stats["PROVISIONAL_GAP_CHAIN_RESCUE_NOT_GT"]==3
    assert all(x["show_in_primary_view"] is True for x in rows)
    assert all(x["original_source_untouched"] for x in rows)


def test_short_hairpin_bridge_across_parallel_tracks_requires_observed_curvature():
    # A 153-degree bridge would produce a fake short U/hair between two
    # unrelated straight parallel strokes. S92 never selects it.
    b=_line(50,85,34)
    c=_line(50,85,46)
    props,sel,_=propose_gap_links([
        {"component_id":"hair_left","points_xy":b},
        {"component_id":"hair_right","points_xy":c},
    ])
    assert sel==[]
    assert all(p["bridge_turn_deg"]<=110 or
               p["review_decision"]!="PRIORITY_GAP_REVIEW_NOT_PHYSICAL_ACCEPT"
               for p in props)


def test_medium_size_v_mark_goes_to_review_without_dropping_true_short_track():
    # A 40-50px hand-drawn tick was not covered by the tiny-stroke cutoff.
    rising=np.column_stack((np.linspace(10,23,26),np.linspace(40,25,26)))
    falling=np.column_stack((np.linspace(23.2,36,26),np.linspace(25.2,40,26)))
    glyph=np.concatenate([rising,falling],axis=0)
    m=describe_trace(glyph)
    assert 24<m["observed_length_px"]<72
    assert m["shape_review_category"]=="LOW_PRIORITY_POSSIBLE_ARTIFACT"
    assert "SHORT_ANGULAR_GLYPH_POSSIBLE" in m["review_flags"]
    assert m["physical_identity"]=="UNKNOWN"
