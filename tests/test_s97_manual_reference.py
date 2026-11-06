"""S97 positive-only native-image-reference geometry; synthetic coordinates.

Neither 3D particle labels, verified neutron interactions, nor real H50Z photos
are available in these tests. Do not report synthetic scores as native quality.
"""
from __future__ import annotations
import copy
import numpy as np
import pytest

from bubble_chamber.s97_manual_reference import (
    SCHEMA,POSITIVE_KINDS,validate_reference_annotation,
    evaluate_partial_positive_coverage,paired_coverage_contrasts,
)
from bubble_chamber.s97_link_audit import (
    map_observed_components_to_reference,audit_provisional_joins
)


def _original_pack():
    return {
        "schema":SCHEMA,
        "source_bmp_name":"ORIGINAL_SYNTHETIC_NOT_H50Z.bmp",
        "source_bmp_sha256":"a"*64,
        "image_shape_yx":[300,350],
        "roi_xyxy_source":[[10,10,200,200]],
    }


def _line(x0,x1,y,step=10):
    x=np.arange(x0,x1+step*.5,step,dtype=float)
    return np.column_stack([x,np.full_like(x,y)])


def _human_stroke(sid,instance,kind="TRACK_LIKE",y=40,certainty="CLEAR"):
    return {
        "stroke_id":sid,"visual_instance_id":instance,
        "kind":kind,"certainty":certainty,
        "points_xy_source_px":_line(20,120,y).tolist(),
        "note":"drawn on independent synthetic original pixel image",
    }


def _review(strokes):
    return {
        **_original_pack(),
        "source_reference_pack_sha256":"0"*64,
        "reviewer_id":"synthetic_expert_a",
        "image_was_unmodified_original":True,
        "machine_proposals_visible_during_annotation":False,
        "full_exhaustive_negative_segmentation":False,
        "data_source":"INDEPENDENT_OPERATOR_ORIGINAL_BMP_ONLY",
        "reference_semantics":"PARTIAL_VISIBLE_2D_STROKES_NO_PARTICLE_GT",
        "strokes":strokes,
    }


def _component(cid,xy):
    return {"component_id":cid,"points_xy":np.asarray(xy,float)}


def test_s97_accepts_multiple_partial_types_and_never_promotes_unknown_to_negative():
    original=[
        _human_stroke("track1","T01"),
        _human_stroke("coil1","T02","COILING_TRACK_LIKE",y=80),
        _human_stroke("scratch","A01","ARTIFACT",y=140),
        _human_stroke("uncertain","T03","TRACK_LIKE",y=160,certainty="UNCERTAIN"),
    ]
    data=validate_reference_annotation(_review(original),_original_pack())
    assert len(data)==4
    assert all(r["human_source"].startswith("ORIGINAL") for r in data)
    machine=[
        _component("source1",_line(20,60,40,step=1)),
        _component("source2",_line(20,120,80,step=1)),
        _component("noise",_line(20,120,140,step=1)),
    ]
    r=evaluate_partial_positive_coverage(data,machine,radius_px=1.5)
    assert r["reference_clear_track_or_coil_strokes"]==2
    assert r["observed_positive_centerline_coverage"] is not None
    assert .60<r["observed_positive_centerline_coverage"]<.75
    by_id={x["stroke_id"]:x for x in r["per_human_stroke"]}
    assert by_id["track1"]["max_uncovered_span_px"]>40
    assert by_id["track1"]["uncovered_spans_at_least_4px"]>=1
    assert by_id["coil1"]["observed_S86_centerline_coverage"]==1.
    assert "scratch" not in by_id and "uncertain" not in by_id
    assert r["true_precision_fpr_f1_unavailable_without_exhaustive_negative_GT"]


def test_s97_empty_machine_and_zero_human_positive_are_not_fake_metrics():
    good=validate_reference_annotation(
        _review([_human_stroke("main","T1")]),_original_pack()
    )
    all_missing=evaluate_partial_positive_coverage(good,[])
    assert all_missing["observed_positive_centerline_coverage"]==0.
    assert all_missing["source_observed_centerline_points"]==0
    assert all_missing["per_human_stroke"][0]["max_uncovered_span_px"]>=100
    only_artifacts=validate_reference_annotation(
        _review([_human_stroke("garbage","garbage","ARTIFACT")]),_original_pack()
    )
    r=evaluate_partial_positive_coverage(only_artifacts,[])
    assert r["reference_samples_total"]==0
    assert r["observed_positive_centerline_coverage"] is None
    assert r["per_human_stroke"]==[]


def test_s97_strict_exact_photo_and_sparse_stroke_sampling_contract():
    one=_review([_human_stroke("s1","T1")])
    wrong=copy.deepcopy(one)
    wrong["source_bmp_sha256"]="f"*64
    with pytest.raises(ValueError,match="different BMP"):
        validate_reference_annotation(wrong,_original_pack())
    wrong=copy.deepcopy(one)
    wrong["machine_proposals_visible_during_annotation"]=True
    with pytest.raises(ValueError,match="original-only"):
        validate_reference_annotation(wrong,_original_pack())
    wrong=copy.deepcopy(one)
    wrong["full_exhaustive_negative_segmentation"]=True
    with pytest.raises(ValueError,match="non-exhaustive"):
        validate_reference_annotation(wrong,_original_pack())
    wrong=copy.deepcopy(one)
    wrong["strokes"][0]["points_xy_source_px"]=[[20.,40.],[95.,40.]]
    with pytest.raises(ValueError,match="32px"):
        validate_reference_annotation(wrong,_original_pack())
    wrong=copy.deepcopy(one)
    wrong["strokes"][0]["points_xy_source_px"]=[[20.,40.],[20.,40.]]
    with pytest.raises(ValueError,match="Repeated"):
        validate_reference_annotation(wrong,_original_pack())
    wrong=copy.deepcopy(one)
    wrong["strokes"].append(wrong["strokes"][0])
    with pytest.raises(ValueError,match="Duplicate"):
        validate_reference_annotation(wrong,_original_pack())
    wrong=copy.deepcopy(one)
    wrong["strokes"][0]["points_xy_source_px"]=[[20.,40.],[21.,600.]]
    with pytest.raises(ValueError,match="outside"):
        validate_reference_annotation(wrong,_original_pack())


def test_matched_s97_same_stroke_denominators_only_descriptive_differences():
    reference=validate_reference_annotation(
        _review([_human_stroke("T1","unique")]),_original_pack()
    )
    baseline=evaluate_partial_positive_coverage(reference,[
        _component("fragment",_line(20,60,40,step=1))
    ])
    challenger=evaluate_partial_positive_coverage(reference,[
        _component("fragment",_line(20,120,40,step=1))
    ])
    baseline["candidate_name"]="S88"
    challenger["candidate_name"]="AZUNET"
    diff=paired_coverage_contrasts([baseline,challenger])
    assert len(diff)==1 and diff[0]["descriptive_delta_pp"]>40.
    assert diff[0]["no_statistical_significance_claim"] is True
    incompatible=copy.deepcopy(challenger)
    incompatible["per_human_stroke"]=[]
    with pytest.raises(ValueError,match="same expert"):
        paired_coverage_contrasts([baseline,incompatible])


def test_partial_human_visual_instances_identify_only_supported_join_disagreement():
    reference=validate_reference_annotation(
        _review([
            _human_stroke("t1","one","TRACK_LIKE",y=40),
            _human_stroke("t2","two","TRACK_LIKE",y=80),
            _human_stroke("scratch","scratch","ARTIFACT",y=120),
        ]),_original_pack()
    )
    fragments=[
        _component("part-a",_line(20,50,40,step=2)),
        _component("part-b",_line(55,120,40,step=2)),
        _component("part-c",_line(20,120,80,step=2)),
        _component("part-d",_line(20,120,120,step=2)),
        _component("unlabelled",_line(240,275,240,step=2)),
    ]
    labels,summary=map_observed_components_to_reference(reference,fragments)
    assert summary["matched_components"]==4
    assert labels["part-a"]["visual_instance_id"]=="one"
    assert labels["part-b"]["visual_instance_id"]=="one"
    assert labels["part-c"]["visual_instance_id"]=="two"
    assert labels["part-d"]["human_kind"]=="ARTIFACT"
    assert labels["unlabelled"] is None

    def edge(a,b,decision):
        return {
            "component_a":a,"component_b":b,
            "review_decision":decision,"status":"UNKNOWN",
            "physical_auto_join":False,
        }
    s92=[
        edge("part-a","part-b","PRIORITY_GAP_REVIEW_NOT_PHYSICAL_ACCEPT"),
        edge("part-a","part-c","PRIORITY_GAP_REVIEW_NOT_PHYSICAL_ACCEPT"),
        edge("part-b","unlabelled","PRIORITY_GAP_REVIEW_NOT_PHYSICAL_ACCEPT"),
        edge("part-a","part-d","AMBIGUOUS_PARTNER_ABSTAIN"),
    ]
    r=audit_provisional_joins(
        edges=s92,component_labels=labels,source_kind="S92_GAP"
    )
    assert r["selected_review_level_links"]==3
    assert r["assessable_source_annotated_links"]==2
    assert r["consistent_with_independent_human_2d_instances"]==1
    assert r["contradicts_independent_human_2d_instances"]==1
    assert r["abstained_unreviewable_links"]==1
    assert r["conditional_disagreement_fraction"]==.5
    assert r["full_unproposed_pair_recall_not_measured"] is True
    s96=[edge("part-b","part-c","PROVISIONAL_2D_CROSSING_CONTINUATION")]
    q=audit_provisional_joins(edges=s96,component_labels=labels,source_kind="S96_CROSSING")
    assert q["conditional_disagreement_fraction"]==1.
    with pytest.raises(ValueError,match="physically asserted"):
        audit_provisional_joins(
            edges=[{**s92[0],"physical_auto_join":True}],
            component_labels=labels,source_kind="S92_GAP",
        )
    with pytest.raises(ValueError,match="non-source"):
        audit_provisional_joins(
            edges=[edge("not-source","part-b","PRIORITY_GAP_REVIEW_NOT_PHYSICAL_ACCEPT")],
            component_labels=labels,source_kind="S92_GAP",
        )
