"""S100 synthetic 2D X geometry; geometrically cross-shaped != fiducial GT."""
from __future__ import annotations

from math import cos,sin,pi
import numpy as np
import pytest

from bubble_chamber.s96_junction_graph import propose_four_arm_continuations
from bubble_chamber.s100_cross_geometry import (
    SCHEMA,SOURCE_ORIGINAL,SOURCE_MASK,ACCEPT,XShapeConfig,
    score_x_shape_junction,build_x_shape_candidate_inventory
)


def _lines(center=(100,100),r=5.,heading=0.,delta=pi/2):
    cx,cy=center
    rays=[]
    for i,ang in enumerate((heading,heading+pi,heading+delta,heading+delta+pi)):
        ray=np.array([cos(ang),sin(ang)])
        near=np.array(center)+r*ray
        far=np.array(center)+40*ray
        rays.append({
            "component_id":f"arm_{i}",
            "points_xy":np.linspace(near,far,41)
        })
    return rays


def _junction(*,delta=pi/2,heading=0.,center=(100,100)):
    a,b,c=propose_four_arm_continuations(
        _lines(center=center,heading=heading,delta=delta)
    )
    assert a and c["candidate_junctions"]>=1
    return a[0]


def test_X_shape_without_any_fiducial_mark_and_rotation_invariance():
    base=_junction()
    for angle in (0.,.31,.8,1.5):
        row=_junction(heading=angle,center=(230,240))
        score=score_x_shape_junction(
            row,representation=SOURCE_ORIGINAL,
            source_bmp_sha256="a"*64
        )
        assert score["cross_shape_review_status"]==ACCEPT
        assert score["crossing_acute_axes_angle_deg"]>=89
        assert score["max_opposite_turn_deflection_deg"]<.02
        assert score["best_vs_second_pairing_margin_deg"]>100
        assert {frozenset(x) for x in score[
            "paired_source_arm_ids_proposed_visual_only"
        ]}=={
            frozenset(("arm_0","arm_1")),
            frozenset(("arm_2","arm_3"))
        }
        assert score["source_representation"]==SOURCE_ORIGINAL
        assert score["not_automatic_calibration_fiducial_identity"] is True
        assert score["overlap_vs_interaction_vs_fiducial_undecidable"] is True
        assert score["physical_auto_join_authorized"] is False
        assert score["model_only_not_independent_expert_truth"] is True


def test_three_arm_Y_bundle_of_five_and_duplicate_component_X_abstain():
    rows=_lines()
    three,_,_=propose_four_arm_continuations(rows[:3])
    y=score_x_shape_junction(
        three[0],representation=SOURCE_ORIGINAL,source_bmp_sha256="b"*64
    )
    assert y["cross_shape_review_status"]=="NON_FOUR_ARM_ABSTAIN"
    assert y["paired_source_arm_ids_proposed_visual_only"]==[]
    extra=[*rows,{
        "component_id":"fifth",
        "points_xy":np.linspace([98,98],[65,65],41)
    }]
    five,_,_=propose_four_arm_continuations(extra)
    assert five
    score=score_x_shape_junction(
        five[0],representation=SOURCE_ORIGINAL,source_bmp_sha256="b"*64
    )
    assert score["cross_shape_review_status"]=="NON_FOUR_ARM_ABSTAIN"
    duplicated=_junction()
    duplicated["ports"][1]["component_id"]=duplicated["ports"][0]["component_id"]
    score=score_x_shape_junction(
        duplicated,representation=SOURCE_ORIGINAL,source_bmp_sha256="b"*64
    )
    assert score["cross_shape_review_status"]=="DUPLICATE_SOURCE_ARM_ABSTAIN"


def test_shallow_cross_angle_and_non_four_arm_do_not_become_true_fiducials():
    shallow=_junction(delta=np.deg2rad(16.))
    score=score_x_shape_junction(
        shallow,representation=SOURCE_ORIGINAL,
        source_bmp_sha256="d"*64
    )
    assert score["cross_shape_review_status"]=="SHALLOW_AXIS_ANGLE_ABSTAIN"
    assert score["paired_source_arm_ids_proposed_visual_only"]==[]
    assert score["crossing_acute_axes_angle_deg"]<35


def test_spread_endpoint_tangent_and_malformed_source_fail_closed():
    row=_junction()
    row["ports"][0]["point_xy"]=[50.,50.]
    score=score_x_shape_junction(
        row,representation=SOURCE_ORIGINAL,source_bmp_sha256="e"*64
    )
    assert score["cross_shape_review_status"]=="SOURCE_PORT_TOO_FAR_FROM_JUNCTION_ABSTAIN"
    j=_junction()
    j["ports"][0]["outward_unit"]=[float("nan"),0]
    with pytest.raises(ValueError,match="finite"):
        score_x_shape_junction(j,representation=SOURCE_ORIGINAL,
                               source_bmp_sha256="e"*64)
    with pytest.raises(ValueError,match="representation"):
        score_x_shape_junction(_junction(),representation="HUMAN_FIDUCIAL_GT",
                               source_bmp_sha256="e"*64)
    with pytest.raises(ValueError,match="bounded"):
        XShapeConfig(min_axes_cross_angle_deg=99.).validate()
    with pytest.raises(ValueError,match="source BMP SHA256"):
        score_x_shape_junction(_junction(),representation=SOURCE_ORIGINAL,
                               source_bmp_sha256="wrong")


def test_mask_and_original_alternatives_remain_two_machine_representations():
    original=_junction()
    mask={**_junction(),"source_mask_tile_id":"t003"}
    raw,summary=build_x_shape_candidate_inventory(
        [original],[mask],source_bmp_sha256="f"*64
    )
    assert summary["schema"]==SCHEMA
    assert len(raw)==2
    assert summary["four_arm_geometric_X_review_candidates"]==2
    assert raw[0]["source_representation"]==SOURCE_ORIGINAL
    assert raw[1]["source_representation"]==SOURCE_MASK
    assert raw[0]["candidate_id"]!=raw[1]["candidate_id"]
    assert raw[1]["nearby_alternate_representation_candidates"][0]["candidate_id"]==raw[0]["candidate_id"]
    assert summary["original_alternative_representations_NOT_counted_as_distinct_physics_events"] is True
    with pytest.raises(ValueError,match="Duplicate S100"):
        build_x_shape_candidate_inventory(
            [original,original],[],source_bmp_sha256="f"*64
        )


def test_source_orientation_source_structure_and_margins_versus_nearly_collinear():
    src=_lines()
    reverse=[{**x,"points_xy":x["points_xy"][::-1]} for x in src]
    j1,_,_=propose_four_arm_continuations(src)
    j2,_,_=propose_four_arm_continuations(reverse)
    a=score_x_shape_junction(j1[0],representation=SOURCE_ORIGINAL,
                              source_bmp_sha256="a"*64)
    b=score_x_shape_junction(j2[0],representation=SOURCE_ORIGINAL,
                              source_bmp_sha256="a"*64)
    assert a["cross_shape_review_status"]==b["cross_shape_review_status"]==ACCEPT
    assert {
        frozenset(x) for x in a["paired_source_arm_ids_proposed_visual_only"]
    }=={
        frozenset(x) for x in b["paired_source_arm_ids_proposed_visual_only"]
    }
