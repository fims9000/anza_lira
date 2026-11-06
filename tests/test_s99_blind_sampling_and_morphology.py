"""S99 reproducible, source-only spatial sample and human-first class audit.

All geometry is synthetic and manually defined in this test. Passing this
suite is NOT an empirical H50Z detection/classification quality statement.
"""
from __future__ import annotations

import numpy as np
import pytest

from bubble_chamber.s99_blind_sampling import (
    SCHEMA,select_blind_original_source_rois
)
from bubble_chamber.s99_morphology_audit import (
    SCHEMA as AUDIT_SCHEMA,
    EXPECTED_S95,VALID_MACHINE_TYPES,
    audit_s95_against_blind_source_strokes,
    descriptive_same_photo_morphology_contrasts
)


def _s95(id,kind="TRACK_LIKE_CANDIDATE"):
    return {
        "component_id":id,
        "primary_review_category":kind,
        "preserved_source_observed":True,
        "physical_auto_join_authorized":False,
    }


def _source(id,x0,x1,y):
    return {
        "component_id":id,
        "points_xy":np.column_stack((
            np.arange(x0,x1+1,dtype=float),
            np.full(x1-x0+1,y,dtype=float)
        ))
    }


def _human(sid,kind,y,x0=20,x1=120,certainty="CLEAR"):
    return {
        "stroke_id":sid,"kind":kind,"certainty":certainty,
        "visual_instance_id":sid,
        "points_xy_source_px":[[float(x),float(y)] for x in range(x0,x1+1,10)],
    }


def test_source_sha_only_design_is_stable_balanced_without_machine_labels():
    first=select_blind_original_source_rois(
        height=24744,width=8358,source_sha256="a"*64,
        seed=20261106,tile_size=512,roi_count=18
    )
    again=select_blind_original_source_rois(
        height=24744,width=8358,source_sha256="a"*64,
        seed=20261106,tile_size=512,roi_count=18
    )
    assert first==again
    assert first["schema"]==SCHEMA
    assert first["sampled_source_roi_count"]==18
    assert first["strata_grid_yx"]==[3,3]
    assert first["selected_spatial_stratum_counts"]=={
        f"{r},{c}":2 for r in range(3) for c in range(3)
    }
    assert first["not_uniform_random_source_pixel_sampling"]
    boxes=[tuple(x["xyxy_source"]) for x in first["sampled_cell_selection"]]
    assert len(set(boxes))==18
    for x0,y0,x1,y1 in boxes:
        assert 0<=x0<x1<=8358
        assert 0<=y0<y1<=24744
        assert x1-x0<=512 and y1-y0<=512
    assert all(x["model_scores_used_to_select_ROI"] is False
               for x in first["sampled_cell_selection"])
    other_seed=select_blind_original_source_rois(
        height=24744,width=8358,source_sha256="a"*64,
        seed=20261107,tile_size=512,roi_count=18
    )
    assert {
        tuple(x["xyxy_source"]) for x in other_seed["sampled_cell_selection"]
    }!={tuple(x["xyxy_source"]) for x in first["sampled_cell_selection"]}


def test_predeclared_reviewer_budget_and_grid_dimensions_fail_closed():
    base=dict(height=256,width=384,source_sha256="a"*64,tile_size=128)
    with pytest.raises(ValueError,match="every nonempty spatial stratum"):
        select_blind_original_source_rois(**base,roi_count=3,strata_rows=2,strata_cols=2)
    with pytest.raises(ValueError,match="more independent source cells"):
        select_blind_original_source_rois(**base,roi_count=7,strata_rows=2,strata_cols=2)
    with pytest.raises(ValueError,match="hexadecimal"):
        select_blind_original_source_rois(**{**base,"source_sha256":"z"*64},roi_count=4)
    with pytest.raises(ValueError,match="integers"):
        select_blind_original_source_rois(**base,seed=1.5,roi_count=4)
    with pytest.raises(ValueError,match="invalid ROI"):
        select_blind_original_source_rois(**base,tile_size=100,roi_count=4)


def test_original_source_human_morphology_and_missing_are_separate():
    humans=[
        _human("physics_line","TRACK_LIKE",40),
        _human("human_artifact","ARTIFACT",80),
        _human("human_coil","COILING_TRACK_LIKE",120),
        _human("human_doubtful","TRACK_LIKE",160,certainty="UNCERTAIN"),
        _human("human_unknown","UNKNOWN",200),
    ]
    source=[
        _source("piece_track_a",20,60,40),
        _source("piece_track_b",71,120,40),
        _source("piece_artifact",20,120,80),
        _source("piece_coil",20,120,120),
    ]
    typed=[
        _s95("piece_track_a","TRACK_LIKE_CANDIDATE"),
        _s95("piece_track_b","TRACK_LIKE_CANDIDATE"),
        _s95("piece_artifact","TRACK_LIKE_CANDIDATE"), # machine wrong vs human
        _s95("piece_coil","COILING_TRACE_CANDIDATE"),
    ]
    r=audit_s95_against_blind_source_strokes(
        humans,source,typed,radius_px=1.5
    )
    assert r["schema"]==AUDIT_SCHEMA
    assert r["source_human_clear_classified_strokes"]==3
    assert len(r["per_human_stroke"])==3
    assert r["per_human_morphology"]["TRACK_LIKE"]["human_strokes"]==1
    assert r["per_human_morphology"]["TRACK_LIKE"][
        "reference_S86_observed_coverage_conditional_on_this_kind"
    ]<1
    assert r["per_human_morphology"]["ARTIFACT"][
        "S95_morphology_agreement_among_assigned_source_points"
    ]==0.
    assert r["per_human_morphology"]["COILING_TRACK_LIKE"][
        "S95_morphology_agreement_among_assigned_source_points"
    ]==1.
    assert r["per_human_morphology"]["FIDUCIAL_CROSS"][
        "S95_morphology_agreement_among_assigned_source_points"
    ] is None
    assert r["human_uncertain_or_unknown_strokes_excluded"]=={
        "TRACK_LIKE":1,"UNKNOWN":1
    }
    for h in r["per_human_stroke"]:
        assert h["human_samples_assigned_plus_unobserved_plus_ambiguous"]==h[
            "human_reference_sampled_points"
        ]
    assert r["unreviewed_and_unmarked_original_pixels_not_negative_labels"]


def test_absent_detector_is_zero_source_coverage_but_not_perfect_machine_class():
    reference=[_human("T","TRACK_LIKE",40)]
    empty=audit_s95_against_blind_source_strokes(reference,[],[])
    row=empty["per_human_morphology"]["TRACK_LIKE"]
    assert row["human_strokes"]==1
    assert row["reference_S86_observed_coverage_conditional_on_this_kind"]==0.
    assert row["S95_morphology_agreement_among_assigned_source_points"] is None
    assert empty["source_component_inventory_count"]==0
    assert empty["human_samples_total_across_explicit_positive_and_explicit_artifact_strokes"]>0


def test_source_type_identity_and_physics_claims_refused():
    reference=[_human("T","TRACK_LIKE",40)]
    src=[_source("src",20,120,40)]
    with pytest.raises(ValueError,match="labels must cover"):
        audit_s95_against_blind_source_strokes(reference,src,[])
    with pytest.raises(ValueError,match="duplicate"):
        audit_s95_against_blind_source_strokes(reference,src,[_s95("src"),_s95("src")])
    with pytest.raises(ValueError,match="physical GT"):
        audit_s95_against_blind_source_strokes(
            reference,src,[{**_s95("src"),"physical_auto_join_authorized":True}]
        )
    with pytest.raises(ValueError,match="type"):
        audit_s95_against_blind_source_strokes(
            reference,src,[_s95("src","TRUE_NEUTRON_PARTICLE")]
        )
    with pytest.raises(ValueError,match="Duplicate original"):
        audit_s95_against_blind_source_strokes(
            reference+reference,src,[_s95("src")]
        )


def test_same_original_strokes_descriptive_only_not_significance():
    human=[_human("T","TRACK_LIKE",40)]
    first=audit_s95_against_blind_source_strokes(
        human,[_source("a",20,60,40)],[_s95("a")]
    )
    second=audit_s95_against_blind_source_strokes(
        human,[_source("b",20,120,40)],[_s95("b")]
    )
    delta=descriptive_same_photo_morphology_contrasts([first,second])
    assert delta[0]["human_morphology_conditional_differences"]["TRACK_LIKE"][
        "delta_observed_source_coverage_percentage_points"
    ]>30.
    assert delta[0]["independent_event_level_CI_not_available"] is True
    assert delta[0]["human_morphology_conditional_differences"]["ARTIFACT"][
        "delta_observed_source_coverage_percentage_points"
    ] is None
    wrong=audit_s95_against_blind_source_strokes(
        [_human("NOT_SAME","TRACK_LIKE",40)],
        [_source("b",20,120,40)],[_s95("b")]
    )
    with pytest.raises(ValueError,match="SAME original"):
        descriptive_same_photo_morphology_contrasts([first,wrong])
