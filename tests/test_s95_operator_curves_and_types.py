"""S95 true operator quadratic reconstruction and trust/ambiguity controls."""
from __future__ import annotations
import math
import numpy as np
import pytest

from bubble_chamber.s95_operator_curves import (
    SCHEMA,fit_measured_polyline,sample_operator_record,
    validate_operator_curve_package,operator_curve_support,
)
from bubble_chamber.s95_all_structure_types import (
    type_all_observed_components,human_mark_objects,
)
from bubble_chamber.s94_sparse_keypoint_guard import parse_photo_labels


def _line(x0,x1,y):
    return np.column_stack((
        np.linspace(x0,x1,61),np.full(61,y),
    ))


def test_human_parametric_parabola_has_exact_start_end_and_vertical_support():
    # x(u)=300-100u, y(u)=40+100u+200u(1-u); valid almost-vertical arc.
    row={
        "track_id":"expert-t01",
        "kind":"parametric_quadratic",
        "start_xy":[300.,40.],
        "end_xy":[200.,140.],
        "x_coefficients_c_b_a":[300.,-100.,0.],
        "y_coefficients_c_b_a":[40.,300.,-200.],
    }
    guide=sample_operator_record(row,shape_yx=(500,500),n=301)
    assert guide["curve_status"]=="EXPLICIT_OPERATOR_QUADRATIC_SOURCE"
    pts=np.asarray(guide["sampled_guide_xy_source"])
    assert np.allclose(pts[0],[300,40])
    assert np.allclose(pts[-1],[200,140])
    assert pts[150,1]>140  # curved away from straight segment
    assert guide["not_a_dense_pixel_mask"] is True
    assert guide["operator_geometry_reference_only"] is True
    with pytest.raises(ValueError,match="start/end disagree"):
        sample_operator_record({**row,"end_xy":[201,152]},shape_yx=(500,500))


def test_pca_local_parabola_is_not_mistaken_for_global_y_of_x():
    row={
        "track_id":"human-local",
        "kind":"local_quadratic",
        "principal_origin_xy":[200.,120.],
        "u_axis_xy":[0.,1.],
        "normal_axis_xy":[-1.,0.],
        "quadratic_v_a_b_c":[.04,0.,0.],
        "u_start_end":[-30.,30.],
        "start_xy":[164.,90.],
        "end_xy":[164.,150.],
    }
    g=sample_operator_record(row,shape_yx=(300,300))
    pts=np.asarray(g["sampled_guide_xy_source"])
    assert np.allclose(pts[0],[164,90])
    assert np.allclose(pts[-1],[164,150])
    assert pts[len(pts)//2,0]==pytest.approx(200,abs=.1)


def test_start_and_end_alone_cannot_establish_curve():
    row={"track_id":"unknown","start_xy":[4,3],"end_xy":[20,20]}
    with pytest.raises(ValueError,match="Unsupported operator curve type"):
        sample_operator_record(row,shape_yx=(100,100))
    with pytest.raises(ValueError,match="3 finite polynomial"):
        sample_operator_record({
            **row,"kind":"parametric_quadratic",
            "x_coefficients_c_b_a":[0,0],
            "y_coefficients_c_b_a":[3,1,0],
        },shape_yx=(100,100))


def test_operator_measured_points_fit_constrained_endpoint_parabola():
    u=np.linspace(0,1,19)
    x=22+105*u+12*u*(1-u)
    y=40+30*u+85*u*(1-u)
    pts=np.column_stack((x,y))
    fitted=fit_measured_polyline(pts,max_rms_px=1.)
    assert fitted["quadratic_fit_usable"] is True
    assert fitted["rms_to_measured_points_px"]<.4
    row={
        "track_id":"operator-measured-01",
        "kind":"measured_points",
        "points_xy":pts.tolist(),
        "start_xy":pts[0].tolist(),
        "end_xy":pts[-1].tolist(),
        "max_rms_px":1.0,
    }
    guide=sample_operator_record(row,shape_yx=(200,200))
    assert guide["curve_status"]=="FITTED_QUADRATIC_FROM_OPERATOR_MEASUREMENTS"
    result=np.asarray(guide["sampled_guide_xy_source"])
    assert np.linalg.norm(result[0]-pts[0])<1e-5
    assert np.linalg.norm(result[-1]-pts[-1])<1e-5


def test_operator_coil_is_not_represented_by_fake_parabola():
    t=np.linspace(0,3*np.pi,120)
    r=18+t*1.3
    pts=np.column_stack((120+r*np.cos(t),115+r*np.sin(t)))
    guide=sample_operator_record({
        "track_id":"human-coil",
        "kind":"measured_points",
        "start_xy":pts[0].tolist(),"end_xy":pts[-1].tolist(),
        "points_xy":pts.tolist(),"max_rms_px":1.,
    },shape_yx=(300,300))
    assert guide["curve_status"]=="MEASURED_POINTS_POLYLINE_NONQUADRATIC_NO_FORCED_PARABOLA"
    assert guide["model_parameters"]["quadratic_fit_usable"] is False


def test_alignment_fails_if_H50Z_photo_or_hash_differs_from_visualmetro():
    row={
        "track_id":"op01","kind":"parametric_quadratic",
        "start_xy":[10,20],"end_xy":[50,20],
        "x_coefficients_c_b_a":[10,40,0],
        "y_coefficients_c_b_a":[20,0,0],
    }
    payload={
        "schema":SCHEMA,
        "source_bmp_name":"H50Z_030_01_405.bmp",
        "source_bmp_sha256":"a"*64,
        "image_shape_yx":[300,300],
        "explicit_photo_alignment_verified":True,
        "curves":[row],
    }
    guides=validate_operator_curve_package(
        payload,source_bmp_name="H50Z_030_01_405.bmp",
        source_bmp_sha256="a"*64,source_shape_yx=(300,300),
    )
    assert len(guides)==1
    with pytest.raises(ValueError,match="mismatch"):
        validate_operator_curve_package(
            payload,source_bmp_name="H50Z_030_01_405.bmp",
            source_bmp_sha256="b"*64,source_shape_yx=(300,300))
    with pytest.raises(ValueError,match="independently verified"):
        validate_operator_curve_package(
            {**payload,"explicit_photo_alignment_verified":False},
            source_bmp_name="H50Z_030_01_405.bmp",
            source_bmp_sha256="a"*64,source_shape_yx=(300,300))


def test_operator_fragment_match_never_creates_negative_GT_from_empty_areas():
    row={
        "track_id":"human-one","kind":"parametric_quadratic",
        "start_xy":[10,20],"end_xy":[90,20],
        "x_coefficients_c_b_a":[10,80,0],
        "y_coefficients_c_b_a":[20,0,0],
    }
    ref=sample_operator_record(row,shape_yx=(200,200))
    pos=operator_curve_support(_line(12,40,20),[ref])
    assert pos["status"]=="NEAR_SINGLE_OPERATOR_GUIDE_REVIEW"
    assert pos["operator_curve_coverage"]<.5
    assert pos["operator_support_is_not_physical_truth"] is True
    empty=operator_curve_support(_line(12,40,120),[ref])
    assert empty["status"]=="NOT_NEAR_OPERATOR_GUIDE_UNLABELED_NOT_NEGATIVE"


def test_all_types_include_hair_spiral_fiducial_bundle_and_unknown():
    t=np.linspace(0,4*np.pi,200)
    spiral=np.column_stack((120+(15+t)*np.cos(t),140+(15+t)*np.sin(t)))
    paths=[
        {"component_id":"short","source_fragment_ids":["short"],"points_xy":_line(15,30,70)},
        {"component_id":"track","source_fragment_ids":["track"],"points_xy":_line(5,95,50)},
        {"component_id":"spiral","source_fragment_ids":["spiral"],"points_xy":spiral},
        {"component_id":"cross","source_fragment_ids":["cross"],"points_xy":_line(172,228,180)},
        {"component_id":"ray1","source_fragment_ids":["ray1"],"points_xy":_line(238,280,220)},
        {"component_id":"ray2","source_fragment_ids":["ray2"],"points_xy":
            np.column_stack((np.full(45,240.),np.linspace(200,240,45)))},
        {"component_id":"ray3","source_fragment_ids":["ray3"],"points_xy":
            np.column_stack((np.linspace(217,263,45),np.linspace(197,243,45)))},
    ]
    shape_categories=[
        "LOW_PRIORITY_POSSIBLE_ARTIFACT","EXTENDED_IMAGE_TRACE_CANDIDATE",
        "COILING_TRACE_REVIEW","EXTENDED_IMAGE_TRACE_CANDIDATE",
        "EXTENDED_IMAGE_TRACE_CANDIDATE","EXTENDED_IMAGE_TRACE_CANDIDATE",
        "EXTENDED_IMAGE_TRACE_CANDIDATE"
    ]
    ann=[{"component_id":c["component_id"],"shape_review_category":kind,
          "observed_length_px":len(c["points_xy"]),
          "total_signed_turn_deg":0,"show_in_primary_view":False}
         for c,kind in zip(paths,shape_categories)]
    context=parse_photo_labels({
        "H50Z_030_01_405.bmp":{
            "original_width":400,"original_height":300,
            "keypoints":[{"label":"f1","x_px":200,"y_px":180,"width_px":18},
                         {"label":"start","x_px":42,"y_px":49,"width_px":12}]
        }
    },"H50Z_030_01_405.bmp",(300,400))
    result,summary=type_all_observed_components(paths,ann,keypoint_context=context)
    byid={r["component_id"]:r for r in result}
    assert byid["short"]["primary_review_category"]=="ARTIFACT_LIKE_CANDIDATE"
    assert byid["track"]["primary_review_category"]=="TRACK_LIKE_CANDIDATE"
    assert byid["spiral"]["primary_review_category"]=="COILING_TRACE_CANDIDATE"
    assert byid["cross"]["primary_review_category"]=="CALIBRATION_CROSS_CONTEXT"
    assert any(byid[k]["primary_review_category"]=="BUNDLE_OR_INTERSECTION_REVIEW"
               for k in ("ray1","ray2","ray3"))
    assert len(result)==len(paths)
    assert all(x["preserved_source_observed"] for x in result)
    assert all(x["physical_track_identity"]=="UNKNOWN" for x in result)
    assert summary["no_source_components_deleted"] is True
    markers=human_mark_objects(context)
    assert len(markers)==2
    assert any(x["category"]=="HUMAN_FIDUCIAL_MARK_LOCATION" for x in markers)
