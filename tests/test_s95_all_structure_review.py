"""S95 no-BMP and real-BMP synthetic integration through frozen S86/S92.

All tests use generated small images; no claim of accuracy on genuine H50Z.
"""
from __future__ import annotations
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from bubble_chamber.bitmap_tiles import sha256_file
from bubble_chamber.full_bmp_segmentation import FullBMPConfig,run_full_bmp
from bubble_chamber.s92_curve_review import run_s92
from bubble_chamber.s95_all_structures_review import run_s95
from bubble_chamber.s95_operator_curves import SCHEMA as OP_SCHEMA


def fake_image(tmp_path):
    pixels=np.full((384,512),210,np.uint8)
    cv2.line(pixels,(25,55),(130,55),35,1)
    cv2.line(pixels,(150,55),(330,55),35,1)
    # Tiny marks / "human hair" and operator annotation in different areas.
    cv2.line(pixels,(65,120),(77,132),30,1)
    cv2.line(pixels,(77,132),(88,120),30,1)
    cv2.line(pixels,(185,180),(230,225),35,2)
    cv2.line(pixels,(185,225),(230,180),35,2)
    theta=np.linspace(0,4*np.pi,200)
    radius=4+1.1*theta
    coil=np.rint(np.column_stack((400+radius*np.cos(theta),
                                  289+radius*np.sin(theta)))).astype(np.int32)
    cv2.polylines(pixels,[coil.reshape(-1,1,2)],False,35,1)
    source=tmp_path/"H50Z_030_01_405.bmp"
    assert cv2.imwrite(str(source),pixels)
    return source


def _meta(tmp_path,source):
    payload={
        source.name:{"original_width":512,"original_height":384,
                     "keypoints_count":2,
                     "keypoints":[
                         {"label":"start","x_px":65,"y_px":120,"width_px":15},
                         {"label":"f1","x_px":208,"y_px":202,"width_px":24},
                     ]}
    }
    file=tmp_path/"labels.json"
    file.write_text(json.dumps(payload),encoding="utf8")
    return file


def _operators(tmp_path,source):
    payload={
        "schema":OP_SCHEMA,
        "source_bmp_name":source.name,
        "source_bmp_sha256":sha256_file(source),
        "image_shape_yx":[384,512],
        "explicit_photo_alignment_verified":True,
        "curves":[{
            "track_id":"OP-T01",
            "kind":"parametric_quadratic",
            "start_xy":[25,55],"end_xy":[330,55],
            "x_coefficients_c_b_a":[25,305,0],
            "y_coefficients_c_b_a":[55,0,0],
        }],
    }
    file=tmp_path/"operator_curves.json"
    file.write_text(json.dumps(payload),encoding="utf8")
    return file


def _s86_s92(tmp_path,monkeypatch,source,labels):
    from bubble_chamber import full_bmp_segmentation
    monkeypatch.setattr(full_bmp_segmentation,"predict_mask",
                        lambda gray,cfg,az_model=None:gray<80)
    s86=tmp_path/"S86"
    run_full_bmp(source,s86,FullBMPConfig(
        method="contrast",core_size=128,halo=32,
        min_component_area=2,min_segment_length_px=4,preview_limit=0
    ))
    s92=tmp_path/"S92"
    run_s92(s86,s92,bmp_path=source,labels_path=labels,preview_limit=2)
    return s86,s92


def test_run_s95_with_image_all_fragments_classes_guides_and_gap_purity(tmp_path,monkeypatch):
    source=fake_image(tmp_path)
    labels=_meta(tmp_path,source)
    guide=_operators(tmp_path,source)
    source_sha=sha256_file(source)
    s86,s92=_s86_s92(tmp_path,monkeypatch,source,labels)
    s86_sha=sha256_file(s86/"manifest.json")
    result=run_s95(s86,s92,tmp_path/"S95",bmp_path=source,labels_path=labels,
                   operator_curves_path=guide,max_side=512)
    assert result["status"]=="ALL_S86_EXTRACTED_FRAGMENTS_TYPED_MACHINE_REVIEW_ONLY"
    assert result["source_bmp_sha256"]==source_sha
    assert result["source_fragments_deleted"]==0
    assert result["all_extracted_line_categories"]["extracted_components"] == (
        result["all_extracted_line_categories"]["typed_components"]
    )
    assert result["not_ground_truth_segmentation"] is True
    assert result["operator_guides"]==1
    assert result["human_annotation_marker_locations"]==2
    assert result["all_classes_comparison_file"]=="S95_all_classes_comparison.jpg"
    assert (tmp_path/"S95"/"S95_original_small.jpg").is_file()
    assert (tmp_path/"S95"/"S95_all_extracted_types_small.jpg").is_file()
    photo=cv2.imread(str(tmp_path/"S95"/"S95_original_small.jpg"))
    pred=cv2.imread(str(tmp_path/"S95"/"S95_all_extracted_types_small.jpg"))
    assert photo.shape==pred.shape==(384,512,3)
    assert np.max(np.abs(photo.astype(int)-pred.astype(int)))>20
    rows=[
        json.loads(line) for line in
        (tmp_path/"S95"/"all_source_fragment_types.ndjson").read_text().splitlines()
    ]
    assert len(rows)==result["all_extracted_line_categories"]["typed_components"]
    assert all(x["preserved_source_observed"] for x in rows)
    assert all(x["physical_auto_join_authorized"] is False for x in rows)
    assert all(x["physical_track_identity"]=="UNKNOWN" for x in rows)
    assert sha256_file(source)==source_sha
    assert sha256_file(s86/"manifest.json")==s86_sha
    with pytest.raises(FileExistsError):
        run_s95(s86,s92,tmp_path/"S95",bmp_path=source,labels_path=labels,
                operator_curves_path=guide,max_side=512)


def test_s95_without_source_BMP_still_types_every_observed_fragment(tmp_path,monkeypatch):
    source=fake_image(tmp_path)
    labels=_meta(tmp_path,source)
    s86,s92=_s86_s92(tmp_path,monkeypatch,source,labels)
    result=run_s95(s86,s92,tmp_path/"S95_NO_IMAGE",labels_path=labels)
    assert result["all_classes_comparison_file"] is None
    assert result["all_extracted_line_categories"]["typed_components"]>0
    assert not (tmp_path/"S95_NO_IMAGE"/"S95_all_extracted_types_small.jpg").exists()
    assert (tmp_path/"S95_NO_IMAGE"/"all_source_fragment_types.ndjson").is_file()
    assert (tmp_path/"S95_NO_IMAGE"/"index.html").is_file()


def test_S95_refuses_operator_curve_from_different_source_or_missing_same_photo_labels(tmp_path,monkeypatch):
    source=fake_image(tmp_path)
    labels=_meta(tmp_path,source)
    guide=_operators(tmp_path,source)
    s86,s92=_s86_s92(tmp_path,monkeypatch,source,labels)
    payload=json.loads(guide.read_text())
    payload["source_bmp_sha256"]="a"*64
    guide.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match="mismatch"):
        run_s95(s86,s92,tmp_path/"INVALID_GUIDE",labels_path=labels,
                operator_curves_path=guide)
    with pytest.raises(ValueError,match="matching original labels"):
        run_s95(s86,s92,tmp_path/"MISSING_LABELS")


def test_s95_operator_curves_none_not_same_as_absent_pixelGT(tmp_path,monkeypatch):
    source=fake_image(tmp_path)
    labels=_meta(tmp_path,source)
    s86,s92=_s86_s92(tmp_path,monkeypatch,source,labels)
    report=run_s95(s86,s92,tmp_path/"NO_PARABOLA",labels_path=labels)
    assert report["operator_guides"]==0
    assert report["full_recall_not_verified"]
    assert report["collisions_neutron_origin_unknown"]
    assert report["all_objects_include_short_garbage_like_and_unknown"]
