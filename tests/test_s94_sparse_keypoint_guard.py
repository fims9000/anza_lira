"""S94 sparse labels.json tests; no actual 207MB H50Z photographs required."""
import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from bubble_chamber.bitmap_tiles import sha256_file
from bubble_chamber.s94_sparse_keypoint_guard import (
    audit_label_collection, parse_photo_labels, apply_sparse_keypoint_guard,
    preserve_start_adjacent_short_marks,
)
from bubble_chamber.s92_curve_review import run_s92
from bubble_chamber.s93_compact_real_overlay import run_one_command
from bubble_chamber.full_bmp_segmentation import FullBMPConfig, run_full_bmp

PHOTO="H50Z_030_01_405.bmp"


def labels():
    return {
        PHOTO:{
            "original_width":512,"original_height":384,"keypoints_count":3,
            "keypoints":[
                {"label":"start","x_px":58,"y_px":120,"width_px":12},
                {"label":"f1","x_px":200,"y_px":190,"width_px":20},
                {"label":"f2","x_px":330,"y_px":310,"width_px":22},
            ],
        },
        "H50Z_030_01_406.bmp":{
            "original_width":512,"original_height":384,"keypoints_count":1,
            "keypoints":[{"label":"f1","x_px":205,"y_px":205,"width_px":24}],
        }
    }


def test_label_metadata_audit_is_not_dense_truth():
    data=labels()
    report=audit_label_collection(data)
    assert report["photos"]==2
    assert report["starts"]==1
    assert report["usable_fiducials"]==3
    assert report["track_masks_present"] is False
    assert report["unlabeled_pixels_are_background"] is False
    assert report["physical_class_labels_present"] is False
    one=parse_photo_labels(data,PHOTO,(384,512))
    assert one["status"]=="SPARSE_KEYPOINTS_ONLY_NOT_TRACK_GT"
    assert len(one["starts"])==1 and len(one["fiducials"])==2
    assert one["has_verified_track_masks"] is False
    no_match=parse_photo_labels(data,"H50Z_030_01_407.bmp",(384,512))
    assert no_match["status"]=="NO_MATCHING_PHOTO_LABELS"
    assert no_match["starts"]==[] and no_match["fiducials"]==[]


def test_duplicate_fiducial_is_quarantined_not_cherry_picked():
    data=labels()
    data[PHOTO]["keypoints"].append(
        {"label":"f1","x_px":201,"y_px":191,"width_px":21}
    )
    data[PHOTO]["keypoints_count"]=4
    photo=parse_photo_labels(data,PHOTO,(384,512))
    assert photo["status"]=="DUPLICATE_FIDUCIAL_QUARANTINED"
    assert photo["duplicate_fiducial_labels"]==["f1"]
    assert [x["label"] for x in photo["fiducials"]]==["f2"]
    assert len(photo["starts"])==1
    audit=audit_label_collection(data)
    assert audit["quarantined_fiducial_marks"]==2
    assert audit["duplicate_fiducial_cases"][0]["fiducial_labels"]==["f1"]


def test_source_label_photo_name_and_dimensions_must_match():
    data=labels()
    with pytest.raises(ValueError,match="dimensions"):
        parse_photo_labels(data,PHOTO,(512,384))
    with pytest.raises(ValueError,match="Expected original"):
        parse_photo_labels(data,"../../etc/passwd",(384,512))
    data[PHOTO]["keypoints"][0]["x_px"]=1e9
    with pytest.raises(ValueError,match="outside"):
        audit_label_collection(data)
    data=labels()
    data[PHOTO]["keypoints_count"]=55
    with pytest.raises(ValueError,match="count"):
        parse_photo_labels(data,PHOTO,(384,512))


def _line(a,b,y):
    return np.column_stack((np.arange(a,b+1,dtype=float),
                            np.full(b-a+1,y,dtype=float)))


def test_keypoint_review_rescues_short_source_line_but_vetoes_near_fiducial_gap():
    data=labels()
    context=parse_photo_labels(data,PHOTO,(384,512))
    components=[
        {"component_id":"tiny","points_xy":_line(52,65,120)},
        {"component_id":"left","points_xy":_line(150,180,190)},
        {"component_id":"right","points_xy":_line(220,249,190)},
    ]
    annotated=[
        {"component_id":"tiny","shape_review_category":"LOW_PRIORITY_POSSIBLE_ARTIFACT",
         "show_in_primary_view":False,"review_flags":[]},
        {"component_id":"left","shape_review_category":"EXTENDED_IMAGE_TRACE_CANDIDATE",
         "show_in_primary_view":True,"review_flags":[]},
        {"component_id":"right","shape_review_category":"EXTENDED_IMAGE_TRACE_CANDIDATE",
         "show_in_primary_view":True,"review_flags":[]},
    ]
    bridge=np.column_stack((np.linspace(180,220,17),np.full(17,190.)))
    p={"component_a":"left","component_b":"right",
       "review_decision":"PRIORITY_GAP_REVIEW_NOT_PHYSICAL_ACCEPT",
       "inferred_bridge_points_xy":bridge.tolist(),
       "physical_auto_join":False,
       "status":"UNKNOWN"}
    saved=bridge.copy()
    chosen,summary=apply_sparse_keypoint_guard(
        components,annotated,[p],[p],context
    )
    assert chosen==[]
    assert p["review_decision"]=="KEYPOINT_REGION_ABSTAIN_NOT_A_TRACK_JOIN"
    assert summary["formerly_selected_gaps_abstained"]==1
    assert summary["components_near_operator_start"]==1
    assert summary["components_near_fiducial"]==2
    assert preserve_start_adjacent_short_marks(annotated)==1
    assert annotated[0]["show_in_primary_view"] is True
    assert annotated[0]["display_reason"]=="NEAR_OPERATOR_START_MANUAL_REVIEW_NOT_GT"
    assert "FIDUCIAL_MARK_REGION_AMBIGUOUS" in annotated[1]["review_flags"]
    assert np.array_equal(bridge,saved)
    assert p["physical_auto_join"] is False


def test_missing_photo_labels_are_unknown_not_certified_safe():
    unknown=parse_photo_labels(labels(),"H50Z_030_01_407.bmp",(384,512))
    proposal={"review_decision":"PRIORITY_GAP_REVIEW_NOT_PHYSICAL_ACCEPT"}
    kept,s=apply_sparse_keypoint_guard([],[],[proposal],[proposal],unknown)
    assert kept==[proposal]
    assert s["no_annotation_does_not_mean_no_tracks"] is True
    assert s["physical_auto_joins_authorized"] is False


def _draw_source(tmp_path):
    gray=np.full((384,512),220,np.uint8)
    cv2.line(gray,(50,120),(64,120),30,1)
    cv2.line(gray,(130,190),(170,190),30,1)
    cv2.line(gray,(185,190),(245,190),30,1)
    cv2.line(gray,(315,300),(359,300),30,1)
    path=tmp_path/PHOTO
    assert cv2.imwrite(str(path),gray)
    return path


def _make_labels_file(tmp_path):
    path=tmp_path/"labels.json"
    path.write_text(json.dumps(labels(),ensure_ascii=False),encoding="utf8")
    return path


def test_s92_can_read_small_metadata_and_keep_source_S86_sha(tmp_path,monkeypatch):
    original=_draw_source(tmp_path)
    annotations=_make_labels_file(tmp_path)
    import bubble_chamber.full_bmp_segmentation as pipeline
    monkeypatch.setattr(pipeline,"predict_mask",
                        lambda gray,cfg,az_model=None:gray<80)
    s86=tmp_path/"PRIVATE_S86"
    run_full_bmp(original,s86,FullBMPConfig(
        method="contrast",core_size=128,halo=32,
        min_component_area=1,min_segment_length_px=4,preview_limit=0
    ))
    first_sha=sha256_file(original)
    mask_sha=sha256_file(s86/"manifest.json")
    output=tmp_path/"PRIVATE_S92"
    r=run_s92(s86,output,bmp_path=original,labels_path=annotations,
              preview_limit=2)
    assert r["status"]=="COMPLETE_PRIVATE_REVIEW_ONLY"
    assert r["sparse_labels_sha256"]==sha256_file(annotations)
    assert r["sparse_keypoint_guard"]["status"]=="SPARSE_KEYPOINTS_ONLY_NOT_TRACK_GT"
    assert r["sparse_keypoints_are_not_track_masks"] is True
    assert r["gap"]["gap_review_links"]==r["sparse_keypoint_guard"]["remaining_geometric_gap_review_links"]
    assert sha256_file(original)==first_sha
    assert sha256_file(s86/"manifest.json")==mask_sha
    rows=[json.loads(line) for line in
          (output/"observed_shape_review.ndjson").read_text().splitlines()]
    assert any("operator_start_nearby_row_indices_PRIVATE" in row for row in rows)
    assert all(row["physical_identity"]=="UNKNOWN" for row in rows)


def test_s93_one_command_binds_labels_sha_and_rejects_tamper(tmp_path,monkeypatch):
    bmp=_draw_source(tmp_path)
    meta=_make_labels_file(tmp_path)
    import bubble_chamber.full_bmp_segmentation as pipeline
    monkeypatch.setattr(pipeline,"predict_mask",
                        lambda gray,cfg,az_model=None:gray<80)
    out=tmp_path/"PRIVATE_405"
    result=run_one_command(bmp,out,method="contrast",
                           max_side=512,preview_limit=1,labels_path=meta)
    assert Path(result["small_annotated_photo"]).is_file()
    rec=json.loads((out/"run_identity.json").read_text())
    assert rec["sparse_labels_sha256"]==sha256_file(meta)
    s92=json.loads((out/"S92"/"report.json").read_text())
    assert s92["sparse_labels_sha256"]==rec["sparse_labels_sha256"]
    recovered=run_one_command(bmp,out,method="contrast",max_side=512,
                              preview_limit=1,labels_path=meta,resume=True)
    assert recovered["small_annotated_photo"]==result["small_annotated_photo"]
    meta.write_text(meta.read_text()+" ",encoding="utf8")
    with pytest.raises(ValueError,match="changed on resume"):
        run_one_command(bmp,out,method="contrast",max_side=512,
                        preview_limit=1,labels_path=meta,resume=True)
