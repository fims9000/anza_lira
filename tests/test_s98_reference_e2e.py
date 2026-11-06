"""S98 end-to-end source-pixel benchmark with independent synthetic 2D tracing.

All pixels are procedurally generated; no actual H50Z image, manual physics
reference, nuclear interaction class, or model-accuracy evidence is claimed.
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
from bubble_chamber.s96_junction_workflow import run_s96
from bubble_chamber.s97_reference_viewer import prepare_independent_source_rois
from scripts.research.bubble_chamber.physicists_s98_trace_continuity import (
    main,run_audit
)


def _bmp(tmp_path):
    image=np.full((256,384),220,dtype=np.uint8)
    # Deliberately fragmented but clearly collinear 2D human-positive trace.
    cv2.line(image,(20,40),(70,40),30,1)
    cv2.line(image,(85,40),(130,40),30,1)
    cv2.line(image,(146,40),(180,40),30,1)
    # A nearby nonpositive "possible artifact" left as source material.
    cv2.line(image,(210,80),(240,95),32,1)
    out=tmp_path/"SYNTHETIC_S98_NOT_H50Z.bmp"
    assert cv2.imwrite(str(out),image)
    return out,image


def _s86(tmp_path,monkeypatch,bmp,name,detected=True):
    from bubble_chamber import full_bmp_segmentation
    monkeypatch.setattr(
        full_bmp_segmentation,"predict_mask",
        (lambda gray,cfg,az_model=None:gray<90)
        if detected else
        (lambda gray,cfg,az_model=None:np.zeros_like(gray,dtype=bool))
    )
    out=tmp_path/name
    run_full_bmp(bmp,out,FullBMPConfig(
        method="contrast",core_size=128,halo=32,preview_limit=0,
        min_component_area=1,min_segment_length_px=4,max_fragments_per_tile=512
    ))
    return out


def _reference(tmp_path,bmp):
    reference_root=tmp_path/"S97_ORIGINAL_ONLY"
    pack=prepare_independent_source_rois(
        bmp,reference_root,roi_xyxy_source=[[0,0,384,256]],tile_size=512
    )
    pack_file=reference_root/"reference_pack.json"
    human={
        "schema":pack["schema"],
        "source_bmp_name":pack["source_bmp_name"],
        "source_bmp_sha256":pack["source_bmp_sha256"],
        "image_shape_yx":pack["image_shape_yx"],
        "roi_xyxy_source":pack["roi_xyxy_source"],
        "source_reference_pack_sha256":sha256_file(pack_file),
        "image_was_unmodified_original":True,
        "machine_proposals_visible_during_annotation":False,
        "full_exhaustive_negative_segmentation":False,
        "data_source":"INDEPENDENT_OPERATOR_ORIGINAL_BMP_ONLY",
        "reference_semantics":"PARTIAL_VISIBLE_2D_STROKES_NO_PARTICLE_GT",
        "reviewer_id":"unit_test_fake_physicist",
        "strokes":[{
            "stroke_id":"independent_001",
            "visual_instance_id":"visual_T01",
            "kind":"TRACK_LIKE","certainty":"CLEAR",
            "points_xy_source_px":[[x,40] for x in range(20,181,10)],
            "note":"synthetic unobserved gap is 2D visual only",
        },{
            "stroke_id":"artifact_001",
            "visual_instance_id":"possible_artifact",
            "kind":"ARTIFACT","certainty":"CLEAR",
            "points_xy_source_px":[[210,80],[225,87.5],[240,95]],
            "note":"not an exhaustive human negative mask",
        }],
    }
    path=tmp_path/"S97_HUMAN_REFERENCE.json"
    path.write_text(json.dumps(human),encoding="utf8")
    return reference_root,path


def _all(tmp_path,monkeypatch):
    bmp,gray=_bmp(tmp_path)
    refdir,human=_reference(tmp_path,bmp)
    source=_s86(tmp_path,monkeypatch,bmp,"S86_DETECTED",True)
    empty=_s86(tmp_path,monkeypatch,bmp,"S86_EMPTY",False)
    s92=tmp_path/"S92_SELECTED"
    run_s92(source,s92,bmp_path=bmp,preview_limit=0)
    s95=tmp_path/"S95_SELECTED"
    run_s95(source,s92,s95,bmp_path=bmp,max_side=512)
    s96=tmp_path/"S96_SELECTED"
    run_s96(source,s95,s96,bmp_path=bmp,preview_limit=6)
    return bmp,gray,refdir,human,source,empty,s92,s95,s96


def test_s98_real_synthetic_BMP_reference_first_full_CLI_and_honest_zero_detection(
    tmp_path,monkeypatch
):
    bmp,gray,refdir,human,source,empty,s92,s95,s96=_all(tmp_path,monkeypatch)
    srcsha=sha256_file(bmp)
    humansha=sha256_file(human)
    s86sha=sha256_file(source/"manifest.json")
    out=tmp_path/"S98_REPORT_PRIVATE"
    main([
        "--pack-dir",str(refdir),
        "--reference-json",str(human),
        "--candidate","DETECTED="+str(source),
        "--candidate","UNDETECTED="+str(empty),
        "--s92","DETECTED="+str(s92),
        "--s96","DETECTED="+str(s96),
        "--private-out-dir",str(out),
        "--max-visual-cases","8",
    ])
    assert (out/"report.json").is_file()
    assert (out/"index.html").is_file()
    report=json.loads((out/"report.json").read_text())
    assert report["schema"]=="S98_INDEPENDENT_VISUAL_CONTINUITY_BOUNDARIES_V1"
    assert report["source_bmp_sha256"]==srcsha
    assert report["human_clear_positive_stroke_count"]==1
    assert report["no_true_physical_trajectory_identity_established"] is True
    assert report["missing_S92_S96_pack_means_NOT_EVALUATED_not_zero_recall"] is True
    assert len(report["candidates"])==2
    detected,undetected=report["candidates"]
    assert detected["model_name"]=="DETECTED"
    assert detected["observed_source_centerline_positive_coverage"]>.94
    assert detected["reference_first_source_piece_boundaries"][
        "assessable_unique_detected_piece_adjacencies"
    ]>=2
    assert detected["reference_first_model_link_recovery"] is not None
    assert detected["reference_first_model_link_recovery"][
        "partial_human_visible_not_particle_GT"
    ] is True
    assert undetected["model_name"]=="UNDETECTED"
    assert undetected["observed_source_centerline_positive_coverage"]==0.
    assert undetected["reference_first_source_piece_boundaries"][
        "assessable_unique_detected_piece_adjacencies"
    ]==0
    assert undetected["reference_first_model_link_recovery"] is None
    assert report["matched_observed_centerline_positive_coverage_pp"][0][
        "descriptive_delta_pp"
    ]< -90.
    preview=report["posthoc_visual_case_preview"]["per_model"]
    assert preview["UNDETECTED"]["candidate_missing_link_claims_generated"] is False
    assert preview["DETECTED"]["total_reference_boundaries_unselected"]>=0
    for case in preview["DETECTED"]["posthoc_cases"]:
        if "photo" not in case:continue
        img=cv2.imread(str(out/case["photo"]))
        assert img is not None
        assert img.shape[1]==img.shape[0] * 2 or img.shape[1]>img.shape[0]
        assert sha256_file(out/case["photo"])==case["photo_sha256"]
        x0,y0,x1,y1=case["crop_xyxy_original_source"]
        # Left half is pixel-identical to private original uncompressed BMP.
        assert np.array_equal(
            cv2.cvtColor(img[:,:x1-x0],cv2.COLOR_BGR2GRAY),
            gray[y0:y1,x0:x1],
        )
        assert case["overlay_right_not_observed_physics_or_ground_truth"]
    assert "Физичес" not in (out/"index.html").read_text() or "не" in (
        out/"index.html").read_text()
    assert sha256_file(bmp)==srcsha
    assert sha256_file(human)==humansha
    assert sha256_file(source/"manifest.json")==s86sha
    with pytest.raises(ValueError,match="fresh PRIVATE"):
        main([
            "--pack-dir",str(refdir),"--reference-json",str(human),
            "--candidate","DETECTED="+str(source),
            "--private-out-dir",str(out)
        ])


def test_s98_rejects_mixed_s92_provenance_and_does_not_write_results(
    tmp_path,monkeypatch
):
    bmp,gray,refdir,human,source,empty,s92,s95,s96=_all(tmp_path,monkeypatch)
    other_s92=tmp_path/"S92_WRONG_MODEL"
    run_s92(empty,other_s92,bmp_path=bmp,preview_limit=0)
    target=tmp_path/"WRONG_LINK_PACK"
    with pytest.raises(ValueError,match="another source S86"):
        run_audit(
            pack_dir=refdir,human_reference_json=human,
            candidates=[("DETECTED",source)],
            s92_packs={"DETECTED":other_s92},s96_packs={},
            private_out_dir=target
        )
    assert not (target/"report.json").exists()


def test_s98_fails_if_reference_original_tiles_changed_after_export(
    tmp_path,monkeypatch
):
    bmp,gray,refdir,human,source,empty,s92,s95,s96=_all(tmp_path,monkeypatch)
    pack=json.loads((refdir/"reference_pack.json").read_text())
    picture=refdir/pack["source_image_tiles"][0]["source_original_png"]
    picture.write_bytes(b"not the original pixel tile")
    with pytest.raises(ValueError,match="tile missing or changed"):
        run_audit(
            pack_dir=refdir,human_reference_json=human,
            candidates=[("DETECTED",source)],
            s92_packs={},s96_packs={},
            private_out_dir=tmp_path/"NOT_CREATED"
        )


def test_s98_bad_model_labels_and_wrong_output_destination_are_blocked(
    tmp_path,monkeypatch
):
    bmp,gray,refdir,human,source,empty,s92,s95,s96=_all(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match="Duplicate"):
        main([
            "--pack-dir",str(refdir),"--reference-json",str(human),
            "--candidate","ONE="+str(source),"--candidate","ONE="+str(empty),
            "--private-out-dir",str(tmp_path/"BAD_DUP")
        ])
    with pytest.raises(ValueError,match="unknown candidate"):
        main([
            "--pack-dir",str(refdir),"--reference-json",str(human),
            "--candidate","ONE="+str(source),
            "--s92","TWO="+str(s92),
            "--private-out-dir",str(tmp_path/"BAD_UNKNOWN")
        ])
    with pytest.raises(ValueError,match="original S86/S97"):
        run_audit(
            pack_dir=refdir,human_reference_json=human,
            candidates=[("ONE",source)],
            s92_packs={},s96_packs={},
            private_out_dir=refdir/"S98_SHOULD_NOT_EXIST"
        )
