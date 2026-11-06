"""S99 complete private experiment over a REAL SYNTHETIC BMP, never H50Z GT."""
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
from bubble_chamber.s99_blind_sampling import prepare_blind_s99_review
from scripts.research.bubble_chamber.physicists_s99_blind_morphology import (
    main,evaluate_s99
)


def _bmp(tmp):
    image=np.full((256,384),225,dtype=np.uint8)
    for y0 in (0,128):
        cv2.line(image,(10,y0+60),(374,y0+60),22,1)
        for x0 in (0,128,256):
            cv2.line(image,(x0+35,y0+80),(x0+45,y0+90),24,1)
            cv2.line(image,(x0+45,y0+90),(x0+55,y0+80),24,1)
    bmp=tmp/"S99_SYNTHETIC_ORIGINAL_NOT_H50Z.bmp"
    assert cv2.imwrite(str(bmp),image)
    return bmp,image


def _human_pack(tmp,bmp):
    root=tmp/"S99_PRESELECTED_BLIND"
    main([
        "prepare","--bmp",str(bmp),
        "--private-out-dir",str(root),
        "--seed","20261106","--tile-size","128",
        "--strata-rows","2","--strata-cols","2",
        "--roi-count","4",
    ])
    packfile=root/"reference_pack.json"
    pack=json.loads(packfile.read_text())
    selection=json.loads((root/"s99_blind_selection.json").read_text())
    assert selection["sampled_source_roi_count"]==4
    assert len(pack["roi_xyxy_source"])==4
    assert pack["human_edit_does_not_show_machine_predictions"]
    assert len(pack["source_image_tiles"])==4
    original_sha=sha256_file(bmp)
    assert pack["source_bmp_sha256"]==original_sha
    for tile in pack["source_image_tiles"]:
        image=cv2.imread(str(root/tile["source_original_png"]),
                         cv2.IMREAD_GRAYSCALE)
        x0,y0,x1,y1=tile["xyxy_source"]
        original=cv2.imread(str(bmp),cv2.IMREAD_GRAYSCALE)
        assert np.array_equal(image,original[y0:y1,x0:x1])
    first=pack["roi_xyxy_source"][0]
    x0,y0,_,_=first
    human={
        "schema":pack["schema"],
        "source_bmp_name":pack["source_bmp_name"],
        "source_bmp_sha256":pack["source_bmp_sha256"],
        "image_shape_yx":pack["image_shape_yx"],
        "roi_xyxy_source":pack["roi_xyxy_source"],
        "source_reference_pack_sha256":sha256_file(packfile),
        "image_was_unmodified_original":True,
        "machine_proposals_visible_during_annotation":False,
        "full_exhaustive_negative_segmentation":False,
        "data_source":"INDEPENDENT_OPERATOR_ORIGINAL_BMP_ONLY",
        "reference_semantics":"PARTIAL_VISIBLE_2D_STROKES_NO_PARTICLE_GT",
        "reviewer_id":"synthetic_manual_source_1",
        "strokes":[{
            "stroke_id":"human_T001","visual_instance_id":"visual_track_1",
            "kind":"TRACK_LIKE","certainty":"CLEAR",
            "points_xy_source_px":[[x0+x,y0+60] for x in range(20,101,10)],
            "note":"artificial human reference drawn on artificial original source",
        },{
            "stroke_id":"human_A001","visual_instance_id":"visual_artifact_1",
            "kind":"ARTIFACT","certainty":"CLEAR",
            "points_xy_source_px":[
                [x0+35,y0+80],[x0+45,y0+90],[x0+55,y0+80]
            ],
            "note":"visual training label, not particle truth",
        },{
            "stroke_id":"human_unclear","visual_instance_id":"visual_unknown",
            "kind":"UNKNOWN","certainty":"UNCERTAIN",
            "points_xy_source_px":[[x0+60,y0+110],[x0+75,y0+110]],
            "note":"not an exhaustive negative example",
        }],
    }
    reference=tmp/"S97_HUMAN_REFERENCE_AFTER_S99_BLIND.json"
    reference.write_text(json.dumps(human),encoding="utf8")
    return root,reference


def _machine(tmp,monkeypatch,bmp):
    from bubble_chamber import full_bmp_segmentation
    monkeypatch.setattr(
        full_bmp_segmentation,"predict_mask",
        lambda gray,cfg,az_model=None:gray<90
    )
    s86=tmp/"S86_ARTIFICIAL_SOURCE"
    run_full_bmp(bmp,s86,FullBMPConfig(
        method="contrast",core_size=128,halo=32,preview_limit=0,
        min_component_area=1,min_segment_length_px=4,
        max_fragments_per_tile=1024
    ))
    s92=tmp/"S92_ARTIFICIAL_SOURCE"
    run_s92(s86,s92,bmp_path=bmp,preview_limit=0)
    s95=tmp/"S95_ARTIFICIAL_SOURCE"
    run_s95(s86,s92,s95,bmp_path=bmp,max_side=512)
    return s86,s92,s95


def _all(tmp,monkeypatch):
    bmp,orig=_bmp(tmp)
    pack,expert=_human_pack(tmp,bmp)
    s86,s92,s95=_machine(tmp,monkeypatch,bmp)
    return bmp,orig,pack,expert,s86,s92,s95


def test_s99_full_source_only_original_annotation_to_photo_morphology_report(
    tmp_path,monkeypatch
):
    bmp,source,pack,expert,s86,s92,s95=_all(tmp_path,monkeypatch)
    original_sha=sha256_file(bmp)
    human_sha=sha256_file(expert)
    original_manifest_sha=sha256_file(s86/"manifest.json")
    out=tmp_path/"S99_HUMAN_BEFORE_MACHINE_REPORT"
    main([
        "evaluate","--pack-dir",str(pack),
        "--reference-json",str(expert),
        "--candidate","ARTIFICIAL="+str(s86),
        "--s92","ARTIFICIAL="+str(s92),
        "--s95","ARTIFICIAL="+str(s95),
        "--max-preview-tiles","2",
        "--private-out-dir",str(out)
    ])
    results=json.loads((out/"morphology_benchmark.json").read_text())
    assert results["status"]=="SOURCE_SHA_VERIFIED_INDEPENDENT_HUMAN_2D_MORPHOLOGY_PARTIAL_BENCHMARK"
    assert results["original_bmp_sha256"]==original_sha
    assert results["sampled_source_roi_count"]==4
    assert results["reviewed_reference_strokes_total"]==3
    assert results["posthoc_private_source_tiles_shown"]==2
    assert results["posterior_model_preview_not_shown_to_original_human_annotator"]
    assert results["no_particle_same_track_or_neutron_collision_truth_established"]
    assert results["no_film_level_confidence_interval"]
    trial=results["candidate_model_scores"][0]
    assert trial["source_s86_manifest_sha256"]==original_manifest_sha
    assert trial["source_human_clear_classified_strokes"]==2
    assert trial["per_human_morphology"]["TRACK_LIKE"][
        "reference_S86_observed_coverage_conditional_on_this_kind"
    ]>.70
    assert trial["per_human_morphology"]["ARTIFACT"][
        "human_strokes"
    ]==1
    assert trial["human_uncertain_or_unknown_strokes_excluded"]["UNKNOWN"]==1
    assert trial["conditioned_on_independent_explicit_strokes_not_dense_image_GT"]
    gallery=json.loads((out/"posthoc_gallery_manifest.json").read_text())
    assert gallery["source_tiles_previewed"]==2
    assert gallery["independent_annotation_editor_never_displays_S95"]
    assert (out/"index.html").is_file()
    for tile in gallery["cases"]:
        source_img=cv2.imread(str(out/tile["original_png"]),cv2.IMREAD_GRAYSCALE)
        x0,y0,x1,y1=tile["source_xyxy"]
        assert np.array_equal(source_img,source[y0:y1,x0:x1])
        pairfile=out/tile["overlays"]["ARTIFICIAL"][
            "source_original_left_machine_and_human_right"
        ]
        comparison=cv2.imread(str(pairfile))
        assert comparison.shape[:2]==(y1-y0,2*(x1-x0))
        assert np.array_equal(
            cv2.cvtColor(comparison[:,:x1-x0],cv2.COLOR_BGR2GRAY),
            source[y0:y1,x0:x1]
        )
    assert sha256_file(bmp)==original_sha
    assert sha256_file(expert)==human_sha
    assert sha256_file(s86/"manifest.json")==original_manifest_sha
    with pytest.raises(ValueError,match="NEW private"):
        main([
            "evaluate","--pack-dir",str(pack),
            "--reference-json",str(expert),
            "--candidate","ARTIFICIAL="+str(s86),
            "--s92","ARTIFICIAL="+str(s92),
            "--s95","ARTIFICIAL="+str(s95),
            "--private-out-dir",str(out)
        ])


def test_source_photo_preselection_tampering_refuses_morphology_eval(
    tmp_path,monkeypatch
):
    bmp,source,pack,expert,s86,s92,s95=_all(tmp_path,monkeypatch)
    path=pack/"s99_blind_selection.json"
    plan=json.loads(path.read_text())
    plan["sampled_cell_selection"][0]["xyxy_source"]=[0,0,128,128]
    plan["sampling_seed_predeclared"]+=1
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match="blind sample"):
        evaluate_s99(
            pack_dir=pack,reference_json=expert,
            candidates=[("MODEL",s86)],
            s92_by_name={"MODEL":s92},
            s95_by_name={"MODEL":s95},
            private_out_dir=tmp_path/"NO_EVALUATION",
        )
    assert not (tmp_path/"NO_EVALUATION").exists()


def test_no_cross_source_S92_S95_mix_allowed(tmp_path,monkeypatch):
    bmp,source,pack,expert,s86,s92,s95=_all(tmp_path,monkeypatch)
    report=s95/"report.json"
    source_report=json.loads(report.read_text())
    source_report["source_s92_report_sha256"]="b"*64
    report.write_text(json.dumps(source_report))
    with pytest.raises(ValueError,match="same source S86/S92"):
        evaluate_s99(
            pack_dir=pack,reference_json=expert,
            candidates=[("MODEL",s86)],
            s92_by_name={"MODEL":s92},
            s95_by_name={"MODEL":s95},
            private_out_dir=tmp_path/"NO_WRONG_SOURCE"
        )
    assert not (tmp_path/"NO_WRONG_SOURCE").exists()


def test_source_tile_corruption_and_case_mismatch_stay_private(tmp_path,monkeypatch):
    bmp,source,pack,expert,s86,s92,s95=_all(tmp_path,monkeypatch)
    index=json.loads((pack/"reference_pack.json").read_text())
    tile=pack/index["source_image_tiles"][0]["source_original_png"]
    tile.write_bytes(b"NOT THE ORIGINAL SOURCE")
    with pytest.raises(ValueError,match="tile missing or changed"):
        evaluate_s99(
            pack_dir=pack,reference_json=expert,
            candidates=[("MODEL",s86)],
            s92_by_name={"MODEL":s92},
            s95_by_name={"MODEL":s95},
            private_out_dir=tmp_path/"REFUSED_CORRUPT_TILE",
        )
    assert not (tmp_path/"REFUSED_CORRUPT_TILE").exists()
