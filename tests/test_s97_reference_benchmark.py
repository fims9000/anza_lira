"""S97 end-to-end private original-BMP independent annotation workflow tests.

All images are artificial grayscale photos manufactured solely in pytest,
no real operator H50Z, no independently verified 3D particle identities.
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
from scripts.research.bubble_chamber.physicists_s97_reference_benchmark import main


def _source(tmp_path):
    gray=np.full((256,384),215,np.uint8)
    cv2.line(gray,(25,80),(178,80),30,1)
    cv2.line(gray,(30,130),(110,130),32,1)
    cv2.line(gray,(220,40),(262,82),30,1)
    bmp=tmp_path/"S97_SYNTHETIC_BMP_NOT_REAL_H50Z.bmp"
    assert cv2.imwrite(str(bmp),gray)
    return bmp,gray


def _reference(pack,pack_sha):
    source={
        "schema":pack["schema"],
        "source_bmp_name":pack["source_bmp_name"],
        "source_bmp_sha256":pack["source_bmp_sha256"],
        "image_shape_yx":pack["image_shape_yx"],
        "roi_xyxy_source":pack["roi_xyxy_source"],
        "source_reference_pack_sha256":pack_sha,
        "image_was_unmodified_original":True,
        "machine_proposals_visible_during_annotation":False,
        "full_exhaustive_negative_segmentation":False,
        "data_source":"INDEPENDENT_OPERATOR_ORIGINAL_BMP_ONLY",
        "reference_semantics":"PARTIAL_VISIBLE_2D_STROKES_NO_PARTICLE_GT",
        "reviewer_id":"fake_human_unit_test",
        "strokes":[
            {
                "stroke_id":"h000001","visual_instance_id":"T01",
                "kind":"TRACK_LIKE","certainty":"CLEAR",
                "points_xy_source_px":[[int(x),80] for x in range(30,171,10)],
                "note":"fake GT only for software invariant test",
            },
            {
                "stroke_id":"h000002","visual_instance_id":"A01",
                "kind":"ARTIFACT","certainty":"CLEAR",
                "points_xy_source_px":[[int(x),130] for x in range(30,111,10)],
                "note":"fake visual class only, not true particle truth",
            },
        ],
    }
    return source


def _segmented_source(tmp_path,bmp,monkeypatch,dirname,foreground=True):
    from bubble_chamber import full_bmp_segmentation
    monkeypatch.setattr(
        full_bmp_segmentation,"predict_mask",
        (lambda gray,cfg,az_model=None:gray<90) if foreground
        else (lambda gray,cfg,az_model=None:np.zeros_like(gray,dtype=bool))
    )
    out=tmp_path/dirname
    run_full_bmp(bmp,out,FullBMPConfig(
        method="contrast",core_size=128,halo=32,preview_limit=0,
        min_component_area=1,min_segment_length_px=4,max_fragments_per_tile=2048
    ))
    return out


def test_prepare_source_reference_is_original_lossless_private_and_shielded_from_machine(
    tmp_path
):
    bmp,original=_source(tmp_path)
    origsha=sha256_file(bmp)
    pack=prepare_independent_source_rois(
        bmp,tmp_path/"REF",roi_xyxy_source=[[0,0,384,256]],
        tile_size=128
    )
    assert pack["status"]=="INDEPENDENT_ORIGINAL_BMP_HUMAN_POSITIVE_REFERENCE_PACK"
    assert pack["source_bmp_sha256"]==origsha
    assert pack["annotation_is_partial_positive_strokes_only"] is True
    assert pack["unreviewed_photo_regions_not_negative"] is True
    assert len(pack["source_image_tiles"])==6
    for tile in pack["source_image_tiles"]:
        png=cv2.imread(str(tmp_path/"REF"/tile["source_original_png"]),
                       cv2.IMREAD_GRAYSCALE)
        x0,y0,x1,y1=tile["xyxy_source"]
        assert np.array_equal(png,original[y0:y1,x0:x1])
        assert sha256_file(tmp_path/"REF"/tile["source_original_png"])==tile[
            "tile_preview_sha256"
        ]
    page=(tmp_path/"REF"/"index.html").read_text(encoding="utf8")
    assert "machine_proposals_visible_during_annotation" in page
    assert "INDEPENDENT_OPERATOR_ORIGINAL_BMP_ONLY" in page
    assert "COILING_TRACK_LIKE" in page
    assert "source_reference_pack_sha256" in page
    assert "Скачать независимую разметку" in page
    assert "S86" not in page or "машинных масок" in page.lower()
    assert sha256_file(bmp)==origsha
    with pytest.raises(ValueError,match="new PRIVATE"):
        prepare_independent_source_rois(bmp,tmp_path/"REF")


def test_S97_large_BMP_requires_explicit_source_roi_and_budget(tmp_path):
    gray=np.full((1100,1100),220,np.uint8)
    src=tmp_path/"LARGE_SYNTHETIC.bmp"
    assert cv2.imwrite(str(src),gray)
    with pytest.raises(ValueError,match="explicit --roi"):
        prepare_independent_source_rois(src,tmp_path/"FAIL_NO_ROI")
    with pytest.raises(ValueError,match="tile count"):
        prepare_independent_source_rois(
            src,tmp_path/"FAIL_TILE_BUDGET",
            roi_xyxy_source=[[0,0,1100,1100]],
            tile_size=256,max_tiles=3
        )
    assert not (tmp_path/"FAIL_TILE_BUDGET").exists()


def test_S97_prepare_evaluate_and_same_photo_baseline_both_S92_S96(
    tmp_path,monkeypatch
):
    bmp,original=_source(tmp_path)
    sha=sha256_file(bmp)
    refdir=tmp_path/"ORIGINAL_ONLY"
    main(["prepare","--bmp",str(bmp),"--private-out-dir",str(refdir),
          "--roi","0","0","384","256","--tile-size","512"])
    packfile=refdir/"reference_pack.json"
    pack=json.loads(packfile.read_text())
    assert len(pack["source_image_tiles"])==1
    annotations=_reference(pack,sha256_file(packfile))
    reference=tmp_path/"human_source_positive.json"
    reference.write_text(json.dumps(annotations),encoding="utf8")
    selected=_segmented_source(tmp_path,bmp,monkeypatch,"S86_SELECTED",True)
    empty=_segmented_source(tmp_path,bmp,monkeypatch,"S86_NO_CANDIDATE",False)
    selected_sha=sha256_file(selected/"manifest.json")
    s92=tmp_path/"S92_BASELINE"
    run_s92(selected,s92,bmp_path=bmp,preview_limit=0)
    s95=tmp_path/"S95_BASELINE"
    run_s95(selected,s92,s95,bmp_path=bmp,max_side=512)
    s96=tmp_path/"S96_BASELINE"
    run_s96(selected,s95,s96,bmp_path=bmp,preview_limit=8)
    report=tmp_path/"S97_REPORT.json"
    main([
        "evaluate","--pack-dir",str(refdir),
        "--reference-json",str(reference),
        "--candidate","S86_DETECTED="+str(selected),
        "--candidate","S86_EMPTY="+str(empty),
        "--s92-dir",str(s92),"--s96-dir",str(s96),
        "--visual-preview-dir",str(tmp_path/"POSTHOC"),
        "--visual-preview-tiles","2",
        "--private-out-json",str(report)
    ])
    body=json.loads(report.read_text())
    assert body["status"]=="MATCHED_INDEPENDENT_2D_HUMAN_POSITIVE_BENCHMARK_PRIVATE_ONLY"
    assert body["source_bmp_sha256"]==sha
    assert body["clear_positive_track_or_coil_strokes"]==1
    assert body["precision_f1_fpr_not_identifiable"] is True
    assert body["no_particle_identity_or_neutron_causality"] is True
    results=body["per_candidate_matched_positive_coverage"]
    assert len(results)==2
    assert results[0]["observed_positive_centerline_coverage"]>.9
    assert results[1]["observed_positive_centerline_coverage"]==0.
    diff=body["descriptive_paired_model_contrasts"]
    assert diff[0]["descriptive_delta_pp"]< -90.
    assert diff[0]["no_statistical_significance_claim"] is True
    audit=body["optional_visual_instance_provisional_link_audits"]
    assert "S92_GAP" in audit and "S96_CROSSING" in audit
    assert audit["S92_GAP"]["not_a_precision_f1_or_particle_identity_measure"]
    assert audit["S96_CROSSING"]["inferred_gap_pixels_never_count_as_observed_centerline"]
    assert body["source_candidate_receipts"][0]["s86_source_manifest_sha256"]==selected_sha
    assert sha256_file(selected/"manifest.json")==selected_sha
    assert sha256_file(bmp)==sha
    vis=body["posthoc_error_maps_PRIVATE"]
    assert vis["displayed_tiles"]==1
    assert Path(vis["index_html"]).is_file()
    assert vis["manifest_sha256"]==sha256_file(
        tmp_path/"POSTHOC"/"posthoc_error_map_report.json"
    )
    post=json.loads((tmp_path/"POSTHOC"/"posthoc_error_map_report.json").read_text())
    assert post["original_source_bmp_sha256"]==sha
    assert post["posthoc_overlay_never_shown_during_original_reference_annotation"]
    assert post["no_unreviewed_negative_pixel_GT"]
    original_tile=cv2.imread(
        str(tmp_path/"POSTHOC"/post["cards"][0]["unmodified_original_png"]),
        cv2.IMREAD_GRAYSCALE
    )
    assert np.array_equal(original_tile,original)
    for model in ("S86_DETECTED","S86_EMPTY"):
        item=post["cards"][0]["model_maps"][model]
        assert (tmp_path/"POSTHOC"/item["image"]).is_file()
    assert post["cards"][0]["model_maps"]["S86_EMPTY"][
        "human_positive_samples_missed_in_this_tile"
    ]>100
    assert post["cards"][0]["model_maps"]["S86_DETECTED"][
        "human_positive_samples_covered_in_this_tile"
    ]>100
    assert sha256_file(refdir/"reference_pack.json")==sha256_file(packfile)
    with pytest.raises(ValueError,match="NEW private"):
        main([
            "evaluate","--pack-dir",str(refdir),
            "--reference-json",str(reference),
            "--candidate","S86_DETECTED="+str(selected),
            "--private-out-json",str(report)
        ])


def test_cross_photo_or_edited_source_tiles_fail_closed(tmp_path,monkeypatch):
    bmp,_=_source(tmp_path)
    refdir=tmp_path/"REF"
    prepare_independent_source_rois(bmp,refdir)
    packfile=refdir/"reference_pack.json"
    pack=json.loads(packfile.read_text())
    ref=_reference(pack,sha256_file(packfile))
    reference=tmp_path/"reference.json"
    reference.write_text(json.dumps(ref))
    selected=_segmented_source(tmp_path,bmp,monkeypatch,"SOURCE",True)
    wrong={**ref,"source_bmp_sha256":"b"*64}
    reference.write_text(json.dumps(wrong))
    with pytest.raises(ValueError,match="different BMP"):
        main([
            "evaluate","--pack-dir",str(refdir),
            "--reference-json",str(reference),
            "--candidate","S88="+str(selected),
            "--private-out-json",str(tmp_path/"INVALID_HASH.json")
        ])
    reference.write_text(json.dumps(ref))
    tile=refdir/pack["source_image_tiles"][0]["source_original_png"]
    tile.write_bytes(b"tampered")
    with pytest.raises(ValueError,match="tile missing or changed"):
        main([
            "evaluate","--pack-dir",str(refdir),
            "--reference-json",str(reference),
            "--candidate","S88="+str(selected),
            "--private-out-json",str(tmp_path/"INVALID_TILE.json")
        ])
