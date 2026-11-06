"""Full synthetic original BMP source pixel audit for S100, NOT H50Z physics GT."""
from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
import json

import cv2
import numpy as np
import pytest

from bubble_chamber.bitmap_tiles import sha256_file
from bubble_chamber.full_bmp_segmentation import FullBMPConfig,run_full_bmp
from bubble_chamber.s92_curve_review import run_s92
from bubble_chamber.s95_all_structures_review import run_s95
from bubble_chamber.s96_junction_workflow import run_s96
from bubble_chamber.s100_cross_geometry import ACCEPT,SOURCE_MASK
from bubble_chamber.s100_cross_review import run_s100_unlabeled_cross_review
from scripts.research.bubble_chamber.physicists_s100_unlabeled_cross_review import main


def _data(tmp,monkeypatch,with_mask=True):
    image=np.full((384,512),219,np.uint8)
    cv2.line(image,(170,210),(250,210),29,1)
    cv2.line(image,(210,170),(210,250),29,1)
    # Second, clearly disconnected track-looking line and intentionally
    # unannotated short squiggle test that no input line is deleted.
    cv2.line(image,(325,110),(475,110),29,1)
    cv2.line(image,(320,270),(355,285),31,1)
    bmp=tmp/"H50Z_030_01_405.bmp"
    assert cv2.imwrite(str(bmp),image)
    from bubble_chamber import full_bmp_segmentation as f
    monkeypatch.setattr(f,"predict_mask",
                        lambda gray,cfg,az_model=None:gray<90)
    s86=tmp/"S86"
    run_full_bmp(bmp,s86,FullBMPConfig(
        method="contrast",core_size=128,halo=32,preview_limit=0,
        min_component_area=1,min_segment_length_px=4,
        max_fragments_per_tile=2048,
    ))
    s92=tmp/"S92"
    run_s92(s86,s92,bmp_path=bmp,preview_limit=0)
    s95=tmp/"S95"
    run_s95(s86,s92,s95,bmp_path=bmp,max_side=512)
    s96=tmp/"S96"
    run_s96(s86,s95,s96,bmp_path=bmp,
            mask_junction_audit=with_mask,preview_limit=2)
    return bmp,image,s86,s92,s95,s96


def test_s100_true_source_pixel_fused_X_mask_split_not_known_fiducial(
    tmp_path,monkeypatch
):
    bmp,gray,s86,s92,s95,s96=_data(tmp_path,monkeypatch)
    source_sha=sha256_file(bmp)
    source_manifest_sha=sha256_file(s86/"manifest.json")
    output=tmp_path/"S100_REAL_SOURCE_ONLY_X_GEOMETRY"
    main([
        "--bmp",str(bmp),"--s86-dir",str(s86),
        "--s92-dir",str(s92),"--s95-dir",str(s95),
        "--s96-dir",str(s96),"--private-out-dir",str(output),
        "--case","SYNTHETIC_FOR_CODE_TEST_ONLY",
        "--max-preview-cases","8","--crop-size","320"
    ])
    report=json.loads((output/"report.json").read_text())
    gallery=json.loads((output/"preview_receipts.json").read_text())
    assert report["status"]=="SOURCE_SHA_VERIFIED_ORIGINAL_BMP_MACHINE_X_SHAPE_REVIEW_NOT_GT"
    assert report["source_bmp_sha256"]==source_sha
    assert report["source_bmp_name"]==bmp.name
    assert report["original_bmp_bytes_SHA_verified"]
    assert report["alternative_verified_S86_mask_junctions_scored"]>=1
    assert report["alternative_mask_source_integrity"][
        "source_alternative_mask_virtual_arms_exact_foreground_verified"
    ]>=4
    assert report["alternative_mask_source_integrity"][
        "S96_alternative_mask_original_components_replaced"
    ] is False
    assert report["four_arm_geometric_X_review_candidates"]>=1
    candidates=[
        json.loads(x) for x in
        (output/"candidates.ndjson").read_text().splitlines() if x.strip()
    ]
    assert report["total_machine_proposed_junctions_scored"]==len(candidates)
    assert any(
        x["cross_shape_review_status"]==ACCEPT and
        x["source_representation"]==SOURCE_MASK
        for x in candidates
    )
    assert all(x["not_automatic_calibration_fiducial_identity"] for x in candidates)
    assert all(x["physical_auto_join_authorized"] is False for x in candidates)
    assert all(x["overlap_vs_interaction_vs_fiducial_undecidable"] for x in candidates)
    assert gallery["source_original_sha256"]==source_sha
    assert len(gallery["cases"])>=1
    assert report["actual_original_film_previews_exported"]==len(gallery["cases"])
    for row in gallery["cases"]:
        x0,y0,x1,y1=row["source_xyxy"]
        original=cv2.imread(str(output/row["original_photo_png"]),cv2.IMREAD_GRAYSCALE)
        paired=cv2.imread(str(output/row["observed_source_overlay_png"]),cv2.IMREAD_COLOR)
        assert original.shape==(320,320)
        assert paired.shape==(320,640,3)
        assert np.array_equal(original,gray[y0:y1,x0:x1])
        assert np.array_equal(
            cv2.cvtColor(paired[:,:320],cv2.COLOR_BGR2GRAY),
            gray[y0:y1,x0:x1]
        )
        assert sha256_file(output/row["original_photo_png"])==row[
            "original_photo_png_sha256"
        ]
        assert sha256_file(output/row["observed_source_overlay_png"])==row[
            "observed_source_overlay_png_sha256"
        ]
        assert row["no_center_bridge_or_same_particle_identity_drawn"]
    archive=output/"S100_REAL_SOURCE_X_SHAPE_DIAGNOSIS_PRIVATE.zip"
    assert archive.is_file()
    assert archive.stat().st_size<8_000_000
    with ZipFile(archive) as z:
        assert "report.json" in z.namelist()
        assert "candidates.ndjson" in z.namelist()
        assert "index.html" in z.namelist()
        assert not any(x.endswith((".bmp",".pt",".pth")) for x in z.namelist())
        assert b"gt" not in z.read("report.json")[:10].lower()
    assert sha256_file(bmp)==source_sha
    assert sha256_file(s86/"manifest.json")==source_manifest_sha
    with pytest.raises(ValueError,match="NEW PRIVATE"):
        run_s100_unlabeled_cross_review(
            bmp_path=bmp,s86_dir=s86,s92_dir=s92,
            s95_dir=s95,s96_dir=s96,private_out_dir=output,
        )


def test_s100_safely_handles_S96_without_optional_mask_experiment(
    tmp_path,monkeypatch
):
    bmp,gray,s86,s92,s95,s96=_data(tmp_path,monkeypatch,with_mask=False)
    out=tmp_path/"S100_WITHOUT_MASK_SOURCE"
    result=run_s100_unlabeled_cross_review(
        bmp_path=bmp,s86_dir=s86,s92_dir=s92,s95_dir=s95,s96_dir=s96,
        private_out_dir=out,case_name="SYNTHETIC_ONLY",
        max_review_cases=0
    )
    assert result["alternative_verified_S86_mask_junctions_scored"]==0
    assert result["actual_original_film_previews_exported"]==0
    assert (out/"candidates.ndjson").is_file()
    assert (out/"index.html").is_file()
    assert (out/"S100_REAL_SOURCE_X_SHAPE_DIAGNOSIS_PRIVATE.zip").is_file()
    assert not any("mask_derived" in x["source_representation"].lower()
                   for x in [
                       json.loads(x) for x in
                       (out/"candidates.ndjson").read_text().splitlines()
                       if x.strip()
                   ])


def test_tampered_mask_alternative_arm_must_not_generate_real_source_photo(
    tmp_path,monkeypatch
):
    bmp,gray,s86,s92,s95,s96=_data(tmp_path,monkeypatch,with_mask=True)
    path=s96/"mask_derived_virtual_arms.ndjson"
    lines=path.read_text().splitlines()
    rows=[json.loads(x) for x in lines if x.strip()]
    assert rows
    rows[0]["points_xy"][0]=[1,1]
    path.write_text("\n".join(json.dumps(x) for x in rows)+"\n")
    target=tmp_path/"INVALID_MASK_SOURCE_NO_GALLERY"
    with pytest.raises(ValueError,match="non-observed pixel"):
        run_s100_unlabeled_cross_review(
            bmp_path=bmp,s86_dir=s86,s92_dir=s92,s95_dir=s95,s96_dir=s96,
            private_out_dir=target,case_name="SYNTHETIC_ONLY"
        )
    assert not target.exists()


def test_tampered_original_s96_endpoint_must_not_be_treated_as_observed(
    tmp_path,monkeypatch
):
    bmp,gray,s86,s92,s95,s96=_data(tmp_path,monkeypatch,with_mask=True)
    path=s96/"junction_hypotheses.ndjson"
    rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if not rows:
        pytest.skip("Synthetic fused cross can yield only mask-derived S96 junction")
    rows[0]["ports"][0]["point_xy"]=[1000,1000]
    path.write_text("\n".join(json.dumps(x) for x in rows)+"\n")
    target=tmp_path/"INVALID_PORT_SOURCE_NO_GALLERY"
    with pytest.raises(ValueError,match="port endpoint was not observed"):
        run_s100_unlabeled_cross_review(
            bmp_path=bmp,s86_dir=s86,s92_dir=s92,s95_dir=s95,s96_dir=s96,
            private_out_dir=target,case_name="SYNTHETIC_ONLY"
        )
    assert not target.exists()


def test_cross_photo_S96_source_mix_is_denied_before_image_export(
    tmp_path,monkeypatch
):
    bmp,gray,s86,s92,s95,s96=_data(tmp_path,monkeypatch,with_mask=True)
    report=s96/"report.json"
    data=json.loads(report.read_text())
    data["source_bmp_sha256"]="a"*64
    report.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="S96 source_bmp_sha256"):
        run_s100_unlabeled_cross_review(
            bmp_path=bmp,s86_dir=s86,s92_dir=s92,s95_dir=s95,s96_dir=s96,
            private_out_dir=tmp_path/"MUST_NOT_EXPORT",
        )
    assert not (tmp_path/"MUST_NOT_EXPORT").exists()
