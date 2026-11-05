"""S92 whole-BMP source-proofed read-only workflow; ALL photos generated."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from bubble_chamber.bitmap_tiles import sha256_file
from bubble_chamber.full_bmp_segmentation import FullBMPConfig,run_full_bmp
from bubble_chamber.s92_curve_review import run_s92


def make_fake_bmp(tmp_path):
    gray=np.full((306,333),210,dtype=np.uint8)
    # Long genuinely observed dashed line over three image tiles.
    cv2.line(gray,(15,55),(107,55),30,1)
    cv2.line(gray,(122,55),(310,55),30,1)
    # Tiny V-shaped glyph, unrelated to the long line.
    cv2.line(gray,(55,180),(63,188),30,1)
    cv2.line(gray,(63,188),(71,180),30,1)
    # Coil with spatially varying radius, not a synthetic classification GT.
    t=np.linspace(0,4*np.pi,500)
    radius=5+1.15*t
    xy=np.rint(np.column_stack((
        215+radius*np.cos(t),205+radius*np.sin(t)
    ))).astype("int32")
    cv2.polylines(gray,[xy.reshape(-1,1,2)],False,25,1)
    bmp=tmp_path/"SYNTHETIC_DATA_ONLY.bmp"
    assert cv2.imwrite(str(bmp),gray)
    return bmp


def make_s86(monkeypatch,tmp_path,original):
    import bubble_chamber.full_bmp_segmentation as s86
    monkeypatch.setattr(s86,"predict_mask",
                        lambda pixels,cfg,az_model=None: pixels<80)
    folder=tmp_path/"PRIVATE_S86"
    config=FullBMPConfig(
        method="contrast",core_size=128,halo=32,
        min_component_area=2,min_segment_length_px=5,preview_limit=0
    )
    run_full_bmp(original,folder,config)
    return folder


def test_private_full_bmp_s92_sees_dashes_and_keeps_raw_masks(tmp_path,monkeypatch):
    source=make_fake_bmp(tmp_path)
    s86=make_s86(monkeypatch,tmp_path,source)
    original_sha=sha256_file(source)
    manifest_sha=sha256_file(s86/"manifest.json")
    shard_hashes={p.name:sha256_file(p) for p in (s86/"masks").glob("*.png")}
    output=tmp_path/"PRIVATE_S92"
    report=run_s92(s86,output,bmp_path=source,preview_limit=12)
    assert report["status"]=="COMPLETE_PRIVATE_REVIEW_ONLY"
    assert report["source_fragments"]>=3
    assert report["observed_components_after_S87"]>=2
    assert report["gap"]["gap_review_links"]>=1
    assert report["physical_auto_joins_authorized"] is False
    assert report["source_line_pixels_mutated"] is False
    assert report["physical_GT_count"]==0
    assert report["image_photometry_available"] is True
    assert sha256_file(source)==original_sha
    assert sha256_file(s86/"manifest.json")==manifest_sha
    assert {p.name:sha256_file(p) for p in (s86/"masks").glob("*.png")}==shard_hashes
    shapes=[json.loads(s) for s in (output/"observed_shape_review.ndjson").read_text().splitlines()]
    assert len(shapes)==report["observed_components_after_S87"]
    assert all(x["source_fragment_remains_available"] for x in shapes)
    proposals=[json.loads(s) for s in (output/"gap_proposals_unknown.ndjson").read_text().splitlines()]
    assert all(p["status"]=="UNKNOWN" and p["physical_auto_join"] is False for p in proposals)
    assert any(p["review_decision"]=="PRIORITY_GAP_REVIEW_NOT_PHYSICAL_ACCEPT" for p in proposals)
    chains=[json.loads(s) for s in (output/"stitched_chain_hypotheses_unknown.ndjson").read_text().splitlines()]
    assert chains and all(x["not_a_verified_particle_track"] for x in chains)
    assert all(all(gap["inferred_only_points_xy"] for gap in c["inferred_gaps"]) for c in chains)
    gallery=(output/"index.html").read_text(encoding="utf8")
    assert "S92" in gallery and "НЕ физическая разметка" in gallery
    assert report["preview_count"]>=1
    assert list((output/"preview").glob("*_source.png"))
    assert list((output/"preview").glob("*_overlay.png"))
    with pytest.raises(FileExistsError):
        run_s92(s86,output,bmp_path=source)


def test_s92_no_BMP_degrades_to_geometry_only_without_claimed_photometry(tmp_path,monkeypatch):
    original=make_fake_bmp(tmp_path)
    s86=make_s86(monkeypatch,tmp_path,original)
    out=tmp_path/"NO_PHOTO"
    result=run_s92(s86,out,preview_limit=8)
    assert result["image_photometry_available"] is False
    assert result["preview_count"]==0
    assert result["source_bmp_sha256"]==sha256_file(original)
    assert (out/"observed_shape_review.ndjson").is_file()


def test_s92_rejects_wrong_photo_and_mutated_S86_masks(tmp_path,monkeypatch):
    bmp=make_fake_bmp(tmp_path)
    s86=make_s86(monkeypatch,tmp_path,bmp)
    wrong=tmp_path/"UNRELATED.bmp"
    arr=np.full((306,333),255,dtype=np.uint8)
    assert cv2.imwrite(str(wrong),arr)
    with pytest.raises(ValueError,match="BMP SHA"):
        run_s92(s86,tmp_path/"INVALID",bmp_path=wrong)
    shard=next((s86/"masks").glob("*.png"))
    shard.write_bytes(b"corrupt")
    with pytest.raises(ValueError,match="modified S86 mask"):
        run_s92(s86,tmp_path/"TAMPERED",bmp_path=bmp)


def test_s92_refuses_repository_or_nested_source_destination(tmp_path,monkeypatch):
    src=make_fake_bmp(tmp_path)
    s86=make_s86(monkeypatch,tmp_path,src)
    import bubble_chamber.s92_curve_review as module
    with pytest.raises(ValueError,match="outside Git"):
        run_s92(s86,Path(module.__file__).resolve().parents[1]/"WRONG_OUTPUT")
    with pytest.raises(ValueError,match="outside Git"):
        run_s92(s86,s86/"nested")
