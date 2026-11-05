"""S93 genuine file-to-JPEG pipeline on synthetic but source-locked BMP.

No huge original BMPs or private physics footage used in CI.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from bubble_chamber.bitmap_tiles import GrayBMP8,sha256_file
from bubble_chamber.s93_compact_real_overlay import (
    low_memory_gray_preview, run_one_command,
)


def create_gray_bmp(tmp_path):
    img=np.full((512,1024),220,dtype=np.uint8)
    # Simulated interrupted long "track" and misleading V shaped scratch.
    cv2.line(img,(15,125),(314,125),35,1,lineType=cv2.LINE_8)
    cv2.line(img,(332,125),(1010,125),35,1,lineType=cv2.LINE_8)
    cv2.line(img,(85,370),(104,351),30,1,lineType=cv2.LINE_8)
    cv2.line(img,(104,351),(123,370),30,1,lineType=cv2.LINE_8)
    t=np.linspace(0.,4*np.pi,240)
    r=5+1.1*t
    coil=np.rint(np.column_stack((590+r*np.cos(t),355+r*np.sin(t)))).astype(np.int32)
    cv2.polylines(img,[coil.reshape(-1,1,2)],False,25,1,lineType=cv2.LINE_8)
    source=tmp_path/"H50Z_TEST_SYNTHETIC.bmp"
    assert cv2.imwrite(str(source),img)
    return source


def test_real_gray_bmp_memory_map_downsample_preserves_coords(tmp_path):
    path=create_gray_bmp(tmp_path)
    bmp=GrayBMP8(path)
    small,scale=low_memory_gray_preview(bmp,max_side=512)
    assert small.shape==(256,512)
    assert scale==pytest.approx(.5)
    assert small.dtype==np.uint8
    assert small[0,0]==220
    # A single original dark pixel line must survive bounded-memory
    # anti-aliased reduction even when it falls between nearest-neighbor rows.
    assert np.min(small[60:66,20:160])<220
    with pytest.raises(ValueError,match="512..4096"):
        low_memory_gray_preview(bmp,max_side=256)


def test_one_command_generates_actual_small_jpg_and_review_sheet(tmp_path,monkeypatch):
    source=create_gray_bmp(tmp_path)
    from bubble_chamber import full_bmp_segmentation
    monkeypatch.setattr(
        full_bmp_segmentation,"predict_mask",
        lambda gray,cfg,az_model=None:gray<80
    )
    prior_sha=sha256_file(source)
    out=tmp_path/"PRIVATE_ANZA_PHOTO_405"
    report=run_one_command(
        bmp_path=source,private_root=out,
        method="contrast",max_side=512,preview_limit=6,
    )
    assert report["status"]=="COMPLETE_REAL_SOURCE_MACHINE_OVERLAY_NOT_PHYSICAL_GROUND_TRUTH"
    assert report["physical_auto_joins_authorized"] is False
    assert report["source_SHA256"]==prior_sha
    original=Path(report["small_original_photo"])
    annotated=Path(report["small_annotated_photo"])
    comparison=Path(report["comparison_montage"])
    assert original.is_file() and annotated.is_file() and comparison.is_file()
    raw=cv2.imread(str(original))
    mask=cv2.imread(str(annotated))
    assert raw is not None and mask is not None
    assert raw.shape==mask.shape==(256,512,3)
    assert np.max(np.abs(raw.astype(int)-mask.astype(int)))>45
    assert comparison.stat().st_size<15_000_000
    receipt=json.loads((out/"SMALL_IMAGE"/"compact_preview_manifest.json").read_text())
    assert receipt["original_small_sha256"]==sha256_file(original)
    assert receipt["annotated_small_sha256"]==sha256_file(annotated)
    assert receipt["source_bmp_sha256"]==prior_sha
    assert receipt["source_photo_not_edited"] is True
    assert receipt["unobserved_bridge_pixels_never_presented_as_source"] is True
    assert sha256_file(source)==prior_sha

    reloaded=run_one_command(
        bmp_path=source,private_root=out,
        method="contrast",max_side=512,preview_limit=6,
        resume=True,
    )
    assert reloaded["small_annotated_photo"]==report["small_annotated_photo"]
    assert sha256_file(annotated)==receipt["annotated_small_sha256"]
    with pytest.raises(FileExistsError):
        run_one_command(
            bmp_path=source,private_root=out,method="contrast",
            max_side=512,preview_limit=6,
        )
    with pytest.raises(ValueError,match="changed on resume"):
        run_one_command(
            bmp_path=source,private_root=out,method="contrast",
            max_side=1024,preview_limit=6,resume=True,
        )


def test_one_command_hard_rejects_untrained_anza_and_tampered_overlay(tmp_path,monkeypatch):
    source=create_gray_bmp(tmp_path)
    output=tmp_path/"PRIVATE_RESULTS"
    with pytest.raises(ValueError,match="checkpoint"):
        run_one_command(source,output,method="azunet")
    from bubble_chamber import full_bmp_segmentation
    monkeypatch.setattr(
        full_bmp_segmentation,"predict_mask",
        lambda gray,cfg,az_model=None:gray<80
    )
    result=run_one_command(
        source,output,method="contrast",max_side=512,preview_limit=0
    )
    assert result["comparison_montage"] is None
    screenshot=Path(result["small_annotated_photo"])
    screenshot.write_bytes(b"tampered")
    with pytest.raises(ValueError,match="resume SHA mismatch"):
        run_one_command(
            source,output,method="contrast",max_side=512,
            preview_limit=0,resume=True
        )
