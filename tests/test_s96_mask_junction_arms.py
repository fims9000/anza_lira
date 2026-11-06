"""S96 mask-derived virtual arm splitting, never altering original S86 masks."""
from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from bubble_chamber.bitmap_tiles import sha256_file
from bubble_chamber.full_bmp_segmentation import FullBMPConfig,run_full_bmp
from bubble_chamber.s87_seam_continuity import _read_source
from bubble_chamber.s92_curve_review import run_s92
from bubble_chamber.s95_all_structures_review import run_s95
from bubble_chamber.s96_mask_junction_arms import analyze_s86_mask_junctions
from bubble_chamber.s96_junction_workflow import run_s96


def _fused_cross(tmp_path,center=(210,210)):
    cx,cy=center
    image=np.full((384,512),220,np.uint8)
    cv2.line(image,(cx-40,cy),(cx+40,cy),28,1)
    cv2.line(image,(cx,cy-40),(cx,cy+40),28,1)
    bmp=tmp_path/"H50Z_030_01_405.bmp"
    assert cv2.imwrite(str(bmp),image)
    return bmp


def _s86(tmp_path,monkeypatch,bmp):
    from bubble_chamber import full_bmp_segmentation as f
    monkeypatch.setattr(f,"predict_mask",lambda gray,cfg,az_model=None:gray<80)
    out=tmp_path/"S86"
    run_full_bmp(bmp,out,FullBMPConfig(
        method="contrast",core_size=128,halo=32,
        min_component_area=1,min_segment_length_px=4,preview_limit=0,
        max_fragments_per_tile=2048
    ))
    return out


def test_moore_pixel_skeleton_four_arms_without_changing_any_original_source_mask(
    tmp_path,monkeypatch
):
    bmp=_fused_cross(tmp_path)
    source_sha=sha256_file(bmp)
    s86=_s86(tmp_path,monkeypatch,bmp)
    manifest,records,_=_read_source(s86)
    pristine={r["tile_id"]:sha256_file(s86/r["mask_file"]) for r in records}
    arms,junctions,pairs,chains,report=analyze_s86_mask_junctions(s86,records)
    assert report["status"]=="ALTERNATIVE_PIXEL_BRANCH_HYPOTHESES_PRIVATE_ONLY"
    assert report["source_mask_junction_candidates"]>=1
    assert report["source_skeleton_arm_hypotheses"]>=4
    assert report["four_arm_crossing_pair_hypotheses"]>=2
    assert report["source_S86_mask_tiles_changed"]==0
    assert report["all_original_S86_fragment_IDs_preserved"] is True
    assert report["physical_auto_joins_authorized"] is False
    assert all(x["original_S86_fragments_replaced"] is False for x in arms)
    assert all(x["source_core_skeleton_observed_pixels_only"] for x in arms)
    assert all(x["physical_auto_join"] is False for x in pairs)
    assert all(x["same_particle_identity_verified"] is False for x in chains)
    assert {r["tile_id"]:sha256_file(s86/r["mask_file"]) for r in records}==pristine
    assert sha256_file(bmp)==source_sha


def test_mask_junction_workflow_reuses_S95_but_never_mixes_virtual_and_original(
    tmp_path,monkeypatch
):
    bmp=_fused_cross(tmp_path)
    s86=_s86(tmp_path,monkeypatch,bmp)
    s92=tmp_path/"S92"
    run_s92(s86,s92,bmp_path=bmp,preview_limit=0)
    s95=tmp_path/"S95"
    run_s95(s86,s92,s95,bmp_path=bmp,max_side=512)
    s86_sha=sha256_file(s86/"manifest.json")
    result=run_s96(
        s86,s95,tmp_path/"WITH_MASK",bmp_path=bmp,
        mask_junction_audit=True,preview_limit=16
    )
    assert result["optional_alternative_mask_junction_audit"] is not None
    r=result["optional_alternative_mask_junction_audit"]
    assert r["source_mask_junction_candidates"]>=1
    assert result["mask_derived_blind_review_cases"]>=1
    assert result["mask_derived_blind_review_page"].endswith("index.html")
    assert Path(result["mask_derived_blind_review_page"]).is_file()
    source_count=result["geometry"]["source_components_preserved"]
    assert result["source_observed_centerline_components"]==source_count
    assert result["source_source_line_pixels_modified"] is False
    assert result["s96_original_centerlines_never_replaced_by_mask_virtual_arms"] is True
    mask_folder=tmp_path/"WITH_MASK"
    synthetic=[json.loads(l) for l in (
        mask_folder/"mask_derived_virtual_arms.ndjson"
    ).read_text().splitlines() if l.strip()]
    original=[json.loads(l) for l in (
        mask_folder/"original_observed_component_inventory.ndjson"
    ).read_text().splitlines() if l.strip()]
    assert {x["component_id"] for x in synthetic}.isdisjoint(
        {x["component_id"] for x in original}
    )
    assert len(original)==source_count
    assert sha256_file(s86/"manifest.json")==s86_sha


def test_crossing_center_at_tile_seam_is_not_declared_safe(tmp_path,monkeypatch):
    bmp=_fused_cross(tmp_path,center=(128,128))
    s86=_s86(tmp_path,monkeypatch,bmp)
    _,records,_=_read_source(s86)
    arms,junctions,pairs,chains,report=analyze_s86_mask_junctions(s86,records)
    assert report["four_arm_crossing_pair_hypotheses"]==0
    assert report["source_mask_junction_tile_status"].get(
        "UNCERTAIN_NEAR_TILE_EDGE_JUNCTION",0
    )>=1 or report["source_mask_junction_candidates"]==0


def test_invalid_S86_mask_sha_or_budget_abstains_not_silent(tmp_path,monkeypatch):
    bmp=_fused_cross(tmp_path)
    s86=_s86(tmp_path,monkeypatch,bmp)
    _,records,_=_read_source(s86)
    arms,junctions,pairs,chains,report=analyze_s86_mask_junctions(
        s86,records,max_skeleton_pixels_per_tile=100,
    )
    assert report["source_mask_junction_tile_status"].get(
        "MASK_SKELETON_BUDGET_ABSTAIN",0
    )>=1
    assert report["tiles_with_dense_or_budget_abstention"]
    shard=s86/next(r["mask_file"] for r in records if r["foreground_pixel_count"]>0)
    shard.write_bytes(b"tampered")
    with pytest.raises(ValueError,match="Changed/missing"):
        analyze_s86_mask_junctions(s86,records)
