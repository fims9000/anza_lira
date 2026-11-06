"""Audit actual source S86-S96 files, never infer true physics tracks.

This synthetic fixture exercises the real write/read interfaces; it does NOT
measure H50Z sensitivity or validate human visual tracking.
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
from bubble_chamber.real_bmp_run_receipts import (
    verify_stage_receipts,summarize_verification,html_summary
)
from scripts.research.bubble_chamber.physicists_verify_real_h50z_saved_results import main


def _inputs(tmp_path,monkeypatch):
    gray=np.full((256,384),220,dtype=np.uint8)
    cv2.line(gray,(20,70),(130,70),22,1)
    cv2.line(gray,(140,70),(220,70),22,1)
    cv2.line(gray,(200,110),(200,205),22,1)
    bmp=tmp_path/"H50Z_030_01_405.bmp"
    assert cv2.imwrite(str(bmp),gray)
    from bubble_chamber import full_bmp_segmentation as m
    monkeypatch.setattr(m,"predict_mask",lambda gray,cfg,az_model=None: gray<80)
    s86=tmp_path/"S86"
    run_full_bmp(bmp,s86,FullBMPConfig(
        method="contrast",core_size=128,halo=32,
        preview_limit=0,min_component_area=1,
        min_segment_length_px=4,max_fragments_per_tile=512
    ))
    s92=tmp_path/"S92"
    run_s92(s86,s92,bmp_path=bmp,preview_limit=0)
    s95=tmp_path/"S95"
    run_s95(s86,s92,s95,bmp_path=bmp,max_side=512)
    s96=tmp_path/"S96"
    run_s96(s86,s95,s96,bmp_path=bmp,preview_limit=4)
    return bmp,s86,s92,s95,s96


def test_all_saved_sources_are_real_and_sha_verified_not_just_report_counts(tmp_path,monkeypatch):
    bmp,s86,s92,s95,s96=_inputs(tmp_path,monkeypatch)
    original_sha=sha256_file(bmp)
    source=verify_stage_receipts(
        case_name="H50Z_405_SYNTHETIC",
        s86_dir=s86,s92_dir=s92,s95_dir=s95,s96_dir=s96,
        original_bmp=bmp
    )
    assert source["status"]=="VERIFIED_SAVED_MACHINE_STAGE_ARTIFACTS_NOT_HUMAN_GT"
    assert source["actual_original_BMP_bytes_verified"] is True
    assert source["original_bmp_sha256"]==original_sha
    assert source["s86"]["candidate_fragments"]>=2
    assert source["s86"]["source_components_after_S87"]>=1
    assert source["s95"]["typed_observed_components"]==source["s86"]["source_components_after_S87"]
    assert sum(source["s95"]["all_categories"].values())==source["s95"]["typed_observed_components"]
    assert source["s96"]["source_S86_components_preserved"]==source["s95"]["typed_observed_components"]
    assert source["s92"]["all_generated_gap_proposals"]>=0
    assert source["true_particle_or_neutron_collision_metrics_computable"] is False
    assert source["full_real_photo_precision_recall_F1_computable"] is False
    assert sha256_file(bmp)==original_sha
    summary=summarize_verification([source])
    assert summary["verified_case_count"]==1
    assert summary["real_bmp_bytes_verified_case_count"]==1
    assert "не является" in html_summary(summary)


def test_missing_bmp_means_only_source_chain_not_original_image_bytes(tmp_path,monkeypatch):
    bmp,s86,s92,s95,s96=_inputs(tmp_path,monkeypatch)
    result=verify_stage_receipts(
        case_name="source_only",s86_dir=s86,s92_dir=s92,s95_dir=s95
    )
    assert result["actual_original_BMP_bytes_verified"] is False
    assert result["s96"] is None
    assert result["original_bmp_sha256"]==sha256_file(bmp)
    assert summarize_verification([result])["cases_missing_S96_evaluation"]==["source_only"]


def test_cross_photo_source_mix_rejected_before_reporting(tmp_path,monkeypatch):
    bmp,s86,s92,s95,s96=_inputs(tmp_path,monkeypatch)
    wrong=tmp_path/"H50Z_030_01_406.bmp"
    wrong.write_bytes(bmp.read_bytes())
    with pytest.raises(ValueError,match="filename"):
        verify_stage_receipts(
            case_name="405",s86_dir=s86,s92_dir=s92,s95_dir=s95,
            s96_dir=s96,original_bmp=wrong
        )
    s95_report=s95/"report.json"
    meta=json.loads(s95_report.read_text())
    meta["source_s92_report_sha256"]="0"*64
    s95_report.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match="S95 source_s92_report_sha256"):
        verify_stage_receipts(
            case_name="405",s86_dir=s86,s92_dir=s92,s95_dir=s95
        )


def test_mutated_real_mask_shard_rejected_as_non_observed_pixels(tmp_path,monkeypatch):
    bmp,s86,s92,s95,s96=_inputs(tmp_path,monkeypatch)
    mask=next((s86/"masks").glob("*.png"))
    mask.write_bytes(b"fake mask cannot silently count as detected source pixels")
    with pytest.raises(ValueError,match="modified S86 mask shard"):
        verify_stage_receipts(
            case_name="405",s86_dir=s86,s92_dir=s92,s95_dir=s95,
            s96_dir=s96,original_bmp=bmp
        )


def test_verify_CLI_generates_local_html_and_json_only_after_success(tmp_path,monkeypatch):
    bmp,s86,s92,s95,s96=_inputs(tmp_path,monkeypatch)
    output=tmp_path/"PRIVATE_RECEIPT"
    case="|".join(["SYNTHETIC_ONLY",str(s86),str(s92),str(s95),str(s96),str(bmp)])
    main(["--case",case,"--private-out-dir",str(output)])
    result=json.loads((output/"source_verified_machine_counts.json").read_text())
    assert result["verified_case_count"]==1
    assert result["per_case"][0]["actual_original_BMP_bytes_verified"]
    assert (output/"index.html").is_file()
    with pytest.raises(ValueError,match="NEW private"):
        main(["--case",case,"--private-out-dir",str(output)])
    with pytest.raises(FileNotFoundError,match="source"):
        main(["--case","MISSING|/unknown/S86|/unknown/S92|/unknown/S95",
              "--private-out-dir",str(tmp_path/"NO_RESULT")])
    assert not (tmp_path/"NO_RESULT").exists()


def test_empty_or_duplicate_multifilm_labels_fail():
    with pytest.raises(ValueError,match="one or more"):
        summarize_verification([])
    with pytest.raises(ValueError,match="one or more"):
        summarize_verification([{"case":"same"},{"case":"same"}])
