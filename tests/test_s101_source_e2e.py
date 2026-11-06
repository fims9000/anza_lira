"""S101 real source-format end-to-end with artificial grayscale original BMP.

Exercises ACTUAL S86 source mask writes and S92 source gap receipts, then
S101 optimized global review alternatives; never claims real H50Z results.
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from bubble_chamber.bitmap_tiles import sha256_file
from bubble_chamber.full_bmp_segmentation import FullBMPConfig,run_full_bmp
from bubble_chamber.s92_curve_review import run_s92
from scripts.research.bubble_chamber.physicists_s101_global_gap_ablation import (
    run_s101_gap_ablation,main
)


def _make(tmp,monkeypatch):
    original=np.full((256,384),225,np.uint8)
    # Three separate observed-source lines on the same manually imagined
    # 2D long trace. No actual same-particle statement.
    for start,end in ((20,75),(85,140),(150,205),(215,270)):
        cv2.line(original,(start,90),(end,90),28,1)
    # separate V-shaped artifact
    cv2.line(original,(300,145),(315,160),28,1)
    cv2.line(original,(315,160),(330,145),28,1)
    bmp=tmp/"ARTIFICIAL_S101_SOURCE_NOT_REAL_H50Z.bmp"
    assert cv2.imwrite(str(bmp),original)
    from bubble_chamber import full_bmp_segmentation as segmentation
    monkeypatch.setattr(segmentation,"predict_mask",lambda gray,cfg,az_model=None:gray<90)
    s86=tmp/"ORIGINAL_SYNTHETIC_S86"
    run_full_bmp(
        bmp,s86,FullBMPConfig(
            method="contrast",core_size=128,halo=32,
            preview_limit=0,min_component_area=1,
            min_segment_length_px=4,max_fragments_per_tile=256
        )
    )
    s92=tmp/"ORIGINAL_SYNTHETIC_S92"
    run_s92(s86,s92,bmp_path=bmp,preview_limit=0)
    return bmp,s86,s92


def test_actual_saved_S86_and_S92_pipelines_yield_immutable_S101_optimization(
    tmp_path,monkeypatch
):
    bmp,s86,s92=_make(tmp_path,monkeypatch)
    hashes={
        p:sha256_file(p) for p in (
            bmp,s86/"manifest.json",
            s92/"report.json",
            s92/"gap_proposals_unknown.ndjson"
        )
    }
    output=tmp_path/"S101_PRIVATE"
    main([
        "--s86-dir",str(s86),"--s92-dir",str(s92),
        "--private-out-dir",str(output)
    ])
    report=json.loads((output/"report.json").read_text())
    rows=[
        json.loads(line)
        for line in (output/"global_gap_review_unknown.ndjson").read_text().splitlines()
        if line.strip()
    ]
    assert report["status"]=="S101_SAME_S92_CANDIDATE_GLOBAL_LINEAR_FOREST_ALTERNATIVE"
    assert report["source_bmp_sha256"]==sha256_file(bmp)
    assert report["source_s86_manifest_sha256"]==sha256_file(s86/"manifest.json")
    assert len(rows)==report["source_S92_all_candidate_count"]
    assert all(not row["physical_auto_join"] for row in rows)
    assert all(row["original_centerline_pixels_altered"] is False for row in rows)
    assert report["no_cycles_in_selected_source_component_graph"]
    assert report["original_s92_greedy_geometry_objective"]>=0
    assert report["independent_S97_S98_human_reference_comparison"] is None
    for path,checksum in hashes.items():
        assert sha256_file(path)==checksum
    with pytest.raises(ValueError,match="NEW PRIVATE"):
        run_s101_gap_ablation(
            s86_dir=s86,s92_dir=s92,private_out_dir=output
        )


def test_wrong_independent_partial_S97_pack_cannot_be_passed_as_same_photo(
    tmp_path,monkeypatch
):
    bmp,s86,s92=_make(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match="must both"):
        run_s101_gap_ablation(
            s86_dir=s86,s92_dir=s92,
            private_out_dir=tmp_path/"WRONG_PARTIAL",
            independent_s97_pack_dir=tmp_path/"WRONG_REFERENCE"
        )
    assert not (tmp_path/"WRONG_PARTIAL").exists()


def test_forged_S92_priorities_and_wrong_source_manifest_rejected_before_writes(
    tmp_path,monkeypatch
):
    bmp,s86,s92=_make(tmp_path,monkeypatch)
    reportpath=s92/"report.json"
    report=json.loads(reportpath.read_text())
    report["gap"]["gap_review_links"]+=1
    reportpath.write_text(json.dumps(report),encoding="utf-8")
    out=tmp_path/"DONT_EXPORT_INVALID"
    with pytest.raises(ValueError,match="decisions/counts"):
        run_s101_gap_ablation(
            s86_dir=s86,s92_dir=s92,private_out_dir=out
        )
    assert not out.exists()



def test_independent_original_only_S97_2D_line_benchmark_is_same_source_locked(
    tmp_path,monkeypatch
):
    from bubble_chamber.s97_reference_viewer import prepare_independent_source_rois
    bmp,s86,s92=_make(tmp_path,monkeypatch)
    root=tmp_path/"HUMAN_ONLY_ORIGINAL_SOURCE"
    pack=prepare_independent_source_rois(
        bmp,root,roi_xyxy_source=[[0,0,384,256]],tile_size=512
    )
    reference={
        "schema":pack["schema"],
        "source_bmp_name":pack["source_bmp_name"],
        "source_bmp_sha256":pack["source_bmp_sha256"],
        "image_shape_yx":pack["image_shape_yx"],
        "roi_xyxy_source":pack["roi_xyxy_source"],
        "source_reference_pack_sha256":sha256_file(root/"reference_pack.json"),
        "image_was_unmodified_original":True,
        "machine_proposals_visible_during_annotation":False,
        "full_exhaustive_negative_segmentation":False,
        "data_source":"INDEPENDENT_OPERATOR_ORIGINAL_BMP_ONLY",
        "reference_semantics":"PARTIAL_VISIBLE_2D_STROKES_NO_PARTICLE_GT",
        "reviewer_id":"synthetic_source_independent",
        "strokes":[{
            "stroke_id":"original_source_human",
            "visual_instance_id":"2D_visually_same_trajectory_only",
            "kind":"TRACK_LIKE","certainty":"CLEAR",
            "points_xy_source_px":[[float(x),90.] for x in range(20,271,10)],
            "note":"Generated synthetic physical-image-like stroke, no particle identity",
        }]
    }
    human=tmp_path/"INDEPENDENT_SOURCE_STROKES.json"
    human.write_text(json.dumps(reference),encoding="utf-8")
    out=tmp_path/"S101_COMPARED_TO_HUMAN_VISUAL_SOURCE"
    main([
        "--s86-dir",str(s86),
        "--s92-dir",str(s92),
        "--s97-pack-dir",str(root),
        "--s97-human-reference",str(human),
        "--private-out-dir",str(out)
    ])
    report=json.loads((out/"report.json").read_text())
    performance=report["independent_S97_S98_human_reference_comparison"]
    assert performance["independent_human_source_pack_sha256"]==sha256_file(
        root/"reference_pack.json"
    )
    assert performance["independent_human_annotations_sha256"]==sha256_file(human)
    assert performance["conditional_visual_2D_source_piece_adjacency_only"]
    assert performance["not_complete_physical_track_link_recall_or_precision"]
    assert performance["assessable_human_positive_source_piece_boundaries"]>=1
    assert performance["old_s92_selected_boundary_coverage"] is not None
    assert performance["s101_global_selected_boundary_coverage"] is not None
    assert report["no_claim_of_validation_improvement_without_independent_S97_S98_reference"]
