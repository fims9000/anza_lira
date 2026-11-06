"""End-to-end user path for SMALL SYNTHETIC BMPs only; not real H50Z GT.

The code under test exports original pixel-exact native BMP crops, two
separate archives for blind human review / AFTER-review machine hypotheses,
source S86-S96 SHA receipts and deterministic class-sampled candidate IDs.
It never uploads any real private source photo to GitHub.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from zipfile import ZipFile
import csv
import io
import json

import cv2
import numpy as np
import pytest

from bubble_chamber.bitmap_tiles import sha256_file
from bubble_chamber.full_bmp_segmentation import FullBMPConfig,run_full_bmp
from bubble_chamber.s92_curve_review import run_s92
from bubble_chamber.s95_all_structures_review import run_s95
from bubble_chamber.s96_junction_workflow import run_s96
from bubble_chamber.real_h50z_compact_evidence import (
    SCHEMA,EXPECTED,package_verified_real_h50z,
    evaluate_blind_component_cases
)
from scripts.research.bubble_chamber.physicists_real_h50z_reproduce import main


def _real_synthetic_bmp(tmp):
    image=np.full((256,384),223,np.uint8)
    cv2.line(image,(25,45),(162,45),28,1)
    cv2.line(image,(180,45),(330,45),28,1)
    cv2.line(image,(35,120),(75,160),22,1)
    cv2.line(image,(75,160),(115,120),22,1)
    cv2.line(image,(185,150),(255,170),23,1)
    bmp=tmp/"SYNTHETIC_H50Z_STYLE_NOT_ORIGINAL_FILM.bmp"
    assert cv2.imwrite(str(bmp),image)
    return bmp,image


def _sources(tmp,monkeypatch,*,include_s96=True):
    bmp,img=_real_synthetic_bmp(tmp)
    from bubble_chamber import full_bmp_segmentation as f
    monkeypatch.setattr(f,"predict_mask",
                        lambda gray,cfg,az_model=None:gray<90)
    s86=tmp/"ACTUAL_SYNTHETIC_S86"
    run_full_bmp(bmp,s86,FullBMPConfig(
        method="contrast",core_size=128,halo=32,
        preview_limit=0,min_component_area=1,
        min_segment_length_px=4,max_fragments_per_tile=1024
    ))
    s92=tmp/"ACTUAL_SYNTHETIC_S92"
    run_s92(s86,s92,bmp_path=bmp,preview_limit=0)
    s95=tmp/"ACTUAL_SYNTHETIC_S95"
    run_s95(s86,s92,s95,bmp_path=bmp,max_side=512)
    s96=None
    if include_s96:
        s96=tmp/"ACTUAL_SYNTHETIC_S96"
        run_s96(s86,s95,s96,bmp_path=bmp,preview_limit=0)
    return bmp,img,s86,s92,s95,s96


def test_compact_source_verified_packages_open_blind_zip_without_leaking_class(
    tmp_path,monkeypatch
):
    bmp,img,s86,s92,s95,s96=_sources(tmp_path,monkeypatch)
    origsha=sha256_file(bmp)
    s86sha=sha256_file(s86/"manifest.json")
    root=tmp_path/"LOCAL_H50Z_EVIDENCE"
    data=package_verified_real_h50z(
        bmp_path=bmp,s86_dir=s86,s92_dir=s92,s95_dir=s95,
        s96_dir=s96,private_out_dir=root,
        max_per_class=2,crop_size=128,case_id="SYNTHETIC_ONLY"
    )
    assert data["status"]=="TWO_SEPARATE_VERIFIED_PRIVATE_H50Z_DIAGNOSTIC_ARCHIVES"
    assert data["source_bmp_sha256"]==origsha
    assert data["blind_source_examples"]>=2
    assert data["original_verified_S86_S95_count"]>=2
    blind=root/data["archives"]["blind"]["file"]
    after=root/data["archives"]["posthoc_machine"]["file"]
    assert blind.stat().st_size<8_000_000
    assert after.stat().st_size<8_000_000
    assert sha256_file(blind)==data["archives"]["blind"]["sha256"]
    assert sha256_file(after)==data["archives"]["posthoc_machine"]["sha256"]
    with ZipFile(blind) as zip:
        names=zip.namelist()
        assert "index.html" in names
        assert "review_template.csv" in names
        assert not any("overlay" in x or "machine" in x or "report" in x for x in names)
        assert all(not x.endswith((".bmp",".pt",".pth")) for x in names)
        text=zip.read("index.html").decode()
        assert "S95" not in text
        assert "machine category" not in text.lower()
        assert "оригинальной" in text or "исходной" in text
        view=csv.DictReader(io.StringIO(zip.read("review_template.csv").decode()))
        rows=list(view)
        assert len(rows)==data["blind_source_examples"]
        assert all(not r["human_label"] and not r["certainty"] for r in rows)
        assert "machine_category" not in view.fieldnames
    with ZipFile(after) as zip:
        assert "verified_machine_only_report.json" in zip.namelist()
        assert "source_machine_classes.csv" in zip.namelist()
        assert not any(x.endswith(".bmp") for x in zip.namelist())
        man=json.loads(zip.read("verified_machine_only_report.json"))
        assert man["source_bmp_sha256"]==origsha
        assert man["human_visual_labels_not_yet_present"] is True
        assert man["sampling_conditioned_on_S95_machine_class_not_unbiased_GT"]
        assert len(man["sampled_original_photo_component_cases"])==len(rows)
        for sample in man["sampled_original_photo_component_cases"]:
            sid=sample["sample_id"]
            raw=cv2.imdecode(np.frombuffer(
                (root/"01_blind_original_only"/"original"/f"{sid}.png").read_bytes(),
                dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
            x0,y0,x1,y1=sample["source_xyxy"]
            assert np.array_equal(raw,img[y0:y1,x0:x1])
            overlay_file=f"overlays/{sid}_S95_source.png"
            comparison=cv2.imdecode(np.frombuffer(zip.read(overlay_file),dtype=np.uint8),
                                     cv2.IMREAD_COLOR)
            assert comparison.shape[:2]==(128,256)
            assert np.array_equal(
                cv2.cvtColor(comparison[:,:128],cv2.COLOR_BGR2GRAY),
                img[y0:y1,x0:x1]
            )
            assert len(sample["source_component_id"])>0
        assert data["blind_source_examples"]==len(man["sampled_original_photo_component_cases"])
    assert sha256_file(bmp)==origsha
    assert sha256_file(s86/"manifest.json")==s86sha
    assert data["do_not_open_posthoc_until_human_review_complete"] is True
    with pytest.raises(ValueError,match="New PRIVATE"):
        package_verified_real_h50z(
            bmp_path=bmp,s86_dir=s86,s92_dir=s92,s95_dir=s95,
            s96_dir=s96,private_out_dir=root,
            max_per_class=2,crop_size=128,case_id="SYNTHETIC_ONLY"
        )


def test_completed_blind_source_review_compares_only_selected_visual_cases(
    tmp_path,monkeypatch
):
    bmp,img,s86,s92,s95,s96=_sources(tmp_path,monkeypatch,include_s96=False)
    output=tmp_path/"LOCAL"
    package_verified_real_h50z(
        bmp_path=bmp,s86_dir=s86,s92_dir=s92,s95_dir=s95,
        s96_dir=None,private_out_dir=output,
        max_per_class=2,crop_size=128,case_id="SYNTHETIC_ONLY"
    )
    reference=output/"02_machine_after_human_review"/"verified_machine_only_report.json"
    report=json.loads(reference.read_text())
    rows=[]
    for idx,item in enumerate(report["sampled_original_photo_component_cases"]):
        human=EXPECTED[item["machine_S95_category"]]
        if idx==0:human="UNKNOWN"
        if idx==1:human="ARTIFACT" if human!="ARTIFACT" else "TRACK_LIKE"
        rows.append({
            "sample_id":item["sample_id"],
            "human_label":human,
            "certainty":"UNCERTAIN" if idx==0 else "CLEAR",
            "note":"synthetic source-only expert simulated for software test"
        })
    completed=tmp_path/"human_review_finished.csv"
    with completed.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=[
            "sample_id","human_label","certainty","note"
        ])
        writer.writeheader()
        writer.writerows(rows)
    scored=tmp_path/"human_source_case_review.json"
    main([
        "score","--machine-report-json",str(reference),
        "--completed-blind-csv",str(completed),
        "--private-out-json",str(scored)
    ])
    result=json.loads(scored.read_text())
    assert result["schema"]==SCHEMA
    assert result["independently_reviewed_machine_candidates"]==len(rows)
    assert result["human_uncertain_cases_abstained"]==1
    assert result["visual_contradictory_cases"]>=1
    assert (result["visual_consistent_cases"]
            +result["visual_contradictory_cases"]
            +result["human_uncertain_cases_abstained"]==len(rows))
    assert result["not_unbiased_population_precision_F1_recall"]
    assert result["physical_track_identity_not_evaluated"]
    with pytest.raises(ValueError,match="new immutable"):
        evaluate_blind_component_cases(
            machine_report_path=reference,
            blind_review_csv=completed,
            private_out_json=scored
        )


def test_invalid_source_or_forged_S95_class_receipt_never_exports_as_real(
    tmp_path,monkeypatch
):
    bmp,img,s86,s92,s95,s96=_sources(tmp_path,monkeypatch)
    cls=s95/"all_source_fragment_types.ndjson"
    rows=[json.loads(x) for x in cls.read_text().splitlines() if x.strip()]
    rows[0]["primary_review_category"]="TRUE_NEUTRON_COLLISION"
    cls.write_text("\n".join(json.dumps(x) for x in rows)+"\n")
    target=tmp_path/"SHOULD_NOT_BE_CREATED"
    with pytest.raises(ValueError,match="S95 reported type counts"):
        package_verified_real_h50z(
            bmp_path=bmp,s86_dir=s86,s92_dir=s92,s95_dir=s95,
            s96_dir=s96,private_out_dir=target,crop_size=128
        )
    assert not target.exists()


def test_single_command_real_source_pipeline_run_with_model_guard(
    tmp_path,monkeypatch
):
    bmp,img=_real_synthetic_bmp(tmp_path)
    from bubble_chamber import full_bmp_segmentation as f
    monkeypatch.setattr(f,"predict_mask",lambda gray,cfg,az_model=None:gray<90)
    root=tmp_path/"PRIVATE_ONE_COMMAND"
    main([
        "run","--bmp",str(bmp),"--work-root",str(root),
        "--method","contrast","--skip-s96",
        "--crop-size","128","--max-side","512",
        "--max-per-class","1","--max-fragments-per-tile","512",
        "--min-component-area","1"
    ])
    assert (root/"S93_SOURCES"/"S86"/"manifest.json").is_file()
    assert (root/"S93_SOURCES"/"S92"/"report.json").is_file()
    assert (root/"S95_ALL_OBJECTS"/"report.json").is_file()
    assert not (root/"S96_JUNCTIONS").exists()
    receipt=json.loads((root/"COMPACT_REAL_EVIDENCE"/"compact_evidence_receipt.json").read_text())
    assert receipt["status"]=="TWO_SEPARATE_VERIFIED_PRIVATE_H50Z_DIAGNOSTIC_ARCHIVES"
    assert receipt["source_bmp_sha256"]==sha256_file(bmp)
    assert receipt["blind_source_examples"]>=1
    with pytest.raises(ValueError,match="already exists"):
        main([
            "run","--bmp",str(bmp),"--work-root",str(root),
            "--method","contrast","--skip-s96",
            "--crop-size","128","--max-side","512",
            "--max-per-class","1","--max-fragments-per-tile","512",
            "--min-component-area","1"
        ])
    with pytest.raises(ValueError,match="actual checkpoint"):
        main([
            "run","--bmp",str(bmp),
            "--work-root",str(tmp_path/"NEVER_GENERATE_UNTRAINED_AZUNET"),
            "--method","azunet",
        ])


def test_partially_completed_or_wrong_blind_operator_csv_fails_closed(
    tmp_path,monkeypatch
):
    bmp,img,s86,s92,s95,s96=_sources(tmp_path,monkeypatch,include_s96=False)
    root=tmp_path/"EVIDENCE"
    report=package_verified_real_h50z(
        bmp_path=bmp,s86_dir=s86,s92_dir=s92,s95_dir=s95,
        s96_dir=None,private_out_dir=root,max_per_class=1,crop_size=128
    )
    reference=root/"02_machine_after_human_review"/"verified_machine_only_report.json"
    bad=tmp_path/"NOT_COMPLETED.csv"
    bad.write_text("sample_id,human_label,certainty,note\nNOT_A_SAMPLE,ARTIFACT,CLEAR,no\n")
    with pytest.raises(ValueError,match="unknown/duplicate"):
        evaluate_blind_component_cases(
            machine_report_path=reference,blind_review_csv=bad,
            private_out_json=tmp_path/"MUST_NOT_EXIST.json"
        )
    assert not (tmp_path/"MUST_NOT_EXIST.json").exists()
