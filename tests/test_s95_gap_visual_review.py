"""S95 blinded gap audit: real source pixels in synthesized BMP, no private H50Z.

Verify prospective machine-only proposals can be reviewed by an expert with
source images and independently marked APPEARS_SAME / DIFFERENT / UNDECIDABLE.
Never use synthetic "GT" as real-particle evidence.
"""
from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from bubble_chamber.bitmap_tiles import sha256_file
from bubble_chamber.full_bmp_segmentation import FullBMPConfig,run_full_bmp
from bubble_chamber.s92_curve_review import run_s92,collapse_observed_components
from bubble_chamber.s87_seam_continuity import _read_source
from bubble_chamber.s95_all_structures_review import run_s95
from bubble_chamber.s95_gap_visual_review import (
    SCHEMA,select_visual_cases,build_visual_gap_pack,
    compile_visual_gap_review,
)
from scripts.research.bubble_chamber.physicists_s95_visual_gap_review import main


def _gap(ident,status):
    return {
        "proposal_id":ident,
        "review_decision":status,
        "status":"UNKNOWN",
        "physical_auto_join":False,
    }


def test_blind_selection_deterministic_stratified_no_label_peeking():
    source=[
        _gap("g1","PRIORITY_GAP_REVIEW_NOT_PHYSICAL_ACCEPT"),
        _gap("g2","PRIORITY_GAP_REVIEW_NOT_PHYSICAL_ACCEPT"),
        _gap("g3","AMBIGUOUS_PARTNER_ABSTAIN"),
        _gap("g4","THIRD_FRAGMENT_NEAR_GAP_ABSTAIN"),
        _gap("g5","PORT_OCCUPIED_ABSTAIN"),
    ]
    first=select_visual_cases(source,max_cases=4)
    second=select_visual_cases(list(reversed(source)),max_cases=4)
    assert [x["proposal_id"] for x in first]==[x["proposal_id"] for x in second]
    assert len(first)==4
    assert {x["review_decision"] for x in first}=={
        "PRIORITY_GAP_REVIEW_NOT_PHYSICAL_ACCEPT",
        "AMBIGUOUS_PARTNER_ABSTAIN",
        "THIRD_FRAGMENT_NEAR_GAP_ABSTAIN",
        "PORT_OCCUPIED_ABSTAIN",
    }
    with pytest.raises(ValueError,match="duplicate"):
        select_visual_cases(source+[source[0]])
    with pytest.raises(ValueError,match="Unsafe"):
        select_visual_cases([_gap("a","PRIORITY_GAP_REVIEW_NOT_PHYSICAL_ACCEPT")
                             |{"physical_auto_join":True}])


def _source_and_outputs(tmp_path,monkeypatch):
    image=np.full((384,512),225,dtype=np.uint8)
    cv2.line(image,(22,75),(76,75),25,1)
    cv2.line(image,(89,75),(158,75),25,1)
    cv2.line(image,(280,180),(320,180),25,1)
    bmp=tmp_path/"H50Z_030_01_405.bmp"
    assert cv2.imwrite(str(bmp),image)
    from bubble_chamber import full_bmp_segmentation
    monkeypatch.setattr(full_bmp_segmentation,"predict_mask",
                        lambda gray,cfg,az_model=None:gray<100)
    s86=tmp_path/"S86"
    run_full_bmp(bmp,s86,FullBMPConfig(
        method="contrast",core_size=128,halo=32,preview_limit=0,
        min_component_area=1,min_segment_length_px=4
    ))
    s92=tmp_path/"S92"
    r92=run_s92(s86,s92,bmp_path=bmp,preview_limit=0)
    assert r92["gap"]["gap_candidates"]>=1
    s95=tmp_path/"S95"
    run_s95(s86,s92,s95,bmp_path=bmp,max_side=512)
    manifest,records,_=_read_source(s86)
    comps,_=collapse_observed_components(records)
    return bmp,s86,s92,s95,comps


def test_blind_photo_pack_real_original_crops_and_conditional_adjudication(
    tmp_path,monkeypatch
):
    bmp,s86,s92,s95,comps=_source_and_outputs(tmp_path,monkeypatch)
    before=sha256_file(bmp)
    out=tmp_path/"GAP_REVIEW"
    pack=build_visual_gap_pack(
        s95_dir=s95,s92_dir=s92,components=comps,bmp_path=bmp,
        out_dir=out,max_cases=12,crop_size=320
    )
    assert pack["source_image_available"] is True
    assert pack["sampled_cases"]>=1
    assert pack["sampled_cases"]<=pack["all_machine_gap_proposals"]
    assert sha256_file(bmp)==before
    case=pack["cases"][0]
    assert case["case_id"].startswith("gap_")
    assert (out/case["relative_case_photo"]).is_file()
    frame=cv2.imread(str(out/case["relative_case_photo"]))
    assert frame is not None and frame.shape==(320,640,3)
    assert sha256_file(out/case["relative_case_photo"])==case["case_photo_sha256"]
    html=(out/"index.html").read_text(encoding="utf8")
    assert "Машинное решение скрыто" in html
    assert case["case_id"] in html
    assert case["machine_stratum_HIDDEN_IN_BLIND_REVIEW"] not in html
    assert case["machine_decision_HIDDEN_IN_BLIND_REVIEW"] not in html
    assert "APPEARS_SAME_2D_TRACE" in html

    human={
        "schema":SCHEMA,
        "source_bmp_sha256":pack["source_bmp_sha256"],
        "source_s95_report_sha256":pack["source_s95_report_sha256"],
        "source_s92_gap_proposals_sha256":pack["source_s92_gap_proposals_sha256"],
        "source_image_available":True,
        "decisions":[{
            "case_id":case["case_id"],
            "choice":"APPEARS_DIFFERENT_2D_TRACES",
            "reason":"CROSSING_OR_BUNDLE",
            "note":"synthetic manual review test only",
        }],
    }
    report=compile_visual_gap_review(pack,human)
    assert report["reviewed_cases"]==1
    assert report["unreviewed_cases_remain_unknown"]==len(pack["cases"])-1
    assert report["source_bmp_sha256"]==before
    assert report["cannot_estimate_unproposed_true_link_recall"] is True
    assert report["all_physical_joins_still_unapproved"] is True
    if case["machine_stratum_HIDDEN_IN_BLIND_REVIEW"]=="SELECTED_FOR_REVIEW":
        assert report["selected_apparent_false_join_fraction_among_decided"]==1.
    else:
        assert report["selected_apparent_false_join_fraction_among_decided"] is None

    with pytest.raises(ValueError,match="source SHA"):
        compile_visual_gap_review(pack,{**human,"source_bmp_sha256":"0"*64})
    with pytest.raises(ValueError,match="Unknown/duplicate"):
        compile_visual_gap_review(pack,{**human,"decisions":human["decisions"]*2})
    with pytest.raises(ValueError,match="Invalid human"):
        bad=[{**human["decisions"][0],"choice":"NEUTRON_PROVEN"}]
        compile_visual_gap_review(pack,{**human,"decisions":bad})
    with pytest.raises(ValueError,match="PRIVATE"):
        build_visual_gap_pack(
            s95_dir=s95,s92_dir=s92,components=comps,bmp_path=bmp,
            out_dir=out,max_cases=12,crop_size=320
        )


def test_no_photo_cannot_be_treated_as_expert_validated_GT(tmp_path,monkeypatch):
    bmp,s86,s92,s95,comps=_source_and_outputs(tmp_path,monkeypatch)
    pack=build_visual_gap_pack(
        s95_dir=s95,s92_dir=s92,components=comps,bmp_path=None,
        out_dir=tmp_path/"NO_PHOTO",max_cases=8
    )
    assert pack["source_image_available"] is False
    assert "ФОТО НЕ ДОСТУПНО" in (tmp_path/"NO_PHOTO"/"index.html").read_text()
    forbidden={
        "schema":SCHEMA,
        "source_bmp_sha256":pack["source_bmp_sha256"],
        "source_s95_report_sha256":pack["source_s95_report_sha256"],
        "source_s92_gap_proposals_sha256":pack["source_s92_gap_proposals_sha256"],
        "source_image_available":False,
        "decisions":[{"case_id":pack["cases"][0]["case_id"],
                      "choice":"APPEARS_SAME_2D_TRACE","reason":"OTHER","note":""}],
    }
    with pytest.raises(ValueError,match="No original image"):
        compile_visual_gap_review(pack,forbidden)
    assert compile_visual_gap_review(pack,{**forbidden,"decisions":[]})[
        "reviewed_cases"
    ]==0


def test_cli_pack_and_compile_use_immutable_artifacts(tmp_path,monkeypatch):
    bmp,s86,s92,s95,comps=_source_and_outputs(tmp_path,monkeypatch)
    dst=tmp_path/"PRIVATE_REVIEW_CLI"
    main(["pack","--s86-dir",str(s86),"--s92-dir",str(s92),
          "--s95-dir",str(s95),"--bmp",str(bmp),
          "--private-out-dir",str(dst),"--max-cases","12"])
    pack=json.loads((dst/"review_pack.json").read_text())
    assert pack["source_image_available"] is True
    human={
        "schema":SCHEMA,
        "source_bmp_sha256":pack["source_bmp_sha256"],
        "source_s95_report_sha256":pack["source_s95_report_sha256"],
        "source_s92_gap_proposals_sha256":pack["source_s92_gap_proposals_sha256"],
        "source_image_available":True,
        "decisions":[{
            "case_id":pack["cases"][0]["case_id"],
            "choice":"UNDECIDABLE","reason":"MISSING_PIXEL_EVIDENCE",
            "note":"could be multiple trajectories",
        }],
    }
    infile=tmp_path/"human_private.json"
    infile.write_text(json.dumps(human),encoding="utf8")
    compile_file=tmp_path/"compiled.json"
    main(["compile","--pack-dir",str(dst),"--review-json",str(infile),
          "--private-out-json",str(compile_file)])
    r=json.loads(compile_file.read_text())
    assert r["reviewed_cases"]==1
    assert r["selected_apparent_false_join_fraction_among_decided"] is None
    assert r["source_human_response_sha256"]==sha256_file(infile)
    with pytest.raises(ValueError,match="new private"):
        main(["compile","--pack-dir",str(dst),"--review-json",str(infile),
              "--private-out-json",str(compile_file)])
