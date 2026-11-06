"""S96 source-locked real-photo integration via *procedurally generated* BMP.

No actual H50Z originals, no 070 measurements and no particle-identity GT.
These tests verify the workflow and human 2D appearance-review semantics ONLY.
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
from bubble_chamber.s96_junction_human_review import (
    compile_junction_review,label_ports,pair_code_for_machine,PAIRINGS
)
from scripts.research.bubble_chamber.physicists_s96_junctions import main


def _scene(tmp_path):
    gray=np.full((384,512),212,np.uint8)
    # Four SOURCE pixels arms with a deliberately empty ~10px central hole.
    cv2.line(gray,(150,210),(205,210),35,1)
    cv2.line(gray,(215,210),(247,210),35,1)
    cv2.line(gray,(210,150),(210,205),35,1)
    cv2.line(gray,(210,215),(210,247),35,1)
    # Long independent smooth candidate outside crossing.
    cv2.line(gray,(34,48),(103,48),35,1)
    bmp=tmp_path/"H50Z_030_01_405.bmp"
    assert cv2.imwrite(str(bmp),gray)
    return bmp


def _stages(tmp_path,monkeypatch):
    bmp=_scene(tmp_path)
    from bubble_chamber import full_bmp_segmentation
    monkeypatch.setattr(
        full_bmp_segmentation,"predict_mask",
        lambda gray,cfg,az_model=None: gray<95
    )
    s86=tmp_path/"S86"
    run_full_bmp(bmp,s86,FullBMPConfig(
        method="contrast",core_size=128,halo=32,
        min_component_area=1,min_segment_length_px=4,preview_limit=0,
        max_fragments_per_tile=256
    ))
    s92=tmp_path/"S92"
    run_s92(s86,s92,bmp_path=bmp,preview_limit=0)
    s95=tmp_path/"S95"
    run_s95(s86,s92,s95,bmp_path=bmp,max_side=512)
    return bmp,s86,s92,s95


def _review_response(pack,pair="AC|BD"):
    return {
        "schema":pack["schema"],
        "source_bmp_sha256":pack["source_bmp_sha256"],
        "source_s96_report_sha256":pack["source_s96_report_sha256"],
        "sampled_cases":pack["sampled_cases"],
        "source_image_available":pack["source_image_available"],
        "decisions":[{
            "case_id":pack["cases"][0]["case_id"],
            "structure":"CROSSING",
            "pairing":pair,
            "note":"synthetic unit test: visual alternative not physical event",
        }],
    }


def test_source_locked_s96_geometry_creates_original_junction_and_blinded_review(
    tmp_path,monkeypatch
):
    bmp,s86,s92,s95=_stages(tmp_path,monkeypatch)
    sha=sha256_file(bmp)
    source_s86_sha=sha256_file(s86/"manifest.json")
    report=run_s96(s86,s95,tmp_path/"S96_FULL",bmp_path=bmp,preview_limit=20)
    assert report["status"]=="REVERSIBLE_2D_JUNCTION_REVIEW_ONLY"
    assert report["geometry"]["source_components_preserved"]>=4
    assert report["geometry"]["four_arm_junctions_with_two_visual_pairs"]>=1
    assert report["geometry"]["provisional_visual_continuation_pairs"]>=2
    assert report["source_s86_skipped_tiles"]==0
    assert report["all_source_components_available"] is True
    assert report["review_image_available"] is True
    assert report["blinded_review_cases"]>=1
    assert report["physical_auto_joins_authorized"] is False
    assert sha256_file(bmp)==sha
    assert sha256_file(s86/"manifest.json")==source_s86_sha
    out=tmp_path/"S96_FULL"
    assert (out/"junction_hypotheses.ndjson").is_file()
    assert (out/"visual_continuation_pairs_unknown.ndjson").is_file()
    assert (out/"visual_chain_hypotheses_unknown.ndjson").is_file()
    assert (out/"original_observed_component_inventory.ndjson").is_file()
    inventory=[json.loads(s) for s in (
        out/"original_observed_component_inventory.ndjson"
    ).read_text().splitlines()]
    assert len(inventory)==report["geometry"]["source_components_preserved"]
    assert all(x["source_centerline_unchanged"] is True for x in inventory)

    pack=json.loads((out/"review"/"review_pack.json").read_text())
    assert pack["source_image_available"] is True
    assert pack["source_bmp_sha256"]==sha
    case=next(x for x in pack["cases"] if x["observed_arm_count"]==4)
    assert case["machine_pairing_HIDDEN_FROM_UI"] in PAIRINGS
    assert len(case["display_ports"])==4
    assert set(x["label"] for x in case["display_ports"])==set("ABCD")
    photo=out/"review"/case["relative_original_vs_arms_image"]
    assert photo.is_file()
    crop=cv2.imread(str(photo))
    assert crop is not None and crop.shape==(320,640,3)
    assert sha256_file(photo)==case["crop_sha256"]
    viewer=(out/"review"/"index.html").read_text(encoding="utf8")
    assert "AB|CD" in viewer and "AD|BC" in viewer
    assert "машинная" not in viewer.lower() or "скрыта" in viewer.lower()
    assert "machine_pairing_HIDDEN_FROM_UI" not in viewer
    assert case["case_id"] in viewer

    with pytest.raises(ValueError,match="fresh PRIVATE"):
        run_s96(s86,s95,out,bmp_path=bmp)


def test_no_BMP_mode_exposes_machine_junctions_but_blocks_expert_pair_labels(
    tmp_path,monkeypatch
):
    bmp,s86,s92,s95=_stages(tmp_path,monkeypatch)
    report=run_s96(s86,s95,tmp_path/"NO_BMP")
    assert report["review_image_available"] is False
    pack=json.loads((tmp_path/"NO_BMP"/"review"/"review_pack.json").read_text())
    assert pack["source_image_available"] is False
    case=next(x for x in pack["cases"] if x["observed_arm_count"]==4)
    test=_review_response(pack)
    test["decisions"][0]["case_id"]=case["case_id"]
    with pytest.raises(ValueError,match="No original photo"):
        compile_junction_review(pack,test)
    assert compile_junction_review(
        pack,{**test,"decisions":[]}
    )["reviewed_cases"]==0


def test_human_four_arm_pair_label_compiler_preserves_unknown_and_physics_firewall(
    tmp_path,monkeypatch
):
    bmp,s86,s92,s95=_stages(tmp_path,monkeypatch)
    run_s96(s86,s95,tmp_path/"WITH_BMP",bmp_path=bmp)
    pack=json.loads((tmp_path/"WITH_BMP"/"review"/"review_pack.json").read_text())
    index=next(i for i,c in enumerate(pack["cases"]) if c["observed_arm_count"]==4)
    case=pack["cases"][index]
    human=_review_response(pack,pair=case["machine_pairing_HIDDEN_FROM_UI"])
    human["decisions"][0]["case_id"]=case["case_id"]
    parsed=compile_junction_review(pack,human)
    assert parsed["status"]=="SOURCE_LOCKED_2D_VISUAL_JUNCTION_REVIEW_COMPILED"
    assert parsed["reviewed_cases"]==1
    assert parsed["unreviewed_cases_unknown"]==len(pack["cases"])-1
    assert parsed["machine_human_visual_pairing_agreement_rate"]==1.0
    assert parsed["not_physical_track_or_collision_accuracy"] is True
    assert parsed["no_verified_native_real_H50Z_GT"] is True

    wrong=next(x for x in ("AB|CD","AC|BD","AD|BC")
               if x!=case["machine_pairing_HIDDEN_FROM_UI"])
    human["decisions"][0]["pairing"]=wrong
    out=compile_junction_review(pack,human)
    assert out["machine_human_visual_pairing_agreement_rate"]==0.
    unknown={**human,"decisions":[{**human["decisions"][0],"pairing":"UNDECIDABLE"}]}
    assert compile_junction_review(pack,unknown)[
        "machine_human_visual_pairing_agreement_rate"
    ] is None
    with pytest.raises(ValueError,match="another"):
        compile_junction_review(pack,{**human,"source_s96_report_sha256":"b"*64})
    with pytest.raises(ValueError,match="Unknown or duplicated"):
        compile_junction_review(pack,{**human,"decisions":human["decisions"]*2})
    with pytest.raises(ValueError,match="Invalid"):
        compile_junction_review(pack,{
            **human,"decisions":[{**human["decisions"][0],"structure":"NEUTRON_COLLISION_PROVEN"}]
        })


def test_cli_build_and_compile_immutable_private_review_receipts(tmp_path,monkeypatch):
    bmp,s86,s92,s95=_stages(tmp_path,monkeypatch)
    out=tmp_path/"S96_CLI"
    main(["run","--s86-dir",str(s86),"--s95-dir",str(s95),
          "--bmp",str(bmp),"--private-out-dir",str(out)])
    packfile=out/"review"/"review_pack.json"
    pack=json.loads(packfile.read_text())
    case=next(x for x in pack["cases"] if x["observed_arm_count"]==4)
    review=_review_response(pack,pair=case["machine_pairing_HIDDEN_FROM_UI"])
    review["decisions"][0]["case_id"]=case["case_id"]
    data=tmp_path/"REVIEW.json"
    data.write_text(json.dumps(review),encoding="utf8")
    rec=tmp_path/"REVIEW_ACCEPTED.json"
    main(["compile","--review-pack",str(packfile),
          "--human-review-json",str(data),"--private-out-json",str(rec)])
    summary=json.loads(rec.read_text())
    assert summary["reviewed_cases"]==1
    assert summary["source_review_pack_sha256"]==sha256_file(packfile)
    assert summary["source_human_review_sha256"]==sha256_file(data)
    assert summary["not_physical_track_or_collision_accuracy"] is True
    with pytest.raises(ValueError,match="NEW private"):
        main(["compile","--review-pack",str(packfile),
              "--human-review-json",str(data),"--private-out-json",str(rec)])



def test_balanced_machine_junction_sampling_includes_rare_Y_and_dense_bundle():
    from bubble_chamber.s96_junction_human_review import balanced_blind_junction_sample
    source=[
        {"junction_id":f"cross{i:03d}","ports":[0,1,2,3],
         "provisional_pairs":[{},{}]}
        for i in range(100)
    ]
    source.extend([
        {"junction_id":"amb","ports":[0,1,2,3],"provisional_pairs":[]},
        {"junction_id":"Y","ports":[0,1,2],"provisional_pairs":[]},
        {"junction_id":"BUNDLE","ports":[0,1,2,3,4],"provisional_pairs":[]},
    ])
    chosen=balanced_blind_junction_sample(
        source,source_sha="a"*64,max_cases=4
    )
    assert len(chosen)==4
    assert {x["junction_id"] for x in chosen}>={"Y","BUNDLE","amb"}
    assert [x["junction_id"] for x in chosen]==[
        x["junction_id"] for x in balanced_blind_junction_sample(
            list(reversed(source)),source_sha="a"*64,max_cases=4
        )
    ]
    with pytest.raises(ValueError,match="Duplicate"):
        balanced_blind_junction_sample(
            source+[source[0]],source_sha="a"*64,max_cases=4
        )


def test_two_blinded_reviewer_visual_consensus_does_not_invent_physics_GT(
    tmp_path,monkeypatch
):
    from bubble_chamber.s96_junction_human_review import compile_two_reviewer_consensus
    bmp,s86,s92,s95=_stages(tmp_path,monkeypatch)
    run_s96(s86,s95,tmp_path/"TWO_RATERS",bmp_path=bmp)
    pack=json.loads((tmp_path/"TWO_RATERS"/"review"/"review_pack.json").read_text())
    crossing=next(x for x in pack["cases"] if x["observed_arm_count"]==4)
    first=_review_response(pack,pair=crossing["machine_pairing_HIDDEN_FROM_UI"])
    first["decisions"][0]["case_id"]=crossing["case_id"]
    independent=json.loads(json.dumps(first))
    same=compile_two_reviewer_consensus(pack,first,independent)
    assert same["both_reviewed"]==1
    assert same["both_decided_visual_cases"]==1
    assert same["visual_consensus_cases"]==1
    assert same["visual_agreement_rate_among_both_decided"]==1.
    assert same["not_native_physical_GT_or_particle_identification"] is True
    assert same["consensus_visual_junction_cases"][0][
        "not_a_verified_physical_particle_identity"
    ] is True

    independent["decisions"][0]["pairing"]=next(
        p for p in ("AB|CD","AC|BD","AD|BC")
        if p!=crossing["machine_pairing_HIDDEN_FROM_UI"]
    )
    mixed=compile_two_reviewer_consensus(pack,first,independent)
    assert mixed["visual_consensus_cases"]==0
    assert mixed["explicit_disagreements"]==1
    assert mixed["disagreements_requiring_adjudication"][0][
        "needs_independent_adjudication"
    ]
    uncertain=json.loads(json.dumps(independent))
    uncertain["decisions"][0]["pairing"]="UNDECIDABLE"
    skipped=compile_two_reviewer_consensus(pack,first,uncertain)
    assert skipped["both_decided_visual_cases"]==0
    assert skipped["uncertain_or_abstained_by_one"]==1
    assert skipped["visual_agreement_rate_among_both_decided"] is None
    with pytest.raises(ValueError,match="distinct"):
        compile_two_reviewer_consensus(pack,first,first,
                                       reviewer_a="same",reviewer_b="same")


def test_two_reviewer_consensus_CLI_writes_sha_receipt(tmp_path,monkeypatch):
    bmp,s86,s92,s95=_stages(tmp_path,monkeypatch)
    run_s96(s86,s95,tmp_path/"REVIEW_ROOT",bmp_path=bmp)
    packfile=tmp_path/"REVIEW_ROOT"/"review"/"review_pack.json"
    pack=json.loads(packfile.read_text())
    crossing=next(c for c in pack["cases"] if c["observed_arm_count"]==4)
    human=_review_response(pack,pair=crossing["machine_pairing_HIDDEN_FROM_UI"])
    human["decisions"][0]["case_id"]=crossing["case_id"]
    first=tmp_path/"physicist_A.json"
    second=tmp_path/"physicist_B.json"
    for file in (first,second):
        file.write_text(json.dumps(human),encoding="utf8")
    out=tmp_path/"private_consensus.json"
    main(["consensus","--review-pack",str(packfile),
          "--first-json",str(first),"--second-json",str(second),
          "--private-out-json",str(out)])
    body=json.loads(out.read_text())
    assert body["visual_consensus_cases"]==1
    assert body["source_independent_review_sha256s"]==[
        sha256_file(first),sha256_file(second)
    ]
    assert body["source_review_pack_sha256"]==sha256_file(packfile)
    with pytest.raises(ValueError,match="NEW private"):
        main(["consensus","--review-pack",str(packfile),
              "--first-json",str(first),"--second-json",str(second),
              "--private-out-json",str(out)])
