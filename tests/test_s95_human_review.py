"""S95 offline human multi-class review is source-SHA and object-ID strict."""
from __future__ import annotations
import json
from pathlib import Path

import pytest

from bubble_chamber.s95_human_review import (
    SCHEMA,build_review_html,compile_human_review,STATUSES,MORPHOLOGIES,
)
from bubble_chamber.bitmap_tiles import sha256_file


def _payload():
    return {
        "source_bmp_sha256":"a"*64,
        "image_shape_yx":[200,300],
        "all_extracted_line_categories":{"typed_components":2},
    }


def _objects():
    return [
        {"component_id":"part_a","primary_review_category":"COILING_TRACE_CANDIDATE",
         "all_morphology_tags":["COILED_2D_SHAPE_NO_CAUSE_IDENTIFIED"]},
        {"component_id":"part_b","primary_review_category":"ARTIFACT_LIKE_CANDIDATE",
         "all_morphology_tags":["SHORT_ANGULAR_GLYPH_POSSIBLE"]},
    ]


def _components():
    return [
        {"component_id":"part_a","points_xy":[[40,50],[50,55],[56,62],[60,70]]},
        {"component_id":"part_b","points_xy":[[80,80],[90,90],[100,80]]},
    ]


def _review_data():
    return {
        "schema":SCHEMA,"source_bmp_sha256":"a"*64,
        "source_s95_report_sha256":"b"*64,
        "object_count":2,
        "decisions":[
            {"component_id":"part_a","status":"TRACK_EVIDENCE",
             "morphology":"COILING_OR_SPIRAL","note":"independent operator choice"},
            {"component_id":"part_b","status":"ARTIFACT_EVIDENCE",
             "morphology":"OTHER","note":"scratch"},
        ],
    }


def test_offline_reviewer_has_all_candidate_ids_and_two_independent_axes():
    html=build_review_html(
        source_report=_payload(),
        source_report_sha256="b"*64,
        objects=_objects(),components=_components(),
        image_shape_yx=[200,300],small_shape_yx=[100,150],
        source_image_available=True,
    )
    assert "S95_original_small.jpg" in html
    assert "part_a" in html and "part_b" in html
    assert "ARTIFACT_EVIDENCE" in html and "TRACK_EVIDENCE" in html
    assert "CALIBRATION_CROSS" in html and "COILING_OR_SPIRAL" in html
    assert "source_s95_report_sha256" in html
    assert "Неразмеченное остаётся UNKNOWN" in html
    assert "fetch(" not in html
    no_image=build_review_html(
        source_report=_payload(),
        source_report_sha256="b"*64,
        objects=_objects(),components=_components(),
        image_shape_yx=[200,300],small_shape_yx=[200,300],
        source_image_available=False,
    )
    assert '<image href=' not in no_image
    assert '<rect' in no_image


def test_review_compiler_never_imputes_unreviewed_negative_or_neutron_cause():
    r=compile_human_review(_review_data(),_payload(),"b"*64,_objects())
    assert r["status"]=="HUMAN_VISUAL_REVIEW_PROVENANCE_CHECKED"
    assert r["human_reviewed"]==2
    assert r["unreviewed_remain_unknown"]==0
    assert r["human_status_counts"]["TRACK_EVIDENCE"]==1
    assert r["human_morphology_counts"]["COILING_OR_SPIRAL"]==1
    assert all(x["particle_species_identified"] is False for x in r["decisions"])
    assert r["no_verified_particle_momentum_mass_or_neutron_cause"] is True
    few=_review_data()
    few["decisions"]=[few["decisions"][0]]
    result=compile_human_review(few,_payload(),"b"*64,_objects())
    assert result["unreviewed_remain_unknown"]==1
    assert result["unreviewed_not_negative"] is True


def test_review_compiler_rejects_mismatch_unapproved_classes_and_duplicate_ID():
    data=_review_data()
    with pytest.raises(ValueError,match="different"):
        compile_human_review({**data,"source_s95_report_sha256":"c"*64},
                             _payload(),"b"*64,_objects())
    with pytest.raises(ValueError,match="duplicate"):
        compile_human_review({**data,"decisions":[data["decisions"][0],
                                                  data["decisions"][0]]},
                             _payload(),"b"*64,_objects())
    bad=[{**data["decisions"][0],"status":"NEUTRON_PROVEN"}]
    with pytest.raises(ValueError,match="Invalid human"):
        compile_human_review({**data,"decisions":bad},
                             _payload(),"b"*64,_objects())
    wrong=[{**data["decisions"][0],"component_id":"invented"}]
    with pytest.raises(ValueError,match="Unknown"):
        compile_human_review({**data,"decisions":wrong},
                             _payload(),"b"*64,_objects())


def test_operator_review_compiler_cli_is_private_and_not_overwritten(tmp_path):
    from scripts.research.bubble_chamber.physicists_s95_compile_human_review import main
    source=tmp_path/"PRIVATE_S95"
    source.mkdir()
    report_file=source/"report.json"
    report_file.write_text(json.dumps(_payload()),encoding="utf8")
    objects_file=source/"all_source_fragment_types.ndjson"
    objects_file.write_text("".join(json.dumps(row)+"\n" for row in _objects()),
                            encoding="utf8")
    review=_review_data()
    review["source_s95_report_sha256"]=sha256_file(report_file)
    human=tmp_path/"review.json"
    human.write_text(json.dumps(review),encoding="utf8")
    out=tmp_path/"RESULT"/"accepted_review.json"
    original_sha=sha256_file(review_file:=human)
    main(["--s95-dir",str(source),"--human-review-json",str(human),
          "--private-out-json",str(out)])
    compiled=json.loads(out.read_text(encoding="utf8"))
    assert compiled["human_reviewed"]==2
    assert compiled["human_review_source_sha256"]==original_sha
    assert sha256_file(human)==original_sha
    with pytest.raises(FileExistsError):
        main(["--s95-dir",str(source),"--human-review-json",str(human),
              "--private-out-json",str(out)])
