"""Audit genuinely saved H50Z S86/S92/S95/S96 real-photo run artifacts.

NO synthetic demonstration success is promoted to medical/physics truth.
This verifier reopens ALL canonical S86 mask tiles and source polylines,
checks cross-stage SHA-256 provenance and actual NDJSON inventories, reports
source-fragment and review-hypothesis counts, and identifies absent stages.

A detached output folder is immutable. Neither original BMP nor any stage
output is ever opened for writing. Raw source photos and model weights remain
PRIVATE and should not be pushed to GitHub.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import html
import json

from .bitmap_tiles import GrayBMP8,sha256_file
from .s87_seam_continuity import _read_source
from .s92_curve_review import collapse_observed_components

SCHEMA="BUBBLE_REAL_H50Z_SOURCE_VERIFIED_STAGE_AUDIT_V1"
STAGE_REPORTS=("S86/manifest.json","S92/report.json","S95/report.json","S96/report.json")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _lines(path,*,key=None):
    """Validate NDJSON records as a stream, detecting duplicate identities."""
    count=0
    seen=set()
    category=Counter()
    with Path(path).open(encoding="utf-8") as f:
        for number,line in enumerate(f,1):
            if not line.strip():
                continue
            try:
                row=json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed NDJSON {Path(path).name}:{number}") from e
            if not isinstance(row,dict):
                raise ValueError("NDJSON record must be an object")
            if key is not None:
                ident=row.get(key)
                if not isinstance(ident,str) or not ident or ident in seen:
                    raise ValueError(f"Duplicate/missing {key} in {Path(path).name}")
                seen.add(ident)
            if "primary_review_category" in row:
                category[row["primary_review_category"]]+=1
            count+=1
    return count,seen,dict(category)


def _require_same(x,y,what):
    if x!=y:
        raise ValueError(f"Cross-stage disagreement: {what}: {x!r} != {y!r}")


def _file_sha(stage,file,base,*,required=True):
    path=(Path(base)/file).resolve()
    root=Path(base).resolve()
    if root not in path.parents:
        raise ValueError("Stage artifact must remain inside source directory")
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"{stage} missing {file}")
        return None
    return sha256_file(path)


def verify_stage_receipts(
    *,
    case_name:str,
    s86_dir:Path,
    s92_dir:Path,
    s95_dir:Path,
    s96_dir:Path|None=None,
    original_bmp:Path|None=None,
):
    """Return machine-only counts, from actual files, or raise on mismatch."""
    if (not isinstance(case_name,str) or not case_name or len(case_name)>60
        or any(not(c.isascii() and (c.isalnum() or c in "_.-"))
               for c in case_name)):
        raise ValueError("Case ID must be a nonempty safe ASCII identifier")
    s86_dir=Path(s86_dir).resolve()
    s92_dir=Path(s92_dir).resolve()
    s95_dir=Path(s95_dir).resolve()
    if s96_dir is not None:
        s96_dir=Path(s96_dir).resolve()
    roots=[s86_dir,s92_dir,s95_dir]+([s96_dir] if s96_dir else [])
    if len(set(roots))!=len(roots):
        raise ValueError("S86/S92/S95/S96 source directories cannot alias")
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Missing original machine-stage directory: {root}")

    # Re-open ALL S86 records and SHA-verified, real stored binary mask tiles.
    # This detects missing/changed shards and source polylines unsupported
    # by the actual frozen mask, not just matching JSON summaries.
    manifest,records,s86_sha=_read_source(s86_dir)
    if not isinstance(manifest.get("source_bmp_sha256"),str):
        raise ValueError("S86 source BMP SHA missing")
    src_sha=manifest["source_bmp_sha256"]
    name=manifest["source_bmp_name"]
    shape=manifest["image_shape_yx"]
    if original_bmp is not None:
        bmp_path=Path(original_bmp).resolve()
        if not bmp_path.is_file() or bmp_path.name!=name:
            raise ValueError("Exact original BMP filename not matched")
        _require_same(sha256_file(bmp_path),src_sha,"original BMP bytes SHA256")
        bmp=GrayBMP8(bmp_path)
        _require_same([bmp.height,bmp.width],shape,"original BMP dimensions")
    observed,seams=collapse_observed_components(records)
    observed_ids={p["component_id"] for p in observed}
    if len(observed_ids)!=len(observed):
        raise ValueError("S87 original source component IDs are nonunique")
    _require_same(sum(r["fragment_count"] for r in records),
                  manifest["candidate_fragments"],
                  "S86 fragment count versus actual source tile records")
    _require_same(sum(r["fragment_extraction_skip_reason"] is not None for r in records),
                  manifest["tiles_with_fragment_extraction_skipped"],
                  "S86 skipped source tile count")

    s92=s92_dir/"report.json"
    s92_sha=_file_sha("S92","report.json",s92_dir)
    s92_report=_json(s92)
    _require_same(s92_report.get("status"),"COMPLETE_PRIVATE_REVIEW_ONLY",
                  "S92 completed status")
    for key,value in (
        ("source_s86_manifest_sha256",s86_sha),
        ("source_bmp_sha256",src_sha),
        ("source_bmp_name",name),
        ("image_shape_yx",shape),
        ("source_fragments",manifest["candidate_fragments"]),
        ("observed_components_after_S87",len(observed)),
    ):
        _require_same(s92_report.get(key),value,"S92 "+key)
    s92_lines,s92_ids,_=_lines(s92_dir/"observed_shape_review.ndjson",
                               key="component_id")
    _require_same(s92_lines,len(observed),"S92 observed shape line count")
    _require_same(s92_ids,observed_ids,"S92 exact original component ID set")
    gap_n,_,_=_lines(s92_dir/"gap_proposals_unknown.ndjson",
                     key="proposal_id")
    if "abstain_reason" not in s92_report["gap"]:
        _require_same(s92_report["gap"]["gap_candidates"],gap_n,
                      "S92 all geometric proposals")
    else:
        if s92_report["gap"]["gap_candidates"]<gap_n:
            raise ValueError("S92 fail-closed geometric proposal count mismatch")
    shape_counts=s92_report["shape_review_categories"]
    _require_same(sum(shape_counts.values()),len(observed),
                  "S92 shape-category inventory")
    _require_same(_file_sha("S92","observed_shape_review.ndjson",s92_dir)!=None,True,
                  "S92 readable line inventory")

    s95=s95_dir/"report.json"
    s95_sha=_file_sha("S95","report.json",s95_dir)
    s95_report=_json(s95)
    _require_same(
        s95_report.get("status"),
        "ALL_S86_EXTRACTED_FRAGMENTS_TYPED_MACHINE_REVIEW_ONLY",
        "S95 status")
    for key,value in (
        ("source_s86_manifest_sha256",s86_sha),
        ("source_s92_report_sha256",s92_sha),
        ("source_bmp_sha256",src_sha),
        ("source_bmp_name",name),
        ("image_shape_yx",shape),
        ("source_s86_extracted_fragments",manifest["candidate_fragments"]),
        ("source_s86_tiles_skipped",manifest["tiles_with_fragment_extraction_skipped"]),
        ("source_fragments_deleted",0),
    ):
        _require_same(s95_report.get(key),value,"S95 "+key)
    s95_count,s95_ids,real_categories=_lines(
        s95_dir/"all_source_fragment_types.ndjson",key="component_id")
    _require_same(s95_count,len(observed),"S95 all observed source components")
    _require_same(s95_ids,observed_ids,"S95 exact original source component IDs")
    _require_same(s95_report["all_extracted_line_categories"]["categories"],
                  real_categories,"S95 reported type counts versus actual typed lines")
    _require_same(s95_report["all_extracted_line_categories"]["typed_components"],
                  len(observed),"S95 typed components")
    s95_types_sha=_file_sha("S95","all_source_fragment_types.ndjson",s95_dir)
    if s95_report.get("all_classes_comparison_file"):
        comparison=s95_report["all_classes_comparison_file"]
        _require_same(s95_report.get("all_classes_comparison_sha256"),
                      _file_sha("S95",comparison,s95_dir),
                      "S95 source-image overlay SHA256")

    s96_metrics=None
    s96_sha=None
    if s96_dir is not None:
        s96=s96_dir/"report.json"
        s96_sha=_file_sha("S96","report.json",s96_dir)
        s96_report=_json(s96)
        if s96_report.get("status") not in (
            "REVERSIBLE_2D_JUNCTION_REVIEW_ONLY",
            "DENSE_ENDPOINT_NEIGHBORHOOD",
            "ENDPOINT_BUDGET_EXCEEDED",
        ):
            raise ValueError("S96 report has unrecognized or incomplete status")
        for key,value in (
            ("source_s86_manifest_sha256",s86_sha),
            ("source_s95_report_sha256",s95_sha),
            ("source_s95_types_sha256",s95_types_sha),
            ("source_bmp_sha256",src_sha),
            ("source_bmp_name",name),
            ("image_shape_yx",shape),
            ("source_observed_centerline_components",len(observed)),
            ("source_s86_fragment_count",manifest["candidate_fragments"]),
            ("source_s86_skipped_tiles",manifest["tiles_with_fragment_extraction_skipped"]),
            ("physical_auto_joins_authorized",False),
        ):
            _require_same(s96_report.get(key),value,"S96 "+key)
        geom=s96_report["geometry"]
        _require_same(geom["source_components_preserved"],len(observed),
                      "S96 original component preservation")
        jcount,_,_=_lines(s96_dir/"junction_hypotheses.ndjson",key="junction_id")
        pc,_,_=_lines(s96_dir/"visual_continuation_pairs_unknown.ndjson")
        _require_same(jcount,geom.get("candidate_junctions",0),
                      "S96 junction hypothesis count")
        _require_same(pc,geom.get("provisional_visual_continuation_pairs",0),
                      "S96 2D provisional continuation count")
        inv_n,inv_ids,_=_lines(s96_dir/"original_observed_component_inventory.ndjson",
                               key="component_id")
        _require_same(inv_n,len(observed),"S96 original component inventory count")
        _require_same(inv_ids,observed_ids,"S96 original exact component ID set")
        s96_metrics={
            "status":s96_report["status"],
            "candidate_junctions":jcount,
            "provisional_visual_continuation_pairs":pc,
            "four_arm_junctions_with_two_visual_pairs":geom.get(
                "four_arm_junctions_with_two_visual_pairs",0),
            "mask_derived_virtual_arm_counts":(
                s96_report.get("optional_alternative_mask_junction_audit")
                or None
            ),
            "source_S86_components_preserved":inv_n,
        }
        # S96 reviewer packs can contain original image crops only when BMP
        # was present. Verify immutable sources if the pack is available.
        pack=s96_dir/"review"/"review_pack.json"
        if pack.is_file():
            payload=_json(pack)
            _require_same(payload["source_bmp_sha256"],src_sha,
                          "S96 human review source BMP")
            _require_same(payload["source_s96_report_sha256"],s96_sha,
                          "S96 human review report provenance")
            _require_same(payload["sampled_cases"],len(payload["cases"]),
                          "S96 blind review sampled case count")
            for x in payload["cases"]:
                rel=x.get("relative_original_vs_arms_image")
                if rel:
                    _require_same(x["crop_sha256"],
                                  _file_sha("S96",rel,s96_dir/"review"),
                                  "S96 blinded source crop SHA")

    return {
        "schema":SCHEMA,
        "status":"VERIFIED_SAVED_MACHINE_STAGE_ARTIFACTS_NOT_HUMAN_GT",
        "case":case_name,
        "original_bmp_name":name,
        "original_bmp_sha256":src_sha,
        "source_shape_yx":shape,
        "actual_original_BMP_bytes_verified":original_bmp is not None,
        "s86":{
            "method":manifest["config"]["method"],
            "tile_count":manifest["tile_count"],
            "candidate_fragments":manifest["candidate_fragments"],
            "source_components_after_S87":len(observed),
            "skipped_fragment_extraction_tiles":manifest[
                "tiles_with_fragment_extraction_skipped"],
            "source_candidate_foreground_pixels":manifest["candidate_foreground_pixels"],
        },
        "s92":{
            "shape_categories":shape_counts,
            "all_generated_gap_proposals":gap_n,
            "geometric_pairs_considered_from_report":s92_report["gap"]["gap_candidates"],
            "priority_gap_review_links_from_report":s92_report["gap"]["gap_review_links"],
            "max_gap_px":s92_report["config"]["max_gap_px"],
            "source_pixels_untouched":True,
        },
        "s95":{
            "all_categories":real_categories,
            "typed_observed_components":s95_count,
            "physical_track_identity_unverified":True,
        },
        "s96":s96_metrics,
        "source_report_sha256":{
            "S86_manifest":s86_sha,"S92_report":s92_sha,
            "S95_report":s95_sha,"S96_report":s96_sha,
        },
        "independent_human_stroke_annotations_provided":False,
        "true_particle_or_neutron_collision_metrics_computable":False,
        "full_real_photo_precision_recall_F1_computable":False,
        "all_machine_suggestions_remain_unverified":True,
    }


def summarize_verification(cases:list[dict]):
    names=[x["case"] for x in cases]
    if not cases or len(set(names))!=len(names):
        raise ValueError("Need one or more distinct verified real-run cases")
    return {
        "schema":SCHEMA,
        "status":"SOURCE_VERIFIED_PRIVATE_MACHINE_ONLY_MULTIFILM_SUMMARY",
        "verified_case_count":len(cases),
        "per_case":cases,
        "real_bmp_bytes_verified_case_count":sum(
            bool(x["actual_original_BMP_bytes_verified"]) for x in cases
        ),
        "cases_missing_S96_evaluation":[x["case"] for x in cases if x["s96"] is None],
        "never_infer_2D_fragment_count_as_physical_track_count":True,
        "same_film_multiple_ROIs_not_independent_physics_events":True,
        "operator_070_files_not_automatically_GT_for_H50Z_405_406_407":True,
        "single_case_performance_comparison_not_expert_validation":True,
    }


def html_summary(report):
    rows=[]
    for r in report["per_case"]:
        z=r["s86"]
        p=r["s92"]
        cls=r["s95"]
        s96=r["s96"]
        fields=(
            r["case"],r["original_bmp_name"],
            "да" if r["actual_original_BMP_bytes_verified"] else "нет, только SHA цепочки отчётов",
            z["tile_count"],z["candidate_fragments"],
            z["source_components_after_S87"],z["skipped_fragment_extraction_tiles"],
            p["all_generated_gap_proposals"],
            p["priority_gap_review_links_from_report"],
            cls["typed_observed_components"],
            s96["candidate_junctions"] if s96 else "не предоставлен",
            s96["provisional_visual_continuation_pairs"] if s96 else "не предоставлен",
        )
        rows.append("<tr>"+"".join("<td>"+html.escape(str(v),quote=True)+"</td>" for v in fields)+"</tr>")
    return (
        "<!doctype html><html lang='ru'><meta charset='utf-8'>"
        "<title>Реальные H50Z: аудит исходных машинных результатов</title>"
        "<style>body{font:15px system-ui;margin:20px;color:#202832}"
        "table{border-collapse:collapse}td,th{padding:8px;border:1px solid #bbb}"
        "p{max-width:1000px}</style>"
        "<h1>Аудит сохранённых оригинальных S86–S96</h1>"
        "<p>Все числа сверены с фактическими S86/S92/S95/S96 файлами, "
        "указанные SHA указывают на одну фотографию. При отсутствии "
        "исходного BMP проверена лишь сохранённая источниковая цепочка."
        "Все оценки — результаты машинной обработки, не GT физических "
        "треков и не медицинские измерения.</p>"
        "<table><tr><th>Случай</th><th>BMP</th><th>Байты BMP проверены</th>"
        "<th>Тайлов</th><th>Фрагментов</th><th>Компонентов S87</th>"
        "<th>Пропущено тайлов</th><th>S92 предложения</th>"
        "<th>S92 приоритет</th><th>Морфологий S95</th>"
        "<th>S96 узлов</th><th>S96 пар продолжения</th></tr>"
        +"".join(rows)+"</table>"
        "<p>Число фрагментов, морфологий, узлов и предложений <b>не является</b> "
        "количеством настоящих физических треков. Для метрик полноты "
        "и ошибочных связей нужен независимый эталон S97 и проверка S98.</p>"
        "</html>"
    )
