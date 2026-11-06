#!/usr/bin/env python3
"""S99: preregister independent original H50Z ROIs and measure S95 classes.

prepare: SOURCE-ONLY spatially balanced 512px ROIs across entire BMP. No
         machine mask, class or model score is read. Exports full-resolution
         ORIGINAL PNGs and existing blind S97 browser editor.
evaluate: only AFTER a human reference JSON has been exported independently:
          verify exact source BMP/ROI/pack, original PNG pixel SHA, exact S86
          masks and matching S92/S95 SHA lineage for every candidate model;
          report class-specific S86 coverage and conditional S95 morphology
          agreement, posthoc original photo+S95 gallery.

No full-photo precision/F1, no physical particle GT, no statistical CI.
"""
from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from bubble_chamber.bitmap_tiles import sha256_file
from bubble_chamber.s99_blind_sampling import (
    SCHEMA as SAMPLE_SCHEMA,select_blind_original_source_rois,
    prepare_blind_s99_review
)
from bubble_chamber.s99_morphology_audit import (
    audit_s95_against_blind_source_strokes,
    descriptive_same_photo_morphology_contrasts
)
from bubble_chamber.s99_posthoc_gallery import export_s99_posthoc_review
from scripts.research.bubble_chamber.physicists_s97_reference_benchmark import (
    _candidate_spec,_read_original_pack,_read_compatible_candidate
)


def _args_map(values,allow,which):
    result={}
    for item in values:
        name,path=_candidate_spec(item)
        if name not in allow or name in result:
            raise ValueError(f"{which}: unknown or duplicated named original-S86 model")
        result[name]=path
    if set(result)!=set(allow):
        raise ValueError(f"{which}: all named original-S86 models require own matching stage")
    return result


def _frozen_selection(pack_dir,pack,pack_sha):
    prereg=pack_dir/"s99_blind_selection.json"
    if not prereg.is_file():
        raise ValueError("S99 evaluation requires independent PRESELECTED original source ROI receipts")
    frozen_sha=sha256_file(prereg)
    source=json.loads(prereg.read_text(encoding="utf-8"))
    if (source.get("schema")!=SAMPLE_SCHEMA
        or source.get("status")!="SOURCE_ONLY_BLIND_PRESELECTED_SAMPLE_NO_MODEL_ACCESS"
        or source.get("source_bmp_sha256")!=pack["source_bmp_sha256"]
        or source.get("source_bmp_name")!=pack["source_bmp_name"]
        or source.get("source_reference_pack_sha256")!=pack_sha
        or source.get("human_editor_shows_machine_prediction") is not False):
        raise ValueError("S99 source original ROI preregistration mismatch or leakage")
    expected=select_blind_original_source_rois(
        height=pack["image_shape_yx"][0],width=pack["image_shape_yx"][1],
        source_sha256=pack["source_bmp_sha256"],
        seed=source["sampling_seed_predeclared"],
        roi_count=source["sampled_source_roi_count"],
        tile_size=source["tile_size_px"],
        strata_rows=source["strata_grid_yx"][0],
        strata_cols=source["strata_grid_yx"][1],
    )
    if (source["sampled_cell_selection"]!=expected["sampled_cell_selection"]
        or pack["roi_xyxy_source"]!=[
            x["xyxy_source"] for x in expected["sampled_cell_selection"]
        ]
        or len(pack["source_image_tiles"])!=len(pack["roi_xyxy_source"])):
        raise ValueError("S99 blind sample is modified, has different ROIs or more source image tiles")
    if sha256_file(prereg)!=frozen_sha:
        raise ValueError("S99 blind design changed while loading")
    return source,frozen_sha


def _read_checked_S95(s86_manifest_sha,manifest,original_components,
                      s92_dir:Path,s95_dir:Path):
    s92_dir=Path(s92_dir).resolve()
    s95_dir=Path(s95_dir).resolve()
    s92_report=s92_dir/"report.json"
    s95_report=s95_dir/"report.json"
    s95_types=s95_dir/"all_source_fragment_types.ndjson"
    for p in (s92_report,s95_report,s95_types):
        if not p.is_file():
            raise FileNotFoundError("S99 missing saved source S92/S95 report or class inventory")
    s92_sha=sha256_file(s92_report)
    s95_sha=sha256_file(s95_report)
    typed_sha=sha256_file(s95_types)
    s92=json.loads(s92_report.read_text(encoding="utf8"))
    s95=json.loads(s95_report.read_text(encoding="utf8"))
    if (s92.get("status")!="COMPLETE_PRIVATE_REVIEW_ONLY"
        or s92.get("source_s86_manifest_sha256")!=s86_manifest_sha
        or s92.get("source_bmp_sha256")!=manifest["source_bmp_sha256"]):
        raise ValueError("S99 S92 not from this source model, exact BMP or complete stage")
    if (s95.get("status")!="ALL_S86_EXTRACTED_FRAGMENTS_TYPED_MACHINE_REVIEW_ONLY"
        or s95.get("source_s86_manifest_sha256")!=s86_manifest_sha
        or s95.get("source_s92_report_sha256")!=s92_sha
        or s95.get("source_bmp_sha256")!=manifest["source_bmp_sha256"]
        or s95.get("source_fragments_deleted")!=0):
        raise ValueError("S99 S95 morphology not from same source S86/S92/BMP")
    source_ids={c["component_id"] for c in original_components}
    rows=[]
    seen=set()
    count=Counter()
    with s95_types.open(encoding="utf-8") as fd:
        for line in fd:
            if not line.strip():
                continue
            item=json.loads(line)
            cid=item["component_id"]
            if cid not in source_ids or cid in seen:
                raise ValueError("S99 S95 source component missing/duplicated/non-source")
            seen.add(cid)
            rows.append(item)
            count[item["primary_review_category"]]+=1
    if (seen!=source_ids
        or dict(count)!=s95["all_extracted_line_categories"]["categories"]
        or s95["all_extracted_line_categories"]["typed_components"]!=len(source_ids)):
        raise ValueError("S99 S95 report totals do not match original source class NDJSON")
    for p,sha in ((s92_report,s92_sha),(s95_report,s95_sha),(s95_types,typed_sha)):
        if sha256_file(p)!=sha:
            raise ValueError("S99 S92/S95 source changed during strict lineage check")
    return rows,{
        "source_s92_report_sha256":s92_sha,
        "source_s95_report_sha256":s95_sha,
        "source_s95_typed_component_ndjson_sha256":typed_sha,
        "source_observed_component_count":len(source_ids),
        "machine_S95_all_source_class_counts":dict(count),
    }


def evaluate_s99(*,pack_dir:Path,reference_json:Path,
                 candidates:list[tuple[str,Path]],
                 s92_by_name:dict[str,Path],s95_by_name:dict[str,Path],
                 private_out_dir:Path,radius_px:float=2.,
                 ambiguity_margin_px:float=.45,max_preview_tiles:int=9):
    pack_dir=Path(pack_dir).resolve()
    reference_json=Path(reference_json).resolve()
    out=Path(private_out_dir).resolve()
    repo=ROOT.resolve()
    if out.exists() or out==repo or repo in out.parents:
        raise ValueError("S99 requires NEW private report directory outside Git")
    paths=[pack_dir,reference_json]+[Path(p).resolve() for _,p in candidates]
    paths+=list(s92_by_name.values())+list(s95_by_name.values())
    if any(out==p or p in out.parents or out in p.parents for p in paths):
        raise ValueError("S99 private reports may not overwrite source S86/S92/S95 or expert input")
    names=[name for name,_ in candidates]
    if not names or len(names)!=len(set(names)) or len(names)>8:
        raise ValueError("S99 expects 1..8 distinct candidate model names")
    if set(s92_by_name)!=set(names) or set(s95_by_name)!=set(names):
        raise ValueError("S99 every S86 model must have same-source S92 and S95")
    pack,human,pack_sha,human_sha=_read_original_pack(pack_dir,reference_json)
    prereg,pre_sha=_frozen_selection(pack_dir,pack,pack_sha)
    scores=[]
    previews=[]
    for name,s86_dir in candidates:
        manifest,components,manifest_sha=_read_compatible_candidate(
            Path(s86_dir).resolve(),pack
        )
        types,lineage=_read_checked_S95(
            manifest_sha,manifest,components,
            s92_by_name[name],s95_by_name[name]
        )
        result=audit_s95_against_blind_source_strokes(
            human,components,types,
            radius_px=radius_px,ambiguity_margin_px=ambiguity_margin_px
        )
        result["candidate_name"]=name
        result["source_s86_method"]=manifest["config"]["method"]
        result["source_s86_manifest_sha256"]=manifest_sha
        result["source_s86_skipped_fragment_extraction_tiles"]=manifest[
            "tiles_with_fragment_extraction_skipped"
        ]
        result["saved_source_S95_provenance"]=lineage
        scores.append(result)
        previews.append((name,components,{
            x["component_id"]:x["primary_review_category"] for x in types
        }))
    report={
        "schema":"S99_MULTIMETHOD_SAMPLED_SOURCE_MORPHOLOGY_AUDIT_V1",
        "status":"SOURCE_SHA_VERIFIED_INDEPENDENT_HUMAN_2D_MORPHOLOGY_PARTIAL_BENCHMARK",
        "original_bmp_name":pack["source_bmp_name"],
        "original_bmp_sha256":pack["source_bmp_sha256"],
        "original_bmp_dimensions_yx":pack["image_shape_yx"],
        "sample_source_reference_pack_sha256":pack_sha,
        "independent_human_stroke_reference_sha256":human_sha,
        "fixed_blind_spatial_preselection_sha256":pre_sha,
        "sampled_source_rois":prereg["sampled_cell_selection"],
        "sampled_source_roi_count":prereg["sampled_source_roi_count"],
        "reviewed_reference_strokes_total":len(human),
        "candidate_model_scores":scores,
        "descriptive_same_photo_model_differences":
            descriptive_same_photo_morphology_contrasts(scores),
        "posterior_model_preview_not_shown_to_original_human_annotator":True,
        "negative_source_image_labels_are_not_exhaustive":True,
        "conditional_morphology_agreement_not_full_photo_precision_F1":True,
        "no_particle_same_track_or_neutron_collision_truth_established":True,
        "source_image_hash_and_all_S86_masks_reverified":True,
        "human_spatial_cells_not_independent_physical_events":True,
        "no_film_level_confidence_interval":True,
    }
    if (sha256_file(pack_dir/"reference_pack.json")!=pack_sha
        or sha256_file(pack_dir/"s99_blind_selection.json")!=pre_sha
        or sha256_file(reference_json)!=human_sha):
        raise ValueError("S99 sealed independent human source reference changed during evaluation")
    gallery=export_s99_posthoc_review(
        pack_dir=pack_dir,pack=pack,human=human,
        candidates=previews,out_dir=out,max_preview_tiles=max_preview_tiles
    )
    report["posthoc_private_preview_manifest_sha256"]=sha256_file(
        out/"posthoc_gallery_manifest.json"
    )
    report["posthoc_private_source_tiles_shown"]=gallery["source_tiles_previewed"]
    path=out/"morphology_benchmark.json"
    with path.open("x",encoding="utf-8") as fd:
        json.dump(report,fd,ensure_ascii=False,indent=2,sort_keys=True)
        fd.write("\n")
    path.chmod(0o600)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="cmd",required=True)
    a=sub.add_parser("prepare",help="Original pixels only, spatially preselected blind S97 annotation")
    a.add_argument("--bmp",required=True,type=Path)
    a.add_argument("--private-out-dir",required=True,type=Path)
    a.add_argument("--seed",type=int,default=20261106)
    a.add_argument("--tile-size",type=int,default=512)
    a.add_argument("--roi-count",type=int,default=18)
    a.add_argument("--strata-rows",type=int,default=3)
    a.add_argument("--strata-cols",type=int,default=3)

    e=sub.add_parser("evaluate",help="AFTER frozen independent S97 human labels, compare machine S95 classes")
    e.add_argument("--pack-dir",type=Path,required=True)
    e.add_argument("--reference-json",type=Path,required=True)
    e.add_argument("--candidate",action="append",required=True,
                   help="NAME=/PRIVATE/S86_FOLDER; 1..8 models")
    e.add_argument("--s92",action="append",required=True,
                   help="NAME=/PRIVATE/S92_FOLDER (must match the corresponding source S86)")
    e.add_argument("--s95",action="append",required=True,
                   help="NAME=/PRIVATE/S95_FOLDER (must match S86/S92)")
    e.add_argument("--private-out-dir",type=Path,required=True)
    e.add_argument("--radius-px",type=float,default=2.)
    e.add_argument("--ambiguity-margin-px",type=float,default=.45)
    e.add_argument("--max-preview-tiles",type=int,default=9)
    args=p.parse_args(argv)
    if args.cmd=="prepare":
        selection=prepare_blind_s99_review(
            args.bmp,args.private_out_dir,
            seed=args.seed,tile_size=args.tile_size,
            roi_count=args.roi_count,
            strata_rows=args.strata_rows,strata_cols=args.strata_cols
        )
        print(json.dumps({
            "status":selection["status"],
            "original_bmp":selection["source_bmp_name"],
            "source_pixel_ROIs":selection["sampled_source_roi_count"],
            "independent_blind_offline_S97_editor":str(args.private_out_dir/"index.html"),
            "source_SHA256":selection["source_bmp_sha256"],
            "machine_hypotheses_visible":False
        },ensure_ascii=False,indent=2))
    else:
        candidates=[_candidate_spec(x) for x in args.candidate]
        names=[x[0] for x in candidates]
        if len(set(names))!=len(names):
            raise ValueError("S99 duplicate source model names")
        results=evaluate_s99(
            pack_dir=args.pack_dir,
            reference_json=args.reference_json,
            candidates=candidates,
            s92_by_name=_args_map(args.s92,names,"--s92"),
            s95_by_name=_args_map(args.s95,names,"--s95"),
            private_out_dir=args.private_out_dir,
            radius_px=args.radius_px,
            ambiguity_margin_px=args.ambiguity_margin_px,
            max_preview_tiles=args.max_preview_tiles
        )
        print(json.dumps({
            "status":results["status"],
            "independent_visible_strokes":results["reviewed_reference_strokes_total"],
            "source_only_original_ROIs":results["sampled_source_roi_count"],
            "candidate_counts":[{
                "name":r["candidate_name"],
                "human_clear_classified_strokes":r["source_human_clear_classified_strokes"],
                "morphology_conditional_counts":r["per_human_morphology"],
            } for r in results["candidate_model_scores"]],
            "private_machine_after_human_review":str(
                args.private_out_dir/"index.html"
            ),
            "original_observed_physical_particle_GTs":0,
        },ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
