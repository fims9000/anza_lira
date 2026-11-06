#!/usr/bin/env python3
"""S97 source-locked independent human 2D trace benchmark, private artifacts.

prepare: lossless grayscale ORIGINAL BMP tiles with NO machine overlay, offline
         human polylines, explicit visual instance ID and morphologies.
evaluate: compare one or more exact-source S86 runs on the SAME recorded
          positive human strokes; optionally audit S92/S96 proposed joins
          against image-level independently marked visible instance IDs.

Not full real-film recall, precision, physical truth, or statistical CI.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from bubble_chamber.bitmap_tiles import sha256_file
from bubble_chamber.s87_seam_continuity import _read_source
from bubble_chamber.s92_curve_review import collapse_observed_components
from bubble_chamber.s97_manual_reference import (
    SCHEMA,EVALUATION_SCHEMA,validate_reference_annotation,
    evaluate_partial_positive_coverage,paired_coverage_contrasts
)
from bubble_chamber.s97_link_audit import (
    map_observed_components_to_reference,audit_provisional_joins
)
from bubble_chamber.s97_reference_viewer import prepare_independent_source_rois


def _json_lines(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _safe_new_private_output(destination:Path,source_paths):
    out=destination.resolve()
    root=ROOT.resolve()
    if out.exists() or out==root or root in out.parents:
        raise ValueError("Output must be a NEW private file outside Git")
    if any(out==src.resolve() or src.resolve() in out.parents for src in source_paths):
        raise ValueError("S97 output must not overwrite any source file/folder")
    return out


def _candidate_spec(text):
    name,separator,path=text.partition("=")
    if not separator or not name or not path or len(name)>60:
        raise ValueError("Use --candidate NAME=/PRIVATE/S86_DIR for every model")
    if not all(x.isascii() and (x.isalnum() or x in "_-.") for x in name):
        raise ValueError("S97 model name must be plain ASCII letters, numbers, '_' '-' '.'")
    return name,Path(path).resolve()


def _read_original_pack(pack_dir:Path,annotation_path:Path):
    pack_dir=pack_dir.resolve()
    file=pack_dir/"reference_pack.json"
    original_sha=sha256_file(file)
    pack=json.loads(file.read_text(encoding="utf-8"))
    if pack.get("schema")!=SCHEMA or pack.get("status")!="INDEPENDENT_ORIGINAL_BMP_HUMAN_POSITIVE_REFERENCE_PACK":
        raise ValueError("S97 needs completed standalone original-photo annotation pack")
    if (pack.get("human_edit_does_not_show_machine_predictions") is not True
        or pack.get("annotation_is_partial_positive_strokes_only") is not True
        or pack.get("tile_pixels_from_unmodified_original_bmp") is not True):
        raise ValueError("S97 refuses non-independent or machine-overlaid reference")
    tiles=pack["source_image_tiles"]
    for row in tiles:
        relative=Path(row["source_original_png"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe source image tile reference")
        source_path=(pack_dir/relative).resolve()
        if pack_dir not in source_path.parents or sha256_file(source_path)!=row["tile_preview_sha256"]:
            raise ValueError("Source original pixel tile missing or changed")
    annotation_sha=sha256_file(annotation_path)
    ref=json.loads(annotation_path.read_text(encoding="utf-8-sig"))
    if ref.get("source_reference_pack_sha256")!=original_sha:
        raise ValueError("Reference JSON was created from another S97 original-photo pack")
    human=validate_reference_annotation(ref,pack)
    if sha256_file(file)!=original_sha or sha256_file(annotation_path)!=annotation_sha:
        raise ValueError("Human reference changed during validation")
    return pack,human,original_sha,annotation_sha


def _read_compatible_candidate(path,pack):
    manifest,records,manifest_sha=_read_source(path)
    if (manifest.get("source_bmp_sha256")!=pack["source_bmp_sha256"]
        or manifest.get("source_bmp_name")!=pack["source_bmp_name"]
        or manifest.get("image_shape_yx")!=pack["image_shape_yx"]):
        raise ValueError("Candidate S86 from a different BMP original/version")
    components,seam_stats=collapse_observed_components(records)
    return manifest,components,manifest_sha


def _read_join_pack(path,kind,manifest_sha,pack):
    file=path/"report.json"
    report_sha=sha256_file(file)
    payload=json.loads(file.read_text(encoding="utf-8"))
    if (payload.get("source_s86_manifest_sha256")!=manifest_sha
        or payload.get("source_bmp_sha256")!=pack["source_bmp_sha256"]):
        raise ValueError(f"{kind} join pack from another source S86 or original photo")
    if kind=="S92_GAP":
        if payload.get("status")!="COMPLETE_PRIVATE_REVIEW_ONLY":
            raise ValueError("Need complete machine-only S92 proposal pack")
        filename="gap_proposals_unknown.ndjson"
    else:
        if payload.get("status")!="REVERSIBLE_2D_JUNCTION_REVIEW_ONLY":
            raise ValueError("Need complete original-fragment S96 proposal pack")
        filename="visual_continuation_pairs_unknown.ndjson"
    source=path/filename
    checksum=sha256_file(source)
    edges=_json_lines(source)
    if sha256_file(source)!=checksum or sha256_file(file)!=report_sha:
        raise ValueError("S92/S96 source changed during link-audit preparation")
    return edges,report_sha,checksum


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    p=sub.add_parser("prepare",help="Export original high-res BMP tiles and independent offline annotator")
    p.add_argument("--bmp",type=Path,required=True)
    p.add_argument("--private-out-dir",type=Path,required=True)
    p.add_argument("--roi",nargs=4,action="append",type=int,
                   metavar=("X0","Y0","X1","Y1"),
                   help="Original pixel rectangle; repeatable, recommended for large BMPs")
    p.add_argument("--whole-photo",action="store_true",
                   help="Export entire photo as tiled original grayscale PNG")
    p.add_argument("--tile-size",type=int,default=512)
    p.add_argument("--max-tiles",type=int,default=256)
    e=sub.add_parser("evaluate",help="Compare matched S86 models on independent human-positive traces")
    e.add_argument("--pack-dir",type=Path,required=True)
    e.add_argument("--reference-json",type=Path,required=True)
    e.add_argument("--candidate",action="append",required=True,
                   help="NAME=/PRIVATE/S86_FULL_RESULTS; repeat for S88/AZUNet/etc")
    e.add_argument("--s92-dir",type=Path,
                   help="Optional original-fragment S92 links from FIRST candidate S86")
    e.add_argument("--s96-dir",type=Path,
                   help="Optional original-fragment S96 links from FIRST candidate S86")
    e.add_argument("--radius-px",type=float,default=2.)
    e.add_argument("--private-out-json",type=Path,required=True)
    args=parser.parse_args(argv)

    if args.action=="prepare":
        if args.roi and args.whole_photo:
            parser.error("Choose explicit --roi OR --whole-photo")
        result=prepare_independent_source_rois(
            args.bmp,args.private_out_dir,
            roi_xyxy_source=args.roi,
            tile_size=args.tile_size,
            max_tiles=args.max_tiles,
            full_photo=args.whole_photo,
        )
        print(json.dumps({
            "status":result["status"],
            "source_name":result["source_bmp_name"],
            "independent_reference_editor":str(args.private_out_dir/"index.html"),
            "lossless_source_tiles":len(result["source_image_tiles"]),
            "source_roi_count":len(result["roi_xyxy_source"]),
            "not_dense_background_GT":result["unreviewed_photo_regions_not_negative"],
        },ensure_ascii=False,indent=2))
        return

    pack_dir=args.pack_dir.resolve()
    reference_path=args.reference_json.resolve()
    candidates=[_candidate_spec(x) for x in args.candidate]
    if len({name for name,_ in candidates})!=len(candidates):
        raise ValueError("Duplicate candidate model label in matched S97 comparison")
    paths=[pack_dir,reference_path]+[x[1] for x in candidates]
    for extra in (args.s92_dir,args.s96_dir):
        if extra is not None:
            paths.append(extra.resolve())
    destination=_safe_new_private_output(args.private_out_json,paths)
    pack,human,pack_sha,annotation_sha=_read_original_pack(
        pack_dir,reference_path
    )
    rows=[]
    links={}
    baseline_components=baseline_manifest_sha=None
    source_receipts=[]
    for i,(name,folder) in enumerate(candidates):
        manifest,components,manifest_sha=_read_compatible_candidate(folder,pack)
        row=evaluate_partial_positive_coverage(
            human,components,radius_px=args.radius_px
        )
        row["candidate_name"]=name
        row["s86_source_manifest_sha256"]=manifest_sha
        row["s86_source_method"]=manifest["config"]["method"]
        row["s86_candidate_extraction_skipped_tiles"]=manifest[
            "tiles_with_fragment_extraction_skipped"
        ]
        row["s86_source_candidate_fragments"]=manifest["candidate_fragments"]
        row["s86_observed_components"]=len(components)
        row["all_extracted_source_fragments_preserved_by_S87"]=True
        rows.append(row)
        source_receipts.append({
            "candidate_name":name,
            "s86_source_manifest_sha256":manifest_sha,
            "s86_path_local_PRIVATE":str(folder),
        })
        if i==0:
            baseline_components=components
            baseline_manifest_sha=manifest_sha

    if args.s92_dir is not None or args.s96_dir is not None:
        mapping,map_stats=map_observed_components_to_reference(
            human,baseline_components,radius_px=args.radius_px
        )
        links["human_to_first_S86_component_assignment"]=map_stats
        for kind,path in (
            ("S92_GAP",args.s92_dir),("S96_CROSSING",args.s96_dir)
        ):
            if path is None:
                continue
            edges,report_sha,edges_sha=_read_join_pack(
                path.resolve(),kind,baseline_manifest_sha,pack
            )
            block=audit_provisional_joins(
                edges=edges,component_labels=mapping,source_kind=kind
            )
            block["source_proposal_report_sha256"]=report_sha
            block["source_proposal_list_sha256"]=edges_sha
            links[kind]=block
    result={
        "schema":EVALUATION_SCHEMA,
        "status":"MATCHED_INDEPENDENT_2D_HUMAN_POSITIVE_BENCHMARK_PRIVATE_ONLY",
        "source_bmp_name":pack["source_bmp_name"],
        "source_bmp_sha256":pack["source_bmp_sha256"],
        "image_shape_yx":pack["image_shape_yx"],
        "operator_visual_reviewer_id_local_PRIVATE":json.loads(
            reference_path.read_text(encoding="utf-8-sig")
        )["reviewer_id"],
        "source_reference_pack_sha256":pack_sha,
        "source_operator_reference_json_sha256":annotation_sha,
        "source_positive_reference_reviewed_ROIs":len(pack["roi_xyxy_source"]),
        "all_human_reference_strokes":len(human),
        "clear_positive_track_or_coil_strokes":sum(
            x["kind"] in ("TRACK_LIKE","COILING_TRACK_LIKE")
            and x["certainty"]=="CLEAR" for x in human
        ),
        "source_candidate_receipts":source_receipts,
        "per_candidate_matched_positive_coverage":rows,
        "descriptive_paired_model_contrasts":paired_coverage_contrasts(rows),
        "optional_visual_instance_provisional_link_audits":links,
        "evaluation_scope":"ONLY_CONFIDENT_EXPERT_POSITIVE_2D_ROI_STROKES",
        "not_full_photo_recall":True,
        "no_dense_negative_mask":True,
        "precision_f1_fpr_not_identifiable":True,
        "no_particle_identity_or_neutron_causality":True,
        "one_photo_no_independent_film_level_CI":True,
        "no_data_from_machine_overlay_visible_during_annotation":True,
        "underlying_original_source_not_modified":True,
    }
    if (sha256_file(pack_dir/"reference_pack.json")!=pack_sha
        or sha256_file(reference_path)!=annotation_sha):
        raise ValueError("Reference pack or operator annotations changed mid-run")
    destination.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    destination.parent.chmod(0o700)
    with destination.open("x",encoding="utf8") as f:
        json.dump(result,f,ensure_ascii=False,indent=2,sort_keys=True)
        f.write("\n")
    destination.chmod(0o600)
    print(json.dumps({
        "status":result["status"],
        "human_clear_visual_tracks":result["clear_positive_track_or_coil_strokes"],
        "candidate_model_coverage":[{
            "name":x["candidate_name"],
            "observed_centerline_coverage":x["observed_positive_centerline_coverage"],
            "missed_original_tiles":x["s86_candidate_extraction_skipped_tiles"],
        } for x in rows],
        "descriptive_same_photo_differences":result["descriptive_paired_model_contrasts"],
        "private_report":str(destination),
        "precision_not_defined":True,
    },ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
