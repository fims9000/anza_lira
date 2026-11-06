#!/usr/bin/env python3
"""S101 compare exact bounded global vs prior S92 greedy source gap hypotheses.

Source photo S86/S92 are already computed: NO rerendered pixels, no
modification of source reports/centerlines. Source original S92 candidate
NDJSON is SHA-checked against S86/BMP and category/frozen decision totals.

Optionally evaluate both variants on the SAME independently source-only S97
expert reference, via source-locked S98 conditional detected-piece adjacency.
This never becomes full physical particle true positive or recall.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from bubble_chamber.bitmap_tiles import sha256_file
from bubble_chamber.s87_seam_continuity import _read_source
from bubble_chamber.s92_curve_review import collapse_observed_components,_write_private,_json,_ndjson
from bubble_chamber.s101_global_gap_matching import (
    GlobalGapConfig,select_global_gap_path_review,SELECT
)
from bubble_chamber.s98_trace_continuity import discover_human_trace_piece_boundaries
from scripts.research.bubble_chamber.physicists_s97_reference_benchmark import (
    _read_original_pack
)


def _checked_s92_source(s86_dir,s92_dir):
    s86_dir=Path(s86_dir).resolve()
    s92_dir=Path(s92_dir).resolve()
    manifest,records,s86sha=_read_source(s86_dir)
    observed,_=collapse_observed_components(records)
    reportfile=s92_dir/"report.json"
    datafile=s92_dir/"gap_proposals_unknown.ndjson"
    if not reportfile.is_file() or not datafile.is_file():
        raise FileNotFoundError("S101 exact S92 report and candidate NDJSON required")
    reportsha=sha256_file(reportfile)
    datasha=sha256_file(datafile)
    report=json.loads(reportfile.read_text(encoding="utf-8"))
    if (report.get("status")!="COMPLETE_PRIVATE_REVIEW_ONLY"
        or report.get("source_s86_manifest_sha256")!=s86sha
        or report.get("source_bmp_sha256")!=manifest["source_bmp_sha256"]
        or report.get("source_bmp_name")!=manifest["source_bmp_name"]
        or report.get("observed_components_after_S87")!=len(observed)):
        raise ValueError("S101 S86/S92 saved lineage or original photo mismatch")
    proposals=[]
    with datafile.open(encoding="utf-8") as fd:
        for line in fd:
            if line.strip():
                proposals.append(json.loads(line))
    if len(proposals)>60000:
        raise ValueError("S101 candidate count exceeds memory-safe source budget")
    # S92 has an explicit density abstention count without emitting proposals;
    # in that case S101 must preserve the conservative source abstention.
    if "abstain_reason" in report["gap"]:
        raise ValueError("S101 refuses to reinterpret aborted dense S92 source gap proposals")
    if (len(proposals)!=report["gap"]["gap_candidates"]
        or sum(p["review_decision"]=="PRIORITY_GAP_REVIEW_NOT_PHYSICAL_ACCEPT"
               for p in proposals)!=report["gap"]["gap_review_links"]):
        raise ValueError("S101 original S92 proposal decisions/counts do not match source report")
    if (sha256_file(reportfile)!=reportsha or sha256_file(datafile)!=datasha
        or sha256_file(s86_dir/"manifest.json")!=s86sha):
        raise ValueError("S101 original S86/S92 changed during load")
    return manifest,observed,proposals,{
        "source_s86_manifest_sha256":s86sha,
        "source_s92_report_sha256":reportsha,
        "source_s92_candidates_sha256":datasha,
    }


def _evaluate_same_human_reference(pack_dir,reference_json,manifest,observed,rows,old):
    pack,expert,packsha,refsha=_read_original_pack(
        Path(pack_dir).resolve(),Path(reference_json).resolve()
    )
    if (pack["source_bmp_sha256"]!=manifest["source_bmp_sha256"]
        or pack["source_bmp_name"]!=manifest["source_bmp_name"]
        or pack["image_shape_yx"]!=manifest["image_shape_yx"]):
        raise ValueError("S101 independent S97 expert labels are from another photo")
    independent=discover_human_trace_piece_boundaries(
        human_strokes=expert,observed_components=observed
    )
    truth={(x["component_a"],x["component_b"])
           for x in independent["conditionally_assessable_piece_adjacencies"]}
    oldpairs={tuple(sorted((p["component_a"],p["component_b"])))
              for p in old if p["review_decision"]==
              "PRIORITY_GAP_REVIEW_NOT_PHYSICAL_ACCEPT"}
    newpairs={tuple(sorted((p["component_a"],p["component_b"])))
              for p in rows if p["global_review_decision"]==SELECT}
    denom=len(truth)
    return {
        "independent_human_source_pack_sha256":packsha,
        "independent_human_annotations_sha256":refsha,
        "assessable_human_positive_source_piece_boundaries":denom,
        "old_s92_selected_boundary_coverage":(
            len(oldpairs&truth)/denom if denom else None
        ),
        "s101_global_selected_boundary_coverage":(
            len(newpairs&truth)/denom if denom else None
        ),
        "old_S92_selected_matched_boundaries":len(oldpairs&truth),
        "s101_global_selected_matched_boundaries":len(newpairs&truth),
        "unselected_human_boundaries_after_S101":[
            {"component_a":a,"component_b":b}
            for a,b in sorted(truth-newpairs)
        ],
        "conditional_visual_2D_source_piece_adjacency_only":True,
        "not_complete_physical_track_link_recall_or_precision":True,
        "reference_first_never_model_selected_GT":True,
    }


def run_s101_gap_ablation(
    *,s86_dir:Path,s92_dir:Path,private_out_dir:Path,
    independent_s97_pack_dir:Path|None=None,
    independent_s97_reference_json:Path|None=None,
    config:GlobalGapConfig=GlobalGapConfig(),
):
    config.validate()
    s86_dir=Path(s86_dir).resolve()
    s92_dir=Path(s92_dir).resolve()
    out=Path(private_out_dir).resolve()
    repo=ROOT.resolve()
    sources=[s86_dir,s92_dir]
    if (independent_s97_pack_dir is None)!=(independent_s97_reference_json is None):
        raise ValueError("S101 S97 original reference pack AND human JSON must both be supplied")
    if independent_s97_pack_dir is not None:
        sources.extend([
            Path(independent_s97_pack_dir).resolve(),
            Path(independent_s97_reference_json).resolve()
        ])
    if (out.exists() or out==repo or repo in out.parents
        or any(out==p or out in p.parents or p in out.parents for p in sources)):
        raise ValueError("S101 requires a NEW PRIVATE source-separated result directory")
    manifest,observed,original,receipts=_checked_s92_source(s86_dir,s92_dir)
    rows,score=select_global_gap_path_review(
        original,
        original_source_component_ids={p["component_id"] for p in observed},
        original_bmp_sha256=manifest["source_bmp_sha256"],
        config=config,
    )
    score.update({
        **receipts,
        "source_original_photo_name":manifest["source_bmp_name"],
        "source_S86_observed_components":len(observed),
        "independent_S97_S98_human_reference_comparison":None,
    })
    if independent_s97_pack_dir is not None:
        score["independent_S97_S98_human_reference_comparison"]=(
            _evaluate_same_human_reference(
                independent_s97_pack_dir,independent_s97_reference_json,
                manifest,observed,rows,original
            )
        )
    if any(sha256_file(p)!=sha for p,sha in (
        (s86_dir/"manifest.json",receipts["source_s86_manifest_sha256"]),
        (s92_dir/"report.json",receipts["source_s92_report_sha256"]),
        (s92_dir/"gap_proposals_unknown.ndjson",
         receipts["source_s92_candidates_sha256"]),
    )):
        raise ValueError("S101 exact original S86/S92 source changed during optimization")
    out.mkdir(parents=True,mode=0o700)
    out.chmod(0o700)
    _write_private(out/"global_gap_review_unknown.ndjson",_ndjson(rows))
    _write_private(out/"report.json",_json(score))
    return score


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s86-dir",type=Path,required=True)
    parser.add_argument("--s92-dir",type=Path,required=True)
    parser.add_argument("--private-out-dir",type=Path,required=True)
    parser.add_argument("--s97-pack-dir",type=Path,
                        help="Optional independently annotated SAME SOURCE original photo pack")
    parser.add_argument("--s97-human-reference",type=Path)
    parser.add_argument("--exact-max-edges",type=int,default=16)
    parser.add_argument("--approx-max-edges",type=int,default=400)
    parser.add_argument("--approx-max-port-nodes",type=int,default=320)
    args=parser.parse_args(argv)
    report=run_s101_gap_ablation(
        s86_dir=args.s86_dir,s92_dir=args.s92_dir,
        private_out_dir=args.private_out_dir,
        independent_s97_pack_dir=args.s97_pack_dir,
        independent_s97_reference_json=args.s97_human_reference,
        config=GlobalGapConfig(
            max_exact_edges_per_group=args.exact_max_edges,
            max_approx_edges_per_group=args.approx_max_edges,
            max_approx_port_nodes=args.approx_max_port_nodes
        ),
    )
    print(json.dumps({
        "status":report["status"],
        "greedy_original_links":report["source_S92_original_greedy_selected_count"],
        "global_source_tentative_links":report["S101_global_tentative_selected_count"],
        "old_S92_geometry_objective":report["original_s92_greedy_geometry_objective"],
        "S101_geometry_objective":report["s101_global_geometry_objective"],
        "complexity_abstained_source_edges":report["number_of_complexity_abstained_S92_edges"],
        "independent_human_2D_boundary_validation":report[
            "independent_S97_S98_human_reference_comparison"
        ],
        "private_report":str(args.private_out_dir.resolve()/"report.json"),
        "physical_particle_truth_proven":False,
    },ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
