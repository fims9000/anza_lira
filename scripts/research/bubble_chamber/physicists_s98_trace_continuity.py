#!/usr/bin/env python3
"""S98 reproducible and private SAME-PHOTO S86 visual trace continuity audit.

Consumes the INDEPENDENT frozen S97 original-BMP human reference, not a
machine-guided set of proposed links. All models are evaluated on the same
expressly annotated positive human strokes. For each candidate, map
observed source pieces to the human-positive curve and measure what fraction
of independently located *detected-piece boundaries* S92/S96 propose.

This is conditional 2D visual continuity, NOT whole-film physical recall,
not neuron/particle identity, and NOT native real H50Z validation in CI.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from bubble_chamber.bitmap_tiles import sha256_file
from bubble_chamber.s97_manual_reference import (
    evaluate_partial_positive_coverage,paired_coverage_contrasts
)
from bubble_chamber.s98_trace_continuity import (
    SCHEMA,discover_human_trace_piece_boundaries,measure_proposal_recovery
)
from bubble_chamber.s98_visual_report import export_s98_private_visual_report
from scripts.research.bubble_chamber.physicists_s97_reference_benchmark import (
    _candidate_spec,_read_original_pack,_read_compatible_candidate,
    _read_join_pack
)


def _specs(specs,valid,flag):
    mapping={}
    for s in specs:
        name,path=_candidate_spec(s)
        if name not in valid:
            raise ValueError(f"{flag} references unknown candidate model {name}")
        if name in mapping:
            raise ValueError(f"Duplicate {flag} for candidate {name}")
        mapping[name]=path
    return mapping


def run_audit(*,
              pack_dir:Path,
              human_reference_json:Path,
              candidates:list[tuple[str,Path]],
              s92_packs:dict[str,Path],
              s96_packs:dict[str,Path],
              private_out_dir:Path,
              radius_px:float=2.,
              ambiguity_margin_px:float=.45,
              min_run_support_px:float=3.,
              max_reference_gap_px:float=80.,
              max_visual_case_count:int=16):
    pack_dir=Path(pack_dir).resolve()
    human_reference_json=Path(human_reference_json).resolve()
    private_out_dir=Path(private_out_dir).resolve()
    repo=ROOT.resolve()
    if (private_out_dir.exists() or private_out_dir==repo
        or repo in private_out_dir.parents
        or private_out_dir==pack_dir or pack_dir in private_out_dir.parents
        or private_out_dir==human_reference_json):
        raise ValueError("S98 must use a fresh PRIVATE folder, not Git/source")
    paths=[pack_dir,human_reference_json]+[path.resolve() for _,path in candidates]
    paths.extend([x.resolve() for x in s92_packs.values()])
    paths.extend([x.resolve() for x in s96_packs.values()])
    if any(p==private_out_dir or p in private_out_dir.parents for p in paths):
        raise ValueError("Never write S98 results inside any original S86/S97 source")
    names=[n for n,_ in candidates]
    if not names or len(names)>8 or len(set(names))!=len(names):
        raise ValueError("Duplicate or out-of-bounds S98 candidate names (1..8 distinct S86 runs required)")
    for name in names:
        _candidate_spec(name+"=/PRIVATE/"+"dummy")  # enforce safe ASCII run label
    if set(s92_packs)-set(names) or set(s96_packs)-set(names):
        raise ValueError("S92/S96 specified for missing candidate")
    pack,expert,pack_sha,reference_sha=_read_original_pack(
        pack_dir,human_reference_json
    )
    trials=[]
    coverage_rows=[]
    source_for_preview={}
    for name,path in candidates:
        manifest,components,manifest_sha=_read_compatible_candidate(path,pack)
        conditional=discover_human_trace_piece_boundaries(
            human_strokes=expert,observed_components=components,
            radius_px=radius_px,
            ambiguity_margin_px=ambiguity_margin_px,
            min_run_support_px=min_run_support_px,
            max_unobserved_gap_along_trace_px=max_reference_gap_px
        )
        positive=evaluate_partial_positive_coverage(
            expert,components,radius_px=radius_px
        )
        positive["candidate_name"]=name
        coverage_rows.append(positive)
        all_known={
            s["stroke_id"]:s for s in positive["per_human_stroke"]
        }
        per_trace=[]
        for trace in conditional["per_human_stroke"]:
            record=all_known.get(trace["stroke_id"])
            if record is None:
                raise ValueError("S97/S98 expert positive denominator inconsistent")
            per_trace.append({
                "human_visual_stroke_id":trace["stroke_id"],
                "human_visual_instance_id":trace["visual_instance_id"],
                "human_visible_type":trace["human_visible_type"],
                "human_reference_length_px":trace["human_curve_length_px"],
                "observed_positive_centerline_coverage":record[
                    "observed_S86_centerline_coverage"
                ],
                "unique_source_components_near_this_human_trace":trace[
                    "unique_stable_detected_pieces"
                ],
                "stable_run_count_on_this_trace":trace[
                    "stable_observed_component_runs"
                ],
                "unique_source_component_ids":trace[
                    "unique_stable_observed_component_ids"
                ],
                "observed_source_component_adjacency_events":trace[
                    "provisional_adjacencies_count"
                ],
                "max_uncovered_source_span_px":record["max_uncovered_span_px"],
                "uncovered_source_spans_at_least_4px":record[
                    "uncovered_spans_at_least_4px"
                ],
            })
        proposal_sets={}
        provenance={}
        for kind,lookup in (
            ("S92_GAP",s92_packs),
            ("S96_CROSSING",s96_packs),
        ):
            if name not in lookup:
                continue
            edges,rep_sha,edge_sha=_read_join_pack(
                lookup[name],kind,manifest_sha,pack
            )
            proposal_sets[kind]=edges
            provenance[kind]={
                "source_proposal_report_sha256":rep_sha,
                "source_proposal_list_sha256":edge_sha,
                "private_proposal_pack_path":str(lookup[name]),
            }
        recovery=(measure_proposal_recovery(
            conditional,proposal_sets=proposal_sets
        ) if proposal_sets else None)
        trial={
            "model_name":name,
            "source_S86_manifest_sha256":manifest_sha,
            "source_S86_private_path":str(path.resolve()),
            "source_S86_extraction_method":manifest["config"]["method"],
            "source_S86_skipped_tiles":manifest[
                "tiles_with_fragment_extraction_skipped"
            ],
            "source_original_candidate_fragments":manifest[
                "candidate_fragments"
            ],
            "human_clear_positive_strokes":conditional[
                "human_clear_positive_strokes"
            ],
            "observed_source_centerline_positive_coverage":positive[
                "observed_positive_centerline_coverage"
            ],
            "human_source_trace_per_stroke_results":per_trace,
            "reference_first_source_piece_boundaries":conditional,
            "proposed_link_packs":provenance,
            "reference_first_model_link_recovery":recovery,
            "visual_reference_is_not_physical_same_particle_GT":True,
            "one_source_component_could_be_multiple_real_tracks":True,
            "more_fragments_not_always_worse_when_detection_coverage_differs":True,
        }
        trials.append(trial)
        source_for_preview[name]=components
    # Immutable positive reference and source folders must still match exactly.
    if (sha256_file(pack_dir/"reference_pack.json")!=pack_sha
        or sha256_file(human_reference_json)!=reference_sha):
        raise ValueError("S98 independent human reference changed during evaluation")
    summary={
        "schema":SCHEMA,
        "status":"S98_SOURCE_LOCKED_CONDITIONAL_HUMAN_FIRST_2D_CONTINUITY_AUDIT",
        "source_bmp_name":pack["source_bmp_name"],
        "source_bmp_sha256":pack["source_bmp_sha256"],
        "source_reference_pack_sha256":pack_sha,
        "source_human_strokes_sha256":reference_sha,
        "source_reference_reviewed_ROIs":pack["roi_xyxy_source"],
        "human_clear_positive_stroke_count":sum(
            x["kind"] in ("TRACK_LIKE","COILING_TRACK_LIKE")
            and x["certainty"]=="CLEAR" for x in expert
        ),
        "candidates":trials,
        "matched_observed_centerline_positive_coverage_pp":paired_coverage_contrasts(
            coverage_rows
        ),
        "fixed_geometry_settings":{
            "radius_px":radius_px,
            "ambiguity_margin_px":ambiguity_margin_px,
            "min_run_support_px":min_run_support_px,
            "max_reference_gap_px":max_reference_gap_px,
        },
        "no_true_physical_trajectory_identity_established":True,
        "no_complete_visible_track_recall_on_unreviewed_photos":True,
        "no_physical_precision_f1_false_positive_rate":True,
        "no_independent_event_level_confidence_interval":True,
        "missing_S92_S96_pack_means_NOT_EVALUATED_not_zero_recall":True,
        "held_out_H50Z_films_not_tested_here":True,
    }
    export_s98_private_visual_report(
        pack_dir=pack_dir,pack=pack,human=expert,
        observed_by_name=source_for_preview,
        output_dir=private_out_dir,
        report=summary,
        max_visual_case_count=max_visual_case_count,
    )
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack-dir",type=Path,required=True)
    p.add_argument("--reference-json",type=Path,required=True)
    p.add_argument("--candidate",action="append",required=True,
                   help="NAME=/PRIVATE/S86_DIR; repeat for every source-model version")
    p.add_argument("--s92",action="append",default=[],
                   help="NAME=/PRIVATE/S92_DIR corresponding to one source candidate")
    p.add_argument("--s96",action="append",default=[],
                   help="NAME=/PRIVATE/S96_DIR corresponding to one source candidate")
    p.add_argument("--radius-px",type=float,default=2.)
    p.add_argument("--ambiguity-margin-px",type=float,default=.45)
    p.add_argument("--min-support-px",type=float,default=3.)
    p.add_argument("--max-reference-gap-px",type=float,default=80.)
    p.add_argument("--max-visual-cases",type=int,default=16)
    p.add_argument("--private-out-dir",type=Path,required=True)
    args=p.parse_args(argv)
    candidates=[_candidate_spec(x) for x in args.candidate]
    valid={n for n,_ in candidates}
    r=run_audit(
        pack_dir=args.pack_dir,
        human_reference_json=args.reference_json,
        candidates=candidates,
        s92_packs=_specs(args.s92,valid,"--s92"),
        s96_packs=_specs(args.s96,valid,"--s96"),
        private_out_dir=args.private_out_dir,
        radius_px=args.radius_px,
        ambiguity_margin_px=args.ambiguity_margin_px,
        min_run_support_px=args.min_support_px,
        max_reference_gap_px=args.max_reference_gap_px,
        max_visual_case_count=args.max_visual_cases,
    )
    print(json.dumps({
        "status":r["status"],
        "human_positive_strokes":r["human_clear_positive_stroke_count"],
        "candidate_results":[{
            "model":item["model_name"],
            "source_centerline_positive_coverage":item[
                "observed_source_centerline_positive_coverage"
            ],
            "independent_piece_adjacencies":item[
                "reference_first_source_piece_boundaries"
            ]["assessable_unique_detected_piece_adjacencies"],
            "selected_proposal_union_conditional_coverage":(
                item["reference_first_model_link_recovery"][
                    "union_selected_conditional_coverage"
                ] if item["reference_first_model_link_recovery"] is not None
                else "NOT_EVALUATED"
            ),
        } for item in r["candidates"]],
        "private_report":str(args.private_out_dir.resolve()/"report.json"),
        "private_original_vs_hypotheses":str(args.private_out_dir.resolve()/"index.html"),
    },ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
