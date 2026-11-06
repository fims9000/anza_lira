"""S96 private source-preserving junction candidate/inspection workflow.

S86/S87 supplied observed centerlines, S95 supplies machine morphology context.
S96 proposes two independent continuation alternatives at a four-arm center
ONLY when already split observed source arms are unambiguous. Y branches,
bundles, self-loops, mixed/crowded endpoints abstain. All source components,
both matched and unmatched, remain available.

Human judgment uses the original BMP crop in a separately blinded viewer;
without BMP only a machine geometry inventory is possible.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json

import numpy as np

from .bitmap_tiles import sha256_file
from .s87_seam_continuity import _read_source
from .s92_curve_review import collapse_observed_components,_write_private,_json,_ndjson
from .s96_junction_graph import (
    JunctionConfig,propose_four_arm_continuations,grouped_2d_instance_hypotheses,SCHEMA
)
from .s96_junction_human_review import prepare_blind_junction_review


def run_s96(s86_dir:Path,s95_dir:Path,out_dir:Path,*,
            bmp_path:Path|None=None,
            cfg:JunctionConfig=JunctionConfig(),
            preview_limit:int=32,crop_size:int=320):
    s86_dir=Path(s86_dir).resolve()
    s95_dir=Path(s95_dir).resolve()
    out_dir=Path(out_dir).resolve()
    repo=Path(__file__).resolve().parents[1]
    if (out_dir.exists() or out_dir==repo or repo in out_dir.parents
        or out_dir in s86_dir.parents or out_dir in s95_dir.parents
        or s86_dir in out_dir.parents or s95_dir in out_dir.parents):
        raise ValueError("S96 needs a fresh PRIVATE directory outside source and Git")
    cfg.validate()
    manifest,records,manifest_sha=_read_source(s86_dir)
    s95_path=s95_dir/"report.json"
    s95=json.loads(s95_path.read_text(encoding="utf-8"))
    if (s95["source_s86_manifest_sha256"]!=manifest_sha
        or s95["source_bmp_sha256"]!=manifest["source_bmp_sha256"]
        or s95["image_shape_yx"]!=manifest["image_shape_yx"]
        or s95["status"]!="ALL_S86_EXTRACTED_FRAGMENTS_TYPED_MACHINE_REVIEW_ONLY"):
        raise ValueError("S96 requires original and complete matching S86/S95")
    paths,stats=collapse_observed_components(records)
    row_path=s95_dir/"all_source_fragment_types.ndjson"
    categories=[json.loads(line) for line in row_path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    if (len(categories)!=len(paths)
        or {x["component_id"] for x in categories}
        !={x["component_id"] for x in paths}):
        raise ValueError("S95 complete-source component class inventory mismatch")
    source_types={x["component_id"]:x["primary_review_category"] for x in categories}
    junctions,edges,overview=propose_four_arm_continuations(paths,cfg)
    chains,unpaired=grouped_2d_instance_hypotheses(paths,edges)
    if overview["source_components_preserved"]!=len(paths):
        raise RuntimeError("S96 lost source centerline components")
    machine_counter=dict(Counter(j["junction_status"] for j in junctions))
    report={
        "schema":SCHEMA,
        "status":overview["status"],
        "source_bmp_name":manifest["source_bmp_name"],
        "source_bmp_sha256":manifest["source_bmp_sha256"],
        "image_shape_yx":manifest["image_shape_yx"],
        "source_s86_manifest_sha256":manifest_sha,
        "source_s95_report_sha256":sha256_file(s95_path),
        "source_s95_types_sha256":sha256_file(row_path),
        "algorithm_parameters":{
            "max_endpoint_center_radius_px":cfg.max_endpoint_center_radius_px,
            "max_cluster_diameter_px":cfg.max_cluster_diameter_px,
            "max_crossing_deflection_deg":cfg.max_crossing_deflection_deg,
            "min_rival_score_margin":cfg.min_rival_score_margin,
            "max_endpoint_count":cfg.max_endpoint_count,
            "max_neighbor_pairs":cfg.max_neighbor_pairs,
        },
        "source_s86_fragment_count":manifest["candidate_fragments"],
        "source_s86_skipped_tiles":manifest["tiles_with_fragment_extraction_skipped"],
        "source_observed_centerline_components":len(paths),
        "source_classes":dict(Counter(source_types.values())),
        "per_junction_machine_types":machine_counter,
        "geometry":overview,
        "hypothetical_visual_chains":len(chains),
        "unassociated_original_source_components":len(unpaired["unassociated_source_components"]),
        "source_source_line_pixels_modified":False,
        "all_source_components_available":True,
        "physical_auto_joins_authorized":False,
        "true_particles_cannot_be_inferred_from_2D_alone":True,
        "critical_limit":"Only already-separated S86/S87 endpoint arms. Fused single-component crossings require verified pixel-junction splitting first.",
    }
    out_dir.mkdir(mode=0o700,parents=True)
    out_dir.chmod(0o700)
    for name,data in (
        ("junction_hypotheses.ndjson",junctions),
        ("visual_continuation_pairs_unknown.ndjson",edges),
        ("visual_chain_hypotheses_unknown.ndjson",chains),
        ("original_observed_component_inventory.ndjson",[
            {
                "component_id":p["component_id"],
                "source_fragment_ids":p["source_fragment_ids"],
                "machine_shape_type":source_types[p["component_id"]],
                "source_centerline_unchanged":True,
                "physical_track_identity":"UNKNOWN"
            } for p in paths
        ]),
    ):
        _write_private(out_dir/name,_ndjson(data))
    if sha256_file(s95_path)!=report["source_s95_report_sha256"]:
        raise ValueError("S95 source changed during S96 geometry analysis")
    if sha256_file(row_path)!=report["source_s95_types_sha256"]:
        raise ValueError("S95 source classes changed during S96 geometry analysis")
    _write_private(out_dir/"report.json",_json(report))
    review=prepare_blind_junction_review(
        junctions=junctions,report=report,
        source_sha=report["source_bmp_sha256"],
        source_report_sha=sha256_file(out_dir/"report.json"),
        bmp_path=bmp_path,out_dir=out_dir/"review",
        max_cases=preview_limit,crop_size=crop_size,
    )
    report_result={**report,
                   "private_junction_review_pack":str(out_dir/"review"/"review_pack.json"),
                   "private_blinded_review_page":str(out_dir/"review"/"index.html"),
                   "review_image_available":review["source_image_available"],
                   "blinded_review_cases":review["sampled_cases"]}
    return report_result
