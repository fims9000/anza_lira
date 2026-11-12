#!/usr/bin/env python3
"""V19 CRACKS real expert-mask audit: availability of close competing *annotations*.

Unbranched connected blue expert tracks define purely annotation-level identity.
No geological fault identity, real negative link, learning or full E2E is claimed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from scripts.cracks_real_oracle_gap_audit import COLORS, paths_from_mask, sections
from scripts.cracks_real_neighbor_path_v17 import rgb

THRESHOLDS = (6, 12, 20, 32, 48, 64)
MIN_VISIBLE = 15
MIN_PATH = 36
GAP_SAMPLES = 12

def closest_interior_pair(path_a, path_b):
    """Pure geometric search; two *different annotated* connected skeletons."""
    if len(path_a) < MIN_PATH or len(path_b) < MIN_PATH:
        return None
    a=np.asarray(path_a[MIN_VISIBLE:-MIN_VISIBLE],dtype=np.float64)
    b=np.asarray(path_b[MIN_VISIBLE:-MIN_VISIBLE],dtype=np.float64)
    if not len(a) or not len(b):
        return None
    dd, indices=cKDTree(b).query(a,k=1)
    pos=int(np.argmin(dd))
    j=int(indices[pos])
    return float(dd[pos]),pos+MIN_VISIBLE,j+MIN_VISIBLE

def audit(annotations:Path,out:Path,cases:Path,max_sections:int=400):
    rows=[]
    counters=Counter()
    sections_by_threshold=defaultdict(set)
    with zipfile.ZipFile(annotations) as za:
        expert=sections(za,expert=True)
        ids=sorted(expert)[:max_sections]
        for sid in ids:
            label=rgb(za,expert[sid])
            blue=np.all(label==np.asarray(COLORS["blue"],np.uint8),axis=2)
            green=np.all(label==np.asarray(COLORS["green"],np.uint8),axis=2)
            clean=blue & ~ndi.binary_dilation(green,iterations=5)
            paths=sorted(
                (p for p in paths_from_mask(clean) if len(p)>=MIN_PATH),
                key=lambda p:(-len(p),p[0][0],p[0][1]))
            counters["expert_sections"]+=1
            counters["eligible_paths"]+=len(paths)
            for i in range(len(paths)):
                for j in range(i+1,len(paths)):
                    closest=closest_interior_pair(paths[i],paths[j])
                    if closest is None:
                        continue
                    dist,ia,ib=closest
                    can_pair=(dist<=20 and
                              len(paths[i])>=GAP_SAMPLES+2*MIN_VISIBLE and
                              len(paths[j])>=GAP_SAMPLES+2*MIN_VISIBLE)
                    row={
                        "section":sid,"path_id_a":i,"path_id_b":j,
                        "len_a":len(paths[i]),"len_b":len(paths[j]),
                        "closest_interior_distance_px":dist,
                        "interior_index_a":ia,"interior_index_b":ib,
                        "position_a_y":paths[i][ia][0],
                        "position_a_x":paths[i][ia][1],
                        "position_b_y":paths[j][ib][0],
                        "position_b_x":paths[j][ib][1],
                        "potential_ambiguous_annotation_case":int(can_pair),
                    }
                    rows.append(row)
                    for th in THRESHOLDS:
                        if dist<=th:
                            counters["pairs_le_"+str(th)]+=1
                            sections_by_threshold[th].add(sid)
                    counters["total_path_pairs"]+=1
                    if can_pair:
                        counters["potential_ambiguities_le20"]+=1
                        sections_by_threshold["ambiguous"].add(sid)
    cases.parent.mkdir(parents=True,exist_ok=True)
    if rows:
        with cases.open("w",newline="",encoding="utf8") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        cases.write_text("",encoding="utf8")
    fractions={str(t):{
        "pairs":int(counters["pairs_le_"+str(t)]),
        "sections":len(sections_by_threshold[t])} for t in THRESHOLDS}
    status=(
        "ENOUGH_ANNOTATED_COMPETITORS_FOR_SYNTHETIC_CONTROL"
        if (counters["potential_ambiguities_le20"]>=10
            and len(sections_by_threshold["ambiguous"])>=5)
        else "INSUFFICIENT_ANNOTATED_COMPETITORS"
    )
    summary={
        "protocol":"CRACKS_V19_COMPETING_CONNECTED_ANNOTATIONS",
        "status":status,
        "source":"CRACKS v2 DOI 10.5281/zenodo.13926822",
        "denominator_expert_sections":int(counters["expert_sections"]),
        "denominator_eligible_paths":int(counters["eligible_paths"]),
        "denominator_distinct_pairs":int(counters["total_path_pairs"]),
        "thresholds_px":list(THRESHOLDS),
        "threshold_counts":fractions,
        "potential_ambiguities_le20":int(counters["potential_ambiguities_le20"]),
        "potential_ambiguous_sections":len(sections_by_threshold["ambiguous"]),
        "min_interior_distance":min(
            (r["closest_interior_distance_px"] for r in rows),default=None),
        "median_interior_distance":float(np.median(
            [r["closest_interior_distance_px"] for r in rows])) if rows else None,
        "fixed_visible_support_samples":MIN_VISIBLE,
        "fixed_gap_samples":GAP_SAMPLES,
        "measured_identity":"annotated connected expert skeleton only",
        "valid_geological_negative_pairs":False,
        "model_learning_performed":False,
        "full_ANZA_LIRA_e2e":False,
        "note":"All threshold counts are descriptive on expert masks; V19 was designed after inspecting V17/V18 real data.",
        "cases_sha256":hashlib.sha256(cases.read_bytes()).hexdigest(),
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--annotations",required=True)
    p.add_argument("--out",default="results/cracks_real_controlled_v1/competitor_v19.json")
    p.add_argument("--cases",default="results/cracks_real_controlled_v1/competitor_v19_cases.csv")
    p.add_argument("--max-sections",type=int,default=400)
    x=p.parse_args()
    audit(Path(x.annotations),Path(x.out),Path(x.cases),x.max_sections)

if __name__=="__main__":
    main()
