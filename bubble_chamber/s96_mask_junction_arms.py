"""S96 optional read-only pixel junction arms from frozen S86 mask PNG tiles.

S86 canonical graph extractor may create one fused crossover component, or
many tiny diagonal stubs at junction pixels. This experimental S96 path uses
S68's Moore 8-neighbor arm count and branch-core excision to build an
*alternative* arm representation. No S86 mask, source segment, or S95 class
is removed/replaced. Mask-derived arms are not called expert GT and any
centerline through a branch core is a new UNVERIFIED visual hypothesis.

All operations use one verified S86 core mask tile at a time; the full
200MB original BMP need never be opened for geometry.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path
import math

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .bitmap_tiles import sha256_file
from .pilot import extract_skeleton_segments
from .s68_branch_preserving_instances import split_skeleton_at_junctions
from .s96_junction_graph import (
    JunctionConfig,propose_four_arm_continuations,grouped_2d_instance_hypotheses
)

SCHEMA="S96_SOURCE_MASK_MOORE_JUNCTION_ARMS_EXPERIMENTAL_V1"


def analyze_s86_mask_junctions(
    s86_dir:Path,records:list[dict],*,
    cfg:JunctionConfig=JunctionConfig(),
    min_arm_length_px:float=8.,
    max_skeleton_pixels_per_tile:int=30000,
    max_junctions_per_tile:int=128,
):
    cfg.validate()
    if (not math.isfinite(min_arm_length_px)
        or not 4<=min_arm_length_px<=100
        or not 100<=max_skeleton_pixels_per_tile<=150000
        or not 1<=max_junctions_per_tile<=1024):
        raise ValueError("Invalid bounded pixel-junction sensitivity")
    s86_dir=Path(s86_dir).resolve()
    arms=[]
    junctions=[]
    pairs=[]
    chains=[]
    tile_status=Counter()
    skipped=[]
    for record in records:
        x0,y0,x1,y1=map(int,record["core_xyxy"])
        h,w=y1-y0,x1-x0
        mask_path=s86_dir/record["mask_file"]
        if not mask_path.is_file() or sha256_file(mask_path)!=record["mask_sha256"]:
            raise ValueError("Changed/missing S86 mask tile: "+record["tile_id"])
        binary=cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE)
        if (binary is None or binary.shape!=(h,w)
            or not np.isin(binary,[0,255]).all()):
            raise ValueError("Incorrect S86 source tile mask: "+record["tile_id"])
        if record.get("fragment_extraction_skip_reason") is not None:
            tile_status["S86_ORIGINAL_FRAGMENT_EXTRACTION_SKIPPED"]+=1
        foreground=binary>0
        skel=skeletonize(foreground)
        count=int(np.count_nonzero(skel))
        if count>max_skeleton_pixels_per_tile:
            tile_status["MASK_SKELETON_BUDGET_ABSTAIN"]+=1
            skipped.append(record["tile_id"])
            continue
        cut,centers=split_skeleton_at_junctions(skel,radius=1)
        if len(centers)>max_junctions_per_tile:
            tile_status["DENSE_MOORE_JUNCTIONS_ABSTAIN"]+=1
            skipped.append(record["tile_id"])
            continue
        # Near tile boundaries, the truncated stencil can turn a continuous
        # original trace into an apparent branch. Refuse branch assessments.
        inside=[
            p for p in centers
            if 8<=p[0]<w-8 and 8<=p[1]<h-8
        ]
        boundary_count=len(centers)-len(inside)
        tile_status["UNCERTAIN_NEAR_TILE_EDGE_JUNCTION"]+=boundary_count
        if not inside:
            tile_status["NO_SAFE_INTERIOR_PIXEL_JUNCTION"]+=1
            continue
        paths=extract_skeleton_segments(cut,min_arm_length_px)
        if len(paths)>4096:
            tile_status["TOO_MANY_SPLIT_SOURCE_ARMS_ABSTAIN"]+=1
            skipped.append(record["tile_id"])
            continue
        near_points=np.asarray(inside,float)
        local_arms=[]
        for idx,path_yx in enumerate(paths):
            if len(path_yx)<4:continue
            xy=path_yx[:,::-1].astype(float)
            # Only use fragments whose observed endpoints actually approach
            # a detected Moore-junction in this *same* verified tile.
            d=np.linalg.norm(
                near_points[None,:,:]-xy[[0,-1],None,:],axis=2
            )
            if not (d<=cfg.max_endpoint_center_radius_px).any():
                continue
            source_xy=xy+np.array([x0,y0])
            cid=f's96_mask_{record["tile_id"]}_a{idx:04d}'
            new={
                "component_id":cid,
                "points_xy":np.round(source_xy,4).tolist(),
                "derived_from_mask_sha256":record["mask_sha256"],
                "source_mask_tile_id":record["tile_id"],
                "source_core_xyxy":record["core_xyxy"],
                "source_core_skeleton_observed_pixels_only":True,
                "original_S86_fragments_replaced":False,
                "physical_track_identity":"UNKNOWN",
            }
            arms.append(new)
            local_arms.append(new)
        local_j,local_e,_=propose_four_arm_continuations(local_arms,cfg)
        in_tile_chains,_=grouped_2d_instance_hypotheses(local_arms,local_e)
        for j in local_j:
            j["source_mask_tile_id"]=record["tile_id"]
            j["source_center_pixel_from_original_mask"]=True
            # Centerline pairing is still UNVERIFIED even though the
            # blob itself is observed in S86 foreground.
            j["not_verified_same_particle_crossing"]=True
        for e in local_e:
            e["source_mask_tile_id"]=record["tile_id"]
            e["physical_auto_join"]=False
            e["inferred_centerline_identity_only"]=True
        for c in in_tile_chains:
            c["source_mask_tile_id"]=record["tile_id"]
            c["original_S86_component_replaced"]=False
        junctions.extend(local_j)
        pairs.extend(local_e)
        chains.extend(in_tile_chains)
        tile_status["MASK_TILE_WITH_INTERIOR_MOORE_JUNCTION"]+=1

    summary={
        "schema":SCHEMA,
        "status":"ALTERNATIVE_PIXEL_BRANCH_HYPOTHESES_PRIVATE_ONLY",
        "source_S86_tiles_examined":len(records),
        "source_S86_mask_tiles_changed":0,
        "source_S86_fragment_count_modified":0,
        "source_skeleton_arm_hypotheses":len(arms),
        "source_mask_junction_candidates":len(junctions),
        "four_arm_crossing_pair_hypotheses":len(pairs),
        "hypothetical_2d_visual_chains":len(chains),
        "source_mask_junction_tile_status":dict(tile_status),
        "tiles_with_dense_or_budget_abstention":skipped,
        "source_film_pixel_mutated":False,
        "all_original_S86_fragment_IDs_preserved":True,
        "physical_track_identity_verified":False,
        "physical_auto_joins_authorized":False,
        "masked_branch_cores_not_claimed_as_real_particle_connections":True,
    }
    return arms,junctions,pairs,chains,summary
