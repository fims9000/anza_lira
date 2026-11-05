"""S92 image-geometry evidence: tiny strokes, hairpins, spirals, broken traces.

All classifications are review priorities, NOT physics truth. The original S86
segmentation, S87 seams, and all observed pixels are immutable. Inferred bridges
are separately labeled as unobserved. No particle identity, momentum, 3D or
automatic physical joins can be concluded from 2D image geometry.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
import math

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class TraceReviewConfig:
    resample_step_px: float = 1.5
    short_stroke_max_px: float = 24.0
    hairpin_max_px: float = 72.0
    hairpin_max_chord_ratio: float = 0.56
    hairpin_min_turn_deg: float = 115.0
    sharp_tick_min_turn_deg: float = 55.0
    spiral_min_px: float = 72.0
    spiral_min_turn_deg: float = 255.0
    spiral_turn_coherence: float = 0.72
    min_gap_px: float = 2.1
    max_gap_px: float = 26.0
    min_heading_cos: float = -0.12
    max_gap_arc_ratio: float = 1.75
    max_gap_turn_deg: float = 165.0
    third_fragment_clearance_px: float = 2.4
    rival_margin: float = 0.075
    max_near_pairs: int = 40000
    min_fragment_for_gap_px: float = 9.0

    def validate(self):
        positive = ("resample_step_px","short_stroke_max_px","hairpin_max_px",
                    "hairpin_max_chord_ratio","hairpin_min_turn_deg",
                    "sharp_tick_min_turn_deg","spiral_min_px","spiral_min_turn_deg",
                    "spiral_turn_coherence","min_gap_px","max_gap_px",
                    "max_gap_arc_ratio","max_gap_turn_deg","third_fragment_clearance_px",
                    "rival_margin","min_fragment_for_gap_px")
        if any(not math.isfinite(getattr(self,key)) or getattr(self,key)<=0
               for key in positive):
            raise ValueError("S92 requires positive finite geometry parameters")
        if (self.min_gap_px>=self.max_gap_px or
            not -1<=self.min_heading_cos<=1 or
            not 0<self.spiral_turn_coherence<=1 or
            not 0<self.hairpin_max_chord_ratio<=1 or
            self.max_near_pairs<10 or
            self.spiral_min_turn_deg<=self.hairpin_min_turn_deg or
            self.max_gap_arc_ratio<=1):
            raise ValueError("Invalid S92 shape or gap review contract")


def _points(points):
    array=np.asarray(points,dtype=np.float64)
    if (array.ndim!=2 or array.shape[1]!=2 or len(array)<4
            or not np.isfinite(array).all()):
        raise ValueError("Expected >=4 finite 2D observed centerline points")
    if np.any(np.linalg.norm(np.diff(array,axis=0),axis=1)<1e-8):
        raise ValueError("Repeated consecutive observed points")
    return array


def resample(points,step:float=1.5):
    p=_points(points)
    if not math.isfinite(step) or step<=0:
        raise ValueError("Invalid resampling step")
    dist=np.linalg.norm(np.diff(p,axis=0),axis=1)
    arc=np.r_[0,np.cumsum(dist)]
    n=max(5,int(np.ceil(arc[-1]/step))+1)
    positions=np.linspace(0,arc[-1],n)
    new=np.column_stack([np.interp(positions,arc,p[:,axis]) for axis in (0,1)])
    return new,float(arc[-1])


def describe_trace(points,cfg:TraceReviewConfig=TraceReviewConfig()):
    cfg.validate()
    p,length=resample(points,cfg.resample_step_px)
    smooth=gaussian_filter1d(p,sigma=0.85,axis=0,mode="nearest")
    v=np.diff(smooth,axis=0)
    angle=np.unwrap(np.arctan2(v[:,1],v[:,0]))
    turn=np.diff(angle)
    degrees=np.degrees(turn)
    total_signed=float(np.degrees(angle[-1]-angle[0]))
    total_abs=float(np.sum(np.abs(degrees)))
    coherence=abs(total_signed)/max(total_abs,1e-9)
    # A 4-step turning window catches a V/checkmark without labeling a
    # gradual long arc as a sharp angular artifact.
    window=min(4,len(degrees))
    local_turn=np.convolve(degrees,np.ones(window),mode="valid")
    sharp=float(np.max(np.abs(local_turn))) if len(local_turn) else 0.
    chord=float(np.linalg.norm(p[-1]-p[0]))
    chord_ratio=min(1.,chord/max(length,1e-9))
    spiral=(length>=cfg.spiral_min_px and
            abs(total_signed)>=cfg.spiral_min_turn_deg and
            coherence>=cfg.spiral_turn_coherence)
    hairpin=(not spiral and length<=cfg.hairpin_max_px and
             abs(total_signed)>=cfg.hairpin_min_turn_deg and
             chord_ratio<=cfg.hairpin_max_chord_ratio and
             coherence>=.55)
    short=length<=cfg.short_stroke_max_px
    tick=(short and sharp>=cfg.sharp_tick_min_turn_deg)
    flags=[]
    if spiral:
        flags.append("COILING_TRACE_KEEP_FOR_REVIEW")
    if hairpin:
        flags.append("COMPACT_HAIRPIN_ARTIFACT_POSSIBLE")
    if tick:
        flags.append("SHORT_ANGULAR_GLYPH_POSSIBLE")
    if short and not tick:
        flags.append("SHORT_STROKE_NOT_VERIFIED")
    if spiral:
        category="COILING_TRACE_REVIEW"
    elif hairpin or short:
        category="LOW_PRIORITY_POSSIBLE_ARTIFACT"
    else:
        category="EXTENDED_IMAGE_TRACE_CANDIDATE"
    return {
        "observed_length_px":round(length,3),
        "chord_ratio":round(chord_ratio,5),
        "total_signed_turn_deg":round(total_signed,3),
        "absolute_turn_deg":round(total_abs,3),
        "turn_direction_coherence":round(float(min(coherence,1.)),5),
        "max_angular_change_over_local_window_deg":round(sharp,3),
        "shape_review_category":category,
        "review_flags":flags,
        "not_a_particle_classification":True,
        "physical_identity":"UNKNOWN",
    }


def _outward(points,port):
    p,length=resample(points,1.)
    back=min(8.,length*.40)
    if port==0:
        tip=p[0]
        inner=p[int(np.ceil(back))]
    else:
        tip=p[-1]
        inner=p[max(0,len(p)-1-int(np.ceil(back)))]
    vector=tip-inner
    norm=float(np.linalg.norm(vector))
    return vector/norm if norm>1e-7 else None


def _hermite(a,b,ua,ub,steps=17):
    # ua: tangent leaving A. ub: tangent leaving B into the other direction.
    chord=float(np.linalg.norm(b-a))
    t=np.linspace(0.,1.,steps)[:,None]
    m0=ua*chord*.80
    m1=-ub*chord*.80
    return ((2*t**3-3*t**2+1)*a+
            (t**3-2*t**2+t)*m0+
            (-2*t**3+3*t**2)*b+
            (t**3-t**2)*m1)


def _link_geometry(a,aport,b,bport,cfg):
    pa=a[0] if aport==0 else a[-1]
    pb=b[0] if bport==0 else b[-1]
    delta=pb-pa
    gap=float(np.linalg.norm(delta))
    if not cfg.min_gap_px<=gap<=cfg.max_gap_px:
        return None
    ua=_outward(a,aport)
    ub=_outward(b,bport)
    if ua is None or ub is None:
        return None
    displacement=delta/gap
    forward=float(np.dot(ua,displacement))
    backward=float(np.dot(ub,-displacement))
    if min(forward,backward)<cfg.min_heading_cos:
        return None
    curve=_hermite(pa,pb,ua,ub)
    step=np.diff(curve,axis=0)
    lengths=np.linalg.norm(step,axis=1)
    if np.any(lengths<1e-6):
        return None
    arc_ratio=float(sum(lengths)/gap)
    if arc_ratio>cfg.max_gap_arc_ratio:
        return None
    angle=np.unwrap(np.arctan2(step[:,1],step[:,0]))
    turn_deg=float(abs(np.degrees(angle[-1]-angle[0])))
    if turn_deg>cfg.max_gap_turn_deg:
        return None
    geometry=(
        .35*(min(1.,max(0.,forward+0.15)/1.15)+
             min(1.,max(0.,backward+0.15)/1.15))*.5+
        .25*(1.-(gap-cfg.min_gap_px)/(cfg.max_gap_px-cfg.min_gap_px))+
        .25*max(0.,1.-(arc_ratio-1.)/(cfg.max_gap_arc_ratio-1.))+
        .15*max(0.,1.-turn_deg/cfg.max_gap_turn_deg)
    )
    return {"gap_px":round(gap,4),"bridge_arc_ratio":round(arc_ratio,4),
            "bridge_turn_deg":round(turn_deg,3),
            "geometry_score":round(float(geometry),6),
            "inferred_bridge_points_xy":np.round(curve,3).tolist()}


def propose_gap_links(components:list[dict],
                      cfg:TraceReviewConfig=TraceReviewConfig()):
    """Only geometric, reversible S92 advisory links. No hidden GT/physics.

    components: [{'component_id':str,'points_xy':array},...]
    Returns all geometrically eligible candidates, an ambiguity-aware matching
    and stitched *hypotheses*. Never modifies observed source coordinates.
    """
    cfg.validate()
    ids=[str(c["component_id"]) for c in components]
    if len(ids)!=len(set(ids)):
        raise ValueError("Duplicate S92 component identity")
    pts=[_points(c["points_xy"]) for c in components]
    lengths=[float(sum(np.linalg.norm(np.diff(p,axis=0),axis=1))) for p in pts]
    n=len(pts)
    if n<2:
        return [],[],{"endpoints_considered":2*n,"gap_candidates":0,
                       "gap_review_links":0,"ambiguous_ports":0}
    endpoint_coordinates=np.concatenate([p[[0,-1]] for p in pts],axis=0)
    candidate_idx=cKDTree(endpoint_coordinates).query_pairs(cfg.max_gap_px)
    if len(candidate_idx)>cfg.max_near_pairs:
        # Density is a safety hazard, not permission to drop hard rivals
        # and silently call the remaining proposal unique.
        return [],[],{"endpoints_considered":2*n,"gap_candidates":len(candidate_idx),
                       "gap_review_links":0,"ambiguous_ports":2*n,
                       "abstain_reason":"DENSE_ENDPOINT_NEIGHBORHOOD"}
    proposals=[]
    for u,v in sorted(candidate_idx):
        i,pa=divmod(u,2)
        j,pb=divmod(v,2)
        if i==j or min(lengths[i],lengths[j])<cfg.min_fragment_for_gap_px:
            continue
        result=_link_geometry(pts[i],pa,pts[j],pb,cfg)
        if result is None:
            continue
        result.update({
            "proposal_id":"s92_"+sha256(
                f"{ids[i]}:{pa}|{ids[j]}:{pb}".encode()).hexdigest()[:20],
            "component_a":ids[i],"port_a":pa,"component_b":ids[j],"port_b":pb,
            "status":"UNKNOWN",
            "inferred_only":True,"physical_auto_join":False,
            "review_decision":"UNDECIDED_CANDIDATE",
        })
        proposals.append(result)
    byport=defaultdict(list)
    for p in proposals:
        byport[(p["component_a"],p["port_a"])].append(p["geometry_score"])
        byport[(p["component_b"],p["port_b"])].append(p["geometry_score"])
    # Any rival within a frozen margin forces abstention. No discretionary
    # choice after looking at synthetic or physical GT.
    for p in proposals:
        ports=[(p["component_a"],p["port_a"]),(p["component_b"],p["port_b"])]
        if any(sum(abs(s-p["geometry_score"])<=cfg.rival_margin+1e-10
                   or s>p["geometry_score"] for s in byport[port])>1
               for port in ports):
            p["review_decision"]="AMBIGUOUS_PARTNER_ABSTAIN"
    ranked=sorted(proposals,key=lambda p:(-p["geometry_score"],p["proposal_id"]))
    used=set()
    parent=list(range(n))
    degree=[0]*n
    look={name:k for k,name in enumerate(ids)}
    def root(index):
        while parent[index]!=index:
            index=parent[index]
        return index
    selected=[]
    for p in ranked:
        if p["review_decision"]!="UNDECIDED_CANDIDATE":
            continue
        i,j=look[p["component_a"]],look[p["component_b"]]
        a,b=(p["component_a"],p["port_a"]),(p["component_b"],p["port_b"])
        if a in used or b in used:
            p["review_decision"]="PORT_OCCUPIED_ABSTAIN"
        elif root(i)==root(j):
            p["review_decision"]="CYCLE_ABSTAIN"
        elif degree[i]>=2 or degree[j]>=2:
            p["review_decision"]="DEGREE_ABSTAIN"
        else:
            p["review_decision"]="PRIORITY_GAP_REVIEW_NOT_PHYSICAL_ACCEPT"
            selected.append(p)
            used.update((a,b))
            degree[i]+=1;degree[j]+=1
            parent[root(j)]=root(i)
    return proposals,selected,{"endpoints_considered":2*n,
              "gap_candidates":len(proposals),"gap_review_links":len(selected),
              "ambiguous_ports":sum(len(v)>1 for v in byport.values()),
              "physical_auto_joins_authorized":False}


def construct_chain_hypotheses(components,selected, cfg=TraceReviewConfig()):
    """Explicitly separate original observed sections and inferred gap pixels."""
    lookup={str(c["component_id"]):_points(c["points_xy"]) for c in components}
    graph=defaultdict(dict)
    for p in selected:
        if p["review_decision"]!="PRIORITY_GAP_REVIEW_NOT_PHYSICAL_ACCEPT":
            raise ValueError("Only unique advisory links may form preview chains")
        a,ap=p["component_a"],p["port_a"]
        b,bp=p["component_b"],p["port_b"]
        if a not in lookup or b not in lookup or ap not in (0,1) or bp not in (0,1):
            raise ValueError("Unknown component/port in S92 proposed chain")
        if ap in graph[a] or bp in graph[b]:
            raise ValueError("Source endpoint assigned to two proposed links")
        graph[a][ap]=(b,bp,p)
        graph[b][bp]=(a,ap,p)
    chains=[]
    visited=set()
    for initial in sorted(graph):
        if initial in visited:
            continue
        members=set()
        pending=[initial]
        while pending:
            current=pending.pop()
            if current in members:continue
            members.add(current)
            pending.extend(x[0] for x in graph[current].values())
        visited|=members
        terminals=sorted(x for x in members if len(graph[x])==1)
        if len(terminals)!=2:
            raise ValueError("Cannot present a cycle/branch as a single observed trace")
        cur=terminals[0]
        input_port=1-next(iter(graph[cur]))
        ordered=[];observed=[];gaps=[];all_points=[]
        while True:
            if cur in ordered:
                raise ValueError("Unexpected repeated component in candidate chain")
            ordered.append(cur)
            p=lookup[cur] if input_port==0 else lookup[cur][::-1]
            observed.append(np.round(p,3).tolist())
            all_points.extend(p.tolist())
            next_port=1-input_port
            if next_port not in graph[cur]:
                break
            nxt,arrives,edge=graph[cur][next_port]
            bridge=np.asarray(edge["inferred_bridge_points_xy"],float)
            if not np.allclose(bridge[0],p[-1],atol=0.005):
                bridge=bridge[::-1]
            if not np.allclose(bridge[0],p[-1],atol=0.005):
                raise ValueError("Reversed or misplaced geometric bridge")
            gaps.append({"proposal_id":edge["proposal_id"],
                         "inferred_only_points_xy":edge["inferred_bridge_points_xy"],
                         "from_component":cur,"to_component":nxt})
            all_points.extend(bridge[1:].tolist())
            cur=nxt
            input_port=arrives
        if set(ordered)!=members:
            raise ValueError("Candidate chain omitted a component")
        classification=describe_trace(np.asarray(all_points,float),cfg)
        chains.append({
            "chain_id":"s92_chain_"+sha256("|".join(sorted(members)).encode()).hexdigest()[:20],
            "ordered_component_ids":ordered,
            "observed_sections_xy":observed,
            "inferred_gaps":gaps,
            "shape_if_all_gaps_real":classification,
            "status":"UNKNOWN","physical_auto_join":False,
            "contains_inferred_pixels":True,
            "not_a_verified_particle_track":True,
        })
    return chains
