"""S96 branch/crossing continuity: keep all observed 2D source components.

Previously S67/S68 separated the arms of crossings but were not demonstrably
better at physical continuity. S96 adds a conservative *review-level* matching
of endpoint arms at a four-arm crossing, with explicit alternatives and reasons
for abstention. The input is S86/S87 source centerline fragments; NO operator
track identity, synthetic GT, pixel class or physical particle enters scoring.

An X-crossing can be two unrelated tracks, an interaction, or an overlap.
Selected pairs are just reversible 2-D visual hypotheses, not physical joins.
Source components are NEVER deleted, replaced by an interpolated centerline,
or reassigned a known particle identity.
"""
from __future__ import annotations

from collections import Counter,defaultdict
from dataclasses import dataclass
from hashlib import sha256
import math

import numpy as np
from scipy.spatial import cKDTree

SCHEMA="S96_REVERSIBLE_JUNCTION_BRANCH_CONTINUATION_V1"


@dataclass(frozen=True)
class JunctionConfig:
    max_endpoint_center_radius_px:float=12.
    max_cluster_diameter_px:float=18.
    max_crossing_deflection_deg:float=35.
    min_rival_score_margin:float=.12
    max_endpoint_count:int=120000
    max_neighbor_pairs:int=250000

    def validate(self):
        if any(not math.isfinite(float(v)) or v<=0 for v in (
            self.max_endpoint_center_radius_px,self.max_cluster_diameter_px,
            self.max_crossing_deflection_deg,self.min_rival_score_margin)):
            raise ValueError("Invalid nonpositive S96 junction parameter")
        if (self.max_cluster_diameter_px>2*self.max_endpoint_center_radius_px
            or self.max_crossing_deflection_deg>80
            or self.min_rival_score_margin>=1
            or not 4<=self.max_endpoint_count<=500000
            or not 2<=self.max_neighbor_pairs<=1500000):
            raise ValueError("Invalid bounded S96 junction configuration")


def _path(value):
    p=np.asarray(value,dtype=float)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<4 or not np.isfinite(p).all():
        raise ValueError("Need >=4 finite observed source centerline points")
    if np.any(np.linalg.norm(np.diff(p,axis=0),axis=1)<1e-7):
        raise ValueError("Degenerate repeated source points")
    return p


def _endpoints(components):
    endpoints=[]
    ids=set()
    for record in components:
        cid=record.get("component_id")
        if not isinstance(cid,str) or not cid or cid in ids:
            raise ValueError("Duplicate/missing S96 source component ID")
        ids.add(cid)
        path=_path(record["points_xy"])
        for port in (0,1):
            endpoint=path[0] if port==0 else path[-1]
            # Local tangent remains tied to original *observed* points.
            # Six samples give a stable direction for long/curved lines.
            inner=path[min(6,len(path)-1)] if port==0 else path[max(0,len(path)-7)]
            outward=endpoint-inner
            length=float(np.linalg.norm(outward))
            if length<1e-7:
                raise ValueError("Degenerate endpoint tangent")
            endpoints.append({
                "component_id":cid,"port":port,
                "point_xy":endpoint.tolist(),"outward_unit":(outward/length).tolist()
            })
    return endpoints


def _clusters(endpoints,cfg):
    """Distance-graph components, fail-closed on long chains or dense scenes."""
    n=len(endpoints)
    if n>cfg.max_endpoint_count:
        return [],"ENDPOINT_BUDGET_EXCEEDED"
    if not n:
        return [],None
    coords=np.array([x["point_xy"] for x in endpoints],dtype=float)
    neighbors=cKDTree(coords).query_pairs(cfg.max_endpoint_center_radius_px)
    if len(neighbors)>cfg.max_neighbor_pairs:
        return [],"DENSE_ENDPOINT_NEIGHBORHOOD"
    parent=list(range(n))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    for i,j in sorted(neighbors):
        a,b=root(i),root(j)
        if a!=b:parent[b]=a
    groups=defaultdict(list)
    for i in range(n):
        groups[root(i)].append(i)
    result=[]
    for indices in groups.values():
        if len(indices)<3:continue
        c=coords[indices]
        center=np.median(c,axis=0)
        diameter=float(np.max(np.linalg.norm(c[:,None,:]-c[None,:,:],axis=2)))
        # Never let a chain of nearby endpoints turn distant segments into one
        # invented collision center.
        status=("CLUSTER_SPREAD_ABSTAIN"
                if diameter>cfg.max_cluster_diameter_px else None)
        ports=[endpoints[i] for i in indices]
        ports.sort(key=lambda x:(x["component_id"],x["port"]))
        ident="s96_j_"+sha256(
            "|".join(f'{x["component_id"]}:{x["port"]}' for x in ports).encode()
        ).hexdigest()[:20]
        result.append({
            "junction_id":ident,
            "center_xy_source":np.round(center,4).tolist(),
            "observed_endpoint_count":len(ports),
            "observed_source_components":sorted(set(x["component_id"] for x in ports)),
            "max_endpoint_pair_distance_px":round(diameter,4),
            "ports":ports,"initial_abstention":status,
        })
    return sorted(result,key=lambda x:x["junction_id"]),None


def _compatible(a,b,center,cfg):
    pa=np.asarray(a["point_xy"],float)
    pb=np.asarray(b["point_xy"],float)
    ua=np.asarray(a["outward_unit"],float)
    ub=np.asarray(b["outward_unit"],float)
    toward=center-pa
    tother=center-pb
    da=float(np.linalg.norm(toward))
    db=float(np.linalg.norm(tother))
    if da>1.5 and float(ua@toward/da)<.05:
        return None
    if db>1.5 and float(ub@tother/db)<.05:
        return None
    # Outward tangents of two opposite arms must point towards each other,
    # so their dot product is near -1; a Y-branch alone is not sufficient.
    alignment=float(-ua@ub)
    if alignment<math.cos(math.radians(cfg.max_crossing_deflection_deg)):
        return None
    return round(alignment-.015*(da+db)/(2*cfg.max_endpoint_center_radius_px),6)


def _port_key(item):
    return f'{item["component_id"]}:{item["port"]}'


def propose_four_arm_continuations(components,cfg:JunctionConfig=JunctionConfig()):
    """Rank BOTH independent X-crossing continuations; abstain on Y/bundles.

    A crossing is proposed only when four distinct already separated source
    arms end near the same center. A single merged four-way source skeleton
    needs upstream S68 pixel-junction splitting; this function does not forge
    split coordinates from ambiguous machine polylines.
    """
    cfg.validate()
    endpoints=_endpoints(components)
    clusters,global_abstain=_clusters(endpoints,cfg)
    junctions=[]
    edges=[]
    if global_abstain:
        return [],[],{
            "schema":SCHEMA,"status":global_abstain,
            "source_components":len(components),
            "source_components_preserved":len(components),
            "total_source_endpoints":len(endpoints),
            "physical_auto_joins_authorized":False,
        }
    for cluster in clusters:
        ports=cluster["ports"]
        count=len(ports)
        if cluster["initial_abstention"] is not None:
            reason=cluster["initial_abstention"]
        elif len(cluster["observed_source_components"])!=count:
            reason="SAME_SOURCE_MULTIPORT_ABSTAIN"
        elif count<4:
            reason="THREE_ARM_BRANCH_ABSTAIN"
        elif count>4:
            reason="DENSE_BUNDLE_ABSTAIN"
        else:
            reason=None
        options=[]
        selected=[]
        center=np.asarray(cluster["center_xy_source"],float)
        if reason is None:
            for i in range(4):
                for j in range(i+1,4):
                    score=_compatible(ports[i],ports[j],center,cfg)
                    if score is not None:
                        options.append({
                            "port_a":_port_key(ports[i]),"port_b":_port_key(ports[j]),
                            "component_a":ports[i]["component_id"],
                            "component_b":ports[j]["component_id"],
                            "endpoint_port_a":ports[i]["port"],
                            "endpoint_port_b":ports[j]["port"],
                            "alignment_score":score,
                            "junction_id":cluster["junction_id"],
                            "inferred_junction_bridge_xy":[
                                ports[i]["point_xy"],cluster["center_xy_source"],
                                ports[j]["point_xy"],
                            ],
                            "inferred_only":True,
                            "status":"UNKNOWN",
                            "physical_auto_join":False,
                        })
            options.sort(key=lambda x:(-x["alignment_score"],x["port_a"],x["port_b"]))
            # Mutual best + unique margin are necessary, but not sufficient.
            # Since four arms should form 2 independent pairs, fail closed
            # unless both disjoint pairs survive.
            confident=[]
            for opt in options:
                rival=max((x["alignment_score"] for x in options
                           if x is not opt
                           and (opt["port_a"] in (x["port_a"],x["port_b"]) or
                                opt["port_b"] in (x["port_a"],x["port_b"]))),
                          default=-1.0)
                opt["nearest_rival_score"]=rival
                opt["unique_margin"]=round(opt["alignment_score"]-rival,6)
                opt["review_decision"]="DIRECTION_RIVAL_ABSTAIN"
                if opt["unique_margin"]>=cfg.min_rival_score_margin:
                    confident.append(opt)
            used=set()
            for opt in confident:
                if opt["port_a"] not in used and opt["port_b"] not in used:
                    selected.append(opt)
                    used.update((opt["port_a"],opt["port_b"]))
            if len(selected)!=2 or len(used)!=4:
                reason="INCOMPLETE_OR_AMBIGUOUS_FOUR_ARM_PAIRING"
                selected=[]
            else:
                for opt in selected:
                    opt["review_decision"]="PROVISIONAL_2D_CROSSING_CONTINUATION"
        # No source evidence for Y/bundles is silently discarded: preserve
        # all arms and identify the uncertain junction for human review.
        cluster["junction_status"]=(
            "TWO_ALTERNATIVE_VISUAL_CROSSING_CONTINUATIONS_UNVERIFIED"
            if not reason else reason
        )
        cluster["all_compatible_pair_options"]=options
        cluster["provisional_pairs"]=selected
        cluster["original_observed_arms_retained"]=True
        cluster["source_film_pixel_mutated"]=False
        cluster["physical_track_identity"]="UNKNOWN"
        junctions.append(cluster)
        edges.extend(selected)

    # A component can have a different junction at each end (valid long
    # curve), but no port may be associated at two competing junctions.
    port_to_junctions=defaultdict(set)
    for row in edges:
        port_to_junctions[row["port_a"]].add(row["junction_id"])
        port_to_junctions[row["port_b"]].add(row["junction_id"])
    collisions={k for k,v in port_to_junctions.items() if len(v)>1}
    if collisions:
        kept=[]
        for edge in edges:
            if edge["port_a"] in collisions or edge["port_b"] in collisions:
                edge["review_decision"]="MULTIPLE_JUNCTIONS_SHARE_PORT_ABSTAIN"
            else:
                kept.append(edge)
        edges=kept

    selected_by_junction=Counter(x["junction_id"] for x in edges)
    # If one of a crossing's two pairs was globally removed, the other is
    # no longer trusted either (two unaccounted arms).
    valid={cid for cid,n in selected_by_junction.items() if n==2}
    edges=[e for e in edges if e["junction_id"] in valid]
    for j in junctions:
        if (j["junction_status"]==
            "TWO_ALTERNATIVE_VISUAL_CROSSING_CONTINUATIONS_UNVERIFIED"
            and j["junction_id"] not in valid):
            j["junction_status"]="GLOBAL_PORT_CONFLICT_ABSTAIN"
            j["provisional_pairs"]=[]
    # Every edge is still an UNKNOWN visual hypothesis, not a guarantee
    # that the crossing is two real tracks.
    return junctions,edges,{
        "schema":SCHEMA,
        "status":"REVERSIBLE_2D_JUNCTION_REVIEW_ONLY",
        "source_components":len(components),
        "source_components_preserved":len(components),
        "total_source_endpoints":len(endpoints),
        "candidate_junctions":len(junctions),
        "four_arm_junctions_with_two_visual_pairs":len(valid),
        "three_arm_branch_abstentions":sum(
            x["junction_status"]=="THREE_ARM_BRANCH_ABSTAIN" for x in junctions
        ),
        "dense_bundle_abstentions":sum(
            x["junction_status"]=="DENSE_BUNDLE_ABSTAIN" for x in junctions
        ),
        "provisional_visual_continuation_pairs":len(edges),
        "source_segments_deleted":0,
        "full_source_line_recall_not_known":True,
        "physical_auto_joins_authorized":False,
        "experimentally_calibrated_on_real_H50Z":False,
    }


def grouped_2d_instance_hypotheses(components,edges):
    """Open chain views with observed sections and inferred gaps DISTINCT.

    Never replace observed sections or change their component identities;
    only aggregate for local human review. Any problematic cycle/port conflict
    fails closed, not a silent reassignment.
    """
    paths={x["component_id"]:_path(x["points_xy"]) for x in components}
    if len(paths)!=len(components):
        raise ValueError("Duplicate 2D source components")
    byport={}
    for e in edges:
        if (e.get("review_decision")!="PROVISIONAL_2D_CROSSING_CONTINUATION"
            or e.get("status")!="UNKNOWN" or e.get("physical_auto_join") is not False):
            raise ValueError("Cannot group unverified or unsafe junction proposal")
        a=(e["component_a"],e["endpoint_port_a"])
        b=(e["component_b"],e["endpoint_port_b"])
        if (a==b or a[0]==b[0] or a in byport or b in byport
            or a[0] not in paths or b[0] not in paths):
            raise ValueError("Invalid or double-occupied endpoint")
        byport[a]=(b,e)
        byport[b]=(a,e)
    graph=defaultdict(set)
    for (a,ap),(other,e) in byport.items():
        graph[a].add(other[0])
    chains=[]
    consumed=set()
    for cid in sorted(graph):
        if cid in consumed:
            continue
        members=set()
        todo=[cid]
        while todo:
            c=todo.pop()
            if c in members:continue
            members.add(c)
            todo.extend(sorted(graph[c]-members))
        terminals=sorted(k for k in members if len(graph[k])==1)
        if len(terminals)!=2:
            raise ValueError("Cycle or branch detected in provisional visual chains")
        cur=terminals[0]
        entry=0 if (cur,0) not in byport else 1
        visited=set()
        order=[];observed=[];missing=[]
        while True:
            if cur in visited:raise ValueError("Cycle in crossing continuation")
            visited.add(cur)
            order.append(cur)
            segment=paths[cur] if entry==0 else paths[cur][::-1]
            observed.append(np.round(segment,5).tolist())
            opposite=1-entry
            link=byport.get((cur,opposite))
            if link is None:break
            (next_c,next_entry),edge=link
            center=edge["inferred_junction_bridge_xy"][1]
            missing.append({
                "junction_id":edge["junction_id"],
                "from_component":cur,"to_component":next_c,
                "inferred_only_points_xy":[
                    np.round(segment[-1],5).tolist(),center,
                    np.round(paths[next_c][next_entry],5).tolist(),
                ],
                "physical_auto_join":False,
            })
            cur,entry=next_c,next_entry
        if visited!=members:
            raise ValueError("Incomplete provisional track grouping")
        consumed|=members
        chains.append({
            "visual_chain_id":"s96_c_"+sha256("|".join(sorted(members)).encode()).hexdigest()[:20],
            "ordered_source_components":order,
            "observed_source_sections_xy":observed,
            "unobserved_junction_interpolations":missing,
            "status":"UNKNOWN",
            "visual_review_required":True,
            "same_particle_identity_verified":False,
            "no_source_pixels_replaced":True,
        })
    return chains,{
        "all_source_components":len(paths),
        "components_in_provisional_chains":len(consumed),
        "unassociated_source_components":sorted(set(paths)-consumed),
        "no_source_components_deleted":True,
        "no_physical_auto_join":True,
    }
