"""S51 reversible 2D collision advisories for *proposed* bridges only.

These are neither native particle labels nor a veto of physical junctions.
At 2-D projection crossings can be real and distinct particle tracks can
visually overlap. Every edge retains UNKNOWN identity and zero auto joins.
"""
from __future__ import annotations

import numpy as np

SCHEMA = 's51_2d_candidate_bridge_collision_advisory_v1'
RADIUS_PX = 3.0


def _valid_line(path, min_points: int = 2) -> np.ndarray:
    x = np.asarray(path, dtype=float)
    if (x.ndim != 2 or x.shape[1] != 2 or len(x) < min_points
            or not np.isfinite(x).all() or (x < 0).any() or (x > 512).any()):
        raise ValueError('Invalid private source polyline coordinates')
    return x


def central_bridge_points(bridge) -> np.ndarray:
    """Fixed central 60% on 101 interpolated Hermite samples."""
    p = _valid_line(bridge, min_points=5)
    u = np.linspace(0., 1.,len(p))
    t = np.linspace(.2,.8,61)
    return np.stack((np.interp(t,u,p[:,0]),np.interp(t,u,p[:,1])),axis=1)


def min_point_to_segments_distance(points,polyline) -> float:
    """Exact point-to-polyline segment distance rather than sample-only KD tree."""
    p=_valid_line(points)
    q=_valid_line(polyline)
    a=q[:-1]
    v=q[1:]-a
    d=p[:,None,:]-a[None,:,:]
    l2=(v*v).sum(axis=1)
    good=l2>1e-9
    if not good.any(): raise ValueError('Degenerate distractor line')
    s=np.clip((d[:,good,:]*v[good][None,:,:]).sum(axis=2)/l2[good][None,:],0.,1.)
    closest=a[good][None,:,:]+s[:,:,None]*v[good][None,:,:]
    return float(np.linalg.norm(p[:,None,:]-closest,axis=2).min())


def collision_advisory(nodes:list[dict],edge:dict,*,radius:float=RADIUS_PX)->dict:
    """Check native graph as-is; never remove edge or assign real identity."""
    if not np.isfinite(radius) or radius <= 0 or radius > 6:
        raise ValueError('Unfrozen collision radius')
    if not isinstance(nodes,list) or len(nodes)<2 or any(not isinstance(n,dict) for n in nodes):
        raise ValueError('Missing graph nodes')
    i,j=edge.get('i'),edge.get('j')
    if (not isinstance(i,int) or not isinstance(j,int) or i==j
            or not (0<=i<len(nodes) and 0<=j<len(nodes))
            or edge.get('physics_join_authorized') is not False):
        raise ValueError('Cannot authorize physical joins or misalign graph')
    points=central_bridge_points(edge.get('hermite_bridge_pixels_PROPOSAL_ONLY'))
    nearest=[]
    for n,node in enumerate(nodes):
        path=_valid_line(node.get('points_xy'),min_points=8)
        if n in (i,j):continue
        distance=min_point_to_segments_distance(points,path)
        nearest.append((n,distance))
    hits=[index for index,distance in nearest if distance<=radius]
    smallest=min((x[1] for x in nearest), default=None)
    return {'i':i,'j':j,'potential_third_line_overlap':bool(hits),
            'third_line_candidate_count':len(hits),
            'minimum_other_line_distance_px_PRIVATE':smallest,
            'other_line_node_indices_PRIVATE':hits,
            'status':'TWO_DIMENSIONAL_TOPOLOGY_AMBIGUITY_NOT_TRUTH' if hits else 'NO_COLLISION_FLAG_STILL_UNVERIFIED',
            'native_particle_relation_truth':'UNKNOWN',
            'physical_auto_join_authorized':False}


def full_graph_diagnostic(graph:dict)->dict:
    if graph.get('physics_auto_links_performed')!=0:
        raise ValueError('Native physical graph cannot contain automatic joins')
    nodes=graph.get('nodes')
    edges=graph.get('candidate_edges')
    if not isinstance(nodes,list) or not isinstance(edges,list):
        raise ValueError('Missing S39 graph')
    info=[collision_advisory(nodes,e) for e in edges]
    return {'schema':SCHEMA, 'original_edges':len(edges),
            'collision_warning_edges':sum(x['potential_third_line_overlap'] for x in info),
            'all_original_edges_preserved':True,
            'auto_physical_joins':0,
            'private_per_edge_advisories':info}
