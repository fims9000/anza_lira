"""S115: recover continuous *observed* centerlines across skeleton junctions.

The join is geometrical: neither a track physical ID nor GT. Only joins that
can be routed using existing original-image skeleton pixels are accepted.
Unobserved gaps remain a separate REVIEW_ONLY layer from S114.
"""
from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import combinations
import math
import numpy as np
from scipy.spatial import cKDTree
from .s114_source_geometry import _spline


@dataclass(frozen=True)
class JunctionConfig:
    max_endpoint_distance_px: float = 7.5
    tangent_span_px: float = 12.0
    min_straight_cos: float = .91
    min_chord_cos: float = .5
    max_observed_connector_px: int = 22
    ambiguity_margin: float = .08
    max_candidate_cluster_ports: int = 12


def _tangent_from_endpoint(xy, end, span):
    p = np.asarray(xy, np.float64)
    if end == 1: p = p[::-1]
    if len(p) < 2: return None
    dist = np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))
    idx = min(len(p) - 1, 1 + int(np.searchsorted(dist, span)))
    vec = p[idx] - p[0]
    norm = np.linalg.norm(vec)
    return None if norm < 3 else vec / norm


def _source_pixel_route(skeleton, p, q, max_len):
    """Shortest 8-connected route restricted to original skeleton; no interpolation."""
    h, w = skeleton.shape
    p = tuple(np.rint(p).astype(int));q = tuple(np.rint(q).astype(int))
    if not (0 <= p[0] < w and 0 <= p[1] < h and 0 <= q[0] < w and 0 <= q[1] < h):return None
    if not skeleton[p[1],p[0]] or not skeleton[q[1],q[0]]:return None
    if p == q:return np.array([p], dtype=np.float64)
    x0=max(0,min(p[0],q[0])-5);x1=min(w,max(p[0],q[0])+6)
    y0=max(0,min(p[1],q[1])-5);y1=min(h,max(p[1],q[1])+6)
    queue=deque([p]);prev={p:None};depth={p:0}
    while queue:
        x,y=queue.popleft()
        if (x,y)==q:break
        if depth[(x,y)]>=max_len:continue
        for dx,dy in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)):
            xx=x+dx; yy=y+dy;v=(xx,yy)
            if not (x0<=xx<x1 and y0<=yy<y1 and skeleton[yy,xx]) or v in prev: continue
            # Keep same corner-shortcut rules as S114 path extraction.
            if dx and dy and (skeleton[y,xx] or skeleton[yy,x]):continue
            prev[v]=(x,y);depth[v]=depth[(x,y)]+1;queue.append(v)
    if q not in prev:return None
    route=[];cur=q
    while cur is not None:route.append(cur);cur=prev[cur]
    route.reverse()
    return np.asarray(route,np.float64)


def _best_matching(edges, n, margin):
    """Exact small-graph matching; abstain where distinct pairings nearly tie.

    Return chosen edge indices, or None when globally ambiguous/overbudget.
    """
    if n > 12: return None
    by_vertex=defaultdict(list)
    for i,e in enumerate(edges):
        by_vertex[e['u']].append(i);by_vertex[e['v']].append(i)
    ranked=[]
    # A small cluster can still have many choices. Recursive exhaustive search
    # is bounded to <=12 endpoint ports (<=10k perfect matchings).
    def scan(unused, score, selection):
        if not unused:
            ranked.append((score, tuple(sorted(selection))))
            return
        u=min(unused)
        scan(unused-{u},score,selection)
        for k in by_vertex[u]:
            e=edges[k];v=e['v'] if e['u']==u else e['u']
            if v in unused:
                scan(unused-{u,v},score+e['score'],selection+[k])
    scan(set(range(n)),0.0,[])
    ranked.sort(key=lambda x:(-x[0], x[1]))
    if not ranked or ranked[0][0] <= 0:return []
    winner=ranked[0]
    if len(ranked)>1 and winner[0]-ranked[1][0]<margin: return None
    return list(winner[1])


def _candidate_join(ti, ei, tj, ej, skeleton, cfg):
    if ti['tracklet_id']==tj['tracklet_id']:return None
    a=np.asarray(ti['observed_pixel_path_xy'],float)
    b=np.asarray(tj['observed_pixel_path_xy'],float)
    if min(len(a),len(b))<5:return None
    p=a[0 if ei==0 else -1];q=b[0 if ej==0 else -1]
    sep=float(np.linalg.norm(p-q))
    if sep>cfg.max_endpoint_distance_px:return None
    u=_tangent_from_endpoint(a,ei,cfg.tangent_span_px)
    v=_tangent_from_endpoint(b,ej,cfg.tangent_span_px)
    if u is None or v is None:return None
    straight=float(-u@v)
    if straight<cfg.min_straight_cos:return None
    if sep>2.5:
        axis=(q-p)/sep
        if float(-u@axis)<cfg.min_chord_cos or float(v@axis)<cfg.min_chord_cos:
            return None
    route=_source_pixel_route(skeleton,p,q,cfg.max_observed_connector_px)
    if route is None:return None
    geom_length=float(np.linalg.norm(np.diff(route,axis=0),axis=1).sum()) if len(route)>1 else 0.
    if geom_length>cfg.max_observed_connector_px:return None
    # Higher confidence for direct skeleton junction, not arbitrary long gaps.
    score=round(float(2*straight-1.55 - .013*sep - .008*geom_length),6)
    if score<=0:return None
    return {'score':score,'cos_straight':round(straight,4),
            'endpoint_distance_px':round(sep,3),'connector_pixel_count':len(route),
            'route_xy':route.round(2).tolist()}


def associate_observed_junctions(tracks, skeleton, cfg:JunctionConfig=JunctionConfig()):
    if skeleton.ndim != 2:raise ValueError('Source skeleton must be HxW')
    sk=skeleton.astype(bool)
    ids=[str(t['tracklet_id']) for t in tracks]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate segment identities')
    ends=[]
    for i,t in enumerate(tracks):
        pts=np.asarray(t['observed_pixel_path_xy'],float)
        if len(pts)<2:continue
        for e in (0,1):ends.append((i,e,pts[0 if e==0 else -1]))
    if not ends:return {'curves':[],'joins':[],'review_ambiguous':[],'report':{}}
    endpoints=np.array([z[2] for z in ends],float)
    pairs=sorted(cKDTree(endpoints).query_pairs(cfg.max_endpoint_distance_px))
    edges=[]
    for i,j in pairs:
        ta,ea,_=ends[i];tb,eb,_=ends[j]
        if ta==tb:continue
        candidate=_candidate_join(tracks[ta],ea,tracks[tb],eb,sk,cfg)
        if candidate is not None:
            edges.append({'u':i,'v':j,'i':ta,'j':tb,'a_end':ea,'b_end':eb,**candidate})
    adj=defaultdict(set)
    for idx, e in enumerate(edges):
        adj[e['u']].add(idx);adj[e['v']].add(idx)
    selected=[];ambiguous=[];seen=set()
    for port in sorted(adj):
        if port in seen:continue
        queue=[port];component=set([port]);seen.add(port);edge_set=set()
        while queue:
            node=queue.pop()
            for ix in adj[node]:
                edge_set.add(ix);z=edges[ix]
                for nxt in (z['u'],z['v']):
                    if nxt not in seen:seen.add(nxt);component.add(nxt);queue.append(nxt)
        ordered=sorted(component)
        local=[dict(edges[k],u=ordered.index(edges[k]['u']),v=ordered.index(edges[k]['v'])) for k in sorted(edge_set)]
        solution=_best_matching(local,len(component),cfg.ambiguity_margin)
        if solution is None:
            ambiguous.append({'reason':'AMBIGUOUS_OR_TOO_COMPLEX',
                              'port_ids':[f'{tracks[ends[k][0]]["tracklet_id"]}:{ends[k][1]}' for k in ordered],
                              'candidate_count':len(local)})
        else:
            global_edge_keys=sorted(edge_set)
            selected.extend(edges[global_edge_keys[k]] for k in solution)
    # Do not join cycles of fragments: ambiguous physical loops stay separate.
    parent=list(range(len(tracks)))
    def root(x):
        while parent[x]!=x:x=parent[x]
        return x
    joins=[];endlinks={}
    for edge in sorted(selected,key=lambda e:(-e['score'],e['i'],e['j'])):
        ri,rj=root(edge['i']),root(edge['j'])
        if ri==rj:continue
        pi=(edge['i'],edge['a_end']);pj=(edge['j'],edge['b_end'])
        if pi in endlinks or pj in endlinks:continue
        parent[rj]=ri
        join={k:v for k,v in edge.items() if k not in ('u','v')}
        join['kind']='OBSERVED_JUNCTION_GEOMETRY_NOT_PARTICLE_ID'
        join['a_tracklet_id']=ids[edge['i']];join['b_tracklet_id']=ids[edge['j']]
        joins.append(join)
        endlinks[pi]=(pj,join);endlinks[pj]=(pi,join)
    # Traverse connected non-branching fragment chains, preserving source path pixels.
    visited=set();curves=[]
    members=defaultdict(list)
    for i in range(len(tracks)):members[root(i)].append(i)
    for component in sorted(members.values(),key=lambda group:min(group)):
        starts=[(i,e) for i in component for e in (0,1) if (i,e) not in endlinks]
        if len(starts)!=2:
            raise RuntimeError('Source fragment association is cyclic or branched')
        start,first_end=min(starts)
        chain=[];observed=[];used_joins=[]
        pos=start;at=first_end
        while pos not in visited:
            visited.add(pos)
            xy=np.asarray(tracks[pos]['observed_pixel_path_xy'],float)
            if at==1:xy=xy[::-1]
            chain.append(ids[pos])
            if len(observed)==0:observed=xy.tolist()
            else:
                if np.linalg.norm(np.asarray(observed[-1])-xy[0]) < 1e-6:
                    observed += xy[1:].tolist()
                else:
                    observed += xy.tolist()
            exit_side=1-at
            next_info=endlinks.get((pos,exit_side))
            if next_info is None:break
            (nxt,nside),join=next_info
            if nxt in visited:break
            route=np.asarray(join['route_xy'],float)
            endpoint=xy[-1];nextpts=np.asarray(tracks[nxt]['observed_pixel_path_xy'],float)
            nextpoint=nextpts[0 if nside==0 else -1]
            if np.linalg.norm(route[0]-endpoint)>np.linalg.norm(route[-1]-endpoint):route=route[::-1]
            # Exclude duplicate edge endpoints; all interior points source-observed.
            if len(route)>2:observed+=route[1:-1].tolist()
            used_joins.append(f'{join["a_tracklet_id"]}:{join["a_end"]}--{join["b_tracklet_id"]}:{join["b_end"]}')
            pos=nxt;at=nside
        if len(chain)!=len(component):
            raise RuntimeError('Broken traversal: joined source fragment was lost')
        if len(used_joins)!=len(chain)-1:
            raise RuntimeError('Broken traversal: source link count mismatch')
        result_xy=np.asarray(observed,float)
        length=float(np.linalg.norm(np.diff(result_xy,axis=0),axis=1).sum()) if len(result_xy)>1 else 0.
        fitted=_spline(result_xy,length)
        # B-spline smoothing does NOT make additional pixels observed.
        curves.append({'candidate_id':f'S115_{len(curves):04d}',
                       'source_tracklet_ids':chain,'num_joined_tracklets':len(chain),
                       'observed_pixel_path_xy':result_xy.round(2).tolist(),
                       'spline_xy':fitted.round(2).tolist(),
                       'geometric_start_A_xy':result_xy[0].round(2).tolist(),
                       'geometric_end_B_xy':result_xy[-1].round(2).tolist(),
                       'observed_centerline_length_px':round(length,2),
                       'junction_join_ids':used_joins,
                       'particle_identity':None,'physical_direction':None,
                       'source_status':'SOURCE_SKELETON_ONLY_NOT_GT'})
    if len(tracks)-len(joins)!=len(curves):
        raise RuntimeError('Source association did not produce a forest of complete curves')
    curves.sort(key=lambda c:(-c['observed_centerline_length_px'],c['candidate_id']))
    for c in curves:
        observed=np.rint(np.asarray(c['observed_pixel_path_xy'])).astype(int)
        if np.any(observed[:,0]<0) or np.any(observed[:,1]<0) or np.any(observed[:,0]>=sk.shape[1]) or np.any(observed[:,1]>=sk.shape[0]):
            raise RuntimeError('Out-of-image observed candidate')
        if not np.all(sk[observed[:,1],observed[:,0]]):
            raise RuntimeError('Joined candidate contains invented observed pixels')
    accepted={frozenset(((e['i'],e['a_end']),(e['j'],e['b_end']))) for e in joins}
    alternatives=[]
    for e in edges:
        if frozenset(((e['i'],e['a_end']),(e['j'],e['b_end']))) in accepted:continue
        alternatives.append({'tracklet_ids':[ids[e['i']],ids[e['j']]],
                             'endpoint_indices':[e['a_end'],e['b_end']],
                             'score':e['score'],'reason':'COMPETING_OR_UNCERTAIN_SOURCE_GEOMETRY_REVIEW_ONLY',
                             'route_xy':e['route_xy'],'not_accepted':True})
    result={'curves':curves,'joins':joins,'review_ambiguous':ambiguous,
            'unaccepted_junction_options_REVIEW_ONLY':alternatives,
            'report':{'input_fragment_count':len(tracks),'candidate_join_count':len(edges),
                      'accepted_source_junction_joins':len(joins),
                      'abstained_ambiguous_junction_clusters':len(ambiguous),
                      'output_curve_count':len(curves),
                      'max_observed_curve_length_px':round(max((c['observed_centerline_length_px'] for c in curves),default=0),2),
                      'all_pixels_source_skeleton':True,'physical_track_accuracy_known':False}}
    return result
