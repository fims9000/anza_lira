"""S110: global, conflict-aware association of S109 cubic B-spline fragments.

Source-backed pixels, inferred connecting geometry, algorithmic group IDs, and
expert-confirmed identity are separate data fields. No fake observations or
unjustified physical particle IDs are manufactured.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from math import acos, degrees
from typing import Any

import cv2
import networkx as nx
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class GlobalConfig:
    max_gap_px: float = 80.0
    min_gap_px: float = 3.0
    min_endpoint_cos: float = .90
    max_curve_to_chord_ratio: float = 1.23
    max_lateral_fraction: float = .17
    max_photo_corridor_blocked: float = .02
    min_photo_support_long: float = .22
    min_photo_support_very_long: float = .34
    minimum_score: float = .70
    auto_candidate_score: float = .77
    endpoint_ambiguity_margin: float = .062
    max_near_rival_px: float = 3.4
    max_path_crossing_px: float = 3.2
    max_endpoint_direction_difference_deg: float = 48.0
    min_endpoint_anchor_length_px: float = 8.0


def _unit(x):
    x=np.asarray(x,float)
    n=float(np.linalg.norm(x))
    return x/max(n,1.e-9)


def endpoint_outward(c:dict,end:int,span_px:float=13):
    """Locally fitted endpoint tangent over arc length, robust to pixel noise."""
    q=np.asarray(c['sampled_xy'],float)
    if end==1:q=q[::-1]
    cumulative=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))]
    idx=int(np.searchsorted(cumulative,span_px,side='left'))
    idx=max(2,min(len(q)-1,idx))
    return _unit(q[0]-q[idx])


def bezier_bridge(p0,p1,out0,out1):
    """Single cubic Hermite bridge with G1-compatible endpoint tangents."""
    p0=np.asarray(p0,float);p1=np.asarray(p1,float)
    d=float(np.linalg.norm(p1-p0))
    # The two controls are symmetric, so a short gap has a well-conditioned
    # gently curved interpolation; this is inferred, never a photo detection.
    h=.33*d
    return [p0.tolist(), (p0+_unit(out0)*h).tolist(),
            (p1+_unit(out1)*h).tolist(), p1.tolist()]


def sample_bezier(b,n=80):
    p=np.asarray(b,float)
    t=np.linspace(0,1,n,dtype=float)[:,None]
    return ((1-t)**3*p[0]+3*(1-t)**2*t*p[1]+3*(1-t)*t*t*p[2]+t**3*p[3])


def _pixel_samples(image,positions):
    h,w=image.shape
    p=np.asarray(positions,float)
    x=np.clip(np.rint(p[:,0]).astype(int),0,w-1)
    y=np.clip(np.rint(p[:,1]).astype(int),0,h-1)
    return image[y,x]


def _rival_cloud(curves):
    arr=[];idx=[]
    for i,c in enumerate(curves):
        q=np.asarray(c['sampled_xy'],float)
        length=float(c['spline_arc_length_px'])
        step=max(1,int(np.floor(len(q)/max(2,length/2.5))))
        sampled=q[::step]
        if not np.array_equal(sampled[-1],q[-1]):sampled=np.vstack((sampled,q[-1]))
        arr.append(sampled);idx.extend([i]*len(sampled))
    if not arr:return None,None
    return cKDTree(np.vstack(arr)),np.asarray(idx,np.int32)


def assess_candidate(ca,ea,cb,eb,dist_photo,blocked,gray,dark,cloud,owner,cfg):
    pa=np.asarray(ca['endpoints'][ea],float)
    pb=np.asarray(cb['endpoints'][eb],float)
    delta=pb-pa;gap=float(np.linalg.norm(delta))
    if gap<cfg.min_gap_px or gap>cfg.max_gap_px:return None,'distance'
    axis=delta/gap
    ta=endpoint_outward(ca,ea)
    tb=endpoint_outward(cb,eb)
    ca_cos=float(ta@axis)
    cb_cos=float(tb@(-axis))
    if min(ca_cos,cb_cos)<cfg.min_endpoint_cos:return None,'tangent'
    turn=degrees(acos(float(np.clip(-ta@tb,-1.,1.))))
    if turn>cfg.max_endpoint_direction_difference_deg:return None,'turn'
    bez=bezier_bridge(pa,pb,ta,tb)
    n=max(12,int(np.ceil(gap*1.7)))
    path=sample_bezier(bez,n)
    arc=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())
    if arc>gap*cfg.max_curve_to_chord_ratio:return None,'curvature'
    perp=np.abs(axis[0]*(path[:,1]-pa[1])-axis[1]*(path[:,0]-pa[0]))
    if float(perp.max())>max(3.,gap*cfg.max_lateral_fraction):return None,'lateral'
    h,w=blocked.shape
    inside=(path[:,0]>=2)&(path[:,0]<w-2)&(path[:,1]>=2)&(path[:,1]<h-2)
    if not bool(np.all(inside)):return None,'out_of_image'
    blockedfrac=float(np.mean(_pixel_samples(blocked,path)))
    if blockedfrac>cfg.max_photo_corridor_blocked:return None,'blocked'
    d=_pixel_samples(dist_photo,path)
    support=float(np.mean(d<=2.5))
    if gap>45 and support<cfg.min_photo_support_very_long:return None,'unsupported_long'
    if gap>18 and support<cfg.min_photo_support_long:return None,'unsupported'
    if cloud is not None:
        # A rival curve intersecting the *interior* of a proposed connector is
        # a genuine ambiguity. Never merge merely because two ends are near.
        ix=np.arange(2,len(path)-2)
        neighbors=cloud.query_ball_point(path[ix],r=cfg.max_near_rival_px)
        rivals=0
        for a in neighbors:
            if any(int(owner[j]) not in (ca['_index'],cb['_index']) for j in a):rivals+=1
        if rivals>=2:return None,'rival_curve'
    # Contrast versus adjacent side bands detects occasional hops between
    # parallel dark lines or dust. It is a weak diagnostic (not expert GT).
    center=float(np.mean(_pixel_samples(dark,path)))
    step=np.gradient(path,axis=0)
    unit=step/np.maximum(np.linalg.norm(step,axis=1)[:,None],1e-6)
    normal=np.column_stack((-unit[:,1],unit[:,0]))
    side=(float(np.mean(_pixel_samples(dark,path+normal*6)))+
          float(np.mean(_pixel_samples(dark,path-normal*6))))/2
    contrast=center-side
    # Does not assign negative score if candidate lies in a blank gap; only
    # stronger photo support in the candidate corridor improves confidence.
    photo_score=np.clip(support,0,1)
    tangent_score=min(ca_cos,cb_cos)
    smooth_score=max(0,1-(arc/gap-1)/.20)
    length_score=max(0,1-gap/cfg.max_gap_px)
    contrast_score=float(np.clip((contrast+1)/7,0,1))
    score=(.28*tangent_score+.31*photo_score+.15*smooth_score+
           .14*length_score+.12*contrast_score)
    rec={'curve_ids':[ca['curve_id'],cb['curve_id']],
         'ends':[int(ea),int(eb)],'ports':[(ca['curve_id'],int(ea)),(cb['curve_id'],int(eb))],
         'gap_px':round(gap,3),'endpoint_alignment':[round(ca_cos,4),round(cb_cos,4)],
         'turn_deg':round(turn,3),'arc_to_chord':round(arc/gap,4),
         'max_lateral_px':round(float(perp.max()),3),
         'photo_support_fraction':round(support,4),'source_dark_minus_side_mean':round(contrast,3),
         'blocked_fraction':round(blockedfrac,3),'score':round(float(score),5),
         'bezier_cubic':[[round(float(v),4) for v in p] for p in bez],
         'sampled_xy':np.round(path,2).tolist(),
         'is_physical_ground_truth':False,'connector_is_observed':False}
    return rec,None


def generate_candidates(curves,photo,blocked,gray,cfg=GlobalConfig()):
    photo=np.asarray(photo,bool);blocked=np.asarray(blocked,bool)
    if photo.shape!=blocked.shape or photo.shape!=gray.shape:raise ValueError('mismatched image masks')
    active=[]
    for i,c in enumerate(curves):
        if not c.get('source_anchored',False):continue
        x=dict(c);x['_index']=i;active.append(x)
    port=[]
    for c in active:
        for e in (0,1):port.append((c,e,np.asarray(c['endpoints'][e],float)))
    photo_dist=distance_transform_edt(~(photo&~blocked))
    f=gray.astype(np.float32)
    dark=cv2.GaussianBlur(f,(0,0),5)-cv2.GaussianBlur(f,(0,0),.8)
    tree_owner,owner=_rival_cloud(curves)
    if len(port)<2:return [],{'ports':len(port),'rejections':{},'candidate_pairs':0}
    pairs=cKDTree(np.vstack([z[2] for z in port])).query_pairs(cfg.max_gap_px)
    results=[];reject=defaultdict(int)
    # Calling the contrast filter for all nearby pairs would repeatedly blur
    # the same image. Defer to assess_candidate after cheap angle/distance.
    for a,b in sorted(pairs):
        ca,ea,_=port[a];cb,eb,_=port[b]
        if ca['curve_id']==cb['curve_id']:continue
        if ca['source_path_id']==cb['source_path_id'] and ca['curve_id']!=cb['curve_id']:
            # Multiple disjoint pieces of same original ordered point path can
            # still be joined, but only under geometric and photo constraints.
            pass
        rec,reason=assess_candidate(ca,ea,cb,eb,photo_dist,blocked,gray,dark,tree_owner,owner,cfg)
        if rec is None:reject[reason]+=1;continue
        rec['candidate_id']='edge_'+sha256(('|'.join(sorted(rec['curve_ids']))+str(rec['ends'])).encode()).hexdigest()[:12]
        rec['decision']='UNRESOLVED_GLOBAL_MATCHING'
        results.append(rec)
    results.sort(key=lambda z:(-z['score'],z['candidate_id']))
    return results,{'ports':len(port),'raw_nearby_pairs':len(pairs),
                    'candidate_pairs':len(results),'rejections':dict(reject)}


def segments_intersect(a,b,minsep=3.0):
    qa=np.asarray(a['sampled_xy'],float)[2:-2]
    qb=np.asarray(b['sampled_xy'],float)[2:-2]
    if len(qa)==0 or len(qb)==0:return False
    d=cKDTree(qa).query(qb)[0]
    return bool(np.any(d<minsep))



def prior_review_edges(curves, original_review, already_proposed):
    """Preserve S109's 22 review-only suggestions without auto-accepting them.

    A former proposal that failed S110 geometry remains visible to a human,
    but is never turned into an automated physical identity. Its cubic gap is
    built from its corresponding S109 endpoint pair and marked inferred.
    """
    byid={c['curve_id']:c for c in curves}
    existing={frozenset(z['curve_ids']):z for z in already_proposed}
    added=[]
    for item in original_review:
        names=item.get('curve_ids',[])
        if len(names)!=2 or any(k not in byid for k in names):continue
        pair=frozenset(names)
        if pair in existing:
            existing[pair]['s109_also_proposed_for_review']=True
            continue
        c0,c1=(byid[names[0]],byid[names[1]])
        pts=np.asarray(item.get('inferred_link_xy',[]),float)
        if pts.ndim!=2 or pts.shape[1]!=2 or len(pts)<2:continue
        e0=min((0,1),key=lambda x:float(np.linalg.norm(np.asarray(c0['endpoints'][x])-pts[0])))
        e1=min((0,1),key=lambda x:float(np.linalg.norm(np.asarray(c1['endpoints'][x])-pts[-1])))
        p0=np.asarray(c0['endpoints'][e0],float)
        p1=np.asarray(c1['endpoints'][e1],float)
        bez=bezier_bridge(p0,p1,endpoint_outward(c0,e0),endpoint_outward(c1,e1))
        sampled=sample_bezier(bez,max(12,int(np.ceil(np.linalg.norm(p1-p0)*2))))
        key='legacy_'+sha256(('|'.join(sorted(names))+str((e0,e1))).encode()).hexdigest()[:12]
        added.append({'candidate_id':key,'curve_ids':names,'ends':[e0,e1],
                      'ports':[(names[0],e0),(names[1],e1)],
                      'gap_px':round(float(np.linalg.norm(p1-p0)),3),
                      'score':round(float(item.get('score',0.0)),5),
                      'photo_support_fraction':float(item.get('photo_corridor_support',0.0)),
                      'blocked_fraction':None,'endpoint_alignment':None,
                      'bezier_cubic':bez,'sampled_xy':np.round(sampled,2).tolist(),
                      'legacy_S109_suggestion_only':True,'auto_eligible':False,
                      'is_physical_ground_truth':False,'connector_is_observed':False,
                      'decision':'REVIEW_S109_LEGACY_NOT_AUTO_ACCEPTED'})
    return added


def solve_global(curves,candidates,cfg=GlobalConfig(),expert_decisions=None):
    """Global maximum weight matching over endpoint ports with abstention.

    Degree of each endpoint <= 1; no curve-level cycles; no crossing bridge
    corridors. Never declares an algorithmic match a confirmed particle ID.
    """
    expert_decisions=expert_decisions or {}
    ids={c['curve_id'] for c in curves}
    decisions={str(k):str(v) for k,v in expert_decisions.items()}
    allowed={'same','different','uncertain'}
    if set(decisions.values())-allowed:raise ValueError('Review values must be same/different/uncertain')
    lookup={e['candidate_id']:e for e in candidates}
    if set(decisions)-set(lookup):raise ValueError('Expert decision references an unknown candidate edge')
    byport=defaultdict(list)
    for e in candidates:
        for p in e['ports']:byport[tuple(p)].append(e)
    ambiguous=set()
    for p,edges in byport.items():
        values=sorted((e['score'] for e in edges if decisions.get(e['candidate_id'])!='different'),reverse=True)
        if len(values)>=2 and values[0]-values[1]<cfg.endpoint_ambiguity_margin:ambiguous.add(p)
    graph=nx.Graph()
    for e in candidates:
        if not e.get('auto_eligible',True):continue
        if decisions.get(e['candidate_id']) in ('different','uncertain'):continue
        if e['score']<cfg.auto_candidate_score:continue
        if any(tuple(p) in ambiguous for p in e['ports']):continue
        a,b=map(tuple,e['ports'])
        # Low-positive total weight allows the optimizer to leave an endpoint
        # unmatched rather than connect for cardinality's sake.
        graph.add_edge(a,b,weight=max(1e-6,e['score']-cfg.minimum_score),candidate_id=e['candidate_id'])
    chosen_ids=set()
    parent={v:v for v in ids}
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    def add(e,forced=False):
        a,b=map(tuple,e['ports'])
        if a in used or b in used:raise ValueError(f'Endpoint conflict for forced edge {e["candidate_id"]}') if forced else RuntimeError('port occupied')
        c0,c1=e['curve_ids']
        if root(c0)==root(c1):
            if forced:raise ValueError('Expert forced edge would close a cycle')
            return False
        for other in accepted:
            if segments_intersect(e,other,cfg.max_path_crossing_px):
                if forced:raise ValueError('Expert forced links spatially conflict')
                return False
        used.update((a,b));parent[root(c1)]=root(c0)
        accepted.append(e);chosen_ids.add(e['candidate_id'])
        return True
    used=set();accepted=[]
    forced=sorted((lookup[k] for k,v in decisions.items() if v=='same'),key=lambda z:z['candidate_id'])
    for e in forced:add(e,True)
    raw_matching=nx.algorithms.matching.max_weight_matching(graph,maxcardinality=False,weight='weight')
    candidates_matched=[]
    for a,b in raw_matching:
        e=lookup[graph[a][b]['candidate_id']]
        candidates_matched.append(e)
    for e in sorted(candidates_matched,key=lambda z:(-z['score'],z['candidate_id'])):
        if e['candidate_id'] in chosen_ids:continue
        a,b=map(tuple,e['ports'])
        if a in used or b in used:continue
        add(e)
    results=[]
    for e in candidates:
        row=dict(e)
        key=e['candidate_id']
        if key in chosen_ids:
            row['decision']='EXPERT_CONFIRMED_SAME' if decisions.get(key)=='same' else 'ALGORITHMIC_GROUPING_NOT_PHYSICAL_GT'
        elif decisions.get(key)=='different':row['decision']='EXPERT_REJECTED_DIFFERENT'
        elif decisions.get(key)=='uncertain':row['decision']='EXPERT_DEFERRED'
        elif not e.get('auto_eligible',True):row['decision']='REVIEW_S109_LEGACY_NOT_AUTO_ACCEPTED'
        elif any(tuple(p) in ambiguous for p in e['ports']):row['decision']='ABSTAIN_AMBIGUOUS_END'
        elif row['score']<cfg.auto_candidate_score:row['decision']='REVIEW_LOW_CONFIDENCE'
        else:row['decision']='REJECTED_GLOBAL_CONFLICT_OR_LOWER_SCORE'
        row['review_status']=decisions.get(key,'unreviewed')
        results.append(row)
    return results,{'raw_proposals':len(candidates),'algorithmic_group_links':len(accepted)-len(forced),
                    'expert_confirmed_group_links':len(forced),
                    'ambiguous_endpoints':len(ambiguous),'matched_by_global_optimizer':len(raw_matching),
                    'no_confirmed_physical_tracks_without_expert_review':True}


def reversed_spans(c):
    spans=c['bezier_cubic_spans']
    return [[*reversed(span)] for span in reversed(spans)]


def path_from_beziers(spans):
    if not spans:return ''
    s=lambda p:f'{p[0]:.3f},{p[1]:.3f}'
    line='M '+s(spans[0][0])
    for p in spans:line+=' C '+s(p[1])+' '+s(p[2])+' '+s(p[3])
    return line


def _make_tracks(curves,edges):
    byid={c['curve_id']:c for c in curves}
    connections=defaultdict(dict)
    selected=[e for e in edges if e['decision'] in ('ALGORITHMIC_GROUPING_NOT_PHYSICAL_GT','EXPERT_CONFIRMED_SAME')]
    for e in selected:
        (i,ei),(j,ej)=map(tuple,e['ports'])
        if ei in connections[i] or ej in connections[j]:raise RuntimeError('two links on one endpoint')
        connections[i][ei]=(j,ej,e);connections[j][ej]=(i,ei,e)
    unseen=set(byid)
    tracks=[]
    while unseen:
        seed=min(unseen)
        # Collect only this component to find a true free endpoint.
        q=[seed];member={seed}
        while q:
            curr=q.pop()
            for other,_,_ in connections[curr].values():
                if other not in member:member.add(other);q.append(other)
        starts=[]
        for i in member:
            for end in (0,1):
                if end not in connections[i]:starts.append((i,end))
        if not starts:raise RuntimeError('global matching formed cycle; cannot orient global path')
        now,entry=min(starts)
        chain=[];links=[];vector_spans=[];gaps=[]
        visited=set()
        while True:
            if now in visited:raise RuntimeError('cycle in assembled group')
            visited.add(now);unseen.remove(now)
            c=byid[now]
            reverse=entry==1
            oriented=np.asarray(c['sampled_xy'],float)[::-1] if reverse else np.asarray(c['sampled_xy'],float)
            flags=np.asarray(c['sampled_source_supported'],bool)[::-1] if reverse else np.asarray(c['sampled_source_supported'],bool)
            spans=reversed_spans(c) if reverse else c['bezier_cubic_spans']
            vector_spans.extend(spans)
            chain.append({'curve_id':now,'reversed':reverse,'source_anchored':c['source_anchored'],
                          'source_backed_sample_count':int(flags.sum()),
                          'inferred_inside_fragment_samples':int((~flags).sum()),
                          'source_spline_svg':path_from_beziers(spans)})
            exitend=1-entry
            if exitend not in connections[now]:break
            nxt,nxtentry,e=connections[now][exitend]
            span=e['bezier_cubic']
            if e['curve_ids'][0]!=now or e['ends'][0]!=exitend:span=list(reversed(span))
            # Every gap connector remains an inferred, reversible hypothesis.
            vector_spans.append(span)
            geom=sample_bezier(span,max(8,int(np.ceil(e['gap_px']*2))))
            gaps.append({'candidate_id':e['candidate_id'],'from_curve_id':now,'to_curve_id':nxt,
                         'status':e['decision'],'connector_svg':path_from_beziers([span]),
                         'sampled_xy':np.round(geom,2).tolist(),
                         'source_supported':False,'gap_px':e['gap_px'],'photo_corridor_support_fraction':e['photo_support_fraction']})
            links.append(e['candidate_id'])
            now=nxt;entry=nxtentry
        members=[c['curve_id'] for c in chain]
        track_id='track_'+sha256(('|'.join(sorted(members))).encode()).hexdigest()[:12]
        status='UNREVIEWED_ASSOCIATION' if len(members)>1 else 'SINGLE_FRAGMENT_UNCONFIRMED_PHYSICAL_ID'
        if len(members)>1 and all(g['status']=='EXPERT_CONFIRMED_SAME' for g in gaps):
            status='ALL_LINKS_EXPERT_CONFIRMED_BUT_PHYSICAL_ID_UNVERIFIED'
        tracks.append({'global_track_id':track_id,'ordered_members':chain,
                       'member_curve_ids':members,'connection_edge_ids':links,
                       'inferred_connectors':gaps,'spans_cubic_bezier':vector_spans,
                       'composite_svg_path':path_from_beziers(vector_spans),
                       'status':status,'confirmed_physical_particle_id':None,
                       'inferred_connector_count':len(gaps),'can_separate_members_without_losing_observations':True})
    tracks.sort(key=lambda t:(-len(t['member_curve_ids']),t['global_track_id']))
    return tracks


def associate(curves,photo,blocked,gray,cfg=GlobalConfig(),decisions=None,legacy_review=None):
    cands,gdiag=generate_candidates(curves,photo,blocked,gray,cfg)
    legacy=prior_review_edges(curves,legacy_review or [],cands)
    cands.extend(legacy)
    cands.sort(key=lambda z:(-z['score'],z['candidate_id']))
    gdiag['legacy_s109_review_only_edges_added']=len(legacy)
    scored,opt_diag=solve_global(curves,cands,cfg,decisions)
    tracks=_make_tracks(curves,scored)
    ncurves=sum(len(t['member_curve_ids']) for t in tracks)
    if ncurves!=len(curves):raise RuntimeError('Each curve must occur exactly once')
    counts={'fragments_in':len(curves),'global_track_candidates':len(tracks),
            'groups_with_multiple_fragments':sum(len(t['member_curve_ids'])>1 for t in tracks),
            'joined_curve_fragments':sum(len(t['member_curve_ids']) for t in tracks if len(t['member_curve_ids'])>1),
            'selected_link_count':sum(t['inferred_connector_count'] for t in tracks),
            'unresolved_candidate_pairs':sum(e['decision'] not in ('ALGORITHMIC_GROUPING_NOT_PHYSICAL_GT','EXPERT_CONFIRMED_SAME') for e in scored),
            'not_verified_true_particle_tracks':True}
    return {'tracks':tracks,'edge_proposals':scored,'association_report':counts,
            'generator_audit':gdiag,'solver_audit':opt_diag}