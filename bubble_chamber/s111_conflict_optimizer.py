"""Exact conflict-aware global matching for ANZA S111.

S110's initial maximum-weight endpoint matching can select two crossing links;
its later greedy crossing removal may then miss a better NONCROSSING set.
This solver jointly enforces ports, geometric crossings, and acyclic groups.
It maximizes the same score utility, without optimizing for link count alone.
"""
from __future__ import annotations
from collections import defaultdict

from .s110_global_association import GlobalConfig, segments_intersect


def _solve_independent_components(curves,candidates,cfg,decisions,max_candidates):
    """Scale exact matching to many geographically independent track groups.

    Candidate groups share an original curve *or* intersect in the image. Thus
    port conflicts, curve-level cycles and spatial crossings cannot cross a
    component boundary, and optimizing components separately remains exact.
    """
    ids={c['curve_id'] for c in curves}
    parent={i:i for i in ids}
    def root(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(x,y):
        rx=root(x);ry=root(y)
        if rx!=ry:parent[ry]=rx
    for edge in candidates:
        union(*edge['curve_ids'])
    # A crossing between two unrelated proposed gaps also connects their
    # optimization components. Bounding boxes cheaply filter most pairs.
    boxes=[]
    for edge in candidates:
        pts=edge['sampled_xy']
        xs=[a[0] for a in pts];ys=[a[1] for a in pts]
        boxes.append((min(xs),max(xs),min(ys),max(ys)))
    margin=float(cfg.max_path_crossing_px)
    for i,e in enumerate(candidates):
        a=boxes[i]
        for j in range(i):
            if root(e['curve_ids'][0])==root(candidates[j]['curve_ids'][0]):continue
            b=boxes[j]
            if (a[1]+margin<b[0] or b[1]+margin<a[0] or
                a[3]+margin<b[2] or b[3]+margin<a[2]):continue
            if segments_intersect(e,candidates[j],cfg.max_path_crossing_px):
                union(e['curve_ids'][0],candidates[j]['curve_ids'][0])
    byroot=defaultdict(list)
    for e in candidates:byroot[root(e['curve_ids'][0])].append(e)
    byname={c['curve_id']:c for c in curves}
    scored_by_id={}
    grouped=[]
    for k,group in sorted(byroot.items(),key=lambda v:str(v[0])):
        # If every edge interacts through curve identity or crossing, there
        # is no independent decomposition. Return an explicit diagnostic,
        # never recurse indefinitely or silently fall back to greedy links.
        if len(group)==len(candidates) and len(group)>max_candidates:
            raise ValueError(f'{len(group)} interacting candidates in a single conflict component; '
                             'exact limit reached; split ROI using verified source boundaries or review this component')
        names=set(n for edge in group for n in edge['curve_ids'])
        subset=[byname[i] for i in sorted(names)]
        decision_subset={e['candidate_id']:decisions[e['candidate_id']]
                         for e in group if e['candidate_id'] in decisions}
        result,meta=solve_exact(subset,group,cfg,decision_subset,max_candidates)
        grouped.append(meta)
        for e in result:scored_by_id[e['candidate_id']]=e
    scored=[scored_by_id[e['candidate_id']] for e in candidates]
    stats={'raw_proposals':len(candidates),
           'algorithmic_group_links':sum(x['algorithmic_group_links'] for x in grouped),
           'expert_confirmed_group_links':sum(x['expert_confirmed_group_links'] for x in grouped),
           'ambiguous_endpoints':sum(x['ambiguous_endpoints'] for x in grouped),
           'auto_candidates_considered':sum(x['auto_candidates_considered'] for x in grouped),
           'exact_branch_and_bound_nodes':sum(x['exact_branch_and_bound_nodes'] for x in grouped),
           'solver_type':'EXACT_CONFLICT_DECOMPOSED_INDEPENDENT_GROUPS',
           'independent_components':len(grouped),
           'objective_weight':round(sum(x['objective_weight'] for x in grouped),6),
           'enforced_constraints':['one-edge-per-endpoint','no-curve-cycle','no-spatially-crossing-connectors'],
           'no_confirmed_physical_tracks_without_expert_review':True}
    return scored,stats


def solve_exact(curves, candidates, cfg:GlobalConfig=GlobalConfig(), decisions=None, max_candidates=28):
    decisions={str(k):str(v) for k,v in (decisions or {}).items()}
    if set(decisions.values())-{'same','different','uncertain'}:
        raise ValueError('Review decisions must be same, different or uncertain')
    ids={c['curve_id'] for c in curves}
    byid={e['candidate_id']:e for e in candidates}
    if len(byid)!=len(candidates):raise ValueError('Duplicate candidate IDs')
    if set(decisions)-set(byid):raise ValueError('Expert decision refers to unknown candidate')
    byport=defaultdict(list)
    for e in candidates:
        if len(e['ports'])!=2 or len(e['curve_ids'])!=2:
            raise ValueError('Invalid edge ports')
        if set(e['curve_ids'])-ids:raise ValueError('Unknown curve in candidate edge')
        for p in e['ports']:byport[tuple(p)].append(e)
    ambiguous=set()
    for port,links in byport.items():
        weights=sorted((e['score'] for e in links if decisions.get(e['candidate_id'])!='different'),reverse=True)
        if len(weights)>1 and weights[0]-weights[1]<cfg.endpoint_ambiguity_margin:ambiguous.add(port)
    forced=sorted((byid[k] for k,v in decisions.items() if v=='same'),key=lambda x:x['candidate_id'])
    active=[]
    for e in candidates:
        cid=e['candidate_id']
        if cid in {x['candidate_id'] for x in forced}:continue
        if not e.get('auto_eligible',True):continue
        if decisions.get(cid) in ('different','uncertain'):continue
        if e['score']<cfg.auto_candidate_score:continue
        if any(tuple(p) in ambiguous for p in e['ports']):continue
        active.append(e)
    # Branch-and-bound for the small, evidence-filtered set; an enormous set
    # must be split or re-reviewed, not silently reduced to a greedy fallback.
    if len(active)>max_candidates:
        # The full H50Z frame can have hundreds of independent candidate
        # groups. Do not fail or switch to greedy: exactly optimize each
        # noninteracting curve/crossing component, preserving global optimum.
        return _solve_independent_components(curves,candidates,cfg,decisions,max_candidates)
    active.sort(key=lambda e:(-max(e['score']-cfg.minimum_score,1e-6),e['candidate_id']))
    all_edges=forced+active
    conflict={e['candidate_id']:set() for e in all_edges}
    for j,e in enumerate(all_edges):
        for k in range(j):
            f=all_edges[k]
            same_port=bool(set(map(tuple,e['ports']))&set(map(tuple,f['ports'])))
            cross=segments_intersect(e,f,cfg.max_path_crossing_px) if not same_port else False
            if same_port or cross:
                conflict[e['candidate_id']].add(f['candidate_id'])
                conflict[f['candidate_id']].add(e['candidate_id'])
    parent={i:i for i in ids}
    def root(p,u):
        while p[u]!=u:u=p[u]
        return u
    def insert(p,e):
        c0,c1=e['curve_ids'];r0=root(p,c0);r1=root(p,c1)
        if r0==r1:return None
        new=p.copy();new[r1]=r0
        return new
    chosen_forced=[];port_used=set();parent_forced=parent.copy()
    for e in forced:
        cid=e['candidate_id']
        if any(x['candidate_id'] in conflict[cid] for x in chosen_forced):
            raise ValueError('Expert forced links have conflicting endpoints or crossed paths')
        new=insert(parent_forced,e)
        if new is None:raise ValueError('Expert forced links create a cycle')
        parent_forced=new;chosen_forced.append(e)
    forced_ids={e['candidate_id'] for e in chosen_forced}
    # Weights are positive above threshold. Bound is total remaining positive
    # weights, deliberately ignoring conflicts (an optimistic upper bound).
    weights=[max(float(e['score'])-cfg.minimum_score,1e-6) for e in active]
    suffix=[0.0]*(len(active)+1)
    for i in range(len(active)-1,-1,-1):suffix[i]=suffix[i+1]+weights[i]
    best_total=-1e100;best_ids=();nodes=0
    def dfs(i,p,selected,selected_set,total):
        nonlocal best_total,best_ids,nodes
        nodes+=1
        if total+suffix[i]<best_total-1e-10:return
        if i==len(active):
            key=tuple(sorted(selected))
            if total>best_total+1e-10 or (abs(total-best_total)<=1e-10 and (not best_ids or key<best_ids)):
                best_total=total;best_ids=key
            return
        e=active[i];cid=e['candidate_id']
        if not conflict[cid].intersection(selected_set|forced_ids):
            new=insert(p,e)
            if new is not None:
                dfs(i+1,new,selected+[cid],selected_set|{cid},total+weights[i])
        dfs(i+1,p,selected,selected_set,total)
    dfs(0,parent_forced,[],set(),0.0)
    selected=set(best_ids)|forced_ids
    scored=[]
    for e in candidates:
        rec=dict(e);cid=e['candidate_id']
        if cid in selected:
            rec['decision']='EXPERT_CONFIRMED_SAME' if cid in forced_ids else 'ALGORITHMIC_GROUPING_NOT_PHYSICAL_GT'
        elif decisions.get(cid)=='different':rec['decision']='EXPERT_REJECTED_DIFFERENT'
        elif decisions.get(cid)=='uncertain':rec['decision']='EXPERT_DEFERRED'
        elif not e.get('auto_eligible',True):rec['decision']='REVIEW_S109_LEGACY_NOT_AUTO_ACCEPTED'
        elif any(tuple(p) in ambiguous for p in e['ports']):rec['decision']='ABSTAIN_AMBIGUOUS_END'
        elif e['score']<cfg.auto_candidate_score:rec['decision']='REVIEW_LOW_CONFIDENCE'
        else:rec['decision']='REJECTED_EXACT_GLOBAL_CONFLICT_OR_LOWER_WEIGHT'
        rec['review_status']=decisions.get(cid,'unreviewed')
        scored.append(rec)
    return scored,{
        'raw_proposals':len(candidates),
        'algorithmic_group_links':len(selected)-len(forced_ids),
        'expert_confirmed_group_links':len(forced_ids),
        'ambiguous_endpoints':len(ambiguous),
        'auto_candidates_considered':len(active),
        'exact_branch_and_bound_nodes':nodes,
        'solver_type':'EXACT_CONFLICT_AWARE_GLOBAL_MATCHING',
        'objective_weight':round(best_total,6),
        'enforced_constraints':['one-edge-per-endpoint','no-curve-cycle','no-spatially-crossing-connectors'],
        'no_confirmed_physical_tracks_without_expert_review':True,
    }