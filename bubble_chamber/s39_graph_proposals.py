"""S39: reversible Graph-LIRA geometry-only candidate graph (not trained relation head).

Missing real particle identity means all links remain optional hypotheses.
Neither link edges nor smooth Hermite curves are authorized native tracks.
"""
from __future__ import annotations
from itertools import combinations
import cv2,numpy as np
from scipy.spatial import cKDTree
from .s35_fragment_link import optimal_endpoints,hermite_gap


def coverage(a,b,max_distance=4.):
    p=np.asarray(a,dtype=float);q=np.asarray(b,dtype=float)
    return float(np.mean(cKDTree(q).query(p)[0]<=max_distance))


def collapse_near_duplicates(proposals:list[dict],*,max_nodes=20)->list[dict]:
    """NMS on near-identical pixel centerlines, not on physically joined paths."""
    if max_nodes<2 or max_nodes>64:raise ValueError('Invalid graph size')
    for p in proposals:
        xy=np.asarray(p['points_xy'],dtype=float)
        if len(xy)<8 or xy.shape[1]!=2 or not np.isfinite(xy).all():raise ValueError('Invalid polyline')
    ranked=sorted(proposals,key=lambda d:(-float(d['review_priority_NOT_PHYSICS_PROBABILITY']),
                                           -len(d['points_xy']),d['draft_id']))
    kept=[]
    for row in ranked:
        duplicate=None
        for item in kept:
            if (coverage(row['points_xy'],item['points_xy'])>=.75 and
                coverage(item['points_xy'],row['points_xy'])>=.75):
                duplicate=item;break
        if duplicate is not None:
            duplicate['source_candidate_ids'].append(row['draft_id'])
        elif len(kept)<max_nodes:
            kept.append({'draft_id':row['draft_id'],'points_xy':row['points_xy'],
                         'model_priority_NOT_GT':row['review_priority_NOT_PHYSICS_PROBABILITY'],
                         'source_candidate_ids':[row['draft_id']],
                         'source':row['source']})
    return kept


def link_graph(proposals:list[dict],*,max_nodes=20,max_gap=36.,margin=.13)->dict:
    """Rank plausible endpoint continuations and explicitly abstain otherwise."""
    if not 5<=max_gap<=64 or margin<0:raise ValueError('Wrong provisional gap controls')
    nodes=collapse_near_duplicates(proposals,max_nodes=max_nodes)
    links=[]
    for i,j in combinations(range(len(nodes)),2):
        a=np.asarray(nodes[i]['points_xy'],dtype='float64')
        b=np.asarray(nodes[j]['points_xy'],dtype='float64')
        opt=optimal_endpoints(a,b)
        dist=opt['gap']
        if not 3<dist<=max_gap:continue
        ap,bp,ta,tb=(opt['endpoint_a'],opt['endpoint_b'],opt['outward_a'],opt['outward_b'])
        chord=(bp-ap)/max(dist,1e-6)
        ca=float(ta@chord);cb=float(tb@-chord)
        aligned=min(ca,cb)
        if aligned<.25:continue
        mutual=float(ta@-tb)
        # A large ambiguous crossing should not be auto-filled.
        if mutual<.0:continue
        # These are deterministic ranking weights, NOT a learned probability.
        score=(.35*np.exp(-dist/20.)+.25*max(ca,0.)+.25*max(cb,0.)+
               .15*max(mutual,0.))
        path=hermite_gap(ap,bp,ta,tb,n=21)
        links.append({'i':i,'j':j,'gap_px':round(float(dist),3),
                      'tangent_to_gap_a':round(ca,4),'tangent_to_gap_b':round(cb,4),
                      'mutual_tangent_alignment':round(mutual,4),
                      'geometry_rank_UNCALIBRATED':round(float(score),4),
                      'hermite_bridge_pixels_PROPOSAL_ONLY':np.round(path,2).tolist(),
                      'physics_join_authorized':False,'native_relation_truth':'UNKNOWN'})
    links.sort(key=lambda p:-p['geometry_rank_UNCALIBRATED'])
    rivals={i:[] for i in range(len(nodes))}
    for pair in links:
        rivals[pair['i']].append(pair)
        rivals[pair['j']].append(pair)
    unique=[];ambiguous=[]
    for pair in links:
        a,b=pair['i'],pair['j'];score=pair['geometry_rank_UNCALIBRATED']
        def has_margin(vertex):
            others=[r['geometry_rank_UNCALIBRATED'] for r in rivals[vertex] if r is not pair]
            return not others or score-max(others)>=margin
        if has_margin(a) and has_margin(b):
            unique.append(pair)
        else:
            ambiguous.append(pair)
    # Select ranked edge set with mutual competition (no double assigned nodes)
    selected=[];used=set()
    for e in unique:
        if e['i'] not in used and e['j'] not in used:
            selected.append(e);used|={e['i'],e['j']}
    return {'schema':'s39_geometry_link_graph_proposals_v1',
            'nodes':nodes,'candidate_edges':links,
            'unambiguous_ranked_edges_UNVERIFIED':selected,
            'ambiguous_edges_abstained':len(ambiguous),
            'conflicting_edges_abstained':len(unique)-len(selected),
            'all_edges_require_future_independent_physics_validation':True,
            'physics_auto_links_performed':0}


def render_graph(gray:np.ndarray,graph:dict)->np.ndarray:
    im=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    for node in graph['nodes']:
        path=np.rint(np.asarray(node['points_xy'])).astype('int32')
        cv2.polylines(im,[path],False,(80,220,120),1,lineType=cv2.LINE_AA)
    for pair in graph['unambiguous_ranked_edges_UNVERIFIED'][:5]:
        pts=np.rint(np.asarray(pair['hermite_bridge_pixels_PROPOSAL_ONLY'])).astype('int32')
        # Draw dashed hypothesis, never solid reconstructed particles.
        for i in range(0,len(pts)-1,3):
            cv2.line(im,tuple(pts[i]),tuple(pts[min(i+1,len(pts)-1)]),(210,60,210),2,cv2.LINE_AA)
    return im
