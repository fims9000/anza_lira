"""S112: dot-level proposals and curve-anchored bead chains (not physical particle IDs).

A bubble is represented by a subpixel center and evidence, not a painted
track pixel. Curves from S111 are immutable. We only extend their *ends* using
multiple, mutually consistent photographed dark bead centers; never infer a
missing bubble as an observed dot. Extensions and free chains are reviewable
hypotheses. Missing pixels, physical interactions and identities are not GT.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
from math import atan2, degrees
from collections import defaultdict
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, gaussian_laplace, maximum_filter, distance_transform_edt
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev
from .s109_vector_tracks import bezier_from_tck,svg_path


@dataclass(frozen=True)
class BubbleConfig:
    sigmas: tuple[float, ...]=(1.0, 1.7, 2.6)
    local_extrema_px: int=5
    min_log_strength: float=9.0
    min_local_darkness: float=5.0
    min_tangent_coherence: float=.24
    min_isolated_coherence: float=.42
    round_blob_log_strength: float=28.0
    round_blob_min_darkness: float=24.0
    independent_round_log_strength: float=48.0
    independent_round_min_darkness: float=36.0
    max_block_near_px: float=4.0
    photo_distance_px: float=2.5
    minimum_peak_separation_px: float=4.0
    max_detected_peaks: int=17000
    # Curves with S111 evidence serve as initial tangents for linking.
    min_curve_anchor_length_px: float=25.0
    min_endpoint_border_px: float=18.0
    min_step_px: float=4.0
    max_step_px: float=22.0
    first_max_step_px: float=27.0
    max_turn_deg: float=25.0
    max_first_turn_deg: float=20.0
    beam_width: int=8
    max_bead_steps: int=15
    min_chain_beads: int=4
    min_chain_source_fraction: float=.35
    min_chain_length_px: float=28.0
    max_other_curve_proximity_px: float=5.5
    min_rival_chain_margin: float=.12
    min_accepted_evidence_score: float=.55
    # Unanchored proposals are never automatically given a particle identity.
    unanchored_min_beads: int=6
    unanchored_max_step_px: float=17.0
    unanchored_max_turn_deg: float=15.0
    unanchored_min_support_fraction: float=.50
    unanchored_max_proposals: int=75


def _coherence_and_orientation(gray):
    g=cv2.GaussianBlur(gray.astype(np.float32),(0,0),1.2)
    gx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3)
    gy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)
    jxx=cv2.GaussianBlur(gx*gx,(0,0),4.0)
    jyy=cv2.GaussianBlur(gy*gy,(0,0),4.0)
    jxy=cv2.GaussianBlur(gx*gy,(0,0),4.0)
    co=np.sqrt((jxx-jyy)**2+4*jxy*jxy)/(jxx+jyy+1e-3)
    # Eigendirection associated with smaller eigenvalue, i.e. along a line.
    alpha=.5*np.arctan2(2*jxy,jxx-jyy)+np.pi/2
    return co,np.cos(alpha),np.sin(alpha)


def _subpixel_peak(response,y,x):
    h,w=response.shape
    if x<2 or x>=w-2 or y<2 or y>=h-2:return float(x),float(y)
    f0=float(response[y,x]);xm=float(response[y,x-1]);xp=float(response[y,x+1]);ym=float(response[y-1,x]);yp=float(response[y+1,x])
    dx_d=xm-2*f0+xp;dy_d=ym-2*f0+yp
    dx=0.5*(xm-xp)/dx_d if dx_d<-.15 else 0.0
    dy=0.5*(ym-yp)/dy_d if dy_d<-.15 else 0.0
    return round(float(x+np.clip(dx,-.45,.45)),3),round(float(y+np.clip(dy,-.45,.45)),3)


def detect_bubble_centers(gray,source_mask,blocked,cfg:BubbleConfig=BubbleConfig()):
    """Multiscale LoG peaks; image evidence, not expert bubble annotations.

    Explicitly exclude protected crosses, image borders and nearby fiducials.
    Only local maxima (bead center points), not thick binary ribbons, are kept.
    """
    gray=np.asarray(gray)
    if gray.dtype!=np.uint8 or gray.ndim!=2:raise ValueError('source grayscale uint8 required')
    if source_mask.shape!=gray.shape or blocked.shape!=gray.shape:raise ValueError('source/mask dimensions mismatch')
    g=gray.astype(np.float32)
    response=np.maximum.reduce([gaussian_laplace(g,sigma)*sigma**2 for sigma in cfg.sigmas]).astype(np.float32)
    # Distinguish a genuinely dark photographic dot from a grain-contrast peak.
    dark=gaussian_filter(g,6.0)-gaussian_filter(g,1.0)
    co,ux,uy=_coherence_and_orientation(gray)
    photod=distance_transform_edt(~source_mask.astype(bool))
    protected=distance_transform_edt(~blocked.astype(bool))
    round_blob=(response>=cfg.round_blob_log_strength)&(dark>=cfg.round_blob_min_darkness)&(
        (photod<=cfg.photo_distance_px)|
        ((response>=cfg.independent_round_log_strength)&(dark>=cfg.independent_round_min_darkness)))
    # Separate morphological channels: anisotropic thin tracks *or* dark
    # approximately circular bead blobs, whose tensor coherence is near zero.
    m=(response>=cfg.min_log_strength)&(dark>=cfg.min_local_darkness)&(
        (co>=cfg.min_tangent_coherence)|round_blob)
    m&=response==maximum_filter(response,size=cfg.local_extrema_px)
    m&=protected>cfg.max_block_near_px
    m[:5,:]=False;m[-5:,:]=False;m[:,:5]=False;m[:,-5:]=False
    yy,xx=np.where(m)
    if len(xx)>cfg.max_detected_peaks:
        raise ValueError('Too many photo-dot candidates. Abort rather than silently truncating evidence.')
    order=np.lexsort((xx,yy,-response[yy,xx]))
    selected=[]
    occupied=np.zeros_like(m,bool)
    rad=int(np.ceil(cfg.minimum_peak_separation_px))
    # Greedy nonmaximum point suppression prevents duplicate scales/shoulders.
    for t in order:
        x=int(xx[t]);y=int(yy[t])
        if occupied[y,x]:continue
        x0=max(0,x-rad);x1=min(gray.shape[1],x+rad+1)
        y0=max(0,y-rad);y1=min(gray.shape[0],y+rad+1)
        occupied[y0:y1,x0:x1]=True
        px,py=_subpixel_peak(response,y,x)
        if (photod[y,x]>cfg.photo_distance_px and co[y,x]<cfg.min_isolated_coherence
            and not round_blob[y,x]):continue
        source=bool(photod[y,x]<=cfg.photo_distance_px)
        # Evidence score is not a calibrated probability of being a bubble.
        strength=float(response[y,x]);contrast=float(dark[y,x]);orient=float(co[y,x])
        score=float(np.clip(.28*np.clip((strength-7)/24,0,1)+.26*np.clip((contrast-4)/26,0,1)+
               .28*np.clip((orient-.18)/.6,0,1)+.18*float(source),0,1))
        selected.append({'point_id':len(selected),'x':px,'y':py,'response':round(strength,2),
                         'morphology_channel':'ROUND_BLOB' if round_blob[y,x] and co[y,x]<cfg.min_tangent_coherence else 'ORIENTED_STREAK',
                         'dark_contrast':round(contrast,2),'coherence':round(orient,3),
                         'tangent':[round(float(ux[y,x]),4),round(float(uy[y,x]),4)],
                         'near_S88_source_mask':source,'mask_distance_px':round(float(photod[y,x]),2),
                         'evidence_score':round(score,4),'physical_bubble_confirmed':False})
    selected.sort(key=lambda p:(p['y'],p['x']))
    for i,p in enumerate(selected):p['point_id']=i
    report={'raw_peaks':int(m.sum()),'candidate_dark_centers':len(selected),
            'near_S88_source_mask':sum(p['near_S88_source_mask'] for p in selected),
            'far_from_S88_source_mask':sum(not p['near_S88_source_mask'] for p in selected),
            'source_mask_unchanged':True,'not_expert_dot_ground_truth':True}
    return selected,report


def _norm(v):
    n=float(np.linalg.norm(v))
    return np.asarray(v,np.float64)/max(n,1e-9)


def _endpoint_outward(curve,end):
    q=np.asarray(curve['sampled_xy'],np.float64)
    k=max(1,min(9,len(q)//8))
    return _norm(q[0]-q[k] if end==0 else q[-1]-q[-1-k])


def _sample_dark(gray,xy):
    p=np.asarray(xy,dtype=np.float32)
    if len(p)==0:return np.empty(0)
    bg=cv2.GaussianBlur(gray.astype(np.float32),(0,0),6.)-cv2.GaussianBlur(gray.astype(np.float32),(0,0),1.)
    return cv2.remap(bg,p[:,0].reshape(-1,1),p[:,1].reshape(-1,1),cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT,borderValue=0).reshape(-1)


def _make_kdtree(points):
    xy=np.array([[z['x'],z['y']] for z in points],dtype=np.float64).reshape(-1,2)
    return xy,cKDTree(xy) if len(xy)>0 else None


def _foreign_curve_distance(curves,own_id):
    pts=[]
    for c in curves:
        if c['curve_id']==own_id:continue
        q=np.asarray(c['sampled_xy'],np.float64)
        # Keep density reasonable for tree searches.
        pts.extend(q[::max(1,len(q)//600)])
    return cKDTree(pts) if pts else None


def _evaluate_chain(points,coords,ids,anchor,initial_dir,cfg):
    xy=np.vstack([anchor,coords[np.asarray(ids,dtype=int)]])
    length=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum())
    photo=np.mean([points[i]['near_S88_source_mask'] for i in ids])
    scores=np.array([points[i]['evidence_score'] for i in ids])
    # Penalize sharp bends beyond smooth spirals, and long jumps.
    v=np.diff(xy,axis=0);vec=v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-6)
    cos=np.einsum('ij,ij->i',vec[:-1],vec[1:]) if len(vec)>1 else np.array([1.0])
    direction=float(np.mean(np.clip(cos,0,1)))
    step=float(np.max(np.linalg.norm(v,axis=1)))
    quality=float(.46*np.mean(scores)+.27*photo+.27*direction)
    return quality,length,float(photo),float(direction),step


def trace_from_anchor(anchor, tangent, all_points, xy, tree, assigned_kdtree=None,
                      cfg:BubbleConfig=BubbleConfig()):
    """Curvature-constrained *beam-search*, with no pixel interpolation as observation.

    Returns best and runner-up paths for ambiguity testing. Different predicted
    directions may branch but are never silently merged into a physical ID.
    """
    if tree is None:return []
    anchor=np.asarray(anchor,np.float64)
    tangent=_norm(tangent)
    beam=[{'ids':(), 'head':anchor,'dir':tangent,'sum':0.0,'turn':0.0}]
    completed=[]
    for step in range(cfg.max_bead_steps):
        nextbeam=[]
        for state in beam:
            nb=tree.query_ball_point(state['head'],cfg.first_max_step_px if step==0 else cfg.max_step_px)
            for j in nb:
                if j in state['ids']:continue
                p=all_points[j];q=xy[j];delta=q-state['head'];dist=float(np.linalg.norm(delta))
                if dist<cfg.min_step_px:continue
                if step>0 and dist>cfg.max_step_px:continue
                direction=delta/dist
                ang=np.degrees(np.arccos(np.clip(np.dot(state['dir'],direction),-1,1)))
                if ang>(cfg.max_first_turn_deg if step==0 else cfg.max_turn_deg):continue
                # Orientation of local structure tensor is sign-invariant.
                ptan=np.asarray(p['tangent'],float)
                if p['coherence']>.47 and abs(np.dot(direction,ptan))<.67:continue
                if assigned_kdtree is not None and assigned_kdtree.query(q)[0]<cfg.max_other_curve_proximity_px:continue
                score=float(p['evidence_score'])+.17*float(p['near_S88_source_mask'])-.006*dist-.00075*ang**2
                ids=state['ids']+(int(j),)
                nstate={'ids':ids,'head':q,'dir':_norm(.6*direction+.4*state['dir']),
                        'sum':state['sum']+score,'turn':state['turn']+ang}
                nextbeam.append(nstate)
                if len(ids)>=cfg.min_chain_beads:
                    quality,length,photo,align,maxstep=_evaluate_chain(all_points,xy,ids,anchor,tangent,cfg)
                    if length>=cfg.min_chain_length_px and photo>=cfg.min_chain_source_fraction and quality>=cfg.min_accepted_evidence_score:
                        completed.append((quality,length,photo,align,ids))
        if not nextbeam:break
        # Keep disjoint *initial directions* where useful, not just greedy 1-step path.
        nextbeam.sort(key=lambda z:(-(z['sum']+len(z['ids'])*.08),z['ids']))
        beam=nextbeam[:cfg.beam_width]
    best_by_initial={}
    for item in completed:
        ids=item[-1]
        key=tuple(ids[:2]) if len(ids)>1 else tuple(ids)
        rank=item[0]+.004*min(item[1],120)
        if key not in best_by_initial or rank>best_by_initial[key][0]:best_by_initial[key]=(rank,item)
    return [u[1] for u in sorted(best_by_initial.values(),key=lambda v:(-v[0],v[1][-1]))[:5]]


def _catmull_hermite_polyline(points,spacing=.7):
    """Smooth observed-peak chain into a *preview*; sampled curve is inferred."""
    q=np.asarray(points,np.float64)
    if len(q)<4:return q
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))]
    if arc[-1]<=0:return q
    try:
        tck,_=splprep([q[:,0],q[:,1]],u=arc/arc[-1],s=1.25*len(q),k=min(3,len(q)-1))
        xx=np.stack(splev(np.linspace(0,1,max(len(q)*2,int(arc[-1]/spacing))),tck),axis=-1)
    except Exception:return q
    return xx


def _serialize_chain_spline(q):
    """Explicit real cubic B-spline parameters for a *review-only* bead chain."""
    q=np.asarray(q,np.float64)
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))]
    if len(q)<4 or arc[-1]<8:return None
    try:
        tck,_=splprep([q[:,0],q[:,1]],u=arc/arc[-1],s=.9*len(q),k=3)
        spans=bezier_from_tck(tck)
        # The spans are identical to the original fitted S109 spline, not a
        # separately approximated interpolating Bezier chain.
        return {'kind':'cubic_parametric_B_spline_inferred_geometry',
                'knots':np.round(tck[0],8).tolist(),
                'coefficients_xy':[np.round(a,8).tolist() for a in tck[1]],
                'degree':int(tck[2]),
                'bezier_cubic_spans':spans,'svg_path':svg_path(spans),
                'not_observed_between_bubbles':True,
                'not_expert_verified_particle_ID':True}
    except (ValueError,TypeError):return None


def extend_curves_from_bubbles(curves, points,blocked,cfg:BubbleConfig=BubbleConfig(),gray=None,source_mask=None):
    """Propose reproducible endpoint extensions without altering S111 curve IDs.

    Competing curves or beams sharing points are demoted to review-only. Only
    locally supported, nonconflicting chains get `AUTO_REVIEW_CANDIDATE` status
    (never an expert-verified same-particle ID).
    """
    if not curves or not points:return [],{'proposals':0,'chain_evidence':0}
    xy,tree=_make_kdtree(points)
    h,w=blocked.shape
    proposed=[]
    for c in curves:
        if not c.get('source_anchored',False) or c.get('spline_arc_length_px',0)<cfg.min_curve_anchor_length_px:continue
        own=c['curve_id']
        foreign=_foreign_curve_distance(curves,own)
        for end in (0,1):
            e=np.array(c['endpoints'][end],dtype=np.float64)
            if min(e[0],e[1],w-e[0],h-e[1])<cfg.min_endpoint_border_px:continue
            t=_endpoint_outward(c,end)
            options=trace_from_anchor(e,t,points,xy,tree,foreign,cfg)
            if not options:continue
            winning=options[0]
            score,length,photo,alignment,ids=winning
            rival=options[1][0] if len(options)>1 else -np.inf
            ambiguous=bool(score-rival<cfg.min_rival_chain_margin and len(options)>1)
            chain=np.vstack([e,xy[np.array(ids,dtype=int)]])
            preview=_catmull_hermite_polyline(chain)
            path_id='bubble_'+sha256((own+str(end)+','.join(map(str,ids))).encode()).hexdigest()[:12]
            status='UNCERTAIN_COMPETING_BEAD_PATH' if ambiguous else 'AUTO_REVIEW_CANDIDATE_NOT_GT'
            # Independent negative checks on the *inferred curve*, not merely
            # on the detected individual bead centers.
            diagnostic={'candidate_path_is_observed_image_line':False}
            if gray is not None and source_mask is not None:
                proxy={'candidate_inferred_spline':_serialize_chain_spline(chain),
                       'source_dot_xy':np.round(xy[np.array(ids,dtype=int)],3).tolist()}
                photo_audit=audit_bead_bridge_against_photo(proxy,gray,source_mask,blocked)
                diagnostic['inferred_path_photo_audit']=photo_audit
                if photo_audit.get('status')!='PHOTO_GAP_CONTRAST_PASS_REVIEW_ONLY':
                    status='UNCERTAIN_PHOTOMETRIC_OR_CURVE_GAP'
            if foreign is not None:
                clearance=float(np.min(foreign.query(preview)[0]))
                diagnostic['min_distance_to_other_S111_curve_px']=round(clearance,3)
                if clearance<8.5:status='UNCERTAIN_NEAR_COMPETING_S111_TRACK'
            indices=np.rint(preview).astype(int)
            indices[:,0]=np.clip(indices[:,0],0,w-1);indices[:,1]=np.clip(indices[:,1],0,h-1)
            invalid_fraction=float(np.mean(blocked[indices[:,1],indices[:,0]]))
            diagnostic['inferred_path_blocked_fraction']=round(invalid_fraction,4)
            if invalid_fraction>.007:status='UNCERTAIN_CROSSES_PROTECTED_REGION'
            proposed.append({'path_id':path_id,'curve_id':own,'anchor_end':int(end),
                         'crossing_and_photo_validation':diagnostic,
                'bubble_ids':list(ids),'bubble_xy':np.round(xy[np.array(ids,dtype=int)],3).tolist(),
                'inferred_spline':_serialize_chain_spline(chain),
                'source_anchor_xy':np.round(e,3).tolist(),
                'preview_spline_xy':np.round(preview,3).tolist(),
                'mean_photo_mask_support':round(photo,4),'mean_geometric_alignment':round(alignment,4),
                'score':round(float(score),5),'competing_second_score':round(float(rival),5) if np.isfinite(rival) else None,
                'length_px':round(length,2),'status':status,
                'observed_photo_dot_positions_only':True,'preview_connection_is_inferred_geometry':True,
                'physical_particle_identity_proven':False})
    # Two tracks may claim the same photographed bubble. Neither is proven.
    occurrences=defaultdict(list)
    for i,prop in enumerate(proposed):
        for k in prop['bubble_ids']:occurrences[k].append(i)
    conflicts={i for indexes in occurrences.values() if len(indexes)>1 for i in indexes}
    for i in conflicts:proposed[i]['status']='UNCERTAIN_CLAIMED_BY_MULTIPLE_TRACKS'
    return proposed,{'endpoints_with_supported_bead_path':len(proposed),
        'auto_review_candidates':sum(x['status']=='AUTO_REVIEW_CANDIDATE_NOT_GT' for x in proposed),
        'uncertain_proposals':sum(x['status']!='AUTO_REVIEW_CANDIDATE_NOT_GT' for x in proposed),
        'total_point_references':sum(len(x['bubble_ids']) for x in proposed),
        'any_bead_as_physical_gt':False,'source_curve_geometry_unchanged':True}


def fit_free_bead_chains(points, curves, blocked, cfg:BubbleConfig=BubbleConfig()):
    """Tracklet discovery on unassigned bead centers, review-only by default.

    Seeds require a local tangent orientation and three collinear nearby dots;
    then beam-search traces the same weakly curved structure. Pure point
    coincidences in grain therefore fail to form a credible long tracklet.
    """
    if len(points)<cfg.unanchored_min_beads:return [],{'candidates':0}
    xy,tree=_make_kdtree(points)
    occupied=[]
    for c in curves:
        q=np.asarray(c['sampled_xy'],float)
        occupied.extend(q[::max(1,len(q)//250)])
    occupied_tree=cKDTree(occupied) if occupied else None
    # Stronger independent candidates, far enough from existing S111 curves.
    potential=[i for i,p in enumerate(points) if p['coherence']>=.55 and p['evidence_score']>=.49
               and p['near_S88_source_mask'] and
               (occupied_tree is None or occupied_tree.query(xy[i])[0]>12)]
    potential=sorted(potential,key=lambda i:(-points[i]['evidence_score'],i))[:500]
    proposals=[];already=set()
    for i in potential:
        if i in already:continue
        p=points[i];dir=np.array(p['tangent'],float)
        options=[]
        for orient in (1,-1):
            # Use the same stringent multi-dot constraints as seeded search.
            opt=trace_from_anchor(xy[i],orient*dir,points,xy,tree,occupied_tree,cfg)
            if opt:options.append(opt[0])
        if not options:continue
        best=max(options,key=lambda a:(a[0]+.004*a[1],a[-1]))
        score,length,photo,alignment,ids=best
        ids=(i,)+tuple(ids)
        if len(ids)<cfg.unanchored_min_beads or photo<cfg.unanchored_min_support_fraction:continue
        if any(j in already for j in ids):continue
        q=xy[list(ids)]
        if len(q)<4:continue
        line=_catmull_hermite_polyline(q)
        name='free_'+sha256(','.join(map(str,ids)).encode()).hexdigest()[:12]
        proposals.append({'path_id':name,'bubble_ids':list(ids),'bubble_xy':np.round(q,3).tolist(),
                 'preview_spline_xy':np.round(line,3).tolist(),'length_px':round(length,2),
                 'inferred_spline':_serialize_chain_spline(q),
                 'mean_photo_mask_support':round(photo,4),'score':round(score,5),
                 'status':'UNANCHORED_BEAD_CHAIN_REVIEW_ONLY',
                 'physical_particle_identity_proven':False})
        already.update(ids)
        if len(proposals)>=cfg.unanchored_max_proposals:break
    return proposals,{'unanchored_review_only_tracklet_count':len(proposals),
                      'no_automatic_physical_identity':True}


def audit_s111_connector_bubbles(edges,points,cfg:BubbleConfig=BubbleConfig()):
    """Independent bead-center evidence on S111 inferred connector paths.

    A dark dot that lies near a previously inferred curve is a photographed
    point candidate, NOT proof both ends are the same physical particle. Every
    S111 decision is preserved; this only prioritizes expert review.
    """
    if not edges:return [],{'old_connections_audited':0,'with_four_or_more_candidate_dots':0}
    xy,ktree=_make_kdtree(points)
    audit=[]
    for edge in edges:
        path=np.asarray(edge.get('sampled_xy',[]),float)
        if len(path)<8 or ktree is None:continue
        stree=cKDTree(path)
        dist,indices=stree.query(xy)
        middle=(indices>int(len(path)*.08))&(indices<int(len(path)*.92))
        center=(dist<=2.5)&middle
        # Make samples spatially distinct; a thick peak cannot count 3 times.
        chosen=np.where(center)[0]
        seen=set()
        keep=[]
        for i in chosen:
            k=int(indices[i]//max(1,int(len(path)/max(1,edge.get('gap_px',40)//5))))
            if k in seen:continue
            seen.add(k);keep.append(int(i))
        num=len(keep)
        support=sum(points[i]['near_S88_source_mask'] for i in keep)
        # Nearby parallel paths are a strong negative control for false matches.
        derivative=np.gradient(path,axis=0)
        d=np.linalg.norm(derivative,axis=1,keepdims=True)
        n=np.column_stack((-derivative[:,1],derivative[:,0]))/np.maximum(d,1e-6)
        side_counts=[]
        for delta in (-10.0,10.0):
            displaced=path+delta*n
            side_tree=cKDTree(displaced)
            sd,si=side_tree.query(xy)
            side_counts.append(int(np.count_nonzero((sd<=2.5)&(si>len(path)*.08)&(si<len(path)*.92))))
        rival=max(side_counts) if side_counts else 0
        label=('SUPPORTS_REVIEW_PRIORITIZATION_NOT_IDENTITY' if num>=4 and
               support>=max(2,.45*num) and num>=1.30*rival else 'INSUFFICIENT_OR_COMPETING_DOT_EVIDENCE')
        audit.append({'s111_edge_id':edge['candidate_id'],'curve_ids':edge['curve_ids'],
             'S111_decision_unchanged':edge['decision'],
             'gap_px':edge.get('gap_px'),'midgap_photometric_dot_count':num,
             'midgap_S88_backed_dot_count':support,'side_control_peak_counts':side_counts,
             'candidate_bubble_ids':keep,'review_priority_label':label,
             'identity_confirmed':False})
    return audit,{'old_connections_audited':len(audit),
        'with_four_or_more_candidate_dots':sum(x['midgap_photometric_dot_count']>=4 for x in audit),
        'source_backed_positive_review_signal':sum(x['review_priority_label'].startswith('SUPPORTS') for x in audit),
        'no_new_physical_ID_declared':True}


def propose_s112_group_bridges(extensions, points, min_shared_beads=3, opposite_cos=-.83):
    """Find review-only candidate joins between *different* S111 groups.

    Both already anchored curves must independently find >=3 identical dark
    photo dot proposals, from opposing directions. Triple-claimed blobs and
    same-direction convergences are not accepted. This does NOT merge particle
    identities; the expert must decide if the two fragments are one track.
    """
    if not extensions:return [],{'opposing_two_group_review_bridges':0}
    byid={p['point_id']:p for p in points}
    membership=defaultdict(set)
    for z in extensions:
        for b in z['bubble_ids']:membership[b].add(z['curve_id'])
    tentative=[]
    import itertools
    for a,b in itertools.combinations(extensions,2):
        ga=a.get('attached_to_s111_global_track_id')
        gb=b.get('attached_to_s111_global_track_id')
        if ga==gb or not ga or not gb:continue
        shared=set(a['bubble_ids'])&set(b['bubble_ids'])
        if len(shared)<min_shared_beads:continue
        pA=np.asarray(a['source_anchor_xy'],np.float64)
        pB=np.asarray(b['source_anchor_xy'],np.float64)
        centers=np.asarray([[byid[i]['x'],byid[i]['y']] for i in sorted(shared)],float)
        mid=centers.mean(axis=0)
        da=_norm(mid-pA);db=_norm(mid-pB)
        opposite=float(np.dot(da,db))
        multiple=bool(any(len(membership[i])>2 for i in shared))
        passed=bool(opposite<=opposite_cos and not multiple and
               a['mean_photo_mask_support']>=.70 and b['mean_photo_mask_support']>=.70)
        # Fit a candidate curve between two anchored ends through photographed
        # *point proposals*, not as a source-backed observed ribbon.
        a_ids=a['bubble_ids']; b_ids=b['bubble_ids']
        pairs=[]
        for id in shared:
            ia=a_ids.index(id);ib=b_ids.index(id)
            # Prefer a meeting point around the median index, robust to local noise.
            pairs.append((abs(ia-.5*(len(a_ids)-1))+abs(ib-.5*(len(b_ids)-1)),id,ia,ib))
        _,chosen,ia,ib=min(pairs)
        qA=np.asarray(a['bubble_xy'][:ia+1],float)
        qB=np.asarray(b['bubble_xy'][:ib],float)
        joined=np.vstack((pA,qA,qB[::-1],pB))
        spline=_serialize_chain_spline(joined)
        if spline is None:passed=False
        key='bridge_'+sha256((min(ga,gb)+max(ga,gb)+str(sorted(shared))).encode()).hexdigest()[:12]
        tentative.append({'bridge_id':key,'curve_ids':[a['curve_id'],b['curve_id']],
                  'source_groups':[ga,gb],'path_ids':[a['path_id'],b['path_id']],
                  'common_bubble_ids':sorted(shared),'shared_dot_count':len(shared),
                  'tangent_opposition_cos':round(opposite,4),
                  'competed_for_by_third_curve':multiple,
                  'candidate_inferred_spline':spline,
                  'source_dot_xy':np.round(centers,3).tolist(),
                  'status':'BEAD_SUPPORTED_INTERGROUP_BRIDGE_REVIEW_ONLY' if passed else
                           'AMBIGUOUS_OR_SAME_DIRECTION_BEAD_JOIN_NOT_AUTO',
                  'physical_IDs_not_merged':True,
                  'source_s111_group_ids_unchanged':True})
    # One curve endpoint can have several plausible links. Keep every one
    # review-only; don't mislabel the highest score as a physical fact.
    used=defaultdict(list)
    for z in tentative:
        for pid in z['path_ids']:used[pid].append(z['bridge_id'])
    for z in tentative:
        if any(len(used[pid])>1 for pid in z['path_ids']):
            z['status']='MULTIPLE_COMPETING_BRIDGES_REVIEW_ONLY'
    return tentative,{'candidate_intergroup_bead_bridges':len(tentative),
        'opposing_two_group_review_bridges':sum(z['status']=='BEAD_SUPPORTED_INTERGROUP_BRIDGE_REVIEW_ONLY' for z in tentative),
        'all_bridge_decisions_review_only':True,'S111_group_ids_immutable':True}


def audit_bead_bridge_against_photo(bridge, gray, source_mask, blocked):
    """Check fitted spline against dark source and independent flanking bands.

    The fitted *curve* remains a hypothesis; even photo-supported bubble points
    do not establish that disconnected fragments belong to one particle.
    """
    spline=bridge.get('candidate_inferred_spline')
    if not spline or not spline.get('bezier_cubic_spans'):
        return {'status':'NO_SPLINE_FOR_REVIEW','verified_particle_ID':False}
    spans=np.asarray(spline['bezier_cubic_spans'],np.float64)
    samples=[]
    for b in spans:
        t=np.linspace(0,1,32,endpoint=False)
        q=((1-t)**3)[:,None]*b[0]+(3*(1-t)**2*t)[:,None]*b[1]+(3*(1-t)*t*t)[:,None]*b[2]+(t**3)[:,None]*b[3]
        samples.extend(q)
    samples.append(spans[-1,-1,:])
    path=np.asarray(samples,dtype=np.float32)
    h,w=gray.shape
    in_bounds=(path[:,0]>2)&(path[:,0]<w-2)&(path[:,1]>2)&(path[:,1]<h-2)
    if not np.all(in_bounds):return {'status':'OUT_OF_BOUNDS_NO_VALIDATED_SPLINE'}
    d=np.gradient(path,axis=0)
    normal=np.column_stack((-d[:,1],d[:,0]))/np.maximum(np.linalg.norm(d,axis=1,keepdims=True),1e-6)
    g=gray.astype(np.float32)
    dark=cv2.GaussianBlur(g,(0,0),6.)-cv2.GaussianBlur(g,(0,0),1.)
    srcdist=distance_transform_edt(~np.asarray(source_mask,bool)).astype(np.float32)
    forbid=blocked.astype(np.uint8)
    def remap(a,q):
        return cv2.remap(a.astype(np.float32),q[:,0].reshape(-1,1),q[:,1].reshape(-1,1),
                   cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0).reshape(-1)
    center_dark=remap(dark,path)
    side_dark=[]
    for step in (-10.,10.):side_dark.append(float(np.mean(remap(dark,path+step*normal))))
    photo_fraction=float(np.mean(remap(srcdist,path)<=2.5))
    blocked_fraction=float(np.mean(remap(forbid,path)>.3))
    dots=np.asarray(bridge['source_dot_xy'],float)
    dotdist=cKDTree(path.astype(float)).query(dots)[0]
    median_offset=float(np.median(dotdist)) if len(dotdist) else np.inf
    max_offset=float(np.max(dotdist)) if len(dotdist) else np.inf
    center_margin=float(np.mean(center_dark)-max(side_dark))
    valid=(photo_fraction>=.38 and center_margin>=2.0 and blocked_fraction<.03 and
           median_offset<=2.5 and max_offset<=4.5)
    return {'status':'PHOTO_GAP_CONTRAST_PASS_REVIEW_ONLY' if valid else 'PHOTO_GAP_CONTRAST_OR_GEOMETRY_REJECTED',
            'source_ridge_fraction':round(photo_fraction,4),
            'mean_dark_minus_best_side':round(center_margin,3),
            'blocked_fraction':round(blocked_fraction,4),
            'median_dot_to_fitted_spline_px':round(median_offset,3),
            'max_dot_to_fitted_spline_px':round(max_offset,3),
            'verified_particle_ID':False}