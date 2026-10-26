"""S40: preserve visualMetro fiducial provenance as uncertainty guards.

A fiducial is a calibration mark, not a certified negative pixel mask.
Spatial risk flags never delete track data or assert native physical identity.
"""
from __future__ import annotations
import re
from typing import Any
import numpy as np
import cv2

SCHEMA='s40_fiducial_spatial_guard_v1'
_FID=re.compile(r'^f[0-9]+$')


def registered_fiducials(labels:dict, photo:str) -> list[dict[str,Any]]:
    if photo not in ('005','006'):
        raise ValueError('H50Z photo must be 005 or 006')
    key=f'H50Z_030_01_{photo}.bmp'
    if key not in labels:
        raise ValueError('Missing source photo labels')
    source=labels[key]
    w=int(source['original_width']); h=int(source['original_height'])
    out=[]; seen=set()
    for item in source['keypoints']:
        name=str(item['label'])
        if not _FID.fullmatch(name):
            continue  # 'start' is a candidate interaction point; not a fiducial.
        if name in seen:
            raise ValueError('Duplicate fiducial in one frame: '+name)
        seen.add(name)
        x=float(item['x_px']);y=float(item['y_px']);width=float(item['width_px'])
        if not (np.isfinite([x,y,width]).all() and 0 <= x < w and 0 <= y < h and 2 <= width <= 150):
            raise ValueError('Fiducial coordinates/size out of bounds')
        out.append({'photo':photo,'label':name,'center_xy_PRIVATE':[x,y],
            'source_marker_width_px':width,
            # Exclusion radius is a conservative REVIEW UNCERTAINTY boundary,
            # not a measured calibration uncertainty and not a particle mask.
            'machine_ambiguity_radius_px':round(max(40.,1.6*width),2),
            'provenance':'H50Z supplied keypoint labels, not predicted particle mask'})
    if not out:
        raise ValueError('No known fiducials found; fail closed')
    return sorted(out,key=lambda r:int(r['label'][1:]))


def infer_fiducial_cross_arms(gray:np.ndarray, center_xy:tuple[float,float]) -> list[float]:
    """Detect the two *calibration X* directions, not particle-track directions.

    Direction estimates arise from image evidence around supplied f-locations;
    missing/ambiguous X evidence returns [], never a fabricated direction.
    Angles are axial (modulo pi) and not calibrated physical field angles.
    """
    if gray.dtype!=np.uint8 or gray.ndim!=2 or min(gray.shape)<256:
        raise ValueError('Expected raw grayscale fiducial crop')
    center=np.asarray(center_xy,dtype=float)
    ed=cv2.Canny(cv2.GaussianBlur(gray,(3,3),.8),50,125)
    lines=cv2.HoughLinesP(ed,1,np.pi/360,threshold=75,minLineLength=75,maxLineGap=18)
    if lines is None:return []
    possible=[]
    for x1,y1,x2,y2 in lines.reshape(-1,4):
        a=np.array([x1,y1],dtype=float);b=np.array([x2,y2],dtype=float)
        v=b-a;norm=np.dot(v,v)
        if norm<75*75:continue
        t=np.clip(np.dot(center-a,v)/norm,0,1)
        near=a+t*v
        if np.linalg.norm(near-center)>22:continue
        angle=float(np.arctan2(v[1],v[0])%np.pi)
        possible.append((float(np.sqrt(norm)),angle))
    if len(possible)<2:return []
    possible.sort(reverse=True)
    first=possible[0][1]
    angular=lambda u,v:abs((u-v+np.pi/2)%np.pi-np.pi/2)
    remaining=[x for x in possible if angular(x[1],first)>np.deg2rad(40)]
    if not remaining:return []
    second=remaining[0][1]
    directions=[]
    for anchor in (first,second):
        around=[(w,ang) for w,ang in possible if angular(ang,anchor)<=np.deg2rad(12)]
        if not around or max(w for w,_ in around)<80:return []
        c=sum(w*np.exp(2j*ang) for w,ang in around)
        directions.append(float((np.angle(c)/2)%np.pi))
    if angular(*directions)<np.deg2rad(40):return []
    return sorted(directions)


def _fiducial_arm_overlap(pts_abs:np.ndarray, mark:dict) -> float:
    angles=mark.get('arms_angles_rad_PRIVATE',[])
    if len(angles)!=2:return 0.
    center=np.asarray(mark['center_xy_PRIVATE'],dtype=float)
    centered=pts_abs-center
    out=[]
    for a in angles:
        d=np.array([np.cos(a),np.sin(a)])
        along=centered@d
        cross=np.abs(centered[:,0]*d[1]-centered[:,1]*d[0])
        overlap=(np.abs(along)<=mark.get('arm_half_span_px',220)) & (cross<=mark.get('arm_tolerance_px',9))
        out.append(float(np.mean(overlap)))
    return max(out)


def _min_distance_to_polyline(center:np.ndarray, pts:np.ndarray) -> float:
    if pts.ndim!=2 or pts.shape[1]!=2 or not len(pts) or not np.isfinite(pts).all():
        raise ValueError('Bad candidate polyline')
    if len(pts)==1:
        return float(np.linalg.norm(center-pts[0]))
    starts=pts[:-1]; vectors=pts[1:]-starts
    length=np.einsum('ij,ij->i',vectors,vectors)
    t=np.clip(np.einsum('ij,ij->i',center[None,:]-starts,vectors)/np.maximum(length,1e-12),0,1)
    near=starts+t[:,None]*vectors
    return float(np.min(np.linalg.norm(near-center[None,:],axis=1)))


def fiducial_guard(polyline:list|np.ndarray, origin_xy:tuple|list, photo:str,
                   zones:list[dict]) -> dict[str,Any]:
    pts=np.asarray(polyline,dtype=np.float64)
    origin=np.asarray(origin_xy,dtype=np.float64)
    if origin.shape!=(2,) or not np.isfinite(origin).all():
        raise ValueError('Bad crop origin')
    absolute=pts+origin
    distances=[]
    arm=[]
    for mark in zones:
        if mark['photo']!=photo:continue
        distance=_min_distance_to_polyline(np.asarray(mark['center_xy_PRIVATE']),absolute)
        if distance<=mark['machine_ambiguity_radius_px']:
            distances.append((mark['label'],round(distance,3)))
        overlap=_fiducial_arm_overlap(absolute,mark)
        if overlap>=.60:
            arm.append((mark['label'],round(overlap,3)))
    return {'fiducial_mark_zone_overlap':bool(distances),
        'fiducial_arm_lookalike':bool(arm),
        'fiducial_uncertainty_any':bool(distances or arm),
        'fiducial_arm_overlaps':[{'label':n,'fraction_of_polyline':v} for n,v in arm],
        'nearby_fiducial_labels':[v[0] for v in distances],
        'min_distance_to_flagged_fiducial_px_PRIVATE':min((v[1] for v in distances),default=None),
        'interpretation':'Review as ambiguous; neither confirm particle nor erase signal'}


def guard_proposals(proposals:list[dict], origin_xy:list, photo:str, zones:list[dict]) -> list[dict]:
    return [{'draft_id':p['draft_id'], **fiducial_guard(p['points_xy'],origin_xy,photo,zones)}
            for p in proposals]


def guard_graph(graph:dict, origin_xy:list, photo:str, zones:list[dict]) ->dict:
    """Annotate S39 graph indices and bridge curves, defaulting to abstention.

    Even edges not near calibration marks remain unverified hypotheses and
    must not be presented as physical same-particle links.
    """
    nodes=guard_proposals(graph['nodes'],origin_xy,photo,zones)
    risky={i for i,x in enumerate(nodes) if x['fiducial_uncertainty_any']}
    raw=graph.get('candidate_edges',graph.get('relations',graph.get('edges',[])))
    chosen={(x.get('i'),x.get('j')) for x in graph.get('unambiguous_ranked_edges_UNVERIFIED',[])}
    out={'nodes':nodes,'candidate_edges':[],
         'guarded_node_count':len(risky),'fiducial_safe_auto_joins':0,
         'guard_applies_only_to_calibration_mark_neighborhoods':True,
         'physically_verified_same_particle_edges':0,
         'preliminary_unambiguous_S39_edges':len(chosen)}
    n_flagged=0;n_selected_flagged=0;n_unknown=0
    for edge in raw:
        i=edge.get('i');j=edge.get('j')
        if not (isinstance(i,int) and isinstance(j,int) and 0<=i<len(nodes) and 0<=j<len(nodes) and i!=j):
            out['candidate_edges'].append({'status':'UNKNOWN_SCHEMA_ABSTAIN'})
            n_unknown+=1
            continue
        bridge=edge.get('hermite_bridge_pixels_PROPOSAL_ONLY',[])
        if bridge:
            bridge_fid=fiducial_guard(bridge,origin_xy,photo,zones)
            bridge_bad=bridge_fid['fiducial_uncertainty_any']
        else:
            bridge_bad=False
        flagged=(i in risky or j in risky or bridge_bad)
        n_flagged+=int(flagged)
        n_selected_flagged+=int(flagged and (i,j) in chosen)
        out['candidate_edges'].append({
            'i':i,'j':j,
            'status':'FIDUCIAL_RISK_ABSTAIN' if flagged else 'MACHINE_UNVERIFIED_NO_AUTOJOIN',
            'fiducial_overlap':bool(flagged),
            'previously_ranked_unambiguous_UNVERIFIED':(i,j) in chosen,
            'bridge_fiducial_risk':bool(bridge_bad),
        })
    out.update({'fiducial_risk_edges':n_flagged,
                'fiducial_risk_among_previously_ranked_edges':n_selected_flagged,
                'unknown_schema_abstain_edges':n_unknown})
    return out
