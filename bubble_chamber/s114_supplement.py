"""S114 ablation: photo-supported tracklets not covered by S111.

Returned candidates are NOT added to S111 physical track identities. S111
curves and their source observations remain immutable. No ground truth claim.
"""
from __future__ import annotations
import numpy as np
import cv2
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates


def _sample(gray,p):
    return cv2.remap(gray.astype(np.float32),p[:,0,None].astype('float32'),
                     p[:,1,None].astype('float32'),cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT).reshape(-1)


def source_center_contrast(gray,xy,offset=5.):
    """Compare spline center brightness with both normal-side bands."""
    p=np.asarray(xy,float)
    if len(p)<3:return 0.
    d=np.gradient(p,axis=0)
    norm=np.linalg.norm(d,axis=1)
    unit=d/np.maximum(norm[:,None],1e-9)
    perp=np.stack((-unit[:,1],unit[:,0]),axis=1)
    inside=(p[:,0]>=offset+1)&(p[:,1]>=offset+1)&(p[:,0]<gray.shape[1]-offset-1)&(p[:,1]<gray.shape[0]-offset-1)
    if inside.sum()<3:return 0.
    p=p[inside];perp=perp[inside]
    return float(((_sample(gray,p+perp*offset)+_sample(gray,p-perp*offset))*.5-_sample(gray,p)).mean())


def estimate_halfdepth_width(gray,xy,maxoffset=14):
    """Median centerline dark-line thickness in raw 8-bit pixels (NOT calibrated)."""
    p=np.asarray(xy,float)[::3]
    if len(p)<3:return 0.
    tangent=np.gradient(p,axis=0)
    tangent=tangent/np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-5)
    normal=np.column_stack((-tangent[:,1],tangent[:,0]))
    offsets=np.arange(-maxoffset,maxoffset+1,dtype=float)
    positions=p[:,None,:]+normal[:,None,:]*offsets[None,:,None]
    vals=map_coordinates(gray.astype(np.float32),[positions[:,:,1],positions[:,:,0]],
                         order=1,mode='nearest')
    bg=np.mean(np.c_[vals[:,:3],vals[:,-3:]],axis=1)
    deficit=bg[:,None]-vals
    maxima=np.maximum(1.,deficit.max(axis=1))
    hits=deficit>(maxima[:,None]*.5)
    middle=maxoffset; widths=[]
    for row in hits:
        if not row[middle]:continue
        a=b=middle
        while a>0 and row[a-1]:a-=1
        while b+1<len(row) and row[b+1]:b+=1
        widths.append(b-a+1)
    return float(np.median(widths)) if widths else 0.


def select_uncovered_tracklets(s114,old_curves,min_length=85.0,
                               min_uncovered_fraction=.60,max_distance=5.0,
                               min_photo_response=15.0,gray=None,
                               min_source_center_contrast=3.0,
                               max_median_line_width_px=8.0):
    cloud=[]
    for c in old_curves:
        if not c.get('source_anchored',False):continue
        p=np.asarray(c['sampled_xy'],float)
        if len(p):cloud.append(p[::max(1,len(p)//300)])
    tree=cKDTree(np.vstack(cloud)) if cloud else None
    chosen=[]
    for t in s114:
        if t['observed_arc_length_px']<min_length:continue
        if t['mean_directional_response']<min_photo_response:continue
        p=np.asarray(t['observed_pixel_path_xy'],float)
        if len(p)<3:continue
        distances=tree.query(p)[0] if tree is not None else np.full(len(p),np.inf)
        missing=float(np.mean(distances>max_distance))
        if missing<min_uncovered_fraction:continue
        extra=dict(t)
        if gray is not None:
            contrast=source_center_contrast(gray,t['spline_xy'])
            if contrast<min_source_center_contrast:continue
            extra['mean_center_vs_side_darkness_8bit']=round(contrast,3)
            width=estimate_halfdepth_width(gray,t['spline_xy'])
            if width>max_median_line_width_px:continue
            extra['median_halfdepth_line_width_px']=round(width,2)
        extra['uncovered_by_S111_fraction']=round(missing,3)
        extra['status']='SOURCE_FIRST_NEW_TRACKLET_REVIEW_ONLY'
        extra['physical_ID_source']='NONE'
        chosen.append(extra)
    return sorted(chosen,key=lambda z:-z['observed_arc_length_px'])


def render_delta(original,old_overlay,new_tracks,marks):
    if old_overlay.shape[:2]!=original.shape:raise ValueError('Input overlays not aligned')
    out=old_overlay.copy()
    for t in new_tracks:
        xy=np.rint(np.asarray(t['spline_xy'],float)).astype(np.int32)
        if len(xy)<2:continue
        cv2.polylines(out,[xy],False,(0,225,255),2,cv2.LINE_AA)
        cv2.circle(out,tuple(np.rint(t['start_A_xy']).astype(int)),5,(40,255,40),-1,cv2.LINE_AA)
        cv2.circle(out,tuple(np.rint(t['end_B_xy']).astype(int)),5,(40,40,255),-1,cv2.LINE_AA)
    for p in marks:
        xy=tuple(map(int,p['xy_local']))
        if p['label'].startswith('f'):
            cv2.circle(out,xy,20,(200,25,200),2,cv2.LINE_AA)
    return out
