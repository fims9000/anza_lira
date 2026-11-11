"""Image-derived cross-like fiducial candidates on low-res FULL-frame previews.

Candidate geometric X's are NOT operator-verified f1..f8 names. Avoid using
this output as ground truth or automatically suppressing a true particle track.
"""
from __future__ import annotations
import cv2,numpy as np


def x_template(radius:int,thickness:int=1)->np.ndarray:
    if radius<6:raise ValueError('radius too small')
    k=2*radius+1
    m=np.full((k,k),210,np.uint8)
    cv2.line(m,(1,1),(k-2,k-2),10,thickness,cv2.LINE_AA)
    cv2.line(m,(k-2,1),(1,k-2),10,thickness,cv2.LINE_AA)
    return cv2.GaussianBlur(m,(3,3),.8)


def detect_cross_candidates(gray:np.ndarray, radii=(10,14,18,22,26),min_score:float=0.52,
                            max_candidates:int=70)->list[dict]:
    if gray.ndim!=2 or gray.dtype!=np.uint8:raise ValueError('expect uint8 gray')
    b=cv2.GaussianBlur(gray,(3,3),.6)
    candidates=[]
    for r in radii:
        for thick in (1,2):
            templ=x_template(r,thick)
            response=cv2.matchTemplate(b,templ,cv2.TM_CCOEFF_NORMED)
            peaked=cv2.dilate(response,np.ones((max(3,r//4),max(3,r//4)),np.uint8))
            yy,xx=np.where((response>=min_score)&(response>=peaked-1e-7))
            for y,x in zip(yy,xx):
                cx=int(x+r);cy=int(y+r)
                if cx<25 or cy<25 or cx>=gray.shape[1]-25 or cy>=gray.shape[0]-25:continue
                # Strong four-armed image evidence: mean on diagonal strokes is
                # darker than side/control locations at similar distance.
                steps=np.round(np.linspace(.35*r,.85*r,7)).astype(int)
                arm=[];side=[]
                for d in steps:
                    for dx,dy in [(d,d),(-d,-d),(-d,d),(d,-d)]:
                        arm.append(float(b[cy+dy,cx+dx]))
                    for dx,dy in [(d,0),(-d,0),(0,d),(0,-d)]:
                        side.append(float(b[cy+dy,cx+dx]))
                contrast=float(np.mean(side)-np.mean(arm))
                if contrast<5.5:continue
                candidates.append({'x':cx,'y':cy,'radius':r,'score':round(float(response[y,x]),4),
                                   'cross_to_side_contrast':round(contrast,2),
                                   'operator_fiducial_id':None,'physical_fiducial_gt':False})
    candidates.sort(key=lambda z:(-z['score'],-z['cross_to_side_contrast'],z['x'],z['y']))
    kept=[]
    for c in candidates:
        if any(np.hypot(c['x']-z['x'],c['y']-z['y'])<min(c['radius'],z['radius'])*.9 for z in kept):continue
        kept.append(c)
        if len(kept)>=max_candidates:break
    return kept


def draw_cross_candidates(img:np.ndarray,points:list[dict])->np.ndarray:
    out=img.copy()
    if out.ndim==2:out=cv2.cvtColor(out,cv2.COLOR_GRAY2BGR)
    for idx,c in enumerate(points,1):
        x=int(c['x']);y=int(c['y']);r=int(c['radius'])
        cv2.circle(out,(x,y),r,(225,35,225),1,cv2.LINE_AA)
        cv2.putText(out,f'X?{idx}',(x+r+2,y-2),cv2.FONT_HERSHEY_SIMPLEX,.36,(230,40,230),1,cv2.LINE_AA)
    return out