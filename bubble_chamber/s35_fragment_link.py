"""S35: abstaining fragment-link suggestions from geometry + photo + original AZUNet.

This module never authorizes a physical particle join. Source photometry and
AZUNet activations are uncalibrated evidence, not physicist-certified labels.
"""
from __future__ import annotations
import cv2
import numpy as np
from scipy.spatial.distance import cdist
from .s33_photometric_domain import image_input

GEOMETRY_FIELDS=(
    'endpoint_gap_px', 'outgoing_tangent_A_to_gap', 'outgoing_tangent_B_to_gap',
    'outgoing_tangent_mutual', 'path_length_ratio', 'chord_curvature_consistency',
)
PHOTO_FIELDS=(
    'raw_endpoint_contrast_diff', 'raw_endpoint_contrast_min',
    'local_endpoint_contrast_diff', 'local_endpoint_contrast_min',
    'local_bridge_center_flank', 'quantile_bridge_center_flank',
)
ANZA_FIELDS=('ANZA_endpoint_contrast_diff', 'ANZA_endpoint_contrast_min',
             'ANZA_bridge_center_flank')


def _assert_polyline(pts):
    a=np.asarray(pts,dtype='float64')
    if a.ndim!=2 or a.shape[1]!=2 or len(a)<8 or not np.isfinite(a).all():
        raise ValueError('Invalid candidate 2D polyline')
    if np.any(a<0) or np.any(a>512):
        raise ValueError('Unbounded source polyline')
    if np.linalg.norm(a[-1]-a[0])<4:
        raise ValueError('Degenerate 2D polyline')
    return a


def _unit(v):
    n=float(np.linalg.norm(v))
    return v/max(n,1e-8)


def _endpoint(pts, index:int):
    """Outward tangent from interior to chosen endpoint (no orientation leak)."""
    if index==0:
        point=pts[0];inside=pts[min(5,len(pts)-1)]
    else:
        point=pts[-1];inside=pts[max(len(pts)-6,0)]
    return point,_unit(point-inside)


def optimal_endpoints(path_a,path_b):
    a=_assert_polyline(path_a);b=_assert_polyline(path_b)
    options=[]
    for i in (0,-1):
        ap,at=_endpoint(a,i)
        for j in (0,-1):
            bp,bt=_endpoint(b,j)
            dist=float(np.linalg.norm(bp-ap))
            # Choose endpoint pair by distance, breaking ties by directional match.
            chord=_unit(bp-ap)
            align=(float(np.dot(at,chord))+float(np.dot(bt,-chord)))/2
            options.append((dist,-align,i,j,ap,bp,at,bt))
    gap,_,ai,bi,ap,bp,at,bt=min(options,key=lambda x:(x[0],x[1]))
    return {'gap':gap,'endpoint_a':ap,'endpoint_b':bp,'outward_a':at,'outward_b':bt,
            'endpoint_index_a':ai,'endpoint_index_b':bi}


def hermite_gap(a,b,ta,tb,n=21):
    """Smooth bridge hypothesis between endpoints; never a recovered physical line."""
    d=float(np.linalg.norm(b-a))
    u=np.linspace(0,1,n)
    h00=2*u**3-3*u**2+1;h10=u**3-2*u**2+u
    h01=-2*u**3+3*u**2;h11=u**3-u**2
    # tb points OUT from second fragment toward first; end derivative = -tb.
    return (h00[:,None]*a + h10[:,None]*(.5*d*ta) +
            h01[:,None]*b + h11[:,None]*(-.5*d*tb))


def _photo_score(map2d,points):
    p=np.asarray(points,float)
    tangent=np.gradient(p,axis=0)
    size=np.maximum(np.linalg.norm(tangent,axis=1),1e-8)
    normals=np.stack([-tangent[:,1],tangent[:,0]],axis=1)/size[:,None]
    h,w=map2d.shape
    keep=(p[:,0]>7)&(p[:,0]<w-8)&(p[:,1]>7)&(p[:,1]<h-8)
    if np.count_nonzero(keep)<5:return 0.
    p=p[keep];normals=normals[keep]
    def sample(coords):
        return cv2.remap(map2d.astype('float32'),coords[:,0].astype('float32')[None],
                         coords[:,1].astype('float32')[None],
                         interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)[0]
    center=sample(p)
    flank=(sample(p+6*normals)+sample(p-6*normals))/2
    return float(np.median(center-flank))


def prepare_evidence(gray, *, anza_map=None):
    if gray.ndim!=2 or gray.dtype!=np.uint8 or min(gray.shape)<96:
        raise ValueError('Need authentic grayscale photographic pixels')
    source={'raw':image_input(gray,'raw'),
            'local':image_input(gray,'local'),
            'quantile':image_input(gray,'quantile')}
    if anza_map is not None:
        a=np.asarray(anza_map,dtype=np.float32)
        if a.shape!=gray.shape or not np.isfinite(a).all() or a.min()<0 or a.max()>1:
            raise ValueError('Untrusted AZUNet score grid')
        source['anza']=a
    return source


def pair_features(path_a,path_b,evidence):
    a=_assert_polyline(path_a);b=_assert_polyline(path_b)
    conn=optimal_endpoints(a,b);gap=conn['gap']
    pA,pB,ta,tb=[conn[k] for k in ('endpoint_a','endpoint_b','outward_a','outward_b')]
    chord=_unit(pB-pA)
    cosA=float(np.dot(ta,chord));cosB=float(np.dot(tb,-chord));mutual=float(np.dot(ta,-tb))
    al=float(np.linalg.norm(np.diff(a,axis=0),axis=1).sum())
    bl=float(np.linalg.norm(np.diff(b,axis=0),axis=1).sum())
    length_similarity=min(al,bl)/max(al,bl,1e-8)
    approx=hermite_gap(pA,pB,ta,tb)
    # Curvature proxy = how much Hermite bends away from straight-line chord.
    chordline=np.linspace(pA,pB,len(approx));deviation=float(np.max(np.linalg.norm(approx-chordline,axis=1)))
    result={'endpoint_gap_px':gap,
            'outgoing_tangent_A_to_gap':cosA,
            'outgoing_tangent_B_to_gap':cosB,
            'outgoing_tangent_mutual':mutual,
            'path_length_ratio':length_similarity,
            'chord_curvature_consistency':float(1/(1+deviation/max(gap,1e-8))),
            'endpoints':{'a':pA.tolist(),'b':pB.tolist()}}
    # Only sample 16 points nearest the putative gap; source orientation agnostic.
    pa=a[-20:] if conn['endpoint_index_a']==-1 else a[:20]
    pb=b[-20:] if conn['endpoint_index_b']==-1 else b[:20]
    for key,prefix in [('raw','raw'),('local','local'),('quantile','quantile'),('anza','ANZA')]:
        if key not in evidence:continue
        A=_photo_score(evidence[key],pa)
        B=_photo_score(evidence[key],pb)
        if prefix=='quantile':
            result['quantile_bridge_center_flank']=_photo_score(evidence[key],approx)
            continue
        result[f'{prefix}_endpoint_contrast_diff']=abs(A-B)
        result[f'{prefix}_endpoint_contrast_min']=min(A,B)
        if key in ('local','anza'):
            result[f'{prefix}_bridge_center_flank']=_photo_score(evidence[key],approx)
    result['source_evidence_status']='UNVERIFIED_MODEL_AND_PHOTOMETRY'
    return result


def feature_matrix(rows,fields):
    if isinstance(fields,(str,bytes)) or not isinstance(fields,(tuple,list)) or not fields:
        raise ValueError('Feature fields must be a nonempty list/tuple of known names')
    if any(not isinstance(f,str) for f in fields):raise ValueError('Invalid feature key')
    if any(any(f not in r for f in fields) for r in rows):raise ValueError('Missing feature fields')
    x=np.asarray([[float(r[k]) for k in fields] for r in rows],dtype=np.float64)
    if x.ndim!=2 or len(x)==0 or not np.isfinite(x).all():
        raise ValueError('Incomplete finite pair evidence')
    return x


def calibrated_negative_threshold(scores,labels,*,max_fpr=.02):
    """Set threshold using CALIBRATION negatives only, no test-label leakage.

    Strictly above observed kth negative, conservatively abstains on ties.
    This empirical threshold is NOT a statistical guarantee on new photos.
    """
    s=np.asarray(scores,dtype=float);y=np.asarray(labels,int)
    if s.shape!=y.shape or not np.isfinite(s).all() or set(np.unique(y))!={0,1}:
        raise ValueError('Need both synthetic classes and finite calibration scores')
    if not 0<=max_fpr<=.2:raise ValueError('invalid FPR target')
    neg=np.sort(s[y==0])[::-1]
    allowed=int(np.floor(max_fpr*len(neg)))
    threshold=float(np.nextafter(neg[allowed],np.inf)) if allowed<len(neg) else float('-inf')
    assert int((neg>=threshold).sum())<=allowed
    return threshold


def select_review_links(pair_rows,score,threshold,*,margin=.04,number_left=2,number_right=3):
    """Assign at most one continuation to each fragment, abstaining on ties.

    Output is a set of review suggestions only. Never physics authorization.
    """
    scores=np.asarray(score,float)
    if not np.isfinite(scores).all() or len(scores)!=len(pair_rows):
        raise ValueError('Invalid scored pair count')
    if margin<0:raise ValueError('Invalid ambiguity margin')
    lookup={(int(row['left_index']),int(row['right_index'])):float(v) for row,v in zip(pair_rows,scores)}
    if len(lookup)!=len(pair_rows):raise ValueError('Duplicated pair candidates')
    for (i,j) in lookup:
        if not 0<=i<number_left or not 0<=j<number_right:raise ValueError('Bad pair index')
    # Conservative mutual-best; no forced Hungarian match, no ties.
    tentative=[]
    for left in range(number_left):
        candidates=sorted([(v,j) for (i,j),v in lookup.items() if i==left],reverse=True)
        if not candidates:continue
        top,j=candidates[0]
        if top<threshold or len(candidates)>1 and top-candidates[1][0]<margin:
            continue
        rivals=[v for (i,k),v in lookup.items() if k==j and i!=left]
        if rivals and top-max(rivals)<margin:continue
        tentative.append((left,j))
    return {'status':'MACHINE_PROVISIONAL_REVIEW_ONLY','suggested_pairs':tentative,
            'physics_auto_joins_authorized':False,'human_verified_pairs_created':0,
            'abstained_left_fragments':number_left-len(tentative)}
